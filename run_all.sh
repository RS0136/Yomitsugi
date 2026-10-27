#!/usr/bin/env bash
# Generate every study from source and configuration; no saved data is required.
set -euo pipefail

case "${1:---resume}" in
  --resume) population_args=(); cycle_args=() ;;
  --force) population_args=(--no-resume); cycle_args=(--force) ;;
  *) printf '%s\n' 'Usage: bash run_all.sh [--resume|--force]' >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
  printf '%s\n' 'Usage: bash run_all.sh [--resume|--force]' >&2
  exit 2
fi

task_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$task_root"
task_python="${YOMITSUGI_PYTHON:-python3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$task_root/results/validation/matplotlib_cache"
mkdir -p results/validation "$MPLCONFIGDIR"

"$task_python" - <<'PY'
from pathlib import Path
import hashlib, importlib, io, json, platform, sys, time, unittest

root = Path.cwd()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'vendor'))
dependencies = {}
for name in ('numpy', 'scipy', 'pandas', 'matplotlib', 'mpmath'):
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        raise SystemExit('Missing dependency. Run: python3 -m pip install -r requirements.txt') from exc
    dependencies[name] = module.__version__

source_hashes = {
    str(source.relative_to(root)): hashlib.sha256(source.read_bytes()).hexdigest()
    for folder in ('analysis', 'scripts', 'tests', 'vendor')
    for source in sorted((root / folder).rglob('*.py'))
}
config_hashes = {
    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((root / 'config').glob('*.json'))
}

log = io.StringIO()
suite_records = []
for label, folder in (('original', root/'vendor/original_tests'), ('publication', root/'tests')):
    started = time.monotonic()
    suite = unittest.TestLoader().discover(str(folder), pattern='test_*.py')
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    suite_records.append(dict(suite=label, tests=result.testsRun, failures=len(result.failures),
                              errors=len(result.errors), skipped=len(result.skipped),
                              passed=result.wasSuccessful(), elapsed_seconds=time.monotonic()-started))

text = log.getvalue()
print(text)
record = dict(python=sys.version, platform=platform.platform(), dependencies=dependencies,
              source_sha256=source_hashes, config_sha256=config_hashes, suites=suite_records,
              all_passed=all(s['passed'] for s in suite_records))
(root/'results/validation/tests.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
if not record['all_passed']:
    raise SystemExit(1)
PY

"$task_python" -m analysis.finite
"$task_python" -m analysis.crossing
"$task_python" -m analysis.original_population --workers 2 "${cycle_args[@]}"
"$task_python" -m analysis.population_extension --workers 2 "${population_args[@]}"
"$task_python" -m analysis.permutation_dynamics
"$task_python" -m analysis.relabeling_diagnostic
"$task_python" scripts/verify_relabeling_witnesses.py
"$task_python" -m analysis.persistence_thresholds
"$task_python" -m analysis.resource_population --workers 2 "${population_args[@]}"
"$task_python" -m analysis.cycle_population --workers 2 "${cycle_args[@]}"
"$task_python" -m analysis.intervention_revision --workers 2 "${population_args[@]}"
"$task_python" -m analysis.editorial_summaries
"$task_python" -m analysis.figures
"$task_python" -m analysis.revision_figures
"$task_python" -m analysis.response_figures
printf '%s\n' 'Completed: scientific tests, all simulations from source, numerical results, and figures.'
