"""Generate the original population study from its settings and recorded seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from yomitsugi.population import (atomic_json, default_config, run_population,
                                  source_hash, stable_hash)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_paths(output, run_id):
    """The current run's files, excluding other historical run directories."""
    return [output / name for name in ("runs.csv", "summary.csv", "contrasts.csv", "latest.json")] + sorted(
        path for path in (output / run_id).rglob("*") if path.is_file()
    )


def reusable(output, signature, run_id):
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf8"))
        if (saved.get("signature") != signature or saved.get("run_id") != run_id
                or saved.get("all_completed") is not True):
            return False
        current = artifact_paths(output, run_id)
        expected = saved["artifacts"]
        if {str(path.relative_to(output)) for path in current} != set(expected):
            return False
        if not all(path.is_file() and sha256(path) == expected[str(path.relative_to(output))]
                   for path in current):
            return False
        latest = json.loads((output / "latest.json").read_text(encoding="utf8"))
        return latest.get("all_completed") is True and latest.get("run_id") == run_id
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(root=ROOT, workers=2, force=False):
    """Run all planned replicates, or reuse a complete byte-verified study."""
    if workers not in (1, 2):
        raise ValueError("Use one or two workers")
    root = Path(root).resolve()
    config_path = root / "config/original_population.json"
    config = json.loads(config_path.read_text(encoding="utf8"))
    settings = default_config("main")
    settings.update(config)
    config_hash, code_hash = stable_hash(settings), source_hash()
    signature = dict(config_sha256=sha256(config_path), config_hash=config_hash,
                     code_hash=code_hash, generator_sha256=sha256(__file__))
    run_id = f"population_main_{config_hash[:12]}_{code_hash[:12]}"
    output = root / "results/original_population"
    if not force and reusable(output, signature, run_id):
        result = json.loads((output / "latest.json").read_text(encoding="utf8"))
        return dict(result, cached=True)

    # The legacy runner resumes individual checkpoints without checking every
    # trajectory. Rebuild the whole study if the wrapper's cache check fails.
    result = run_population(output, config=config, profile="main", resume=False, workers=workers)
    if (not result["all_completed"] or result["run_id"] != run_id
            or result["config_hash"] != config_hash or result["code_hash"] != code_hash):
        raise RuntimeError("Original population study did not complete: "
                           + json.dumps(result.get("termination_counts", {}), sort_keys=True))
    artifacts = {str(path.relative_to(output)): sha256(path)
                 for path in artifact_paths(output, run_id)}
    trajectories = {name: digest for name, digest in artifacts.items()
                    if name.endswith("/trajectory.csv")}
    expected_runs = len(settings["conditions"]) * len(settings["modes"]) * settings["repeats"]
    if len(trajectories) != expected_runs or result["task_count"] != expected_runs:
        raise RuntimeError("Original population trajectory count differs from the plan")
    atomic_json(output / "manifest.json", dict(
        signature=signature, run_id=run_id, all_completed=True,
        config_path="config/original_population.json", task_count=expected_runs,
        artifacts=artifacts, trajectories=trajectories,
    ))
    return dict(result, cached=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true")
    mode.add_argument("--resume", action="store_true", help="Reuse a complete verified run (default)")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.root, arguments.workers, arguments.force), indent=2))
