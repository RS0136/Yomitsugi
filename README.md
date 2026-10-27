# Yomitsugi

Simulation code for coding cycles and population persistence in a self-interpreting artificial chemistry.

Requires Python 3.12 and Bash (Linux, macOS, or WSL). Run from this directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
YOMITSUGI_PYTHON="$PWD/.venv/bin/python" bash run_all.sh --force
```

Results are written to `results/` and `figures/`.
Replace `--force` with `--resume` to reuse compatible completed simulations.
