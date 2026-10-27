"""Run-level bootstrap summaries for the finite-resource composition figures.

Resampling units are independent trajectory means, never individual sweeps.
Existing scale-one/four width-0.8 primary summaries are reused exactly so the
main, supplement and individual-run figure report the same estimates.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESAMPLES = 20000


def run():
    output = ROOT / "results/editorial_summaries"
    output.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / "results/population_extension/finite_run_windows.csv",
              ROOT / "results/population_extension/original_run_windows.csv",
              ROOT / "results/intervention_revision/finite_size_mixture_summary.csv"]
    existing = pd.read_csv(inputs[2])
    rows = []
    for cohort, path in zip(("validation", "cohort"), inputs[:2]):
        data = pd.read_csv(path)
        if cohort == "cohort":
            data = data[data["mode"] == "single_a"].copy()
            data["scale"] = 1
        for (sigma, scale, start), group in data.groupby(["sigma", "scale", "window_start"], sort=True):
            for metric in ("reference_fraction_zero_filled", "reference_fraction_alive"):
                values = group[metric].dropna().to_numpy()
                key = f"{cohort}/{sigma:.12g}/{scale}/{start}/{metric}"
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                boot = values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))].mean(axis=1)
                low, high = np.quantile(boot, [.025, .975])
                origin = "run-level percentile bootstrap"
                if (cohort == "validation" and np.isclose(sigma, .8) and
                        start == 501 and metric == "reference_fraction_zero_filled"):
                    row = existing[existing.scale == scale].iloc[0]
                    low, high = row.bootstrap95_low, row.bootstrap95_high
                    seed = int(row.bootstrap_seed)
                    origin = "same saved summary as the full run-distribution analysis"
                rows.append(dict(cohort=cohort, sigma=sigma, scale=int(scale),
                                 window_start=int(start), window_end=1000, metric=metric,
                                 n=len(values), mean=float(values.mean()),
                                 median=float(np.median(values)), q25=float(np.quantile(values,.25)),
                                 q75=float(np.quantile(values,.75)), bootstrap95_low=float(low),
                                 bootstrap95_high=float(high), resamples=RESAMPLES,
                                 bootstrap_seed=seed, construction=origin))
    frame = pd.DataFrame(rows)
    assert len(frame) == 48
    assert (frame.n >= 1).all()
    assert ((frame.bootstrap95_low >= 0) & (frame.bootstrap95_high <= 1)).all()
    frame.to_csv(output / "composition_bootstrap.csv", index=False)
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [*inputs, Path(__file__)]}
    (output / "sources.json").write_text(json.dumps({"input_sha256": manifest,
        "resampling_unit": "independent trajectory time-window mean",
        "resamples": RESAMPLES, "rows": len(rows)}, indent=2)+"\n")
    print(json.dumps({"composition_bootstrap_rows": len(rows), "resamples": RESAMPLES}))


if __name__ == "__main__":
    run()
