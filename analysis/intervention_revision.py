"""Raw-reader intervention, innovation, and finite-size review analyses.

All states are nonliving four-bit digital readers. Census reference recovery
means an alive 0011 or 1100 reader after the independent death step. Exact
unrestricted PGFs and resource-limited simulations use that same event.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from analysis.finite import build_matrix, parameters, spectral_quantities
from analysis.persistence_thresholds import offspring_pgf, ultimate_approximation
from analysis.resource_population import (
    WORDS, WORD_CLASSES, atomic_json, atomic_npz, candidate_laws, class_matrix,
    first_true, fluid_trajectory, raw_to_classes, rewire_counts, sha256,
    stable_digest, step_counts, wilson, write_csv,
)

ROOT = Path(__file__).resolve().parents[1]
REF = np.isin(np.arange(16), (3, 12))
H = WORD_CLASSES >= 2
ARMS = ("unchanged", "pairs1", "pairs2", "pairs5", "pairs10", "pairs20",
        "sham10", "isolated10")


def mutation_matrix(mu):
    if not 0 <= mu <= .5:
        raise ValueError("Use an independent bit-flip probability in [0,1/2]")
    distances = (WORDS[:, None, :] != WORDS[None, :, :]).sum(axis=2)
    return mu**distances * (1-mu)**(4-distances)


def mutated_matrix(sigma, mu, g=(0, 0, 1, 1)):
    matrix, states = build_matrix(sigma, g=g)
    return matrix @ mutation_matrix(mu), states


def mutated_class_matrix(sigma, mu):
    u, a = parameters(sigma)
    b = 2*mu*(1-mu)
    v = np.array([(1-b)**2, b*(1-b), b*(1-b), b*b])
    changed_a = a+b*(1-2*a)
    return np.array([u*v, v, np.repeat(u/4, 4),
                     [changed_a**2, changed_a*(1-changed_a),
                      changed_a*(1-changed_a), (1-changed_a)**2]])


def prepare_arm(before, arm):
    before = np.asarray(before, dtype=np.int64)
    if arm == "unchanged":
        return before.copy(), {"intervention": "none", "pairs": 0}
    pairs = 10 if arm in ("sham10", "isolated10") else int(arm[5:])
    rewired, report = rewire_counts(before, pairs)
    if arm == "sham10":
        after = before.copy()
        after[[0, 15]] -= pairs
        after[[5, 10]] += pairs
        checks = {
            "nontrivial_raw_state_change": not np.array_equal(before, after),
            "population_preserved": before.sum() == after.sum(),
            "positions_preserved": np.array_equal(before@WORDS, after@WORDS),
            "actor_keys_preserved": all(np.array_equal(WORDS[i, ::2], WORDS[j, ::2])
                                        for i, j in ((0, 5), (15, 10))),
            "class_counts_preserved": np.array_equal(raw_to_classes(before), raw_to_classes(after)),
            "reference_zero": before[REF].sum() == after[REF].sum() == 0,
            "repertoire_zero": before[H].sum() == after[H].sum() == 0,
        }
        assert all(checks.values())
        return after, {"intervention": "exchange both value slots", "pairs": pairs,
                       "checks": {k: bool(v) for k, v in checks.items()},
                       "class_count_resource_process": "exactly unchanged in distribution"}
    if arm == "isolated10":
        after = np.zeros(16, dtype=np.int64)
        after[[1, 14]] = pairs
        return after, {"intervention": "retain only the twenty introduced A readers",
                       "pairs": pairs, "background_removed": int(before.sum()-2*pairs),
                       "scope": "background-removal contrast; population and material inventories change"}
    return rewired, {"intervention": "exchange one value slot", "pairs": pairs,
                     "invariants": report["checks"]}


def unrestricted_curves(matrix, initial, horizon=500, death=.2):
    """Exact census survival and first-reference appearance probabilities."""
    initial = np.asarray(initial, dtype=np.int64)
    z = np.zeros(16)
    x = (~H).astype(float)
    no_hit = (~REF).astype(float)
    rows = []
    for h in range(horizon+1):
        rows.append({"sweep": h,
                     "N_survival": float(1-np.prod(z**initial)),
                     "H_survival": float(1-np.prod(x**initial)),
                     "reference_ever": float(1-np.prod(no_hit**initial))})
        z = np.clip(offspring_pgf(z, matrix, death), 0, 1)
        x = np.clip(offspring_pgf(x, matrix, death), 0, 1)
        no_hit = np.clip((~REF)*offspring_pgf(no_hit, matrix, death), 0, 1)
    return rows


def seed_for(kind, checkpoint, arm, repeat):
    return int(np.random.SeedSequence([2026102419, kind, checkpoint, arm, repeat])
               .generate_state(1, dtype=np.uint64)[0])


def simulate_raw(initial, resource, matrix, death, supply, capacity, horizon, seed):
    initial = np.asarray(initial, dtype=np.int64)
    assert initial.sum()+horizon*capacity < np.iinfo(np.int32).max
    counts = np.zeros((horizon+1, 16), dtype=np.int32)
    resources = np.zeros(horizon+1, dtype=np.int32)
    births = np.zeros(horizon+1, dtype=np.int32)
    counts[0] = initial
    resources[0] = resource
    laws = candidate_laws(matrix)
    rng = np.random.default_rng(int(seed))
    current = initial.copy()
    for h in range(1, horizon+1):
        if not current.any():
            resources[h:] = np.minimum(capacity, resource+supply*np.arange(1, horizon-h+2))
            break
        current, resource, number = step_counts(current, resource, laws, death, supply, capacity, rng)
        counts[h] = current
        resources[h] = resource
        births[h] = number
    return counts, resources, births


def raw_case_worker(job):
    path = Path(job["path"])
    meta = path.with_suffix(".json")
    signature = stable_digest({k: v for k, v in job.items() if k not in ("path", "resume")})
    if job["resume"] and path.exists() and meta.exists():
        previous = json.loads(meta.read_text())
        if (previous.get("complete") and previous.get("case_signature") == signature
                and previous.get("trajectory_sha256") == sha256(path)):
            return {"case": job["case"], "cached": True, "sha256": sha256(path)}
    start = time.perf_counter()
    matrix, _ = build_matrix(job["sigma"])
    shape = (len(job["seeds"]), job["horizon"]+1)
    counts = np.empty((*shape, 16), dtype=np.int32)
    resources = np.empty(shape, dtype=np.int32)
    births = np.empty(shape, dtype=np.int32)
    for repeat, seed in enumerate(job["seeds"]):
        counts[repeat], resources[repeat], births[repeat] = simulate_raw(
            job["initial_counts"], job["initial_resource"], matrix, job["death"],
            job["supply"], job["capacity"], job["horizon"], seed)
    atomic_npz(path, counts=counts, resources=resources, births=births,
               seeds=np.array(job["seeds"], dtype=np.uint64), words=WORDS,
               sweep=np.arange(job["horizon"]+1, dtype=np.int32))
    record = {k: v for k, v in job.items() if k not in ("path", "resume")}
    record.update(complete=True, case_signature=signature, trajectory_sha256=sha256(path),
                  elapsed_seconds=time.perf_counter()-start)
    atomic_json(meta, record)
    return {"case": job["case"], "cached": False, "sha256": record["trajectory_sha256"]}


def planned_jobs(out, signature, resume):
    checkpoints = json.loads((ROOT/"results/resource_population/checkpoints.json").read_text())
    jobs, invariants = [], []
    for cp in checkpoints:
        for arm_id, arm in enumerate(ARMS):
            initial, report = prepare_arm(cp["before_counts"], arm)
            invariants.append({"checkpoint": cp["checkpoint_id"], "arm": arm,
                               "initial_counts": initial.tolist(), **report})
            case = f"checkpoint{cp['checkpoint_id']:03d}_{arm}"
            jobs.append(dict(case=case, kind="intervention", checkpoint=cp["checkpoint_id"],
                             arm=arm, pairs=report["pairs"], initial_counts=initial.tolist(),
                             initial_resource=cp["resource"], sigma=.8, death=.2, supply=30,
                             capacity=120, horizon=500,
                             seeds=[seed_for(1, cp["checkpoint_id"], arm_id, r) for r in range(100)],
                             path=str(out/"raw"/(case+".npz")), signature=signature, resume=resume))
    initial = np.zeros(16, dtype=np.int64)
    initial[3] = 60*16
    jobs.append(dict(case="sigma0.8_scale16", kind="scale16", checkpoint=-1, arm="native_start",
                     pairs=0, initial_counts=initial.tolist(), initial_resource=60*16,
                     sigma=.8, death=.2, supply=30*16, capacity=120*16, horizon=1000,
                     seeds=[seed_for(2, 16, 0, r) for r in range(40)],
                     path=str(out/"raw/sigma0.8_scale16.npz"), signature=signature, resume=resume))
    atomic_json(out/"intervention_invariants.json", invariants)
    return jobs


def validation(jobs, out):
    total = transitions = 0
    all_seeds = []
    for job in jobs:
        with np.load(job["path"], allow_pickle=False) as a:
            n = a["counts"].astype(np.int64)
            r = a["resources"].astype(np.int64)
            b = a["births"].astype(np.int64)
            seeds = a["seeds"].tolist()
        assert np.array_equal(n[:, 0], np.broadcast_to(job["initial_counts"], n[:, 0].shape))
        assert np.all(r[:, 0] == job["initial_resource"])
        assert np.all(n >= 0) and np.all(r >= 0)
        offered = np.minimum(job["capacity"], r[:, :-1]+job["supply"])
        assert np.array_equal(r[:, 1:], offered-b[:, 1:])
        assert np.all(b[:, 1:] <= offered)
        assert np.all(b[:, 1:] <= n[:, :-1].sum(axis=2))
        assert np.all(n[:, 1:].sum(axis=2) <= n[:, :-1].sum(axis=2)+b[:, 1:])
        assert np.all(np.diff((n[:, :, H].sum(axis=2) == 0).astype(np.int8), axis=1) >= 0)
        assert seeds == job["seeds"]
        all_seeds.extend(seeds)
        total += len(n)
        transitions += len(n)*job["horizon"]
    assert len(all_seeds) == len(set(all_seeds))
    atomic_json(out/"trajectory_validation.json", dict(passed=True, trajectories=total,
                                                       transitions=transitions, unique_seeds=True,
                                                       exact_resource_bookkeeping=True,
                                                       irreversible_repertoire_loss_verified=True))


def bootstrap_mean(values, seed, resamples=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)),
            "bootstrap95_low": float(low), "bootstrap95_high": float(high),
            "resamples": resamples, "bootstrap_seed": seed}


def summarize_interventions(jobs, out):
    matrix, _ = build_matrix(.8)
    raw_rows, case_rows, curves, exact_rows = [], [], [], []
    for job in jobs:
        if job["kind"] != "intervention":
            continue
        with np.load(job["path"], allow_pickle=False) as archive:
            counts = archive["counts"].astype(np.int64)
        n = counts.sum(axis=2)
        hcount = counts[:, :, H].sum(axis=2)
        refs = counts[:, :, REF].sum(axis=2)
        hit = np.maximum.accumulate(refs > 0, axis=1)
        exact = unrestricted_curves(matrix, job["initial_counts"])
        for repeat in range(len(counts)):
            raw_rows.append(dict(checkpoint=job["checkpoint"], arm=job["arm"], pairs=job["pairs"],
                                 repeat=repeat, seed=job["seeds"][repeat],
                                 N500=int(n[repeat, 500] > 0), H500=int(hcount[repeat, 500] > 0),
                                 reference_present500=int(refs[repeat, 500] > 0),
                                 reference_ever500=int(hit[repeat, 500]),
                                 reference_ever50=int(hit[repeat, 50]),
                                 first_reference_census=first_true(refs[repeat] > 0),
                                 reference_reappearance_episodes=int(((refs[repeat, :-1] == 0)&(refs[repeat, 1:] > 0)).sum()),
                                 first_H_loss=first_true(hcount[repeat] == 0),
                                 extinction_census=first_true(n[repeat] == 0),
                                 endpoint_N=int(n[repeat, -1]), endpoint_H=int(hcount[repeat, -1]),
                                 endpoint_reference=int(refs[repeat, -1])))
        group = dict(checkpoint=job["checkpoint"], arm=job["arm"], pairs=job["pairs"],
                     independent_replicates=len(counts), initial_N=int(n[0, 0]),
                     initial_H=int(hcount[0, 0]), initial_resource=job["initial_resource"])
        for key, values in (("N500", n[:, -1] > 0), ("H500", hcount[:, -1] > 0),
                            ("reference_present500", refs[:, -1] > 0), ("reference_ever500", hit[:, -1])):
            successes = int(values.sum())
            low, high = wilson(successes, len(values))
            group.update({key+"_count": successes, key+"_fraction": successes/len(values),
                          key+"_wilson_low": low, key+"_wilson_high": high})
        group.update(unrestricted_N500=exact[-1]["N_survival"],
                     unrestricted_H500=exact[-1]["H_survival"],
                     unrestricted_reference_ever500=exact[-1]["reference_ever"])
        case_rows.append(group)
        for sweep in range(501):
            curves.append(dict(checkpoint=job["checkpoint"], arm=job["arm"], pairs=job["pairs"], sweep=sweep,
                               N_survival=float((n[:, sweep] > 0).mean()),
                               H_survival=float((hcount[:, sweep] > 0).mean()),
                               reference_ever=float(hit[:, sweep].mean()),
                               reference_present=float((refs[:, sweep] > 0).mean()),
                               mean_N=float(n[:, sweep].mean()), mean_H=float(hcount[:, sweep].mean()),
                               mean_reference=float(refs[:, sweep].mean())))
            exact_rows.append(dict(checkpoint=job["checkpoint"], arm=job["arm"], pairs=job["pairs"], **exact[sweep]))
    frame = pd.DataFrame(raw_rows)
    cases = pd.DataFrame(case_rows)
    summary, contrasts = [], []
    radius = math.sqrt(math.log(2/.05)/(2*1000))
    for arm in ARMS:
        rows = frame[frame.arm == arm]
        selected = cases[cases.arm == arm]
        item = {"arm": arm, "pairs": int(rows.pairs.iloc[0]), "fixed_checkpoints": 10,
                "independent_continuations": len(rows), "pointwise_hoeffding95_radius": radius}
        for metric in ("N500", "H500", "reference_ever500", "reference_ever50", "reference_present500"):
            mean = float(rows[metric].mean())
            item.update({metric+"_count": int(rows[metric].sum()), metric+"_fraction": mean,
                         metric+"_hoeffding95_low": max(0., mean-radius),
                         metric+"_hoeffding95_high": min(1., mean+radius)})
        reached = rows[rows.first_reference_census.notna()].first_reference_census
        item.update(first_reference_conditional_median=float(reached.median()) if len(reached) else None,
                     first_reference_conditional_q25=float(reached.quantile(.25)) if len(reached) else None,
                     first_reference_conditional_q75=float(reached.quantile(.75)) if len(reached) else None,
                     unrestricted_N500=float(selected.unrestricted_N500.mean()),
                     unrestricted_H500=float(selected.unrestricted_H500.mean()),
                     unrestricted_reference_ever500=float(selected.unrestricted_reference_ever500.mean()))
        summary.append(item)
        for metric in ("N500", "H500", "reference_ever500"):
            exact = item["unrestricted_"+metric]
            mean = item[metric+"_fraction"]
            contrasts.append(dict(arm=arm, metric=metric, finite_resource=mean,
                                  unrestricted_exact=exact, unrestricted_minus_finite=exact-mean,
                                  conditional_hoeffding95_low=max(-1., exact-mean-radius),
                                  conditional_hoeffding95_high=min(1., exact-mean+radius),
                                  interpretation="same-state same-horizon resource-restriction contrast"))
    pooled = pd.DataFrame(curves).groupby(["arm", "pairs", "sweep"], as_index=False).mean(numeric_only=True)
    exact_pooled = pd.DataFrame(exact_rows).groupby(["arm", "pairs", "sweep"], as_index=False).mean(numeric_only=True)
    for data in (pooled, exact_pooled):
        data.drop(columns="checkpoint", inplace=True)
    write_csv(out/"raw_intervention_endpoints.csv", raw_rows)
    write_csv(out/"intervention_checkpoint_summary.csv", case_rows)
    write_csv(out/"dose_response.csv", summary)
    write_csv(out/"resource_contrasts.csv", contrasts)
    write_csv(out/"intervention_curves_by_checkpoint.csv", curves)
    pooled.to_csv(out/"reference_first_hit_curves.csv", index=False)
    write_csv(out/"unrestricted_curves_by_checkpoint.csv", exact_rows)
    exact_pooled.to_csv(out/"unrestricted_curves.csv", index=False)
    Q = class_matrix(.8)[2:, 2:]
    ultimate = ultimate_approximation(Q, .2)
    benchmark = dict(x_A=ultimate["values"][0], x_P=ultimate["values"][1],
                     twenty_A_ultimate_H_survival=1-ultimate["values"][0]**20,
                     approximation_details=ultimate)
    sham = next(r for r in summary if r["arm"] == "sham10")
    unchanged = next(r for r in summary if r["arm"] == "unchanged")
    isolated = next(r for r in summary if r["arm"] == "isolated10")
    matched = next(r for r in summary if r["arm"] == "pairs10")
    benchmark.update(
        matched_pairs10=matched,
        isolated_minus_full_N500=isolated["N500_fraction"]-matched["N500_fraction"],
        isolated_minus_full_H500=isolated["H500_fraction"]-matched["H500_fraction"],
        two_arm_conditional_hoeffding95_radius=math.sqrt(math.log(2/.05)/1000),
        sham_minus_unchanged_N500=sham["N500_fraction"]-unchanged["N500_fraction"],
        census_reference_definition="Alive0011/1100 after deaths; newborns dying before census are not counted.",
        scope="Conditional on ten fixed post-loss checkpoints,100 independent continuations per arm per checkpoint.")
    atomic_json(out/"intervention_summary.json", benchmark)
    return benchmark


def fluid_conditional_errors(out):
    rows = []
    for sigma in (.5, .8):
        target, _, _ = fluid_trajectory([0, 0, 0, 60], 60, sigma, .2, 30, 120, 200)
        for scale in (1, 2, 4, 8, 16):
            path = ROOT/"results/resource_population/fluid"/f"sigma{sigma:g}_scale{scale}.npz"
            with np.load(path, allow_pickle=False) as archive:
                counts = archive["counts"].astype(float)
            for sweep in (100, 200):
                squared = np.square(counts[:, sweep]/scale-target[sweep]).sum(axis=1)
                alive = counts[:, sweep].sum(axis=1) > 0
                retained = counts[:, sweep, 2:].sum(axis=1) > 0
                rows.append(dict(sigma=sigma, scale=scale, sweep=sweep,
                                 role="primary" if sweep == 100 else "secondary",
                                 replicates=len(counts), alive_count=int(alive.sum()), H_count=int(retained.sum()),
                                 mse_unconditional=float(squared.mean()),
                                 mse_alive_conditional=float(squared[alive].mean()),
                                 mse_H_conditional=float(squared[retained].mean()) if retained.any() else None,
                                 source_sha256=sha256(path)))
    write_csv(out/"fluid_conditional_errors.csv", rows)
    return rows


def mixture_and_scale16(jobs, out):
    windows = pd.read_csv(ROOT/"results/population_extension/finite_run_windows.csv")
    ends = pd.read_csv(ROOT/"results/population_extension/finite_run_endpoints.csv")
    selected = windows[(windows.sigma == .8)&(windows.window_start == 501)&(windows.window_end == 1000)].copy()
    selected = selected.merge(ends[["case", "repeat_id", "extinct"]], on=["case", "repeat_id"], validate="one_to_one")
    source_rows = []
    for row in selected.itertuples():
        source_rows.append(dict(scale=int(row.scale), repeat=int(row.repeat_id),
                                value=float(row.reference_fraction_zero_filled), extinct1000=bool(row.extinct),
                                source="stored raw-state finite-resource runs"))
    job = next(j for j in jobs if j["kind"] == "scale16")
    with np.load(job["path"], allow_pickle=False) as archive:
        counts = archive["counts"].astype(np.int64)
    n = counts.sum(axis=2)
    reference = counts[:, :, REF].sum(axis=2)
    fractions = np.divide(reference, n, out=np.zeros_like(n, dtype=float), where=n > 0)
    for repeat in range(len(n)):
        source_rows.append(dict(scale=16, repeat=repeat, value=float(fractions[repeat, 501:1001].mean()),
                                extinct1000=bool(n[repeat, -1] == 0), source="new raw-state continuation"))
    groups = []
    prediction = spectral_quantities(.8)["native_spectral_fraction"]
    for scale in (1, 4, 16):
        selected = [row for row in source_rows if row["scale"] == scale]
        values = [r["value"] for r in selected]
        extinct = sum(r["extinct1000"] for r in selected)
        low, high = wilson(extinct, len(selected))
        alive_values = [r["value"] for r in selected if not r["extinct1000"]]
        extinct_values = [r["value"] for r in selected if r["extinct1000"]]
        item = dict(scale=scale, **bootstrap_mean(values, 2026102490+scale),
                    extinct1000=extinct, extinction_fraction=extinct/len(selected),
                    extinction_wilson95_low=low, extinction_wilson95_high=high,
                    alive_endpoint_conditional_mean=float(np.mean(alive_values)) if alive_values else None,
                    extinct_endpoint_conditional_mean=float(np.mean(extinct_values)) if extinct_values else None,
                    fixed_point_prediction=prediction, finite_window_mean_minus_fixed_point=float(np.mean(values)-prediction))
        groups.append(item)
    write_csv(out/"finite_size_run_distribution.csv", source_rows)
    write_csv(out/"finite_size_mixture_summary.csv", groups)
    result = dict(scale16=groups[-1], all_scales=groups,
                  interpretation="Each datum is one run's501–1000 mean with zeros after extinction; intervals resample independent runs.",
                  theorem_scope="MSE O(1/Omega) implies only an O(1/sqrt(Omega)) bound on mean error without additional expansion.",
                  no_bias_rate_claim=True)
    atomic_json(out/"raw_scale16_summary.json", result)
    return result


def mutation_analysis(out):
    mus = (0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2)
    widths = np.unique(np.r_[np.linspace(.4, 2., 161), .5, .8, .9953945221067288, 1., 2.])
    rows, crossovers = [], []
    for mu in mus:
        for sigma in widths:
            L = mutated_class_matrix(float(sigma), mu)
            values, vectors = np.linalg.eig(L.T)
            index = int(np.argmax(values.real))
            rho = float(values[index].real)
            v = vectors[:, index].real
            v /= v.sum()
            v[np.abs(v) < 1e-13] = 0
            others = np.delete(values, index)
            second = float(max(others.real))
            rho_D = float(max(np.linalg.eigvals(L[:2, :2]).real))
            rho_H = float(max(np.linalg.eigvals(L[2:, 2:]).real))
            raw, _ = mutated_matrix(float(sigma), mu)
            residual = max(np.max(np.abs(raw[:, WORD_CLASSES == j].sum(axis=1)-L[WORD_CLASSES, j])) for j in range(4))
            rows.append(dict(sigma=float(sigma), mu=mu, rho=rho, second_eigenvalue_real=second,
                             eigenvalue_gap=rho-second, demographic_death_threshold=rho/(1+rho),
                             killed_D_death_threshold=rho_D/(1+rho_D), killed_H_death_threshold=rho_H/(1+rho_H),
                             perron_H_fraction=float(v[2:].sum()),
                             Z_to_H_probability=float(L[0, 2:].sum()),
                             lumping_max_residual=float(residual), all_entries_positive=bool(raw.min() > 0)))
        def gap(sigma):
            vals = np.linalg.eigvals(mutated_class_matrix(float(sigma), mu))
            ordered = np.sort(vals.real)
            return float(ordered[-1]-ordered[-2])
        optimized = minimize_scalar(gap, bounds=(.85, 1.15), method="bounded", options={"xatol": 1e-13})
        critical = .9953945221067288
        crossovers.append(dict(mu=mu, minimum_gap_sigma=float(optimized.x), minimum_eigenvalue_gap=gap(optimized.x),
                               gap_at_zero_mutation_crossing=gap(critical),
                               gap_over_sqrt_mu=gap(critical)/math.sqrt(mu) if mu else None,
                               interpretation="minimum spectral-gap location, not a separate persistence threshold"))
    write_csv(out/"mutation_thresholds.csv", rows)
    write_csv(out/"mutation_crossover.csv", crossovers)
    atomic_json(out/"mutation_summary.json", {
        "mechanism": "Each successfully emitted output bit independently flips with probability mu; description remains fixed.",
        "raw_kernel": "M_mu = M U_mu; U_mu(z,w)=mu^Hamming(z,w)(1-mu)^(4-Hamming(z,w))",
        "four_class_lumping": "exactly preserved under symmetric independent binary mutation",
        "positive_mu": "For finite positive width and0<mu<1, all entries positive; H/D are no longer autonomous reproductive blocks.",
        "threshold_interpretation": "One Perron demographic threshold; killed-block thresholds are diagnostics only, not persistence phase boundaries.",
        "maximum_lumping_residual": max(r["lumping_max_residual"] for r in rows),
        "spectral_crossover": crossovers})
    return crossovers


def run(workers=2, resume=True, plan_only=False):
    start = time.perf_counter()
    out = ROOT/"results/intervention_revision"
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": 1, "base_seed": 2026102419, "design": "Fixed-checkpoint raw16 continuation",
        "checkpoint_source": "results/resource_population/checkpoints.json",
        "checkpoint_sha256": sha256(ROOT/"results/resource_population/checkpoints.json"),
        "arms": list(ARMS), "pairs": [0, 1, 2, 5, 10, 20],
        "replicates_per_arm_per_checkpoint": 100, "fixed_checkpoints": 10,
        "sigma": .8, "death": .2, "supply": 30, "capacity": 120, "horizon": 500,
        "primary": ["finite500 total-N survival versus exact unrestricted same-state500", "finite500 H survival versus exact unrestricted same-state500"],
        "secondary": ["dose response", "first alive reference census CDF", "reference present500",
                      "sham exact class-law control", "isolated20A background-removal contrast"],
        "reference_event": "Post-death census contains0011 or1100; not birth before death.",
        "uncertainty": "Pointwise Hoeffding bounds conditional on fixed checkpoints; per-checkpoint Wilson summaries.",
        "scale16": {"sigma": .8, "scale": 16, "replicates": 40, "horizon": 1000,
                    "initial_native": 960, "initial_resource": 960, "supply": 480, "capacity": 1920,
                    "window": [501, 1000], "summary": "run-level percentile bootstrap20000; no1/Omega bias theorem"},
        "deterministic_checks": ["unrestricted PGFs", "symmetric bit-flip innovation and exact lumping", "conditional MSE at100and200"],
        "publication_status": "internally recorded before new simulation; not external preregistration",
    }
    plan_hash = stable_digest(plan)
    plan_path = out/"plan_executed.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text())["plan_sha256"] == plan_hash, "Frozen plan changed"
    else:
        atomic_json(plan_path, dict(plan=plan, plan_sha256=plan_hash,
                                    recorded_before_simulation_utc=datetime.now(timezone.utc).isoformat()))
    sources = [Path(__file__), ROOT/"analysis/finite.py", ROOT/"analysis/resource_population.py",
               ROOT/"analysis/persistence_thresholds.py"]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    runtime = {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()}
    signature = stable_digest(dict(plan_sha256=plan_hash, sources=source_hashes, runtime=runtime))
    jobs = planned_jobs(out, signature, resume)
    atomic_json(out/"jobs_planned.json", [{k: v for k, v in j.items() if k not in ("path", "resume")} for j in jobs])
    atomic_json(out/"inputs.json", dict(source_sha256=source_hashes, runtime=runtime, signature=signature))
    if plan_only:
        print(json.dumps({"planned_cases": len(jobs), "planned_trajectories": sum(len(j["seeds"]) for j in jobs), "plan_sha256": plan_hash}))
        return
    print("Executing frozen raw16 continuation plan", flush=True)
    if workers == 1:
        completion = [raw_case_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            completion = list(pool.map(raw_case_worker, jobs))
    print("Raw trajectories complete; validating and computing exact comparisons", flush=True)
    validation(jobs, out)
    intervention = summarize_interventions(jobs, out)
    conditional = fluid_conditional_errors(out)
    scale16 = mixture_and_scale16(jobs, out)
    mutation = mutation_analysis(out)
    assert all(sha256(ROOT/p) == h for p, h in source_hashes.items()), "Source changed during run"
    result = dict(completed=True, plan_sha256=plan_hash, elapsed_seconds=time.perf_counter()-start,
                  cases=len(jobs), trajectories=sum(len(j["seeds"]) for j in jobs),
                  reused_cases=sum(c["cached"] for c in completion),
                  completion=completion, primary=intervention["matched_pairs10"],
                  scale16=scale16["scale16"], mutation_crossover=mutation,
                  validation="trajectory_validation.json")
    atomic_json(out/"execution.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "completion"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    run(args.workers, not args.no_resume, args.plan_only)
