"""Exact resource-limited count process and prospective revision experiments.

All states are abstract digital readers. Candidate children are generated for
all starting actors; uniform hypergeometric selection implements the original
random-permutation resource stopping law exactly. The four live classes are
Z, C, A, P. This module does not approximate individual count dynamics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import scipy
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("Z", "C", "A", "P")
WORDS = np.array(list(itertools.product((0, 1), repeat=4)), dtype=np.int64)
WORD_CLASSES = (WORDS[:, 0] != WORDS[:, 2]).astype(int) + 2 * (WORDS[:, 1] != WORDS[:, 3]).astype(int)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf8") as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Empty result table")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def class_matrix(sigma):
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError("A finite nonnegative width is required")
    u = math.exp(-1 / sigma**2) if sigma else 0.0
    t = math.exp(-1 / (2 * sigma**2)) if sigma else 0.0
    a = 2 * t / (1 + t)**2
    return np.array([[u, 0, 0, 0], [1, 0, 0, 0], [u/4]*4,
                     [a*a, a*(1-a), a*(1-a), (1-a)**2]])


def candidate_laws(live_matrix):
    live_matrix = np.asarray(live_matrix, dtype=float)
    if live_matrix.ndim != 2 or live_matrix.shape[0] != live_matrix.shape[1]:
        raise ValueError("Square live matrix required")
    if live_matrix.min() < 0 or live_matrix.sum(axis=1).max() > 1 + 1e-12:
        raise ValueError("Substochastic matrix required")
    result = np.column_stack((live_matrix, np.maximum(0, 1-live_matrix.sum(axis=1))))
    result /= result.sum(axis=1, keepdims=True)
    return result


def step_counts(counts, resource, laws, death, supply, capacity, rng):
    """Exact conditional count/resource sweep; laws include a final failure bin."""
    counts = np.asarray(counts, dtype=np.int64)
    offered = min(int(capacity), int(resource) + int(supply))
    potential = np.zeros(len(counts), dtype=np.int64)
    if offered:
        for count, law in zip(counts, laws):
            if count:
                potential += rng.multinomial(int(count), law)[:-1]
    successes = int(potential.sum())
    births = min(offered, successes)
    if births == successes:
        accepted = potential
    elif births:
        accepted = rng.multivariate_hypergeometric(potential, births)
    else:
        accepted = np.zeros_like(counts)
    survivors = rng.binomial(counts + accepted, 1-death)
    return survivors, offered-births, births


def simulate(initial_counts, initial_resource, sigma, death, supply, capacity, horizon, seed):
    initial_counts = np.asarray(initial_counts, dtype=np.int64)
    if len(initial_counts) != 4 or np.any(initial_counts < 0):
        raise ValueError("Four nonnegative class counts required")
    if initial_counts.sum() + horizon*capacity >= np.iinfo(np.int32).max:
        raise ValueError("Declared deterministic count bound exceeds int32 storage")
    if not (0 <= death <= 1 and 0 <= initial_resource <= capacity and supply >= 0):
        raise ValueError("Invalid population parameters")
    counts = np.zeros((horizon+1, 4), dtype=np.int32)
    resources = np.zeros(horizon+1, dtype=np.int32)
    births = np.zeros(horizon+1, dtype=np.int32)
    counts[0] = initial_counts
    resources[0] = initial_resource
    laws = candidate_laws(class_matrix(sigma))
    rng = np.random.default_rng(int(seed))
    current = initial_counts.copy()
    resource = int(initial_resource)
    for sweep in range(1, horizon+1):
        if not current.any():
            resources[sweep:] = np.minimum(capacity, resource + supply*np.arange(1, horizon-sweep+2))
            break
        current, resource, number = step_counts(current, resource, laws, death, supply, capacity, rng)
        counts[sweep] = current
        resources[sweep] = resource
        births[sweep] = number
    return counts, resources, births


def exposure_probabilities(success_probabilities, resource):
    """Exact Poisson-binomial exposure, for small-state verification/diagnosis."""
    success_probabilities = np.asarray(success_probabilities, dtype=float)
    if np.any((success_probabilities < 0) | (success_probabilities > 1)):
        raise ValueError("Probabilities outside [0,1]")
    values = []
    for index in range(len(success_probabilities)):
        mass = np.array([1.0])
        for j, probability in enumerate(success_probabilities):
            if j != index:
                mass = np.convolve(mass, [1-probability, probability])
        factors = np.minimum(1.0, resource/(1+np.arange(len(mass))))
        values.append(float(mass @ factors))
    return np.array(values)


def binary_exposures(n_high, n_low, u, resource):
    """Two exposure values as finite binomial sums; absent groups return None."""
    from scipy.stats import binom
    def expected(offset, number):
        k = np.arange(number+1)
        return float(binom.pmf(k, number, u) @ np.minimum(1., resource/(1+offset+k)))
    return dict(high=expected(n_high-1, n_low) if n_high else None,
                low=expected(n_high, n_low-1) if n_low else None)


def fluid_step(count_density, resource_density, live_matrix, death, supply, capacity):
    x = np.asarray(count_density, dtype=float)
    offered = min(capacity, resource_density+supply)
    candidates = x @ live_matrix
    total = float(candidates.sum())
    accepted = candidates * min(1., offered/total) if total > 0 else np.zeros_like(x)
    return (1-death)*(x+accepted), offered-float(accepted.sum()), float(accepted.sum())


def fluid_trajectory(initial_counts, initial_resource, sigma, death, supply, capacity, horizon):
    L = class_matrix(sigma)
    counts = np.zeros((horizon+1, 4))
    resources = np.zeros(horizon+1)
    births = np.zeros(horizon+1)
    counts[0] = initial_counts
    resources[0] = initial_resource
    for sweep in range(1, horizon+1):
        counts[sweep], resources[sweep], births[sweep] = fluid_step(
            counts[sweep-1], resources[sweep-1], L, death, supply, capacity)
    return counts, resources, births


def fluid_fixed_point(sigma, death=.2, supply=30., capacity=120.):
    L = class_matrix(sigma)
    eigenvalues, eigenvectors = np.linalg.eig(L.T)
    index = int(np.argmax(eigenvalues.real))
    rho = float(eigenvalues[index].real)
    if abs(eigenvalues[index].imag) > 1e-12:
        raise ValueError("Nonreal Perron root")
    v = eigenvectors[:, index].real
    v /= v.sum()
    v[np.abs(v) < 1e-15] = 0
    if v.min() < -1e-12:
        raise ValueError("Nonnegative class vector required")
    survival = 1-death
    if not (0 < death < 1 and 0 < supply <= capacity and survival*rho > death):
        raise ValueError("Strictly saturated supercritical case required")
    total = survival*supply/death
    counts = total*v
    C = L.sum(axis=1)
    jacobian = np.zeros((5, 5))
    jacobian[:4, :4] = survival*np.eye(4) + death/rho*(L-np.outer(C, v))
    # At R=0 and J=K, physical positive stock perturbations are capped.
    if supply < capacity:
        jacobian[4, :4] = survival*v
    other = np.delete(eigenvalues, index)
    predicted = np.r_[0., survival, survival+death*other/rho]
    observed = np.linalg.eigvals(jacobian)
    state = np.r_[counts, 0.]
    finite_difference = np.zeros_like(jacobian)
    step = 1e-4
    for i in range(5):
        delta = np.eye(5)[i]*step
        plus = fluid_step((state+delta)[:4], (state+delta)[4], L, death, supply, capacity)
        if i == 4 and supply == capacity:
            center = fluid_step(state[:4], state[4], L, death, supply, capacity)
            finite_difference[i] = (np.r_[plus[0], plus[1]]-np.r_[center[0], center[1]])/step
        else:
            minus = fluid_step((state-delta)[:4], (state-delta)[4], L, death, supply, capacity)
            finite_difference[i] = (np.r_[plus[0], plus[1]]-np.r_[minus[0], minus[1]])/(2*step)
    nxt = fluid_step(counts, 0., L, death, supply, capacity)
    return dict(sigma=float(sigma), rho=rho, composition=v.tolist(), total=total,
                counts=counts.tolist(), resource=0., saturation_margin=float(counts@C-supply),
                fixed_point_residual=float(np.max(np.abs(np.r_[nxt[0]-counts, nxt[1]]))),
                jacobian_row_convention=jacobian.tolist(),
                jacobian_eigenvalues=np.sort(observed.real).tolist(),
                predicted_eigenvalues=np.sort(predicted.real).tolist(),
                maximum_eigenvalue_modulus=float(max(abs(observed))),
                finite_difference_step=step,
                finite_difference_max_error=float(np.max(np.abs(jacobian-finite_difference))))


def raw_to_classes(raw_counts):
    raw_counts = np.asarray(raw_counts, dtype=np.int64)
    if raw_counts.shape != (16,) or raw_counts.min() < 0:
        raise ValueError("Sixteen nonnegative integer raw counts required")
    return np.bincount(WORD_CLASSES, weights=raw_counts, minlength=4).astype(np.int64)


def word_success(word, sigma):
    word = np.asarray(word)
    keys = word[::2]
    if sigma == 0:
        return float(all(any(keys == c) for c in (0, 0, 1, 1)))
    return float(np.prod([math.exp(-min((c-keys)**2)/(2*sigma*sigma)) for c in (0, 0, 1, 1)]))


def rewire_counts(raw_counts, pairs, sigma=.8):
    """One-time value-slot exchange; does not modify the baseline birth law."""
    before = np.asarray(raw_counts, dtype=np.int64)
    if before.shape != (16,) or pairs < 0 or min(before[0], before[15]) < pairs:
        raise ValueError("Insufficient disjoint homogeneous pairs")
    after = before.copy()
    after[[0, 15]] -= pairs
    after[[1, 14]] += pairs
    class_before = raw_to_classes(before)
    class_after = raw_to_classes(after)
    maps = ((0, 1), (15, 14))
    cp_pairs = [[word_success(WORDS[a], sigma), word_success(WORDS[b], sigma)] for a, b in maps]
    checks = dict(population_preserved=bool(before.sum() == after.sum()),
                  every_position_symbol_totals_preserved=bool(np.array_equal(before@WORDS, after@WORDS)),
                  each_changed_actor_key_slots_preserved=bool(all(np.array_equal(WORDS[a, ::2], WORDS[b, ::2]) for a, b in maps)),
                  each_changed_actor_success_preserved=bool(all(x == y for x, y in cp_pairs)),
                  success_probability_inventory_preserved=bool(class_before[[1, 3]].sum() == class_after[[1, 3]].sum()),
                  reference_count_zero=bool(before[[3, 12]].sum() == after[[3, 12]].sum() == 0),
                  initial_decoding_information_zero=bool(class_before[3] == class_after[3] == 0),
                  no_initial_two_value_readers=bool(class_before[2:].sum() == 0),
                  exactly_two_pairs_added_to_repertoire=bool(class_after[2:].sum() == 2*pairs))
    if not all(checks.values()):
        raise AssertionError(checks)
    return after, dict(checks=checks, before_counts=before.tolist(), after_counts=after.tolist(),
                       before_class_counts=class_before.tolist(), after_class_counts=class_after.tolist(),
                       before_position_one_totals=(before@WORDS).tolist(), after_position_one_totals=(after@WORDS).tolist(),
                       before_position_zero_totals=(before.sum()-before@WORDS).tolist(),
                       after_position_zero_totals=(after.sum()-after@WORDS).tolist(),
                       actor_success_before_after=cp_pairs)


def seed_for(base_seed, mode, sigma, scale_or_checkpoint, arm, repeat):
    key = [int(base_seed), int(mode), int(round(sigma*10000)), int(scale_or_checkpoint), int(arm), int(repeat)]
    return int(np.random.SeedSequence(key).generate_state(1, dtype=np.uint64)[0])


def extract_checkpoints(root, plan):
    source = root/"results/population_extension"
    loss_path = source/"finite_repertoire_loss.csv"
    frame = pd.read_csv(loss_path)
    selected = frame[(frame.sigma == .8) & (frame.scale == 1) & frame.first_repertoire_loss_time.notna()]
    selected = selected.sort_values("repeat_id")
    if len(selected) != 10:
        raise AssertionError("The planned ten fixed checkpoints were not found")
    records = []
    source_hashes = {str(loss_path.relative_to(root)): sha256(loss_path)}
    for row in selected.itertuples():
        path = source/"finite"/f"{row.case}__r{row.repeat_id:03d}"/"trajectory.csv.gz"
        trajectory = pd.read_csv(path)
        sweep = int(row.first_repertoire_loss_time)
        match = trajectory[trajectory.sweep == sweep]
        if len(match) != 1:
            raise AssertionError("Checkpoint must match the stored sweep field")
        checkpoint = match.iloc[0]
        raw = np.array([int(checkpoint["p"+"".join(map(str, word))]) for word in WORDS], dtype=np.int64)
        if raw.sum() != int(checkpoint.N) or raw_to_classes(raw)[2:].sum():
            raise AssertionError("Invalid first-loss checkpoint")
        prior = trajectory[trajectory.sweep < sweep]
        if len(prior) and (prior.two_value_count == 0).any():
            raise AssertionError("Selected checkpoint is not the first loss")
        _, comparison = rewire_counts(raw, int(plan["matched_rewiring"]["pairs_per_checkpoint"]))
        path_name = str(path.relative_to(root))
        source_hashes[path_name] = sha256(path)
        comparison.update(checkpoint_id=int(row.repeat_id), original_case=row.case,
                          original_sweep=sweep, original_seed=int(row.seed),
                          population=int(checkpoint.N), resource=int(checkpoint.S),
                          resource_before=int(checkpoint.S), resource_after=int(checkpoint.S),
                          source_path=path_name, source_sha256=source_hashes[path_name])
        comparison["checks"]["resource_preserved"] = True
        records.append(comparison)
    return records, source_hashes


def case_worker(job):
    path = Path(job["path"])
    metadata_path = path.with_suffix(".json")
    case_signature = stable_digest({k: v for k, v in job.items() if k not in ("resume", "path")})
    if job["resume"] and path.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("complete") and previous.get("case_signature") == case_signature and previous.get("trajectory_sha256") == sha256(path):
            return dict(case=job["case"], path=str(path), cached=True, sha256=previous["trajectory_sha256"])
    nrep = len(job["seeds"])
    horizon = job["horizon"]
    counts = np.empty((nrep, horizon+1, 4), dtype=np.int32)
    resources = np.empty((nrep, horizon+1), dtype=np.int32)
    births = np.empty_like(resources)
    start = time.perf_counter()
    for rep, seed in enumerate(job["seeds"]):
        counts[rep], resources[rep], births[rep] = simulate(
            job["initial_counts"], job["initial_resource"], job["sigma"], job["death"],
            job["supply"], job["capacity"], horizon, seed)
    atomic_npz(path, counts=counts, resources=resources, births=births,
               seeds=np.array(job["seeds"], dtype=np.uint64), class_names=np.array(CLASSES),
               sweep=np.arange(horizon+1, dtype=np.int32))
    metadata = {k: v for k, v in job.items() if k not in ("resume", "path")}
    metadata.update(complete=True, case_signature=case_signature, trajectory_sha256=sha256(path),
                    elapsed_seconds=time.perf_counter()-start,
                    trajectories=nrep, count_shape=list(counts.shape), storage_dtype="int32")
    atomic_json(metadata_path, metadata)
    return dict(case=job["case"], path=str(path), cached=False, sha256=metadata["trajectory_sha256"])


def wilson(successes, n, z=1.959963984540054):
    p = successes/n
    divisor = 1+z*z/n
    center = (p+z*z/(2*n))/divisor
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/divisor
    return center-half, center+half


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    radius = float(student_t.ppf(.975, len(values)-1)*sd/math.sqrt(len(values)))
    return mean, mean-radius, mean+radius, sd


def summarize_fluid(root, plan, jobs):
    model = plan["exact_resource_model"]
    setting = plan["fluid_limit"]
    comparisons, trajectory_rows, replicate_rows = [], [], []
    horizon = setting["horizon"]
    fixed = []
    for sigma in setting["widths"]:
        fixed.append(fluid_fixed_point(sigma, model["d"], model["supply"], model["capacity"]))
    for job in jobs:
        if job["kind"] != "fluid":
            continue
        with np.load(job["path"], allow_pickle=False) as archive:
            counts = archive["counts"].astype(float)
            resources = archive["resources"].astype(float)
            births = archive["births"].astype(float)
        scale = job["scale"]
        density = counts/scale
        target, target_resource, target_births = fluid_trajectory(setting["initial_class_counts"], setting["initial_resource"],
                                                               job["sigma"], model["d"], model["supply"], model["capacity"], horizon)
        for sweep in range(horizon+1):
            values = density[:, sweep]
            row = dict(case=job["case"], sigma=job["sigma"], scale=scale, sweep=sweep, independent_replicates=len(values),
                       mean_population_density=float(values.sum(axis=1).mean()),
                       fluid_population_density=float(target[sweep].sum()),
                       mean_resource_density=float(resources[:, sweep].mean()/scale),
                       fluid_resource_density=float(target_resource[sweep]),
                       mean_birth_density=float(births[:, sweep].mean()/scale),
                       fluid_birth_density=float(target_births[sweep]),
                       extinct_count=int((counts[:, sweep].sum(axis=1) == 0).sum()))
            for i, name in enumerate(CLASSES):
                mean, low, high, sd = interval(values[:, i])
                row.update({"mean_"+name: mean, "ci95_low_"+name: low, "ci95_high_"+name: high, "fluid_"+name: float(target[sweep, i])})
            trajectory_rows.append(row)
        for sweep in (25, 100, 200):
            differences = density[:, sweep]-target[sweep]
            squared_error = np.square(differences).sum(axis=1)
            mean, low, high, sd = interval(squared_error)
            extinct = int((counts[:, sweep].sum(axis=1) == 0).sum())
            wlow, whigh = wilson(extinct, len(counts))
            comparisons.append(dict(case=job["case"], sigma=job["sigma"], scale=scale, sweep=sweep,
                                    role="primary" if sweep == 100 else "secondary",
                                    independent_replicates=len(counts), mse=mean, mse_ci95_low=low, mse_ci95_high=high,
                                    squared_error_sd=sd, rmse=math.sqrt(mean),
                                    mean_bias_l2=float(np.linalg.norm(differences.mean(axis=0))),
                                    extinct_count=extinct, extinction_fraction=extinct/len(counts),
                                    extinction_wilson_low=wlow, extinction_wilson_high=whigh))
            for rep, value in enumerate(squared_error):
                replicate_rows.append(dict(case=job["case"], sigma=job["sigma"], scale=scale, sweep=sweep,
                                           repeat=rep, squared_density_error=float(value)))
    slopes = []
    for sigma in setting["widths"]:
        for sweep in (25, 100, 200):
            rows = sorted([r for r in comparisons if r["sigma"] == sigma and r["sweep"] == sweep], key=lambda r: r["scale"])
            x = np.log([r["scale"] for r in rows]); y = np.log([r["mse"] for r in rows])
            slope, intercept = np.polyfit(x, y, 1)
            r2 = 1-float(np.square(y-(intercept+slope*x)).sum()/np.square(y-y.mean()).sum())
            slopes.append(dict(sigma=sigma, sweep=sweep, role="descriptive", log_mse_log_scale_slope=float(slope),
                               log_intercept=float(intercept), r_squared=r2,
                               mse_scale1=rows[0]["mse"], mse_scale16=rows[-1]["mse"],
                               mse_ratio_scale1_to16=rows[0]["mse"]/rows[-1]["mse"]))
    out = root/"results/resource_population"
    write_csv(out/"fluid_errors.csv", comparisons)
    write_csv(out/"fluid_mean_trajectories.csv", trajectory_rows)
    write_csv(out/"fluid_replicate_errors.csv", replicate_rows)
    write_csv(out/"fluid_scaling_descriptive.csv", slopes)
    atomic_json(out/"fluid_fixed_points.json", fixed)
    return comparisons, slopes, fixed


def first_true(array, start=0):
    positions = np.flatnonzero(array[start:])
    return int(positions[0]+start) if len(positions) else None


def summarize_rewiring(root, plan, jobs):
    horizon = int(plan["matched_rewiring"]["horizon"])
    rows, groups, curves = [], [], []
    for job in jobs:
        if job["kind"] != "rewiring":
            continue
        with np.load(job["path"], allow_pickle=False) as archive:
            counts = archive["counts"].copy()
        n = counts.sum(axis=2)
        repertoire = counts[:, :, 2:].sum(axis=2)
        for rep, path in enumerate(counts):
            living = n[rep] > 0
            first_extinction = first_true(~living)
            first_loss = first_true(repertoire[rep] == 0)
            rows.append(dict(checkpoint_id=job["checkpoint_id"], original_sweep=job["original_sweep"], arm=job["arm"],
                             repeat=rep, seed=job["seeds"][rep], population_initial=int(n[rep, 0]),
                             resource_initial=job["initial_resource"], survival500=int(living[horizon]),
                             P_ever_by500=int((path[:horizon+1, 3] > 0).any()),
                             P_ever_by50=int((path[:51, 3] > 0).any()),
                             repertoire_at100=int(repertoire[rep, 100] > 0),
                             rmst500=int(living[:horizon].sum()),
                             first_P_time=first_true(path[:, 3] > 0),
                             first_repertoire_loss_time=first_loss,
                             first_extinction_time=first_extinction,
                             endpoint_N=int(n[rep, horizon])))
        selected = rows[-len(counts):]
        group = dict(checkpoint_id=job["checkpoint_id"], original_sweep=job["original_sweep"],
                     arm=job["arm"], independent_continuations=len(counts))
        for metric in ("survival500", "P_ever_by500", "P_ever_by50", "repertoire_at100"):
            successes = sum(r[metric] for r in selected)
            low, high = wilson(successes, len(selected))
            group.update({metric+"_count": successes, metric+"_fraction": successes/len(selected),
                          metric+"_wilson_low": low, metric+"_wilson_high": high})
        mean, low, high, sd = interval([r["rmst500"] for r in selected])
        group.update(rmst500_mean=mean, rmst500_ci95_low=low, rmst500_ci95_high=high)
        groups.append(group)
        for sweep in range(horizon+1):
            curve = dict(checkpoint_id=job["checkpoint_id"], arm=job["arm"], sweep=sweep,
                         independent_continuations=len(counts), survival_fraction=float((n[:, sweep] > 0).mean()),
                         mean_N=float(n[:, sweep].mean()),
                         repertoire_fraction=float((repertoire[:, sweep] > 0).mean()))
            for i, name in enumerate(CLASSES):
                curve["mean_"+name] = float(counts[:, sweep, i].mean())
            curves.append(curve)
    checkpoints = sorted({row["checkpoint_id"] for row in groups})
    m = int(plan["matched_rewiring"]["repeats_per_arm_per_checkpoint"])
    radius = math.sqrt(math.log(2/.05)/(len(checkpoints)*m))
    contrasts = []
    for metric in ("survival500", "P_ever_by500", "P_ever_by50", "repertoire_at100"):
        arm_means = {arm: float(np.mean([r[metric+"_fraction"] for r in groups if r["arm"] == arm]))
                     for arm in ("unchanged", "rewired")}
        difference = arm_means["rewired"]-arm_means["unchanged"]
        contrasts.append(dict(metric=metric, role="primary" if metric == "survival500" else "secondary",
                              fixed_checkpoints=len(checkpoints), continuations_per_arm_per_checkpoint=m,
                              unchanged_mean=arm_means["unchanged"], rewired_mean=arm_means["rewired"],
                              difference=difference, conditional_hoeffding95_radius=radius,
                              ci95_low=max(-1., difference-radius), ci95_high=min(1., difference+radius),
                              uncertainty_scope="Monte Carlo conditional on these fixed checkpoints; pointwise"))
    rmst = {arm: float(np.mean([r["rmst500_mean"] for r in groups if r["arm"] == arm]))
            for arm in ("unchanged", "rewired")}
    overall = dict(analysis="prospective continuation conditional on post hoc checkpoints",
                   primary=contrasts[0], secondary_binary=contrasts[1:],
                   secondary_rmst500=dict(unchanged=rmst["unchanged"], rewired=rmst["rewired"],
                                          difference=rmst["rewired"]-rmst["unchanged"]),
                   rmst_definition="sum of Pr(nonempty at t) for t=0,...,499; equivalently min(T_extinction,500)",
                   primary_interval_derivation="2 J m independent bounded Bernoulli terms with weights +/-1/(Jm); Hoeffding radius sqrt(log(2/alpha)/(Jm))",
                   secondary_timing_note="P by 50, repertoire at100 and RMST are extra secondary diagnostics fixed in the implementation before outcomes; primary and planned P by500 unchanged")
    out = root/"results/resource_population"
    write_csv(out/"rewiring_endpoints.csv", rows)
    write_csv(out/"rewiring_groups.csv", groups)
    write_csv(out/"rewiring_mean_trajectories.csv", curves)
    write_csv(out/"rewiring_contrasts.csv", contrasts)
    atomic_json(out/"rewiring_summary.json", overall)
    return overall


def run(root=ROOT, workers=2, resume=True):
    root = Path(root)
    if workers not in (1, 2):
        raise ValueError("Use one or two workers")
    started = time.perf_counter()
    plan_path = root/"config/revision_plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = sha256(plan_path)
    out = root/"results/resource_population"
    out.mkdir(parents=True, exist_ok=True)
    previous_plan = out/"plan_executed.json"
    if previous_plan.exists() and json.loads(previous_plan.read_text())["plan_sha256"] != plan_hash:
        raise ValueError("The prospective plan changed; use a fresh output directory")
    checkpoints, evidence_hashes = extract_checkpoints(root, plan)
    sources = [Path(__file__), root/"vendor/yomitsugi/core.py", root/"vendor/yomitsugi/population.py"]
    source_hashes = {str(p.relative_to(root)): sha256(p) for p in sources}
    runtime = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                   platform=platform.platform(), rng="NumPy Generator/PCG64")
    signature = stable_digest(dict(plan_sha256=plan_hash, sources=source_hashes, evidence=evidence_hashes, runtime=runtime))
    if not previous_plan.exists():
        atomic_json(previous_plan, dict(plan_sha256=plan_hash, plan=plan,
                                        recorded_before_simulation_utc=datetime.now(timezone.utc).isoformat(),
                                        extra_secondary_diagnostics=["P appearance by50", "repertoire presence100", "RMST500"]))
    atomic_json(out/"checkpoints.json", checkpoints)
    atomic_json(out/"inputs.json", dict(plan_sha256=plan_hash, source_hashes=source_hashes,
                                       evidence_hashes=evidence_hashes, runtime=runtime, signature=signature))
    model = plan["exact_resource_model"]
    base = plan["base_seed"]
    jobs = []
    fluid = plan["fluid_limit"]
    for sigma in fluid["widths"]:
        for scale in fluid["scales"]:
            case = f"sigma{sigma:g}_scale{scale}"
            jobs.append(dict(case=case, kind="fluid", sigma=float(sigma), scale=int(scale),
                             initial_counts=(np.array(fluid["initial_class_counts"])*scale).astype(int).tolist(),
                             initial_resource=int(fluid["initial_resource"]*scale),
                             supply=int(model["supply"]*scale), capacity=int(model["capacity"]*scale), death=model["d"],
                             horizon=fluid["horizon"],
                             seeds=[seed_for(base, 1, sigma, scale, 0, rep) for rep in range(fluid["repeats_per_condition"])],
                             path=str(out/"fluid"/(case+".npz")), signature=signature, resume=resume))
    setting = plan["matched_rewiring"]
    for checkpoint in checkpoints:
        for arm_id, arm in enumerate(setting["arms"]):
            case = f"checkpoint{checkpoint['checkpoint_id']:03d}_{arm}"
            jobs.append(dict(case=case, kind="rewiring", sigma=.8,
                             checkpoint_id=checkpoint["checkpoint_id"], original_sweep=checkpoint["original_sweep"], arm=arm,
                             initial_counts=checkpoint["before_class_counts"] if arm == "unchanged" else checkpoint["after_class_counts"],
                             initial_resource=checkpoint["resource"], supply=model["supply"], capacity=model["capacity"], death=model["d"],
                             horizon=setting["horizon"],
                             seeds=[seed_for(base, 2, .8, checkpoint["checkpoint_id"], arm_id, rep) for rep in range(setting["repeats_per_arm_per_checkpoint"])],
                             path=str(out/"rewiring"/(case+".npz")), signature=signature,
                             checkpoint_digest=stable_digest(checkpoint), resume=resume))
    all_seeds = [seed for job in jobs for seed in job["seeds"]]
    if len(all_seeds) != len(set(all_seeds)):
        raise AssertionError("Duplicate continuation seeds")
    atomic_json(out/"jobs_planned.json", [{k: v for k, v in job.items() if k not in ("resume", "path")} for job in jobs])
    if workers == 1:
        completion = [case_worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            completion = list(pool.map(case_worker, jobs))
    fluid_results, slopes, fixed = summarize_fluid(root, plan, jobs)
    rewiring = summarize_rewiring(root, plan, jobs)
    if any(sha256(root/path) != digest for path, digest in source_hashes.items()):
        raise RuntimeError("Source changed during the analysis")
    if sha256(plan_path) != plan_hash:
        raise RuntimeError("Plan changed during analysis")
    if any(sha256(root/path) != digest for path, digest in evidence_hashes.items()):
        raise RuntimeError("Checkpoint evidence changed during analysis")
    execution = dict(completed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.perf_counter()-started, workers=workers,
                     plan_sha256=plan_hash, signature=signature, runtime=runtime,
                     trajectories_fluid=sum(len(j["seeds"]) for j in jobs if j["kind"] == "fluid"),
                     trajectories_rewiring=sum(len(j["seeds"]) for j in jobs if j["kind"] == "rewiring"),
                     reused_cases=sum(r["cached"] for r in completion), completed_cases=len(completion),
                     seed_count=len(all_seeds), unique_seeds=True,
                     cases=[dict(case=r["case"], cached=r["cached"], trajectory_sha256=r["sha256"]) for r in completion])
    atomic_json(out/"execution.json", execution)
    print(json.dumps(dict(completed=True, fluid_trajectories=execution["trajectories_fluid"],
                          rewiring_trajectories=execution["trajectories_rewiring"],
                          primary_rewiring=rewiring["primary"],
                          fluid_primary=[r for r in slopes if r["sweep"] == 100],
                          elapsed_seconds=execution["elapsed_seconds"], reused_cases=execution["reused_cases"]), indent=2))
    return execution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--no-resume", action="store_true", help="Force deterministic regeneration with the same planned seeds")
    args = parser.parse_args()
    run(workers=args.workers, resume=not args.no_resume)
