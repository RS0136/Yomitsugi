"""Exact finite-state probabilities and an independent high-precision reference."""
from __future__ import annotations
import itertools
import json
import math
from pathlib import Path
import numpy as np
from .core import Kernel, NumericalPrecisionError, standard_chemistry
from .exact import encode, permutation_reader, permutation_word, write_csv

HORIZONS = (1, 2, 4, 8, 16, 32)
SIGMAS = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)
EPSILONS = (0.0, 0.01, 0.05, 0.1)


def reader_states(residues=(0, 1), length=4):
    return list(itertools.product(residues, repeat=length))


def build_matrix(kernel, b, g, states=None):
    """Raw product matrix, without silently normalizing or zeroing entries."""
    states = reader_states(kernel.residues, len(g)) if states is None else list(states)
    residue_index = {a: i for i, a in enumerate(kernel.residues)}
    M = np.empty((len(states), len(states)), dtype=float)
    log_C = np.array([kernel.log_success(p, b, g) for p in states])
    for i, p in enumerate(states):
        kernels = []
        for c in g:
            log_s, conditional = kernel.distribution(p, b, c)
            success = math.exp(log_s)
            if success == 0 and math.isfinite(log_s):
                raise NumericalPrecisionError("Matrix one-position success underflow")
            unconditional = success * np.asarray(conditional)
            if success > 0 and np.any((conditional > 0) & (unconditional == 0)):
                raise NumericalPrecisionError("Positive matrix output factor underflow")
            kernels.append(unconditional)
        for j, child in enumerate(states):
            factors = [kernels[r][residue_index[a]] for r, a in enumerate(child)]
            M[i, j] = math.prod(factors)
            if M[i, j] == 0 and all(factor > 0 for factor in factors):
                raise NumericalPrecisionError("Positive matrix word transition underflow")
            if M[i, j] > 0:
                assert set(child) <= set(p[1::2])
    C = np.exp(log_C)
    if np.any(np.isfinite(log_C) & (C == 0)):
        raise NumericalPrecisionError("Positive analytic matrix row sum underflow")
    failure = -np.expm1(log_C)
    T = np.zeros((len(states) + 1, len(states) + 1))
    T[:-1, :-1] = M
    T[:-1, -1] = failure
    T[-1, -1] = 1.0
    audit = dict(max_row_C_error=float(np.max(np.abs(M.sum(axis=1) - C))),
                 max_stochastic_row_error=float(np.max(np.abs(T.sum(axis=1) - 1))),
                 max_raw_row_sum=float(M.sum(axis=1).max()), min_entry=float(M.min()),
                 positive_entries=int(np.count_nonzero(M)),
                 analytic_birth_underflow=int(np.count_nonzero(np.isfinite(log_C) & (C == 0))))
    assert audit["max_row_C_error"] < 2e-13
    assert audit["max_stochastic_row_error"] < 2e-13
    assert audit["min_entry"] >= 0
    return states, M, T, C, audit


def survival(M, horizons=HORIZONS):
    values, current = {}, np.ones(len(M))
    for h in range(1, max(horizons) + 1):
        current = M @ current
        if h in horizons:
            values[h] = current.copy()
    return values


def hitting(M, native_mask, max_h=32):
    """First hit includes time zero; native rows absorb only in this helper."""
    mask = np.asarray(native_mask, dtype=bool)
    cdf = mask.astype(float)
    result = {0: cdf.copy()}
    for h in range(1, max_h + 1):
        cdf = M @ cdf
        cdf[mask] = 1.0
        result[h] = cdf.copy()
    return result


def independent_mp_matrix(sigma, g, dps=80):
    """Independent raw-slot formula for standard q=2; never calls core.Kernel."""
    import mpmath as mp
    states = reader_states()
    with mp.workdps(dps):
        width = mp.mpf(str(sigma))
        M = mp.matrix(16)
        for i, p in enumerate(states):
            columns = []
            for c in g:
                weights = [mp.mpf(int(c == k)) if width == 0 else
                           mp.exp(-mp.mpf(c - k) ** 2 / (2 * width ** 2)) for k in p[::2]]
                total = sum(weights)
                if not total:
                    columns.append([mp.mpf(0), mp.mpf(0)])
                    continue
                success = max(weights)
                columns.append([success * sum(w for w, v in zip(weights, p[1::2]) if v == a) / total
                                for a in (0, 1)])
            for j, child in enumerate(states):
                M[i, j] = mp.fprod(columns[r][a] for r, a in enumerate(child))
        return M


def high_precision_audit(sigma, g, M, native_index, dps=80):
    import mpmath as mp
    with mp.workdps(dps):
        reference = independent_mp_matrix(sigma, g, dps=dps)
        max_error = max(abs(reference[i, j] - mp.mpf(float(M[i, j])))
                        for i in range(16) for j in range(16))
        positive_lost = sum(reference[i, j] > 0 and M[i, j] == 0
                            for i in range(16) for j in range(16))
        vals = (reference ** 32) * mp.matrix([1] * 16)
        approx = survival(M, (32,))[32]
        return dict(dps=dps, max_entry_absolute_error=str(max_error),
                    max_S32_absolute_error=str(max(abs(vals[i] - mp.mpf(float(approx[i]))) for i in range(16))),
                    native_S32_mp=str(vals[native_index]), positive_entries_lost=int(positive_lost))


def low_noise_times():
    """Small QT diagnostic; precision doubling, not an asymptotic proof."""
    import mpmath as mp
    g = (0, 0, 1, 1)
    native_index = reader_states().index(g)
    rows = []
    for sigma in (0.1, 0.15, 0.2, 0.25):
        values = []
        for dps in (70, 100):
            with mp.workdps(dps):
                M = independent_mp_matrix(sigma, g, dps=dps)
                expected = mp.lu_solve(mp.eye(16) - M, mp.matrix([1] * 16))[native_index]
                scaled = 2 * mp.mpf(str(sigma)) ** 2 * mp.log(expected)
                values.append((str(expected), str(scaled)))
        with mp.workdps(100):
            relative_delta = abs(mp.mpf(values[0][0]) / mp.mpf(values[1][0]) - 1)
        rows.append(dict(sigma=sigma, target="identity", dps=100,
                         expected_trials_including_failure=values[1][0],
                         two_sigma_squared_log_expected=values[1][1],
                         relative_precision_doubling_difference=str(relative_delta),
                         interpretation="finite_parameter_diagnostic_no_limit_claim"))
    return rows


def monte_carlo_check(kernel, b, g, initial, M, n=6000, seed=2026102407, horizon=4):
    """Trajectory sampler against a finite-h matrix probability."""
    states = reader_states(kernel.residues, len(g))
    exact = float(survival(M, (horizon,))[horizon][states.index(initial)])
    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(n):
        p = initial
        for _step in range(horizon):
            p = kernel.read_word(p, b, g, rng)
            if p is None:
                break
        else:
            successes += 1
    estimate = successes / n
    se = math.sqrt(exact * (1 - exact) / n)
    z = (estimate - exact) / se if se else 0.0
    assert abs(estimate - exact) <= max(6 * se, 1 / n)
    return dict(seed=seed, n=n, h=horizon, successes=successes, exact=exact,
                estimate=estimate, standard_error_under_exact=se, z_score=z,
                validation_rule="absolute error <= max(6 standard errors, 1/n)")


def q3_cycle_diagnostic(n=1000, seed=2026102408, horizon=32):
    """Finite trajectories from a nonnative 2-cycle, with all failures retained."""
    from .exact import direct_permutation_step, effective_map
    target, initial = (1, 0, 2), (1, 2, 0)
    g, initial_p = permutation_word(target), permutation_reader(initial)
    first = direct_permutation_step(initial, target)
    second = direct_permutation_step(first, target)
    cycle = {first, second}
    rows = []
    for sigma in (0.0, 0.2, 0.5):
        kernel = Kernel(standard_chemistry(3), sigma)
        for replicate in range(n):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(100 * sigma), replicate]))
            p, first_exit, first_failure = initial_p, None, None
            for h in range(1, horizon + 1):
                p = kernel.read_word(p, 0, g, rng)
                if p is None:
                    first_failure = h
                    break
                mapping = effective_map(p)
                in_cycle = (mapping is not None and set(mapping) == {0, 1, 2}
                            and tuple(mapping[i] for i in range(3)) in cycle)
                if not in_cycle and first_exit is None:
                    first_exit = h
            rows.append(dict(sigma=sigma, replicate_id=replicate, seed=seed,
                             seed_sequence=f"{seed},{int(100*sigma)},{replicate}", h=horizon,
                             first_cycle_exit="" if first_exit is None else first_exit,
                             first_read_failure="" if first_failure is None else first_failure,
                             successful_births=horizon if first_failure is None else first_failure - 1,
                             alive_serial_at_h=int(first_failure is None),
                             cycle_exit_censored=int(first_exit is None and first_failure is None),
                             termination_reason="horizon" if first_failure is None else "read_failed"))
    return rows


def q3_sampling_precision_audit(horizon=32, n=1000):
    """Bound CDF rounding error over every raw q3 reader, including rare states.

    This audits the CDF representation of float kernel probabilities. It does
    not claim exact arithmetic or a continuous-uniform random-number generator.
    """
    g = permutation_word((1, 0, 2))
    rows = []
    for sigma in (0.0, 0.2, 0.5):
        kernel = Kernel(standard_chemistry(3), sigma)
        max_word_tv, max_collapsed_mass, collapsed_bins = 0.0, 0.0, 0
        for p in reader_states((0, 1, 2), 6):
            word_tv = 0.0
            for c in g:
                log_s, probs = kernel.distribution(p, 0, c)
                if log_s == -math.inf:
                    continue
                cdf = np.clip(np.cumsum(probs), 0, 1)
                cdf[-1] = 1.0  # reader's final-bin fallback
                represented = np.diff(np.r_[0., cdf])
                lost = (probs > 0) & (represented == 0)
                collapsed_bins += int(lost.sum())
                max_collapsed_mass = max(max_collapsed_mass, float(probs[lost].sum()))
                word_tv += float(np.abs(represented - probs).sum() / 2)
            max_word_tv = max(max_word_tv, word_tv)
        rows.append(dict(sigma=sigma, raw_readers=729,
                         collapsed_positive_bins_across_all_reader_positions=collapsed_bins,
                         largest_collapsed_mass_one_position=max_collapsed_mass,
                         max_word_CDF_TV_bound=max_word_tv,
                         single_trajectory_32_step_CDF_TV_bound=horizon * max_word_tv,
                         all_1000_trajectories_CDF_union_bound=n * horizon * max_word_tv,
                         scope="CDF rounding only; kernel rounding and RNG grid separately limited"))
    return rows


def run_markov(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, recoveries, distance_rows, audits, hit_rows = [], [], [], [], []
    summary_rows = []
    panel = ((0, 0), (0, 1))
    for target_name, permutation in (("identity", (0, 1)), ("swap", (1, 0))):
        g, native = permutation_word(permutation), permutation_reader(permutation)
        for sigma in SIGMAS:
            kernel = Kernel(standard_chemistry(2), sigma)
            states, M, T, C, audit = build_matrix(kernel, 0, g)
            idx = states.index(native)
            distances = np.array([kernel.distance(p, native, panel) for p in states])
            native_mask = distances == 0.0
            assert native_mask[idx]
            assert native_mask.sum() == 2
            S = survival(M)
            hit = hitting(M, native_mask)
            occupancy = np.asarray(native_mask, float)
            occupancies = {}
            for h in range(1, 33):
                occupancy = M @ occupancy
                if h in HORIZONS:
                    occupancies[h] = occupancy.copy()
            rho = float(max(abs(np.linalg.eigvals(M))))
            condition = float(np.linalg.cond(np.eye(16) - M))
            expected = np.full(16, np.nan)
            status = "singular_sigma_zero" if sigma == 0 else "ill_conditioned_not_reported"
            if sigma > 0 and condition < 1e12 and rho < 1:
                expected = np.linalg.solve(np.eye(16) - M, np.ones(16))
                assert np.all(expected >= 1 - 1e-10)
                status = "double_precision_well_conditioned"
            constants = [states.index((a,) * 4) for a in kernel.residues]
            epsilon_bound = min(sum(M[i, j] * (1 - M[j, j]) for j in constants)
                                for i in range(16))
            mp_audit = high_precision_audit(sigma, g, M, idx)
            assert mp_audit["positive_entries_lost"] == 0
            assert float(mp_audit["max_entry_absolute_error"]) < 2e-13
            audits.append(dict(target=target_name, sigma=sigma, **audit, **mp_audit,
                               spectral_radius_raw=rho, I_minus_M_condition=condition,
                               expected_time_status=status, epsilon_two_step_bound=epsilon_bound,
                               raw_matrix_modified=False,
                               finite_h_probability_clipping="display only, raw values retained"))
            stem = f"{target_name}_sigma_{sigma:g}"
            np.savez_compressed(output_dir / f"{stem}.npz", M=M, T=T, states=np.array(states),
                                g=np.array(g), native=np.array(native), distances=distances,
                                native_mask=native_mask, C=C)
            write_csv(output_dir / f"{stem}_matrix.csv",
                      [dict(actor="failure" if i == 16 else encode(states[i]),
                            **{("failure" if j == 16 else encode(states[j])): float(T[i, j])
                               for j in range(17)}) for i in range(17)])
            summary_rows.append(dict(target=target_name, sigma=sigma, C=float(C[idx]),
                                     S32_raw=float(S[32][idx]), S32=float(np.clip(S[32][idx], 0, 1)),
                                     expected_trials=expected[idx] if np.isfinite(expected[idx]) else "",
                                     expected_time_status=status))
            for i, p in enumerate(states):
                for h in HORIZONS:
                    raw = float(S[h][i])
                    assert -1e-13 <= raw <= 1 + 1e-12
                    metrics.append(dict(target=target_name, sigma=sigma, actor=encode(p),
                                        h=h, C=float(C[i]), S_h_raw=raw, S_h=float(np.clip(raw, 0, 1)),
                                        S_h_given_first_birth=raw / C[i] if C[i] else "",
                                        first_hit_by_h=float(hit[h][i]),
                                        native_occupancy_at_h=float(occupancies[h][i]),
                                        D_to_native=float(distances[i]),
                                        expected_trials=expected[i] if np.isfinite(expected[i]) else "",
                                        expected_successes=expected[i] - 1 if np.isfinite(expected[i]) else "",
                                        expected_time_status=status))
                for h in range(33):
                    hit_rows.append(dict(target=target_name, sigma=sigma, actor=encode(p), h=h,
                                         first_hit_cdf=float(hit[h][i]),
                                         first_hit_pmf=float(hit[h][i] - hit[h - 1][i]) if h else float(hit[0][i])))
                for eps in EPSILONS:
                    joint = float(M[i, distances <= eps].sum())
                    recoveries.append(dict(target=target_name, sigma=sigma, actor=encode(p), epsilon=eps,
                                           C=float(C[i]), R_given_birth=joint / C[i] if C[i] else "",
                                           birth_and_recovery=joint))
                for j, child in enumerate(states):
                    distance_rows.append(dict(target=target_name, sigma=sigma, actor=encode(p),
                                              child=encode(child), D_to_native=float(distances[j]),
                                              birth_and_child=float(M[i, j]),
                                              child_given_birth=float(M[i, j] / C[i]) if C[i] else ""))
    write_csv(output_dir / "metrics.csv", metrics)
    write_csv(output_dir / "native_summary.csv", summary_rows)
    write_csv(output_dir / "recovery.csv", recoveries)
    write_csv(output_dir / "child_distance_distribution.csv", distance_rows)
    write_csv(output_dir / "first_hitting.csv", hit_rows)
    write_csv(output_dir / "numerical_audit.csv", audits)
    qt = low_noise_times()
    write_csv(output_dir / "low_noise_QT.csv", qt)
    test_kernel = Kernel(standard_chemistry(2), 0.5)
    test_g = (0, 0, 1, 1)
    _, test_M, _, _, _ = build_matrix(test_kernel, 0, test_g)
    mc = monte_carlo_check(test_kernel, 0, test_g, test_g, test_M)
    write_csv(output_dir / "monte_carlo_validation.csv", [mc])
    q3 = q3_cycle_diagnostic()
    write_csv(output_dir / "q3_cycle_trajectories.csv", q3)
    q3_summary = [dict(sigma=sigma, n=sum(r["sigma"] == sigma for r in q3),
                       survived_to_32=sum(r["alive_serial_at_h"] for r in q3 if r["sigma"] == sigma),
                       cycle_exited=sum(r["first_cycle_exit"] != "" for r in q3 if r["sigma"] == sigma))
                  for sigma in (0.0, 0.2, 0.5)]
    write_csv(output_dir / "q3_cycle_summary.csv", q3_summary)
    write_csv(output_dir / "q3_sampling_precision.csv", q3_sampling_precision_audit())
    qg_summary = approximate_graph_scan(output_dir)
    result = dict(conditions=12, states_including_failure=17,
                  native_summary=summary_rows, monte_carlo_validation=mc,
                  max_row_error=max(r["max_row_C_error"] for r in audits),
                  max_high_precision_S32_error=max(float(r["max_S32_absolute_error"]) for r in audits),
                  qt_conditions=len(qt), q3_cycle_summary=q3_summary, qg_summary=qg_summary)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def approximate_graph_scan(output_dir, delta=0.05, epsilon=0.1):
    """Bounded QG check with fixed capacity and fixed feature/probe tables."""
    candidates, summaries, counterexamples = [], [], []
    states = reader_states()
    panel = ((0, 0), (0, 1))
    for sigma in (0.0, 0.2, 0.5, 1.0):
        kernel = Kernel(standard_chemistry(2), sigma)
        distance_matrix = np.array([[kernel.distance(p, q, panel) for q in states] for p in states])
        eligible = []
        for g in states:
            _, M, _, C, _ = build_matrix(kernel, 0, g)
            for i, p in enumerate(states):
                probability = float(M[i, distance_matrix[i] <= delta].sum())
                qualifies = probability >= 1 - delta
                candidates.append(dict(sigma=sigma, q=2, m=2, delta=delta, epsilon=epsilon,
                                       g=encode(g), p=encode(p), self_C=float(C[i]),
                                       self_birth_and_D_le_delta=probability, qualifies=int(qualifies)))
                if qualifies:
                    eligible.append((g, p))
        matrix = np.array([[kernel.birth_probability(pi, 0, gj) for gj, _ in eligible]
                           for _, pi in eligible])
        clear = (matrix >= 1 - epsilon) | (matrix <= epsilon)
        edges = matrix >= 1 - epsilon
        evaluated, ambiguous, violations = 0, 0, 0
        for inds in itertools.combinations(range(len(eligible)), 3):
            if not clear[np.ix_(inds, inds)].all():
                ambiguous += 1
                continue
            evaluated += 1
            violated = any(edges[i, j] and edges[j, k] and not edges[i, k]
                           for i, j, k in itertools.permutations(inds))
            if violated:
                violations += 1
                counterexamples.append(dict(sigma=sigma, states=json.dumps([eligible[i] for i in inds]),
                                            birth_matrix=json.dumps(matrix[np.ix_(inds, inds)].tolist())))
        summaries.append(dict(sigma=sigma, q=2, m=2, delta=delta, epsilon=epsilon,
                              candidates=len(states) ** 2, qualifying_states=len(eligible),
                              clear_three_state_sets=evaluated,
                              threshold_ambiguous_three_state_sets=ambiguous,
                              nontransitive_three_state_sets=violations,
                              conclusion="bounded_scan_only_no_general_classification"))
    write_csv(Path(output_dir) / "QG_candidates.csv", candidates)
    write_csv(Path(output_dir) / "QG_summary.csv", summaries)
    (Path(output_dir) / "QG_counterexamples.json").write_text(json.dumps(counterexamples, indent=2), encoding="utf-8")
    return summaries
