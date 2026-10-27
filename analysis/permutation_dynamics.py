"""Exact zero-width inheritance dynamics on complete permutation readers.

The descriptor is fixed. Only its reader is reconstructed at each birth.
Permutations act on the left: compose(f, g)[x] == f[g[x]]. A target code pi
induces T_pi(f) = f pi f^-1. One target per cycle type suffices because of
simultaneous relabeling equivariance. This symmetry is NOT assumed for the
positive-width model with ordered numeric feature distances.

Run: python -m analysis.permutation_dynamics --max-q 8
No q^(2q)-state word enumeration or q!-by-q! matrix is constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

import numpy as np


def compose(f, g):
    """Return f after g, as a tuple."""
    return tuple(f[x] for x in g)


def inverse(f):
    result = [0] * len(f)
    for x, value in enumerate(f):
        result[value] = x
    return tuple(result)


def conjugate(f, target):
    return compose(compose(f, target), inverse(f))


def standard_commutator(x, g):
    """[x,g] = x^-1 g^-1 x g; the standard left-Engel convention."""
    return compose(compose(inverse(x), inverse(g)), compose(x, g))


def partitions(n, minimum=1):
    """Increasing integer partitions, including fixed points explicitly."""
    if n == 0:
        yield ()
    for first in range(minimum, n + 1):
        for rest in partitions(n - first, first):
            yield (first,) + rest


def canonical_permutation(cycle_type):
    """Place each cycle on a successive interval of labels."""
    result = list(range(sum(cycle_type)))
    offset = 0
    for length in cycle_type:
        if length < 1:
            raise ValueError("Cycle lengths must be positive")
        for j in range(length):
            result[offset + j] = offset + (j + 1) % length
        offset += length
    return tuple(result)


def centralizer_size(cycle_type):
    return math.prod(length**count * math.factorial(count)
                     for length, count in Counter(cycle_type).items())


def permutation_ranks(permutations):
    """Zero-based lexicographic (Lehmer) ranks for a two-dimensional array."""
    values = np.asarray(permutations)
    if values.ndim != 2:
        raise ValueError("Expected an array of permutations")
    n, q = values.shape
    ranks = np.zeros(n, dtype=np.int32)
    for i in range(q - 1):
        smaller = (values[:, i + 1:] < values[:, i, None]).sum(axis=1)
        ranks += smaller.astype(np.int32) * math.factorial(q - 1 - i)
    return ranks


def successor_indices(actors, target):
    """Vectorized conjugation, with one successor index per actor."""
    actors = np.asarray(actors)
    target = np.asarray(target)
    if actors.ndim != 2 or len(target) != actors.shape[1]:
        raise ValueError("Target and actor alphabets must agree")
    # T(f)(f(k)) = f(pi(k)): no inverse table or pairwise matrix is needed.
    children = np.empty_like(actors)
    children[np.arange(len(actors))[:, None], actors] = actors[:, target]
    return permutation_ranks(children)


def functional_graph(next_index):
    """Resolve every orbit in linear time, retaining a canonical attractor ID.

    attractor_index is the minimum lexicographic state index on the cycle.
    distance is the number of transitions before first entering that cycle.
    period is the cycle length, including for transient basin states.
    """
    successor = np.asarray(next_index, dtype=np.int32)
    n = len(successor)
    if successor.ndim != 1 or np.any(successor < 0) or np.any(successor >= n):
        raise ValueError("Successors must form a finite total function")
    attractor = np.full(n, -1, dtype=np.int32)
    period = np.zeros(n, dtype=np.int32)
    distance = np.zeros(n, dtype=np.int32)
    for start in range(n):
        if attractor[start] >= 0:
            continue
        path, local = [], {}
        state = start
        while attractor[state] < 0 and state not in local:
            local[state] = len(path)
            path.append(state)
            state = int(successor[state])
        if attractor[state] < 0:
            cycle_start = local[state]
            cycle = path[cycle_start:]
            canonical, length = min(cycle), len(cycle)
            attractor[cycle] = canonical
            period[cycle] = length
            path = path[:cycle_start]
        for previous in reversed(path):
            following = successor[previous]
            attractor[previous] = attractor[following]
            period[previous] = period[following]
            distance[previous] = distance[following] + 1
    return dict(attractor_index=attractor, period=period, distance=distance)


def transposition_nonrecovery_fraction(q):
    if q < 2:
        raise ValueError("A transposition requires at least two symbols")
    return Fraction(4 * (q - 2), q * (q - 1))


def finite_width_shadowing(q, sigma_d, horizon):
    """Exact probability of following a noiseless permutation-word trajectory.

    Every descriptor contains every input twice and every on-path reader has
    every key once. At each input the exact slot has conditional probability
    1/Z_c, with Z_c=sum_k exp(-(c-k)^2/(2 sigma_d^2)). Recognition is one for
    any recognition width because the exact key is present. This is a finite-
    horizon path probability, not a probability of an infinite noisy cycle.
    """
    if not isinstance(q, int) or q < 1:
        raise ValueError("Alphabet size must be a positive integer")
    if not math.isfinite(sigma_d) or sigma_d < 0:
        raise ValueError("Decoding width must be finite and nonnegative")
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("Horizon must be a nonnegative integer")
    log_step = 0.0
    if sigma_d > 0:
        for symbol in range(q):
            off_match = math.fsum(math.exp(-(symbol - key)**2 / (2 * sigma_d**2))
                                 for key in range(q) if key != symbol)
            log_step -= 2 * math.log1p(off_match)
    return dict(q=q, sigma_D=float(sigma_d), horizon=horizon,
                step_probability=math.exp(log_step),
                log_step_probability=log_step,
                path_probability=math.exp(horizon * log_step),
                log_path_probability=horizon * log_step)


def is_universally_recovering_cycle_type(cycle_type):
    """Baer's theorem plus the Fitting subgroup of S_q; not an orbit fit."""
    q = sum(cycle_type)
    part = tuple(sorted(cycle_type))
    return (q <= 2 or part == (1,) * q or
            (q == 3 and part == (3,)) or
            (q == 4 and part == (2, 2)))


def analyse_target(actors, cycle_type):
    target = canonical_permutation(cycle_type)
    q, n = len(target), len(actors)
    target_index = int(permutation_ranks([target])[0])
    successors = successor_indices(actors, target)
    graph = functional_graph(successors)
    native = graph['attractor_index'] == target_index
    first_hit = np.where(native, graph['distance'], -1).astype(np.int32)
    # The native actor is already native at h=0, and is the unique fixed point.
    assert successors[target_index] == target_index
    assert np.count_nonzero(successors == np.arange(n)) == 1
    assert first_hit[target_index] == 0
    first_child_count = int(np.count_nonzero(successors == target_index))
    expected_centralizer = centralizer_size(cycle_type)
    assert first_child_count == expected_centralizer
    assert len(np.unique(successors)) == n // expected_centralizer
    native_count = int(native.sum())
    assert (native_count == n) == is_universally_recovering_cycle_type(cycle_type)
    is_transposition = tuple(cycle_type) == (1,) * (q - 2) + (2,) if q >= 2 else False
    expected = transposition_nonrecovery_fraction(q) if is_transposition else None
    if expected is not None:
        assert Fraction(n - native_count, n) == expected
        assert int(graph['period'].max()) == (1 if q == 2 else 2)
    cycle_ids, basin_sizes = np.unique(graph['attractor_index'], return_counts=True)
    type_name = '-'.join(map(str, cycle_type))
    cycle_rows = [dict(q=q, cycle_type=type_name, attractor_index=int(a),
                       period=int(graph['period'][a]), basin_size=int(size),
                       is_native=bool(a == target_index))
                  for a, size in zip(cycle_ids, basin_sizes)]
    period_rows = [dict(q=q, cycle_type=type_name, period=int(period),
                        number_of_cycles=sum(c['period'] == period for c in cycle_rows),
                        cycle_members=int(np.count_nonzero((graph['period'] == period) &
                                                          (graph['distance'] == 0))),
                        actor_count=int(np.count_nonzero(graph['period'] == period)),
                        actor_fraction=float(np.mean(graph['period'] == period)))
                   for period in np.unique(graph['period'])]
    summary = dict(q=q, cycle_type=type_name, total_actors=n,
                   target_index=target_index, centralizer_size=expected_centralizer,
                   conjugacy_class_size=n // expected_centralizer,
                   first_child_native_count=first_child_count,
                   eventual_native_count=native_count,
                   never_native_count=n - native_count,
                   never_native_fraction=(n - native_count) / n,
                   number_of_cycles=len(cycle_ids),
                   max_period=int(graph['period'].max()),
                   max_transient_distance=int(graph['distance'].max()),
                   max_native_first_hit=int(first_hit.max()),
                   all_actors_recover=native_count == n,
                   transposition_prediction_numerator=expected.numerator if expected is not None else '',
                   transposition_prediction_denominator=expected.denominator if expected is not None else '')
    arrays = dict(target=np.asarray(target, dtype=np.uint8),
                  cycle_type=np.asarray(cycle_type, dtype=np.uint8),
                  next_index=successors, native_basin=native,
                  first_native_time=first_hit, **graph)
    return summary, cycle_rows, period_rows, arrays


def _write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root=None, max_q=8):
    if not isinstance(max_q, int) or not 1 <= max_q <= 8:
        raise ValueError("This reproducibility enumeration supports 1 <= max_q <= 8")
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    out = root / 'results' / 'permutation_dynamics'
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summaries, cycles, periods = [], [], []
    for q in range(1, max_q + 1):
        actors = np.asarray(list(itertools.permutations(range(q))), dtype=np.uint8)
        assert np.array_equal(permutation_ranks(actors), np.arange(len(actors)))
        np.savez_compressed(out / f'actors_q{q:02d}.npz', actors=actors)
        for cycle_type in partitions(q):
            row, cycle_rows, period_rows, arrays = analyse_target(actors, cycle_type)
            summaries.append(row)
            cycles.extend(cycle_rows)
            periods.extend(period_rows)
            np.savez_compressed(out / f'q{q:02d}_type_{row["cycle_type"]}.npz', **arrays)
    _write_csv(out / 'target_summary.csv', summaries)
    _write_csv(out / 'attractor_basins.csv', cycles)
    _write_csv(out / 'period_distribution.csv', periods)
    shadowing = [finite_width_shadowing(q, sigma_d, horizon)
                 for q in (3, 8) for sigma_d in (.15, .2, .25)
                 for horizon in (32, 192)]
    _write_csv(out / 'finite_width_shadowing.csv', shadowing)
    execution = dict(
        scope='Complete permutation readers, exact matching at zero width; fixed descriptors',
        enumeration='Every actor in S_q; one canonical target per conjugacy class',
        counting_measure='Uniform initial actor f_0 in S_q; native first hit includes h=0',
        symmetry='Simultaneous conjugation of target and actor; zero width only',
        q_min=1, q_max=max_q, target_types=len(summaries),
        actor_target_pairs=sum(row['total_actors'] for row in summaries),
        total_attractors=len(cycles), largest_period=max(row['max_period'] for row in summaries),
        per_reader_files='actors_qXX.npz stores shared lexicographic catalog; qXX_type_*.npz stores the functional graph',
        finite_width_shadowing='Exact probability r_q(sigma_D)^h of following the zero-width literal-word trajectory; lower bound on no native hit through h for an initially never-native zero-width actor',
        shadowing_rows=len(shadowing),
        validation=['Every first-child native count equals its centralizer size',
                    'Every image size equals its target conjugacy-class size',
                    'The target is the unique fixed point in every graph',
                    'All universal-recovery classifications agree with the Fitting criterion',
                    'All transposition nonrecovery counts equal the all-q formula'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python_version=platform.python_version(), numpy_version=np.__version__,
        elapsed_seconds=time.perf_counter() - started)
    (out / 'execution.json').write_text(json.dumps(execution, indent=2) + '\n', encoding='utf8')
    print(json.dumps(execution, indent=2))
    return execution


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--max-q', type=int, default=8)
    arguments = parser.parse_args()
    run(arguments.root, arguments.max_q)
