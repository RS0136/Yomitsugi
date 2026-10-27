"""Independent finite calculations for the deterministic permutation limit.

Composition acts right-to-left. Direct decoding uses key/value lookups and
does not invoke the conjugation formula used by the comparison implementation.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path


def encode(values):
    return " ".join(map(str, values))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compose(a, b):
    return tuple(a[b[i]] for i in range(len(a)))


def inverse(a):
    result = [0] * len(a)
    for i, value in enumerate(a):
        result[value] = i
    return tuple(result)


def conjugate(f, target):
    return compose(compose(f, target), inverse(f))


def permutation_reader(f):
    return tuple(x for pair in enumerate(f) for x in pair)


def permutation_word(target):
    inv = inverse(target)
    return tuple(x for k in range(len(target)) for x in (inv[k], k))


def effective_map(p):
    """Return a deterministic finite map; None denotes ambiguous duplicate keys."""
    result = {}
    for k, v in zip(p[::2], p[1::2]):
        if k in result and result[k] != v:
            return None
        result[k] = v
    return result


def direct_read(p, g):
    lookup = effective_map(p)
    if lookup is None:
        raise ValueError("direct_read is restricted to deterministic readers")
    if any(c not in lookup for c in g):
        return None
    return tuple(lookup[c] for c in g)


def direct_permutation_step(f, target):
    child = direct_read(permutation_reader(f), permutation_word(target))
    lookup = effective_map(child)
    assert lookup is not None and set(lookup) == set(range(len(f)))
    return tuple(lookup[k] for k in range(len(f)))


def cycle_type(f):
    seen, sizes = set(), []
    for start in range(len(f)):
        if start in seen:
            continue
        cur, size = start, 0
        while cur not in seen:
            seen.add(cur)
            size += 1
            cur = f[cur]
        sizes.append(size)
    return tuple(sorted(sizes, reverse=True))


def orbit(start, step):
    seen, path, cur = {}, [], start
    while cur not in seen:
        seen[cur] = len(path)
        path.append(cur)
        cur = step(cur)
    transient = seen[cur]
    return path, transient, len(path) - transient


def enumerate_permutations(q):
    perms = list(itertools.permutations(range(q)))
    rows, transitions = [], []
    for target in perms:
        for actor in perms:
            direct = direct_permutation_step(actor, target)
            group = conjugate(actor, target)
            assert direct == group
            assert (direct == target) == (compose(actor, target) == compose(target, actor))
            path, transient, period = orbit(actor, lambda f: conjugate(f, target))
            recovery = path.index(target) if target in path else None
            rows.append(dict(q=q, target=encode(target), actor=encode(actor),
                             target_cycle_type=encode(cycle_type(target)),
                             first_child=encode(direct), birth_probability=1,
                             one_step_recovery=int(direct == target),
                             eventual_recovery=int(recovery is not None),
                             recovery_time="" if recovery is None else recovery,
                             transient=transient, period=period,
                             orbit=json.dumps(path),
                             cycle=json.dumps(path[transient:])))
            transitions.append(dict(q=q, target=encode(target), actor=encode(actor),
                                    child=encode(direct), is_native=int(actor == target),
                                    in_native_basin=int(recovery is not None)))
        size = Counter(cycle_type(target))
        centralizer = math.prod(length ** n * math.factorial(n) for length, n in size.items())
        assert sum(conjugate(f, target) == target for f in perms) == centralizer
    return rows, transitions


def enumerate_self_descriptions():
    words = list(itertools.product(range(2), repeat=4))
    valid = []
    for p in words:
        mapping = effective_map(p)
        if mapping is None:
            continue
        for g in words:
            child = direct_read(p, g)
            if child is not None and effective_map(child) == mapping:
                assert set(g) == set(mapping)
                assert set(mapping.values()) == set(mapping)
                valid.append((g, p))
    pairs = []
    for gi, pi in valid:
        for gj, pj in valid:
            actual = direct_read(pi, gj) is not None
            predicted = set(pj[::2]) <= set(pi[::2])
            assert actual == predicted
            pairs.append(dict(actor_g=encode(gi), actor_p=encode(pi),
                              target_g=encode(gj), target_p=encode(pj),
                              birth_probability=int(actual), inclusion=int(predicted)))
    assert len(valid) == 10 and len({p for _, p in valid}) == 6
    return valid, pairs


def enumerate_preorders(n=3):
    """Exhaustively validate the reverse construction for all n-node preorders."""
    offdiag = [(i, j) for i in range(n) for j in range(n) if i != j]
    rows = []
    for bits in itertools.product((0, 1), repeat=len(offdiag)):
        rel = {(i, i) for i in range(n)} | {ij for ij, bit in zip(offdiag, bits) if bit}
        if any((i, k) not in rel for i, j in rel for j2, k in rel if j == j2):
            continue
        keys = [{j for ii, j in rel if ii == i} for i in range(n)]
        m = max(map(len, keys))
        states = []
        for keyset in keys:
            seq = sorted(keyset)
            seq += [seq[0]] * (m - len(seq))
            p = tuple(x for k in seq for x in (k, k))
            states.append(p)
        for i, j in itertools.product(range(n), repeat=2):
            assert (direct_read(states[i], states[j]) is not None) == ((i, j) in rel)
        rows.append(dict(n=n, relation=json.dumps(sorted(rel)), slots=m,
                         realized_states=json.dumps(states), verified=True))
    return rows


def counterexamples():
    target = (0, 0, 1, 1, 0, 0)
    actors = [("identity", (0, 0, 1, 1, 2, 2), (0, 0, 1, 1, 2, 2)),
              ("cycle", (0, 1, 1, 2, 2, 0), (2, 0, 0, 1, 1, 2))]
    rows = []
    for label, p, own_g in actors:
        assert effective_map(direct_read(p, own_g)) == effective_map(p)
        child = direct_read(p, target)
        rows.append(dict(case=label, target_g=encode(target), actor_p=encode(p),
                         actor_self_word=encode(own_g), actor_self_reproduces=True,
                         C=1, child=encode(child), S2=int(direct_read(child, target) is not None),
                         parent_copy_control_S2=1))
    # The separately specified swap-target example, with duplicated first slot.
    swap_g = (1, 0, 0, 1, 1, 0)
    cycle_p = actors[1][1]
    child = direct_read(cycle_p, swap_g)
    assert set(child[::2]) == {1, 2} and direct_read(child, swap_g) is None
    rows.append(dict(case="cycle_to_swap", target_g=encode(swap_g), actor_p=encode(cycle_p),
                     actor_self_word=encode(actors[1][2]), actor_self_reproduces=True,
                     C=1, child=encode(child), S2=0, parent_copy_control_S2=1))
    return rows


def alternating_targets():
    perms = list(itertools.permutations(range(3)))
    rows = []
    for a, b, initial in itertools.product(perms, repeat=3):
        step = lambda f: conjugate(conjugate(f, a), b)
        path, transient, period = orbit(initial, step)
        phases = [(conjugate(f, a), step(f)) for f in path]
        recurrent_phases = phases[transient:]
        native_pair = any(x == a and y == b for x, y in recurrent_phases)
        rows.append(dict(target_a=encode(a), target_b=encode(b), actor=encode(initial),
                         transient_periods=transient, cycle_periods=period,
                         single_a_period=orbit(initial, lambda f: conjugate(f, a))[2],
                         single_b_period=orbit(initial, lambda f: conjugate(f, b))[2],
                         recurrent_phase_native_pair=int(native_pair),
                         initial_or_transient_native_a=int(any(x == a for x, _ in phases)),
                         initial_or_transient_native_b=int(any(y == b for _, y in phases)),
                         stroboscopic_orbit=json.dumps(path), phase_outputs=json.dumps(phases)))
    assert len(rows) == 216
    return rows


def run_exact(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, transitions, summary = [], [], []
    expected = {2: (4, 4, 4), 3: (36, 18, 24), 4: (576, 120, 240)}
    for q in (2, 3, 4):
        rows, edges = enumerate_permutations(q)
        all_rows.extend(rows)
        transitions.extend(edges)
        values = (len(rows), sum(r["one_step_recovery"] for r in rows),
                  sum(r["eventual_recovery"] for r in rows))
        assert values == expected[q]
        summary.append(dict(q=q, ordered_pairs=values[0], one_step_recovery=values[1],
                            eventual_recovery=values[2], not_recovered=values[0] - values[2]))
    write_csv(output_dir / "permutation_orbits.csv", all_rows)
    write_csv(output_dir / "permutation_transitions.csv", transitions)
    write_csv(output_dir / "summary.csv", summary)
    typed = []
    for q in (2, 3, 4):
        for typ in sorted({r["target_cycle_type"] for r in all_rows if r["q"] == q}):
            group = [r for r in all_rows if r["q"] == q and r["target_cycle_type"] == typ]
            targets = len({r["target"] for r in group})
            typed.append(dict(q=q, cycle_type=typ, number_targets=targets,
                              actors_per_target=math.factorial(q),
                              recovered_per_target=sum(r["eventual_recovery"] for r in group) // targets,
                              nonrecovery_period_counts=json.dumps(dict(sorted(Counter(
                                  r["period"] for r in group if not r["eventual_recovery"]).items())))))
    write_csv(output_dir / "cycle_types.csv", typed)
    valid, inclusion = enumerate_self_descriptions()
    write_csv(output_dir / "self_descriptions.csv",
              [dict(g=encode(g), p=encode(p), effective_map=json.dumps(effective_map(p))) for g, p in valid])
    write_csv(output_dir / "inclusion_checks.csv", inclusion)
    write_csv(output_dir / "preorder_constructions.csv", enumerate_preorders())
    write_csv(output_dir / "equal_first_birth_counterexamples.csv", counterexamples())
    qm = alternating_targets()
    write_csv(output_dir / "alternating_q3.csv", qm)
    result = dict(summary=summary, direct_group_agreements=len(all_rows),
                  self_descriptions=len(valid), distinct_raw_readers=len({p for _, p in valid}),
                  inclusion_checks=len(inclusion), q3_preorders=len(enumerate_preorders()),
                  alternating_conditions=len(qm),
                  alternating_period_counts=dict(sorted(Counter(r["cycle_periods"] for r in qm).items())))
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
