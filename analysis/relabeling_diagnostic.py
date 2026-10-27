"""Exact residual-relabeling audit of two published Yomitsugi orbit witnesses.

Standard library only. This script neither imports the publication's group code
nor enumerates any new target family. It treats reader codes modulo the
centralizer of the fixed target and distinguishes this statewise quotient from
one simultaneous relabeling of an entire fixed-label trajectory.
"""

import json
from itertools import product
from pathlib import Path


def compose(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def inverse(p):
    r = [0] * len(p)
    for i, v in enumerate(p):
        r[v] = i
    return tuple(r)


def conjugate(c, p):
    return compose(compose(c, p), inverse(c))


def case(pi, initial, centralizer):
    assert len(set(centralizer)) == len(centralizer)
    assert all(compose(c, pi) == compose(pi, c) for c in centralizer)
    orbit, seen, z = [], {}, initial
    while z not in seen:
        seen[z] = len(orbit)
        orbit.append(z)
        z = conjugate(z, pi)
    transient = seen[z]
    cycle = orbit[transient:]

    def canonical(p):
        return min(conjugate(c, p) for c in centralizer)

    quotient = [canonical(p) for p in cycle]
    period = next(
        h for h in range(1, len(cycle) + 1)
        if all(quotient[i] == quotient[(i + h) % len(cycle)]
               for i in range(len(cycle)))
    )
    # Equivariance is checked at every recorded cycle point and relabeling.
    for p in cycle:
        for c in centralizer:
            assert conjugate(conjugate(c, p), pi) == conjugate(c, conjugate(p, pi))
    witnesses = [
        c for c in centralizer
        if conjugate(c, cycle[0]) == cycle[period % len(cycle)]
    ]
    assert witnesses
    return {
        "target": pi, "initial": initial, "centralizer": centralizer,
        "centralizer_size": len(centralizer), "transient": transient,
        "literal_code_period": len(cycle), "quotient_period": period,
        "distinct_quotient_classes": len(set(quotient)),
        "distinct_raw_codes": len(set(cycle)),
        "quotient_shift_witnesses": witnesses,
        "native_equivalent_to_cycle": any(canonical(p) == canonical(pi) for p in cycle),
        "cycle": cycle, "quotient_cycle": quotient[:period],
    }


def main():
    pi3, f3 = (0, 2, 1), (1, 0, 2)
    c3 = [(0, 1, 2), pi3]
    pi8 = (0, 2, 3, 1, 5, 6, 7, 4)
    f8 = (1, 4, 2, 6, 7, 3, 5, 0)
    c8 = []
    # Distinct cycle lengths 1, 3, 4 imply that these independent rotations
    # are the full centralizer, with 1 * 3 * 4 = 12 elements.
    for a, b in product(range(3), range(4)):
        c = list(range(8))
        for i in range(3):
            c[1 + i] = 1 + (i + a) % 3
        for i in range(4):
            c[4 + i] = 4 + (i + b) % 4
        c8.append(tuple(c))
    result = {
        "scope": (
            "Small exact diagnostic of two published witnesses; not a new broad "
            "enumeration. Statewise equivalence uses C(pi); literal code "
            "trajectories retain fixed labels."
        ),
        "q3": case(pi3, f3, c3),
        "q8": case(pi8, f8, c8),
    }
    assert result["q3"]["literal_code_period"] == 2
    assert result["q3"]["quotient_period"] == 1
    assert result["q8"]["literal_code_period"] == 192
    assert result["q8"]["quotient_period"] == 16
    assert not result["q3"]["native_equivalent_to_cycle"]
    assert not result["q8"]["native_equivalent_to_cycle"]
    destination = Path(__file__).resolve().parents[1] / "results/relabeling_diagnostic/orbits.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    for name in ("q3", "q8"):
        print(name, {k: result[name][k] for k in (
            "literal_code_period", "quotient_period", "centralizer_size",
            "distinct_quotient_classes", "native_equivalent_to_cycle"
        )})
    print("saved", destination)


if __name__ == "__main__":
    main()
