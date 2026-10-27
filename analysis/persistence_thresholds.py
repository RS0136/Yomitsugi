"""Exact binary repertoire/demographic thresholds and occupation times.

All primary curves are deterministic finite-horizon generating-function or
matrix recursions. Ultimate extinction values, where supplied, are explicitly
labelled numerical fixed-point approximations; phase classification never
uses convergence of that iteration. No Monte Carlo simulation is performed.
"""
from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analysis.finite import parameters, productive_eigenvalue

ROOT = Path(__file__).resolve().parents[1]


def _fraction(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


def matrix_from_parameters(u, a):
    """Class order is Z,C,A,P; finite interior parameters are required."""
    u, a = float(u), float(a)
    if not 0 < u < 1 or not 0 < a < .5:
        raise ValueError("Require 0 < u < 1 and 0 < a < 1/2")
    return np.array([[u, 0, 0, 0], [1, 0, 0, 0], [u/4]*4,
                     [a*a, a*(1-a), a*(1-a), (1-a)**2]])


def occupation_times(u, a):
    """Expected attempts by current repertoire, from any P-class reader."""
    matrix_from_parameters(u, a)
    u, a = float(u), float(a)
    retained = (4-u+4*a-4*a*a)/(a*(8-4*a-3*u+2*a*u))
    depleted = 1/(1-u)
    return dict(expected_two_value_attempts=retained,
                expected_depleted_attempts=depleted,
                expected_total_attempts=retained+depleted)


def classify(u, a, d, *, has_repertoire=True, nonempty=True):
    """Classify by exact polynomial comparisons, without eigensolver guesses.

    Fractions are exact. Floats are interpreted as their supplied decimal
    strings. A rounded irrational critical parameter is not the exact
    boundary; use a rational witness for equality tests.
    """
    u, a, d = map(_fraction, (u, a, d))
    if not 0 < u < 1 or not 0 < a < Fraction(1, 2) or not 0 < d < 1:
        raise ValueError("Interior u, a, d are required")
    x = d/(1-d)
    aa, bb = u/4, u/4
    cc, dd = a*(1-a), (1-a)**2
    if x <= max(aa, dd):
        h_state = "supercritical"  # Perron root strictly exceeds both diagonals.
    else:
        determinant = (x-aa)*(x-dd)-bb*cc
        h_state = ("subcritical" if determinant > 0 else
                   "critical" if determinant == 0 else "supercritical")
    z_state = ("supercritical" if u > x else
               "critical" if u == x else "subcritical")
    h_positive = bool(nonempty and has_repertoire and h_state == "supercritical")
    z_positive = bool(nonempty and z_state == "supercritical")
    if not nonempty:
        regime = "empty"
    elif h_positive and z_positive:
        regime = "both_persistence_fates_possible"
    elif h_positive:
        regime = "repertoire_required"
    elif z_positive:
        regime = "demographic_only" if has_repertoire else "depleted_survival_only"
    else:
        regime = "extinction_almost_sure"
    return dict(repertoire_block=h_state, depleted_block=z_state, regime=regime,
                total_survival_positive=h_positive or z_positive,
                repertoire_survival_positive=h_positive,
                demographic_only_survival_positive=z_positive,
                survival_events_equal=(not z_positive))


def offspring_pgf(z, kernel, d):
    """Independent retained parent and surviving newborn; substochastic kernel."""
    z, kernel = np.asarray(z, dtype=float), np.asarray(kernel, dtype=float)
    s = 1-float(d)
    return (d+s*z)*(1-s*kernel.sum(axis=1)+s*(kernel@z))


def extinction_curves(u, a, d, horizon=200, counts=(0, 0, 0, 1)):
    """Joint finite-horizon outcomes; H={A,P} is irreversibly closed to entry."""
    if int(horizon) != horizon or horizon < 0:
        raise ValueError("Nonnegative integer horizon required")
    counts = np.asarray(counts)
    if counts.shape != (4,) or np.any(counts < 0) or np.any(counts != counts.astype(int)):
        raise ValueError("Four nonnegative integer founder counts required")
    counts = counts.astype(int)
    L = matrix_from_parameters(u, a)
    Q = L[2:, 2:]
    B = (1-d)*(np.eye(4)+L)
    BH = (1-d)*(np.eye(2)+Q)
    z, x = np.zeros(4), np.zeros(2)
    mean, mean_h = counts.astype(float), counts[2:].astype(float)
    rows = []
    for h in range(horizon+1):
        p_ext = float(np.prod(z**counts))
        p_h_ext = float(np.prod(x**counts[2:]))
        remaining_depleted = p_h_ext-p_ext
        if remaining_depleted < -2e-13:
            raise ArithmeticError("Repertoire extinction must contain total extinction")
        rows.append(dict(h=h, total_extinction_probability=p_ext,
                         repertoire_extinction_probability=p_h_ext,
                         total_survival_probability=1-p_ext,
                         repertoire_survival_probability=1-p_h_ext,
                         alive_without_two_value_probability=max(0., remaining_depleted),
                         expected_total=float(mean.sum()),
                         expected_two_value=float(mean_h.sum())))
        z = np.clip(offspring_pgf(z, L, d), 0, 1)
        x = np.clip(offspring_pgf(x, Q, d), 0, 1)
        mean, mean_h = mean@B, mean_h@BH
    return rows


def ultimate_approximation(kernel, d, *, exact_extinction=False,
                           tolerance=2e-15, max_steps=200000):
    """Explicitly numerical minimal-fixed-point approximation away from criticality.

    Exact critical/subcritical conclusions must come from classify(), not
    from convergence. A small residual is not a certified probability error.
    """
    kernel = np.asarray(kernel, float)
    if exact_extinction:
        return dict(values=[1.]*len(kernel), status="exact_by_extinction_criterion",
                    iterations=0, residual=0.)
    rho = max(np.linalg.eigvals((1-d)*(np.eye(len(kernel))+kernel)).real)
    if abs(rho-1) < 1e-6:
        raise ValueError("Do not infer a critical classification by fixed-point iteration")
    z = np.zeros(len(kernel))
    for iteration in range(1, max_steps+1):
        nxt = np.clip(offspring_pgf(z, kernel, d), 0, 1)
        if max(abs(nxt-z)) < tolerance:
            z = nxt
            break
        z = nxt
    else:
        raise ArithmeticError("Fixed-point iteration did not reach the recorded tolerance")
    residual = float(max(abs(offspring_pgf(z, kernel, d)-z)))
    return dict(values=z.tolist(), status="numerical_approximation_not_certified_error_bound",
                iterations=iteration, residual=residual,
                stopping_increment=tolerance,
                interpretation="Residual records an algebraic check, not a rigorous error bound.")


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(root=ROOT):
    root = Path(root)
    out = root/"results/persistence_thresholds"
    out.mkdir(parents=True, exist_ok=True)
    widths = np.unique(np.r_[np.geomspace(.15, 8, 181), .2, .5, .8,
                            .9953945221067288, 1, 2, 4])
    diagonal = []
    for sigma in widths:
        u, a = parameters(float(sigma))
        lam = productive_eigenvalue(u, a)
        diagonal.append(dict(sigma=float(sigma), u=u, a=a,
                             lambda_repertoire=lam,
                             death_threshold_repertoire=lam/(1+lam),
                             death_threshold_depleted=u/(1+u),
                             death_threshold_total=max(u, lam)/(1+max(u, lam)),
                             **occupation_times(u, a)))
    _csv(out/"diagonal_thresholds.csv", diagonal)

    curves, cases = [], []
    for sigma, d in ((.8, .2), (2., .35)):
        u, a = parameters(sigma); L = matrix_from_parameters(u, a)
        classification = classify(u, a, d)
        total = ultimate_approximation(L, d,
                   exact_extinction=not classification["total_survival_positive"])
        retained = ultimate_approximation(L[2:, 2:], d,
                   exact_extinction=not classification["repertoire_survival_positive"])
        cases.append(dict(sigma=sigma, d=d, u=u, a=a,
                          classification=classification,
                          ultimate_total_extinction=total,
                          ultimate_repertoire_extinction=retained))
        for founders in (1, 60):
            for row in extinction_curves(u, a, d, counts=(0, 0, 0, founders)):
                curves.append(dict(case=f"sigma{sigma:g}_d{d:g}_n{founders}",
                                   sigma=sigma, death=d, founders=founders, **row))
    _csv(out/"finite_horizon_extinction.csv", curves)

    # Rational equality case: both irreducible blocks are exactly critical.
    u, a, d = Fraction(5, 8), Fraction(1, 4), Fraction(5, 13)
    critical = extinction_curves(float(u), float(a), float(d))
    for row in critical:
        h = row["h"]
        row["expected_total_closed"] = float(Fraction(63, 221)*h+Fraction(241, 289)+
                                             Fraction(48, 289)*Fraction(35, 52)**h)
    _csv(out/"critical_rational_witness.csv", critical)
    summary = dict(
        status="complete", deterministic_only=True, diagonal_points=len(diagonal),
        horizon=200, class_order=["Z", "C", "A", "P"],
        founder_class="P; both reference and alternative P readers obey the same class law",
        assumptions=["one fixed binary identity or swap target", "finite positive widths",
                     "one independent attempt per parent per sweep", "unlimited resources",
                     "independent parent and newborn survival 1-d", "no immigration or innovation"],
        cases=cases,
        critical_witness=dict(u="5/8", a="1/4", death="5/13",
                              classification=classify(u, a, d),
                              expected_total="(63/221) h + 241/289 + (48/289)(35/52)^h",
                              interpretation="Extinction is almost sure although the mean grows linearly."),
        broad_diagonal_limits=dict(expected_two_value_attempts=2,
                                   death_threshold_repertoire=1/3,
                                   death_threshold_depleted=1/2),
        rounding="CSV curves are floating-point evaluations of exact recursions; ultimate numerical fixed points are separately labelled.")
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    sources=[root/"analysis/persistence_thresholds.py", root/"analysis/finite.py"]
    source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    artifacts=[p for p in out.iterdir() if p.name!="manifest.json"]
    manifest=dict(source_sha256=source_hashes,
                  artifact_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(artifacts)})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"persistence_thresholds":"complete", "diagonal_points":len(diagonal),
                      "primary_curve_rows":len(curves), "critical_witness_rows":len(critical)}))


if __name__ == "__main__":
    run()
