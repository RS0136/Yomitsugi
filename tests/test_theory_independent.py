"""Independent model/proof checks, not generated from the closed formulas.

Run from project root with:
    python -m unittest discover -s tests -p 'test_theory_independent.py' -v

The direct enumerator deliberately avoids the production vectorization.
Exact rational first-step calculations independently test the lifetime law.
"""
import itertools
import math
import unittest
from fractions import Fraction

import numpy as np

from analysis.finite import (
    build_matrix, curve_row, decoding_information_p, lifetime_closed,
    lumped_matrix, native_mask, parameters, reader_classes,
)


WORDS = tuple(itertools.product((0, 1), repeat=4))
TEMPLATES = ((0, 0, 1, 1), (1, 0, 0, 1))


def direct_class(word):
    return int(word[0] != word[2]) + 2 * int(word[1] != word[3])


def direct_channel(word, sigma_r, sigma_d):
    """Original slot-level gate and conditional draw, evaluated literally."""
    gates, conditional = [], []
    slots = tuple(zip(word[::2], word[1::2]))
    for symbol in (0, 1):
        recognition = [math.exp(-(symbol-k)**2/(2*sigma_r*sigma_r))
                       for k, _ in slots]
        weights = [math.exp(-(symbol-k)**2/(2*sigma_d*sigma_d))
                   for k, _ in slots]
        gates.append(max(recognition))
        conditional.append([
            sum(w for w, (_, v) in zip(weights, slots) if v == residue)/sum(weights)
            for residue in (0, 1)
        ])
    return gates, conditional


def direct_matrix(sigma_r, sigma_d, template):
    matrix = np.zeros((16, 16))
    for i, parent in enumerate(WORDS):
        gates, conditional = direct_channel(parent, sigma_r, sigma_d)
        for j, child in enumerate(WORDS):
            matrix[i, j] = math.prod(
                gates[symbol]*conditional[symbol][residue]
                for symbol, residue in zip(template, child))
    return matrix


def exact_first_step_lifetime(u, alpha):
    """Rational enumeration, followed by exact elimination of E=1+L E."""
    beta = 1-alpha
    representatives = ((0, 0, 0, 0), (0, 0, 1, 0),
                       (0, 0, 0, 1), (0, 0, 1, 1))
    reduced = [[Fraction(0) for _ in range(4)] for _ in range(4)]
    for i, parent in enumerate(representatives):
        gate = u if parent[0] == parent[2] else Fraction(1)
        for child in WORDS:
            if parent[1] == parent[3]:
                chance = Fraction(int(all(x == parent[1] for x in child)))
            elif parent[0] == parent[2]:
                chance = Fraction(1, 16)
            else:
                chance = math.prod(alpha if x == c else beta
                                   for c, x in zip(TEMPLATES[0], child))
            reduced[i][direct_class(child)] += gate*chance
    augmented = [
        [Fraction(i == j)-reduced[i][j] for j in range(4)] + [Fraction(1)]
        for i in range(4)
    ]
    for column in range(4):
        pivot = next(row for row in range(column, 4) if augmented[row][column])
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [x/scale for x in augmented[column]]
        for row in range(4):
            if row != column:
                scale = augmented[row][column]
                augmented[row] = [x-scale*y for x, y in
                                  zip(augmented[row], augmented[column])]
    return augmented[3][-1]


class IndependentTheoryTests(unittest.TestCase):
    def test_complete_transition_law_against_slot_enumerator(self):
        for sr, sd in ((.2, .2), (.5, .5), (1., 1.), (4., 4.),
                       (.3, 1.2), (1.5, .4)):
            for template in TEMPLATES:
                with self.subTest(sr=sr, sd=sd, template=template):
                    observed, states = build_matrix(sr, sd, template)
                    self.assertEqual(tuple(map(tuple, states)), WORDS)
                    expected = direct_matrix(sr, sd, template)
                    np.testing.assert_allclose(observed, expected, atol=5e-16,
                                               rtol=4e-14)

    def test_strong_lumping_and_repertoire_support(self):
        classes = np.array([direct_class(p) for p in WORDS])
        for sr, sd in ((.25, .7), (.5, .5), (1.3, .35), (3., 3.)):
            for template in TEMPLATES:
                matrix, states = build_matrix(sr, sd, template)
                np.testing.assert_array_equal(reader_classes(states), classes)
                reduced = lumped_matrix(sr, sd)
                for source, parent in enumerate(WORDS):
                    for destination in range(4):
                        self.assertAlmostEqual(
                            matrix[source, classes == destination].sum(),
                            reduced[classes[source], destination], places=14)
                    values = set(parent[1::2])
                    for destination, child in enumerate(WORDS):
                        if not set(child).issubset(values):
                            self.assertEqual(matrix[source, destination], 0.)

    def test_reference_and_homogeneous_rewards_against_raw_states(self):
        homogeneous = np.array([len(set(p)) == 1 for p in WORDS])
        for sr, sd in ((0., 0.), (.5, .5), (1., 1.), (3., 3.), (.3, 1.2), (1.5, .4)):
            for template in TEMPLATES:
                matrix, states = build_matrix(sr, sd, template)
                native = native_mask(states, template)
                expected_native = {template, template[2:]+template[:2]}
                self.assertEqual({p for p, keep in zip(WORDS, native) if keep},
                                 expected_native)
                vector = np.zeros(16)
                vector[WORDS.index(template)] = 1
                for horizon in range(33):
                    if horizon in (0, 1, 2, 4, 8, 16, 32):
                        row = curve_row(sr, sd, horizon=horizon)
                        with self.subTest(sr=sr, sd=sd, template=template,
                                          horizon=horizon):
                            self.assertAlmostEqual(row['S_h'], vector.sum(), places=13)
                            self.assertAlmostEqual(row['native_joint'],
                                                   vector[native].sum(), places=13)
                            self.assertAlmostEqual(row['homogeneous_joint'],
                                                   vector[homogeneous].sum(), places=13)
                    vector = vector @ matrix

    def test_lifetime_against_exact_rational_first_step_solution(self):
        for u in (Fraction(1, 7), Fraction(2, 5), Fraction(5, 8),
                  Fraction(4, 5), Fraction(11, 12)):
            for alpha in (Fraction(3, 5), Fraction(2, 3), Fraction(3, 4),
                          Fraction(4, 5), Fraction(9, 10)):
                sr = math.sqrt(-1/math.log(float(u)))
                td = float((1-alpha)/alpha)
                sd = math.sqrt(-1/(2*math.log(td)))
                expected = float(exact_first_step_lifetime(u, alpha))
                self.assertTrue(math.isclose(lifetime_closed(sr, sd), expected,
                                             rel_tol=5e-14))

    def test_endpoint_lifetime_scalings(self):
        sigma = .1
        scaled = 4*math.exp(-1/(2*sigma*sigma))*lifetime_closed(sigma)
        self.assertAlmostEqual(scaled, 1., places=13)
        sigma = 200.
        self.assertLess(abs(lifetime_closed(sigma)/sigma**2-1), 7e-5)

    def test_decoding_information_from_each_explicit_channel(self):
        for sd in (.2, .5, 1., 4.):
            for word in WORDS:
                _, q = direct_channel(word, .7, sd)
                marginal = [(q[0][a]+q[1][a])/2 for a in (0, 1)]
                information = sum(.5*q[c][a]*math.log2(q[c][a]/marginal[a])
                                  for c in (0, 1) for a in (0, 1)
                                  if q[c][a] > 0)
                expected = decoding_information_p(sd) if direct_class(word) == 3 else 0.
                self.assertAlmostEqual(information, expected, places=14)

    def test_dominant_eigenvalues_and_crossing_classifier(self):
        for sr, sd in ((.2, .2), (.5, .5), (1., 1.), (4., 4.),
                       (.3, 1.2), (1.5, .4)):
            matrix, _ = build_matrix(sr, sd)
            reduced = lumped_matrix(sr, sd)
            u, a = parameters(sr, sd)
            lam = max(np.linalg.eigvals(reduced[2:, 2:]).real)
            actual = max(abs(np.linalg.eigvals(matrix)))
            self.assertAlmostEqual(actual, max(u, lam), places=13)
            self.assertEqual(lam > u, 3*u < (1-a)*(3-2*a))

    def test_critical_resonance_and_opposite_asymptotic_regimes(self):
        u, a = .625, .25
        td = a/(1-a+math.sqrt(1-2*a))
        sr = math.sqrt(-1/math.log(u))
        sd = math.sqrt(-1/(2*math.log(td)))
        reduced = lumped_matrix(sr, sd)
        normalized_survivals, repertoire_fractions = [], []
        for h in (32, 64, 128):
            vector = np.array([0., 0., 0., 1.]) @ np.linalg.matrix_power(reduced, h)
            normalized_survivals.append(vector.sum()/u**h)
            repertoire_fractions.append(vector[2:].sum()/vector.sum())
        # A linear h factor, rather than a pure exponential, at the crossing.
        slope_ratio = ((normalized_survivals[2]-normalized_survivals[1]) /
                       (normalized_survivals[1]-normalized_survivals[0]))
        self.assertAlmostEqual(slope_ratio, 2., places=11)
        self.assertTrue(1.9 < repertoire_fractions[1]/repertoire_fractions[2] < 2.1)
        below = curve_row(.5, horizon=128)
        above = curve_row(2., horizon=128)
        self.assertGreater(below['two_value_joint']/below['S_h'], .1)
        self.assertLess(above['two_value_joint']/above['S_h'], 1e-10)

    def test_unique_lifetime_minimum_certificate(self):
        # Exact rational-polynomial derivative and its t=x/(1+x) transform.
        numerator = [4, 40, 35, 6, -29, -30, 1, 4, 1]
        denominator = [0, 16, 16, -6, -20, -16, 4, 6]
        def multiply(left, right):
            answer = [0]*(len(left)+len(right)-1)
            for i, x in enumerate(left):
                for j, y in enumerate(right):
                    answer[i+j] += x*y
            return answer
        def derivative(poly):
            return [i*poly[i] for i in range(1, len(poly))]
        first = multiply(derivative(numerator), denominator)
        second = multiply(numerator, derivative(denominator))
        polynomial = [x-y for x, y in zip(first, second)]
        degree = len(polynomial)-1
        transformed = [sum(polynomial[k]*math.comb(degree-k, j-k)
                           for k in range(j+1)) for j in range(degree+1)]
        self.assertTrue(all(x < 0 for x in transformed[:7]))
        self.assertTrue(all(x > 0 for x in transformed[7:]))
        # Relate the certificate to the actual model, not only its algebra.
        for t in (.05, .25, .5, .75, .95):
            polynomial_value = (sum(x*t**i for i, x in enumerate(numerator)) /
                                sum(x*t**i for i, x in enumerate(denominator)))
            sigma = math.sqrt(-1/(2*math.log(t)))
            self.assertTrue(math.isclose(polynomial_value, lifetime_closed(sigma),
                                         rel_tol=2e-13))


if __name__ == '__main__':
    unittest.main()
