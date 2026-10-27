"""Independent raw-state, rational-boundary and occupation checks."""
from fractions import Fraction as F
import unittest

import numpy as np

from analysis.finite import build_matrix, parameters, reader_classes
from analysis.persistence_thresholds import (
    classify, extinction_curves, matrix_from_parameters, occupation_times,
    offspring_pgf,
)


class PersistenceThresholdTests(unittest.TestCase):
    def test_reduced_offspring_law_against_every_raw_parent(self):
        for sr, sd in ((.5, .5), (.8, .8), (2., 2.), (.4, 1.1), (1.2, .4)):
            u, a = parameters(sr, sd); L = matrix_from_parameters(u, a)
            for template in ((0, 0, 1, 1), (1, 0, 0, 1)):
                M, words = build_matrix(sr, sd, template)
                classes = reader_classes(words)
                for z, d in ((np.array([.1, .3, .6, .8]), .2),
                             (np.array([.8, .2, .7, .4]), .35)):
                    # Direct full offspring enumeration: zero, parent only,
                    # newborn only, parent plus newborn.
                    s = 1-d; mapped = z[classes]
                    raw = d*(1-s*M.sum(axis=1))
                    raw += s*mapped*(1-s*M.sum(axis=1))
                    raw += d*s*(M@mapped)
                    raw += s*s*mapped*(M@mapped)
                    np.testing.assert_allclose(raw, offspring_pgf(z, L, d)[classes],
                                               rtol=3e-14, atol=5e-16)

    def test_repertoire_pgf_and_curves_against_full_eight_type_restriction(self):
        for sigma, d in ((.8, .2), (2., .35)):
            M, words = build_matrix(sigma)
            classes = reader_classes(words); keep = classes >= 2
            full_h = M[np.ix_(keep, keep)]
            native = list(map(tuple, words)).index((0, 0, 1, 1))
            native_h = list(np.flatnonzero(keep)).index(native)
            u, a = parameters(sigma)
            rows = extinction_curves(u, a, d, horizon=200)
            z, x = np.zeros(16), np.zeros(8)
            for h, row in enumerate(rows):
                if h in (0, 1, 2, 10, 50, 200):
                    self.assertAlmostEqual(row['total_extinction_probability'], z[native], places=13)
                    self.assertAlmostEqual(row['repertoire_extinction_probability'], x[native_h], places=13)
                    self.assertGreaterEqual(row['alive_without_two_value_probability'], 0)
                    self.assertAlmostEqual(row['total_survival_probability'],
                         row['repertoire_survival_probability']+row['alive_without_two_value_probability'], places=14)
                z = offspring_pgf(z, M, d)
                x = offspring_pgf(x, full_h, d)

    def test_physical_occupation_decomposition_against_raw_inverse(self):
        for sr, sd in ((.5, .5), (.8, .8), (2., 2.), (.4, 1.1), (1.2, .4)):
            M, states = build_matrix(sr, sd)
            cl = reader_classes(states)
            u, a = parameters(sr, sd); prediction = occupation_times(u, a)
            for initial in np.flatnonzero(cl == 3):
                occupation = np.linalg.solve((np.eye(16)-M).T, np.eye(16)[initial])
                self.assertAlmostEqual(occupation[cl >= 2].sum(), prediction['expected_two_value_attempts'], places=12)
                self.assertAlmostEqual(occupation[cl < 2].sum(), prediction['expected_depleted_attempts'], places=12)
                self.assertAlmostEqual(occupation[cl == 1].sum(), occupation[cl == 2].sum(), places=13)
                self.assertAlmostEqual((1-u)*occupation[np.isin(cl, [0, 2])].sum(), 1., places=13)

    def test_exact_critical_and_marginal_boundaries(self):
        # Equal Perron/depleted rate =5/8; no floating-point boundary heuristic.
        result = classify(F(5, 8), F(1, 4), F(5, 13))
        self.assertEqual(result['repertoire_block'], 'critical')
        self.assertEqual(result['depleted_block'], 'critical')
        self.assertEqual(result['regime'], 'extinction_almost_sure')
        # lambda=3/5 > u=2/5: depleted criticality still permits H survival.
        result = classify(F(2, 5), F(1, 4), F(2, 7))
        self.assertEqual(result['depleted_block'], 'critical')
        self.assertEqual(result['regime'], 'repertoire_required')
        result = classify(F(2, 5), F(1, 4), F(3, 8))
        self.assertEqual(result['repertoire_block'], 'critical')
        self.assertFalse(result['total_survival_positive'])
        # lambda=2/3 < u=20/21: critical H dies but depleted population may survive.
        result = classify(F(20, 21), F(1, 4), F(2, 5))
        self.assertEqual(result['repertoire_block'], 'critical')
        self.assertEqual(result['regime'], 'demographic_only')
        result = classify(F(20, 21), F(1, 4), F(20, 41))
        self.assertEqual(result['depleted_block'], 'critical')
        self.assertFalse(result['total_survival_positive'])
        epsilon = F(1, 10**9)
        self.assertTrue(classify(F(5, 8), F(1, 4), F(5, 13)-epsilon)['total_survival_positive'])
        self.assertFalse(classify(F(5, 8), F(1, 4), F(5, 13)+epsilon)['total_survival_positive'])

    def test_initial_support_is_not_replaced_by_global_spectral_radius(self):
        u, a = parameters(.8)
        self.assertEqual(classify(u, a, .2)['regime'], 'repertoire_required')
        result = classify(u, a, .2, has_repertoire=False)
        self.assertEqual(result['regime'], 'extinction_almost_sure')
        result = classify(*parameters(2), .35, has_repertoire=False)
        self.assertEqual(result['regime'], 'depleted_survival_only')
        self.assertFalse(classify(*parameters(2), .35, nonempty=False)['total_survival_positive'])
        for row in extinction_curves(u, a, .2, horizon=5, counts=(3, 2, 0, 0)):
            self.assertEqual(row['repertoire_extinction_probability'], 1.)
            self.assertEqual(row['expected_two_value'], 0.)

    def test_first_step_events_and_independent_founders(self):
        u, a = parameters(.8); d = .2; s = 1-d
        one = extinction_curves(u, a, d, horizon=1)[1]
        many = extinction_curves(u, a, d, horizon=1, counts=(0, 0, 0, 3))[1]
        self.assertAlmostEqual(one['total_extinction_probability'], d*d)
        self.assertAlmostEqual(one['repertoire_extinction_probability'], d*(1-s*(1-a)))
        self.assertAlmostEqual(many['total_extinction_probability'], (d*d)**3)
        self.assertAlmostEqual(many['repertoire_extinction_probability'], (d*(1-s*(1-a)))**3)

    def test_critical_linear_mean_with_exact_rational_arithmetic(self):
        u, a, s = F(5, 8), F(1, 4), F(8, 13)
        L = [[u, 0, 0, 0], [1, 0, 0, 0], [u/4]*4,
             [a*a, a*(1-a), a*(1-a), (1-a)**2]]
        B = [[s*(F(i == j)+L[i][j]) for j in range(4)] for i in range(4)]
        vector = [F(0), F(0), F(0), F(1)]
        for h in range(12):
            exact = F(63, 221)*h+F(241, 289)+F(48, 289)*F(35, 52)**h
            self.assertEqual(sum(vector), exact)
            vector = [sum(vector[i]*B[i][j] for i in range(4)) for j in range(4)]

    def test_broad_width_retained_lifetime_remains_two(self):
        u, a = parameters(1000.)
        value = occupation_times(u, a)
        self.assertAlmostEqual(value['expected_two_value_attempts'], 2-1/(2*1000**2), places=11)
        self.assertGreater(value['expected_depleted_attempts'], 10**6)


if __name__ == '__main__':
    unittest.main()
