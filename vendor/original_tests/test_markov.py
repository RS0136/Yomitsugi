import unittest
import numpy as np
from yomitsugi.core import Kernel, NumericalPrecisionError, standard_chemistry
from yomitsugi.markov import build_matrix, survival, hitting, high_precision_audit, monte_carlo_check


class MarkovTests(unittest.TestCase):
    def test_positive_transition_underflow_rejected(self):
        with self.assertRaises(NumericalPrecisionError):
            build_matrix(Kernel(standard_chemistry(2), .05), 0, (0, 0, 1, 1))

    def test_rows_and_native_reference_values(self):
        expected = {0.: 1., .2: .99954545334499, .5: 6.1174762368946e-7, 1.: 9.0098596260593e-13}
        for g, p in (((0, 0, 1, 1), (0, 0, 1, 1)), ((1, 0, 0, 1), (0, 1, 1, 0))):
            for sigma, value in expected.items():
                kernel = Kernel(standard_chemistry(2), sigma)
                states, M, T, C, audit = build_matrix(kernel, 0, g)
                np.testing.assert_allclose(T.sum(axis=1), 1., atol=1e-14)
                np.testing.assert_allclose(M.sum(axis=1), C, atol=1e-14)
                self.assertEqual(C[states.index(p)], 1.)
                self.assertAlmostEqual(survival(M, (32,))[32][states.index(p)] / value, 1., places=11)

    def test_independent_high_precision_reference(self):
        kernel = Kernel(standard_chemistry(2), .2)
        g = (0, 0, 1, 1)
        states, M, _, _, _ = build_matrix(kernel, 0, g)
        audit = high_precision_audit(.2, g, M, states.index(g))
        self.assertEqual(audit['positive_entries_lost'], 0)
        self.assertLess(float(audit['max_S32_absolute_error']), 1e-13)

    def test_native_first_hit_does_not_absorb_actual_dynamics(self):
        kernel = Kernel(standard_chemistry(2), .5)
        p = (0, 0, 1, 1)
        states, M, _, _, _ = build_matrix(kernel, 0, p)
        mask = np.array([kernel.distance(s, p, ((0, 0), (0, 1))) == 0 for s in states])
        self.assertEqual(mask.sum(), 2)
        original = M.copy()
        H = hitting(M, mask)
        idx = states.index(p)
        self.assertEqual(H[32][idx], 1.)
        self.assertLess((M @ mask.astype(float))[idx], 1.)
        np.testing.assert_array_equal(M, original)

    def test_value_set_and_two_step_bound(self):
        kernel = Kernel(standard_chemistry(2), .5)
        states, M, _, _, _ = build_matrix(kernel, 0, (0, 0, 1, 1))
        for i, p in enumerate(states):
            for j, q in enumerate(states):
                if M[i, j] > 0:
                    self.assertLessEqual(set(q[1::2]), set(p[1::2]))
        constant = [states.index((a,) * 4) for a in (0, 1)]
        epsilon = min(sum(M[i, j] * (1 - M[j, j]) for j in constant) for i in range(16))
        self.assertGreater(epsilon, 0)
        self.assertTrue(np.all(survival(M, (32,))[32] <= (1 - epsilon) ** 16 + 1e-13))

    def test_matrix_matches_independent_trajectories(self):
        kernel = Kernel(standard_chemistry(2), .5)
        p = (0, 0, 1, 1)
        _, M, _, _, _ = build_matrix(kernel, 0, p)
        result = monte_carlo_check(kernel, 0, p, p, M, n=2000, seed=91372)
        self.assertLess(abs(result['z_score']), 6)


if __name__ == '__main__':
    unittest.main()
