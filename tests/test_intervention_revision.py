"""Independent event-law checks for the raw16 review extensions."""
import itertools
import math
import unittest
from fractions import Fraction as F

import numpy as np

from analysis.finite import build_matrix, channels
from analysis import intervention_revision as ir
from analysis.resource_population import WORDS, WORD_CLASSES, raw_to_classes


class InterventionRevisionTests(unittest.TestCase):
    def test_mutation_against_direct_position_law_and_exact_lumping(self):
        for sigma in (.2, .8, 2.):
            for mu in (0., 1e-6, .03, .5):
                for g in ((0, 0, 1, 1), (1, 0, 0, 1)):
                    raw, _ = ir.mutated_matrix(sigma, mu, g)
                    states, gate, output = channels(sigma, g=g)
                    changed_output = (1-mu)*output+mu*output[:, :, ::-1]
                    direct = np.ones((16, 16))
                    for j, symbol in enumerate(g):
                        direct *= gate[:, symbol, None]*changed_output[:, symbol, states[:, j]]
                    np.testing.assert_allclose(raw, direct, rtol=3e-13, atol=1e-15)
                    L = ir.mutated_class_matrix(sigma, mu)
                    for c in range(4):
                        np.testing.assert_allclose(raw[:, WORD_CLASSES == c].sum(axis=1),
                                                   L[WORD_CLASSES, c], rtol=3e-13, atol=1e-15)

    def test_mutation_endpoints_and_return_to_repertoire(self):
        for sigma in (.5, .8, 2.):
            matrix, _ = build_matrix(sigma)
            mutated, _ = ir.mutated_matrix(sigma, 0.)
            np.testing.assert_array_equal(matrix, mutated)
            uniform, _ = ir.mutated_matrix(sigma, .5)
            np.testing.assert_allclose(uniform, np.broadcast_to(matrix.sum(axis=1)[:, None]/16, (16, 16)), atol=1e-15)
            positive, _ = ir.mutated_matrix(sigma, 1e-6)
            self.assertTrue(np.all(positive > 0))
            self.assertGreater(positive[0, ir.H].sum(), 0)
            b = 2*1e-6*(1-1e-6)
            self.assertAlmostEqual(positive[0, ir.H].sum(), math.exp(-1/sigma**2)*b, places=15)
            initial = np.zeros(16, dtype=int); initial[0] = 1
            self.assertAlmostEqual(ir.unrestricted_curves(positive, initial, 1)[1]["H_survival"],
                                   .8*math.exp(-1/sigma**2)*b, places=14)

    def test_nonrepertoire_control_preserves_all_required_invariants(self):
        initial = np.zeros(16, dtype=int)
        initial[0], initial[15], initial[2], initial[5] = 40, 45, 3, 2
        changed, report = ir.prepare_arm(initial, "sham10")
        self.assertTrue(all(report["checks"].values()))
        self.assertFalse(np.array_equal(initial, changed))
        np.testing.assert_array_equal(raw_to_classes(initial), raw_to_classes(changed))
        np.testing.assert_array_equal(initial@WORDS, changed@WORDS)
        for sigma in (.5, .8, 2.):
            M, _ = build_matrix(sigma)
            self.assertAlmostEqual(M[0].sum(), M[5].sum())
            self.assertAlmostEqual(M[15].sum(), M[10].sum())
            a = ir.unrestricted_curves(M, initial, 20)
            b = ir.unrestricted_curves(M, changed, 20)
            for x, y in zip(a, b):
                self.assertAlmostEqual(x["N_survival"], y["N_survival"], places=13)
                self.assertEqual(x["H_survival"], 0.)
                self.assertEqual(x["reference_ever"], 0.)

    def test_unrestricted_reference_census_pgf_one_step(self):
        M, _ = build_matrix(.8)
        initial = np.zeros(16, dtype=int)
        initial[1] = initial[14] = 10
        curve = ir.unrestricted_curves(M, initial, 2)
        no_reference = np.prod((1-.8*M[:, ir.REF].sum(axis=1))**initial)
        self.assertAlmostEqual(curve[1]["reference_ever"], 1-no_reference, places=14)
        # A birth that dies before the census is not a reference appearance.
        pre_death_probability = 1-np.prod((1-M[:, ir.REF].sum(axis=1))**initial)
        self.assertLess(curve[1]["reference_ever"], pre_death_probability)
        self.assertEqual(curve[0]["reference_ever"], 0.)
        native = np.zeros(16, dtype=int); native[3] = 1
        self.assertEqual(ir.unrestricted_curves(M, native, 0)[0]["reference_ever"], 1.)

    def test_unrestricted_pgf_matches_literal_one_parent_offspring(self):
        M, _ = build_matrix(.8)
        death = .2
        for parent in (0, 1, 3, 5, 14, 15):
            no_N = no_H = no_ref = 0.
            for parent_survives in (False, True):
                pp = (1-death) if parent_survives else death
                for child in range(17):
                    cp = (1-death)*M[parent, child] if child < 16 else 1-(1-death)*M[parent].sum()
                    offspring = ([parent] if parent_survives else [])+([child] if child < 16 else [])
                    probability = pp*cp
                    no_N += probability*bool(not offspring)
                    no_H += probability*all(not ir.H[i] for i in offspring)
                    no_ref += probability*all(not ir.REF[i] for i in offspring)
            initial = np.zeros(16, dtype=int); initial[parent] = 1
            row = ir.unrestricted_curves(M, initial, 1)[1]
            self.assertAlmostEqual(row["N_survival"], 1-no_N, places=14)
            self.assertAlmostEqual(row["H_survival"], 1-no_H, places=14)
            self.assertAlmostEqual(row["reference_ever"], 1. if ir.REF[parent] else 1-no_ref, places=14)

    def test_raw_sampler_deterministic_and_census_reference_boundaries(self):
        M, _ = build_matrix(0.)
        initial = np.zeros(16, dtype=int); initial[3] = 3
        counts, resources, births = ir.simulate_raw(initial, 0, M, 0., 2, 2, 3, 771)
        np.testing.assert_array_equal(counts.sum(axis=1), [3, 5, 7, 9])
        np.testing.assert_array_equal(counts[:, ir.REF].sum(axis=1), [3, 5, 7, 9])
        np.testing.assert_array_equal(births, [0, 2, 2, 2])
        counts, _, _ = ir.simulate_raw(initial, 0, M, 1., 2, 2, 3, 771)
        self.assertFalse(counts[1:].any())
        self.assertTrue(counts[0, ir.REF].any())

    def test_raw_seed_reproduction_and_extinction_absorption(self):
        M, _ = build_matrix(.8)
        initial = np.zeros(16, dtype=int); initial[1] = initial[14] = 2
        seed = ir.seed_for(1, 0, 1, 0)
        first = ir.simulate_raw(initial, 0, M, .2, 3, 6, 100, seed)
        second = ir.simulate_raw(initial, 0, M, .2, 3, 6, 100, seed)
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)
        counts, resources, births = first
        self.assertTrue(np.all(np.diff((counts[:, ir.H].sum(axis=1) == 0).astype(int)) >= 0))
        for h in range(1, len(counts)):
            self.assertEqual(resources[h], min(6, resources[h-1]+3)-births[h])
            self.assertLessEqual(counts[h].sum(), counts[h-1].sum()+births[h])

    def test_square_root_mode_splitting_by_exact_determinant_derivative(self):
        # A rational crossing checks the determinant coefficient without
        # numerical differentiation or an eigensolver.
        u, a = F(5, 8), F(1, 4)
        self.assertEqual(3*u, (1-a)*(3-2*a))
        L = [[u, 0, 0, 0], [1, 0, 0, 0], [u/4]*4,
             [a*a, a*(1-a), a*(1-a), (1-a)**2]]
        derivative = [[-4*u, 2*u, 2*u, 0], [-4, 2, 2, 0], [0]*4,
                      [4*a*(1-2*a), 2*(1-2*a)**2, 2*(1-2*a)**2, -4*(1-a)*(1-2*a)]]
        A = [[u*int(i == j)-L[i][j] for j in range(4)] for i in range(4)]
        def determinant(matrix):
            value = F(0)
            for permutation in itertools.permutations(range(len(matrix))):
                sign = (-1)**sum(permutation[i] > permutation[j]
                                for i in range(len(matrix)) for j in range(i+1, len(matrix)))
                term = F(sign)
                for i, j in enumerate(permutation): term *= matrix[i][j]
                value += term
            return value
        coefficient = F(0)
        for i in range(4):
            for j in range(4):
                minor = [[A[r][c] for c in range(4) if c != j] for r in range(4) if r != i]
                coefficient -= (-1)**(i+j)*determinant(minor)*derivative[i][j]
        expected = -u*u*a*(4*(1-a)+u*(1+2*a))/6
        self.assertEqual(coefficient, expected)
        lower = (1-a)*(1-2*a)/4
        self.assertGreater(-coefficient/(u*(u-lower)), 0)
        # Independent floating check on the actual one-width crossing.
        from analysis.finite import parameters
        sigma = .9953945221067288
        u, a = parameters(sigma)
        lower = (1-a)*(1-2*a)/4
        k = u*a*(4*(1-a)+u*(1+2*a))/(6*(u-lower))
        for mu in (1e-8, 1e-7, 1e-6):
            values = np.sort(np.linalg.eigvals(ir.mutated_class_matrix(sigma, mu)).real)
            ratio = (values[-1]-values[-2])/(2*math.sqrt(k*mu))
            self.assertLess(abs(ratio-1), 5e-6)


if __name__ == "__main__":
    unittest.main()
