"""Independent checks of exact resource stopping and the revision experiment."""
from collections import Counter, defaultdict
from fractions import Fraction as F
from itertools import permutations, product
from math import comb
from pathlib import Path
import sys
import unittest

import numpy as np

from analysis import resource_population as rp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"vendor"))
from yomitsugi.core import Kernel, standard_chemistry


def literal_accepted(laws, resource):
    """Exact actor-by-actor stopping; no hypergeometric argument is used."""
    n = len(laws)
    live = len(laws[0])-1
    orders = list(permutations(range(n)))
    result = defaultdict(F)
    for outcome in product(range(live+1), repeat=n):
        probability = F(1)
        for i, choice in enumerate(outcome):
            probability *= laws[i][choice]
        for order in orders:
            accepted = [0]*live
            for actor in order:
                if sum(accepted) >= resource:
                    break
                if outcome[actor] != live:
                    accepted[outcome[actor]] += 1
            result[tuple(accepted)] += probability/len(orders)
    return {k: v for k, v in result.items() if v}


def hypergeom_accepted(laws, resource):
    n = len(laws)
    live = len(laws[0])-1
    result = defaultdict(F)
    for outcome in product(range(live+1), repeat=n):
        probability = F(1)
        for i, choice in enumerate(outcome):
            probability *= laws[i][choice]
        inventory = [outcome.count(j) for j in range(live)]
        total = sum(inventory)
        take = min(resource, total)
        for sample in product(range(take+1), repeat=live):
            if sum(sample) != take or any(k > y for k, y in zip(sample, inventory)):
                continue
            numerator = 1
            for y, k in zip(inventory, sample):
                numerator *= comb(y, k)
            result[sample] += probability*F(numerator, comb(total, take))
    return {k: v for k, v in result.items() if v}


class ExactResourcePopulationTests(unittest.TestCase):
    def test_rational_permutation_law_and_sampler(self):
        laws = ((F(1,2), F(1,2), F(0), F(0)),
                (F(1,6), F(1,3), F(0), F(1,2)),
                (F(1,6), F(1,6), F(0), F(2,3)))
        for capacity in range(4):
            expected = literal_accepted(laws, capacity)
            self.assertEqual(expected, hypergeom_accepted(laws, capacity))
            self.assertEqual(sum(expected.values()), F(1))
            if capacity in (1, 2):
                rng = np.random.default_rng(81900+capacity)
                observed = Counter()
                initial = np.ones(3, dtype=np.int64)
                for _ in range(6000):
                    counts, resource, births = rp.step_counts(initial, capacity, np.array(laws, dtype=float), 0., 0, capacity, rng)
                    observed[tuple(counts-initial)] += 1
                    self.assertEqual(resource+births, capacity)
                self.assertTrue(set(observed).issubset(expected))
                for state, probability in expected.items():
                    p = float(probability)
                    self.assertLessEqual(abs(observed[state]/6000-p), 6*np.sqrt(p*(1-p)/6000)+1e-12)

    def test_exposure_against_literal_success_and_order_enumeration(self):
        probabilities = [F(1), F(1,2), F(1,3)]
        for resource in range(4):
            expected = [F(0)]*3
            for successes in product((0, 1), repeat=3):
                mass = F(1)
                for p, success in zip(probabilities, successes):
                    mass *= p if success else 1-p
                for order in permutations(range(3)):
                    used = 0
                    for i in order:
                        if used >= resource:
                            break
                        expected[i] += mass/6
                        used += successes[i]
            actual = rp.exposure_probabilities([float(x) for x in probabilities], resource)
            np.testing.assert_allclose(actual, [float(x) for x in expected], atol=2e-15, rtol=0)
            for i in range(3):
                for j in range(3):
                    self.assertLessEqual(abs(actual[i]-actual[j]), float(abs(probabilities[i]-probabilities[j]))/(resource+1)+1e-14)
        binary = rp.binary_exposures(2, 3, .21, 2)
        full = rp.exposure_probabilities([1., 1., .21, .21, .21], 2)
        self.assertAlmostEqual(binary["high"], full[0])
        self.assertAlmostEqual(binary["low"], full[2])

    def test_original_kernel_implies_four_class_birth_law(self):
        states = list(product((0, 1), repeat=4))
        for sigma in (0., .5, .8):
            kernel = Kernel(standard_chemistry(), sigma)
            L = rp.class_matrix(sigma)
            for target in ((0, 0, 1, 1), (1, 0, 0, 1)):
                for reader in states:
                    parent_class = int(reader[0] != reader[2])+2*int(reader[1] != reader[3])
                    output_laws = [kernel.full_distribution(reader, 0, c) for c in target]
                    summed = np.zeros(4)
                    for child in states:
                        probability = np.prod([output_laws[j][child[j]] for j in range(4)])
                        child_class = int(child[0] != child[2])+2*int(child[1] != child[3])
                        summed[child_class] += probability
                    np.testing.assert_allclose(summed, L[parent_class], atol=1e-15, rtol=2e-13)

    def test_matched_rewiring_preserves_slots_success_and_information(self):
        raw = np.zeros(16, dtype=np.int64)
        raw[0] = 15
        raw[15] = 17
        raw[2] = 3  # A mixed-key single-value C type remains unchanged.
        changed, audit = rp.rewire_counts(raw, 10)
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(raw.sum(), changed.sum())
        np.testing.assert_array_equal(raw@rp.WORDS, changed@rp.WORDS)
        kernel = Kernel(standard_chemistry(), .8)
        for before, after in (((0,0,0,0), (0,0,0,1)), ((1,1,1,1), (1,1,1,0))):
            self.assertEqual(before[::2], after[::2])
            self.assertAlmostEqual(kernel.birth_probability(before, 0, (0,0,1,1)), kernel.birth_probability(after, 0, (0,0,1,1)))
            # The conditional successful-output distribution is input-independent
            # for both the homogeneous and same-key/two-value readers.
            for reader in (before, after):
                laws = [np.array(kernel.full_distribution(reader, 0, c)[:-1]) for c in (0,1)]
                np.testing.assert_allclose(laws[0]/laws[0].sum(), laws[1]/laws[1].sum(), atol=1e-15)
        old_c = np.repeat([kernel.birth_probability(tuple(p), 0, (0,0,1,1)) for p in rp.WORDS], raw)
        new_c = np.repeat([kernel.birth_probability(tuple(p), 0, (0,0,1,1)) for p in rp.WORDS], changed)
        np.testing.assert_allclose(np.sort(old_c), np.sort(new_c), atol=1e-15)

    def test_fluid_jacobian_by_independent_finite_differences(self):
        for sigma in (.5, .8):
            result = rp.fluid_fixed_point(sigma)
            L = rp.class_matrix(sigma)
            point = np.r_[result["counts"], 0.]
            def direct_map(z):
                available = min(120., z[4]+30.)
                potential = z[:4]@L
                amount = min(available, potential.sum())
                children = potential*amount/potential.sum()
                return np.r_[.8*(z[:4]+children), available-amount]
            h = 2e-4
            numerical = np.array([(direct_map(point+h*e)-direct_map(point-h*e))/(2*h) for e in np.eye(5)])
            np.testing.assert_allclose(numerical, result["jacobian_row_convention"], atol=2e-8, rtol=0)
            np.testing.assert_allclose(direct_map(point), point, atol=1e-12, rtol=0)
            np.testing.assert_allclose(result["jacobian_eigenvalues"], result["predicted_eigenvalues"], atol=2e-14)
            self.assertLess(result["maximum_eigenvalue_modulus"], 1.)
            self.assertGreater(result["saturation_margin"], 0.)

    def test_fluid_cap_boundary_uses_physical_one_sided_derivative(self):
        for sigma in (.5, .8):
            result = rp.fluid_fixed_point(sigma, supply=30., capacity=30.)
            L = rp.class_matrix(sigma)
            point = np.r_[result["counts"], 0.]
            def direct_map(z):
                available = min(30., 30.+z[4])
                candidate = z[:4]@L
                births = min(available, candidate.sum())
                return np.r_[.8*(z[:4]+births*candidate/candidate.sum()), available-births]
            h = 2e-4
            numerical = np.zeros((5, 5))
            for i, e in enumerate(np.eye(5)):
                if i == 4:
                    numerical[i] = (direct_map(point+h*e)-direct_map(point))/h
                else:
                    numerical[i] = (direct_map(point+h*e)-direct_map(point-h*e))/(2*h)
            np.testing.assert_allclose(numerical, result["jacobian_row_convention"], atol=2e-8, rtol=0)
            np.testing.assert_array_equal(result["jacobian_row_convention"][4], np.zeros(5))
            np.testing.assert_allclose(direct_map(point), point, atol=1e-12, rtol=0)
            np.testing.assert_allclose(direct_map(point+np.array([0,0,0,0,.25])), point, atol=1e-12, rtol=0)
            self.assertLess(result["finite_difference_max_error"], 2e-8)
            np.testing.assert_allclose(result["jacobian_eigenvalues"], result["predicted_eigenvalues"], atol=2e-14)

    def test_resource_cap_and_death_order_deterministic_endpoints(self):
        initial = np.array([0, 0, 0, 3])
        laws = rp.candidate_laws(rp.class_matrix(0.))
        for death in (0., 1.):
            counts, remaining, births = rp.step_counts(initial, 0, laws, death, 2, 2, np.random.default_rng(12))
            self.assertEqual(births, 2)
            self.assertEqual(remaining, 0)
            self.assertEqual(counts.sum(), 5 if death == 0 else 0)
        counts, resources, births = rp.simulate(initial, 0, .5, 1., 2, 5, 5, 777)
        self.assertFalse(counts[1:].any())
        self.assertEqual(resources[-1], 5)
        self.assertFalse(births[2:].any())

    def test_exact_conditional_mean_and_seed_identity(self):
        L = rp.class_matrix(.8)
        laws = rp.candidate_laws(L)
        initial = np.array([2, 1, 1, 1])
        success = np.repeat(L.sum(axis=1), initial)
        exposure = rp.exposure_probabilities(success, 2)
        repeated_rows = np.repeat(L, initial, axis=0)
        expected = .8*(initial+(exposure[:, None]*repeated_rows).sum(axis=0))
        rng = np.random.default_rng(78211)
        values = np.array([rp.step_counts(initial, 2, laws, .2, 0, 2, rng)[0] for _ in range(12000)])
        tolerance = 6*values.std(axis=0, ddof=1)/np.sqrt(len(values))+1e-12
        self.assertTrue(np.all(np.abs(values.mean(axis=0)-expected) <= tolerance))
        seeds = [rp.seed_for(2026102407, 2, .8, checkpoint, arm, repeat)
                 for checkpoint in (0,1,2) for arm in (0,1) for repeat in range(20)]
        self.assertEqual(len(seeds), len(set(seeds)))
        one = rp.simulate(initial, 3, .8, .2, 2, 8, 30, seeds[0])
        two = rp.simulate(initial, 3, .8, .2, 2, 8, 30, seeds[0])
        for a, b in zip(one, two):
            np.testing.assert_array_equal(a, b)
        counts, resources, births = one
        for t in range(1, len(resources)):
            self.assertEqual(resources[t], min(8, int(resources[t-1])+2)-births[t])
            self.assertLessEqual(counts[t].sum(), counts[t-1].sum()+births[t])


if __name__ == "__main__":
    unittest.main()
