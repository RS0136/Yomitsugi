"""Independent checks for the exact permutation-reader inheritance result.

Literal slot lookup is separate from the production conjugation vectorization.
The small alphabets test every target, including noncanonical relabelings.
"""
import itertools
import math
import unittest
from fractions import Fraction

import numpy as np

from analysis.permutation_dynamics import (
    analyse_target, canonical_permutation, compose, conjugate, functional_graph,
    finite_width_shadowing,
    inverse, partitions, permutation_ranks, standard_commutator,
    successor_indices, transposition_nonrecovery_fraction,
)


def direct_inverse(f):
    return tuple(f.index(k) for k in range(len(f)))


def direct_product(*factors):
    result = []
    for symbol in range(len(factors[0])):
        for factor in reversed(factors):
            symbol = factor[symbol]
        result.append(symbol)
    return tuple(result)


def literal_offspring(f, target):
    """Build a native word, obtain its descriptor, then read every symbol."""
    q = len(f)
    target_inverse = direct_inverse(target)
    native_word = tuple(x for key in range(q) for x in (key, target[key]))
    descriptor = tuple(target_inverse[x] for x in native_word)
    actor_slots = [(key, f[key]) for key in reversed(range(q))]
    child_word = []
    for symbol in descriptor:
        matching = [value for key, value in actor_slots if key == symbol]
        assert len(matching) == 1
        child_word.append(matching[0])
    child_slots = list(zip(child_word[::2], child_word[1::2]))
    assert len({key for key, _ in child_slots}) == q
    assert len({value for _, value in child_slots}) == q
    return tuple(next(value for key, value in child_slots if key == symbol)
                 for symbol in range(q))


def direct_orbit(start, successor):
    seen, sequence = {}, []
    state = start
    while state not in seen:
        seen[state] = len(sequence)
        sequence.append(state)
        state = int(successor[state])
    tail = seen[state]
    cycle = sequence[tail:]
    return min(cycle), len(cycle), tail


def direct_noiseless_child_and_probability(word, descriptor, sigma_r, sigma_d):
    """Original positionwise Gaussian kernel on literal, unordered slots."""
    slots = tuple(zip(word[::2], word[1::2]))
    child, probability = [], 1.0
    for symbol in descriptor:
        output = next(value for key, value in slots if key == symbol)
        gate = (float(any(key == symbol for key, _ in slots)) if sigma_r == 0
                else max(math.exp(-(symbol-key)**2/(2*sigma_r*sigma_r))
                         for key, _ in slots))
        weights = [math.exp(-(symbol-key)**2/(2*sigma_d*sigma_d))
                   for key, _ in slots]
        conditional = sum(weight for weight, (_, value) in zip(weights, slots)
                          if value == output) / sum(weights)
        probability *= gate * conditional
        child.append(output)
    return tuple(child), probability


class PermutationDynamicsTests(unittest.TestCase):
    def test_literal_reading_every_actor_and_target_through_q4(self):
        checks = 0
        for q in range(1, 5):
            catalog = list(itertools.permutations(range(q)))
            actors = np.array(catalog, dtype=np.uint8)
            for target in catalog:
                observed = successor_indices(actors, target)
                for index, f in enumerate(catalog):
                    expected = literal_offspring(f, target)
                    self.assertEqual(tuple(actors[observed[index]]), expected)
                    self.assertEqual(conjugate(f, target), expected)
                    checks += 1
        self.assertEqual(checks, 617)

    def test_lexicographic_rank_and_small_partition_counts(self):
        for q in range(1, 7):
            actors = np.array(list(itertools.permutations(range(q))))
            np.testing.assert_array_equal(permutation_ranks(actors),
                                          np.arange(math.factorial(q)))
        self.assertEqual([len(list(partitions(q))) for q in range(1, 9)],
                         [1, 2, 3, 5, 7, 11, 15, 22])

    def test_graph_decomposition_against_independent_orbits(self):
        # Includes merging tails, a fixed point, and cycles of lengths 2 and 3.
        successor = np.array([0, 2, 3, 2, 1, 6, 7, 5, 7, 8, 0])
        graph = functional_graph(successor)
        for index in range(len(successor)):
            expected = direct_orbit(index, successor)
            self.assertEqual(tuple(int(graph[name][index]) for name in
                                   ('attractor_index', 'period', 'distance')), expected)
        actors = np.array(list(itertools.permutations(range(4))))
        for target in actors:
            successor = successor_indices(actors, target)
            graph = functional_graph(successor)
            for index in range(len(actors)):
                self.assertEqual(tuple(int(graph[name][index]) for name in
                                       ('attractor_index', 'period', 'distance')),
                                 direct_orbit(index, successor))

    def test_transposition_family_counts_and_first_hit(self):
        for q in range(2, 8):
            catalog = list(itertools.permutations(range(q)))
            actors = np.array(catalog)
            part = (1,) * (q - 2) + (2,)
            row, cycles, periods, arrays = analyse_target(actors, part)
            expected = Fraction(4 * (q - 2), q * (q - 1))
            self.assertEqual(Fraction(row['never_native_count'], len(catalog)), expected)
            self.assertEqual(transposition_nonrecovery_fraction(q), expected)
            self.assertEqual(row['max_period'], 1 if q == 2 else 2)
            self.assertEqual(sum(c['basin_size'] for c in cycles), len(catalog))
            self.assertEqual(sum(p['actor_count'] for p in periods), len(catalog))
            target = canonical_permutation(part)
            native_index = catalog.index(target)
            self.assertEqual(arrays['first_native_time'][native_index], 0)
            self.assertEqual(np.count_nonzero(arrays['first_native_time'] == 1),
                             row['first_child_native_count'] - 1)
            self.assertEqual(np.count_nonzero(arrays['first_native_time'] == 2),
                             math.comb(q - 2, 2) * 2 * math.factorial(q - 2))

    def test_standard_engel_identity_with_literal_inheritance(self):
        checks = 0
        for q in range(1, 5):
            identity = tuple(range(q))
            catalog = list(itertools.permutations(range(q)))
            for target in catalog:
                for actor in catalog:
                    reader, target_power = actor, identity
                    engel = direct_product(target, direct_inverse(actor))
                    for _ in range(6):
                        expected_commutator = direct_product(direct_inverse(engel),
                                                             direct_inverse(target),
                                                             engel, target)
                        self.assertEqual(standard_commutator(engel, target), expected_commutator)
                        engel = expected_commutator
                        reader = literal_offspring(reader, target)
                        target_power = direct_product(target_power, target)
                        self.assertEqual(direct_product(target, direct_inverse(reader)),
                                         direct_product(target_power, engel,
                                                        direct_inverse(target_power)))
                        self.assertEqual(reader == target, engel == identity)
                        checks += 1
        self.assertEqual(checks, 3702)

    def test_relabeling_equivariance_every_triple_through_q4(self):
        checks = 0
        for q in range(1, 5):
            catalog = list(itertools.permutations(range(q)))
            for target, actor, relabel in itertools.product(catalog, repeat=3):
                relabel_inverse = direct_inverse(relabel)
                relabeled_target = direct_product(relabel, target, relabel_inverse)
                relabeled_actor = direct_product(relabel, actor, relabel_inverse)
                self.assertEqual(literal_offspring(relabeled_actor, relabeled_target),
                                 direct_product(relabel, literal_offspring(actor, target),
                                                relabel_inverse))
                checks += 1
        self.assertEqual(checks, 14049)

    def test_fitting_classification_finite_check_through_q6(self):
        observed = {}
        for q in range(1, 7):
            actors = np.array(list(itertools.permutations(range(q))))
            observed[q] = []
            for part in partitions(q):
                row, _, _, _ = analyse_target(actors, part)
                if row['eventual_native_count'] == len(actors):
                    observed[q].append(part)
        self.assertEqual(observed, {
            1: [(1,)], 2: [(1, 1), (2,)],
            3: [(1, 1, 1), (3,)],
            4: [(1, 1, 1, 1), (2, 2)],
            5: [(1, 1, 1, 1, 1)], 6: [(1, 1, 1, 1, 1, 1)],
        })

    def test_192_cycle_witness_by_literal_reading(self):
        # An independently replayable witness for the largest reported period.
        # This does not call the graph builder, ranks, or conjugation formula.
        target = (0, 2, 3, 1, 5, 6, 7, 4)
        initial = (1, 4, 2, 6, 7, 3, 5, 0)
        seen, reader = set(), initial
        for _ in range(192):
            self.assertNotIn(reader, seen)
            self.assertNotEqual(reader, target)
            seen.add(reader)
            reader = literal_offspring(reader, target)
        self.assertEqual(reader, initial)

    def test_finite_width_shadowing_against_literal_gaussian_kernel(self):
        for q in range(1, 5):
            catalog = list(itertools.permutations(range(q)))
            for target, actor in itertools.product(catalog, repeat=2):
                descriptor = tuple(x for k in range(q) for x in (k, target[k]))
                word = tuple(x for k in reversed(range(q)) for x in (k, actor[k]))
                for sigma_d in (.15, .2, .25):
                    expected = finite_width_shadowing(q, sigma_d, 1)['path_probability']
                    for sigma_r in (0., .1, 1.):
                        child, observed = direct_noiseless_child_and_probability(
                            word, descriptor, sigma_r, sigma_d)
                        self.assertAlmostEqual(observed, expected, places=13)
                        self.assertEqual(len(set(child[::2])), q)
                        self.assertEqual(len(set(child[1::2])), q)
        # Follow all 192 literal words; their slot order need not stay canonical.
        target = (0, 2, 3, 1, 5, 6, 7, 4)
        actor = (1, 4, 2, 6, 7, 3, 5, 0)
        descriptor = tuple(x for k in range(8) for x in (k, target[k]))
        word = tuple(x for k in range(8) for x in (k, actor[k]))
        observed = 1.0
        for _ in range(192):
            word, chance = direct_noiseless_child_and_probability(word, descriptor, .7, .2)
            observed *= chance
            self.assertNotEqual(dict(zip(word[::2], word[1::2])), dict(enumerate(target)))
        self.assertAlmostEqual(observed,
                               finite_width_shadowing(8, .2, 192)['path_probability'], places=12)
        self.assertAlmostEqual(observed, .980164937287047, places=12)
        self.assertAlmostEqual(finite_width_shadowing(3, .2, 32)['path_probability'],
                               .9990464343873394, places=13)


if __name__ == '__main__':
    unittest.main()
