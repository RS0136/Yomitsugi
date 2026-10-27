import unittest
from collections import Counter
from yomitsugi.exact import (
    enumerate_permutations, enumerate_self_descriptions, enumerate_preorders,
    counterexamples, alternating_targets, effective_map, direct_read,
    permutation_reader, permutation_word,
)


class ExactLimitTests(unittest.TestCase):
    def test_complete_permutation_regression(self):
        expected = {2: (4, 4, 4), 3: (36, 18, 24), 4: (576, 120, 240)}
        for q, values in expected.items():
            rows, _ = enumerate_permutations(q)
            self.assertEqual((len(rows), sum(r['one_step_recovery'] for r in rows),
                              sum(r['eventual_recovery'] for r in rows)), values)
        self.assertEqual(Counter(r['period'] for r in rows if not r['eventual_recovery']),
                         {2: 96, 3: 144, 4: 96})

    def test_native_self_description_is_effective_not_slot_order(self):
        for permutation in ((1, 2, 0), (2, 0, 1), (1, 0, 3, 2)):
            p, g = permutation_reader(permutation), permutation_word(permutation)
            child = direct_read(p, g)
            self.assertEqual(effective_map(child), effective_map(p))
            reordered = tuple(x for pair in reversed(list(zip(p[::2], p[1::2]))) for x in pair)
            self.assertEqual(effective_map(reordered), effective_map(p))

    def test_inclusion_and_reverse_preorder_construction(self):
        states, checks = enumerate_self_descriptions()
        self.assertEqual(len(states), 10)
        self.assertEqual(len({p for _, p in states}), 6)
        self.assertEqual(len(checks), 100)
        self.assertTrue(all(row['birth_probability'] == row['inclusion'] for row in checks))
        self.assertEqual(len(enumerate_preorders(3)), 29)

    def test_birth_does_not_imply_continuation(self):
        rows = counterexamples()
        self.assertEqual([r['C'] for r in rows], [1, 1, 1])
        self.assertEqual([r['S2'] for r in rows], [1, 0, 0])
        self.assertTrue(all(r['parent_copy_control_S2'] == 1 for r in rows))

    def test_alternating_targets_complete(self):
        rows = alternating_targets()
        self.assertEqual(len(rows), 216)
        self.assertEqual(Counter(row['cycle_periods'] for row in rows), {1: 180, 3: 36})


if __name__ == '__main__':
    unittest.main()
