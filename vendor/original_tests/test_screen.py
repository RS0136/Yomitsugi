"""Scientific invariants of the one-axis designs and offspring calculations."""
import itertools
import unittest
import numpy as np
from yomitsugi.core import Chemistry, Kernel, standard_chemistry, NumericalPrecisionError
from yomitsugi.screen import conditions, offspring_law, serial


class ZeroRng:
    def random(self, size=None):
        return 0. if size is None else np.zeros(size)


class ScreenTests(unittest.TestCase):
    def test_zero_random_cannot_select_absent_value(self):
        k=Kernel(standard_chemistry(3),0)
        self.assertEqual(k.read_word((1,2,2,2),0,(1,2),ZeroRng()),(2,2))

    def test_offspring_law_agrees_with_position_products(self):
        for sigma in (0,.2,.5,1):
            k=Kernel(standard_chemistry(2),sigma)
            for p in itertools.product(range(2),repeat=4):
                g=(0,0,1,1)
                states,weights=offspring_law(k,p,0,g)
                reference=np.array([np.prod([k.full_distribution(p,0,c)[a] for c,a in zip(g,s)]) for s in states])
                np.testing.assert_allclose(weights,reference,rtol=3e-14,atol=0)
                self.assertAlmostEqual(weights.sum(),k.birth_probability(p,0,g),places=13)
                for child,w in zip(states,weights):
                    if w>0:
                        self.assertLessEqual(set(child),set(p[1::2]))

    def test_positive_child_underflow_is_reported(self):
        with self.assertRaises(NumericalPrecisionError):
            offspring_law(Kernel(standard_chemistry(2),.04),(0,0,1,1),0,(0,0,1,1))

    def test_unused_catalogs_do_not_change_birth_or_serial_law(self):
        p=g=(0,0,1,1)
        for sigma in (0,.2,.5,1):
            base=Kernel(standard_chemistry(2),sigma)
            base_states,base_weights=offspring_law(base,p,0,g)
            base_serial=serial(base,p,0,g,base_states,(1,2,4),np.random.default_rng(1),10)[0]
            for cond in conditions():
                if cond['sigma']!=sigma or cond.get('control')!='unused':
                    continue
                k=Kernel(Chemistry(**cond['chemistry']),sigma)
                states,weights=offspring_law(k,p,0,g)
                self.assertEqual(states,base_states)
                np.testing.assert_allclose(weights,base_weights,atol=0,rtol=1e-14)
                observed=serial(k,p,0,g,states,(1,2,4),np.random.default_rng(1),10)[0]
                np.testing.assert_allclose(list(observed.values()),list(base_serial.values()),atol=0,rtol=1e-14)

    def test_aliases_are_feature_equivalent_but_residue_ids_differ(self):
        for sigma in (0,.2,.5,1):
            for cond in conditions():
                if cond['sigma']!=sigma or cond.get('control')!='alias':
                    continue
                k=Kernel(Chemistry(**cond['chemistry']),sigma)
                first,second=cond['types']
                p,q=tuple(first['p']),tuple(second['p'])
                panel=list(itertools.product(k.chemistry.backbone_features,k.chemistry.symbol_features))
                self.assertAlmostEqual(k.distance(p,q,panel,group_features=True),0)
                if cond['axis']=='residue':
                    self.assertGreater(k.distance(p,q,panel),0)
                self.assertAlmostEqual(k.birth_probability(p,first['b'],first['g']),
                                       k.birth_probability(q,second['b'],second['g']))

    def test_capacity_comparisons_hold_length_and_slot_capacity(self):
        cases=[c for c in conditions() if c['name'].startswith('code_capacity')]
        self.assertEqual(len(cases),8)
        for cond in cases:
            a,b=cond['types']
            self.assertEqual([len(x['g']) for x in (a,b)],[6,6])
            self.assertEqual([len(x['p'])//2 for x in (a,b)],[3,3])
            self.assertEqual([len(set(x['p'][1::2])) for x in (a,b)],[2,3])
            self.assertEqual(cond['base_alphabet_size'],2)
            self.assertEqual(cond['available_composite_ids'],4)
            k=Kernel(Chemistry(**cond['chemistry']),0)
            for typ in (a,b):
                self.assertEqual(k.birth_probability(tuple(typ['p']),typ['b'],typ['g']),1)
                self.assertEqual(k.read_word(tuple(typ['p']),typ['b'],typ['g'],np.random.default_rng(12)),tuple(typ['p']))

    def test_exact_equal_first_birth_has_distinct_next_generation(self):
        k=Kernel(standard_chemistry(3),0)
        g=(0,0,1,1,0,0)
        for p,expected in [((0,0,1,1,2,2),1),((0,1,1,2,2,0),0)]:
            states,weights=offspring_law(k,p,0,g)
            self.assertEqual(weights.sum(),1)
            # An independent weighted sum over first children checks S2.
            s2=sum(w*k.birth_probability(child,0,g) for child,w in zip(states,weights))
            self.assertEqual(s2,expected)
            continuation,_,_=serial(k,p,0,g,states,(1,2),np.random.default_rng(8),8)
            self.assertEqual(continuation,{1:1.,2:float(expected)})


if __name__=='__main__':
    unittest.main()
