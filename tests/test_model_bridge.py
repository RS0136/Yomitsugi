"""Check the new finite analysis against the unchanged supplied kernel."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'vendor'))
from yomitsugi.core import Kernel, standard_chemistry
from analysis.finite import build_matrix
from analysis.crossing import EXAMPLES, decode, rows

class SuppliedKernelBridgeTests(unittest.TestCase):
    def test_three_symbol_full_matrix_against_supplied_kernel(self):
        g=(1,0,0,1,2,2)
        for sigma in (0., .2, .5, 1., 2.):
            observed, states=build_matrix(sigma,g=g)
            kernel=Kernel(standard_chemistry(3),sigma)
            for i,p in enumerate(states):
                expected=np.ones(len(states))
                for position,symbol in enumerate(g):
                    probabilities=kernel.full_distribution(tuple(p),0,symbol)[:-1]
                    expected*=probabilities[states[:,position]]
                np.testing.assert_allclose(observed[i],expected,atol=6e-16,rtol=4e-13)

    def test_same_feature_counterexample_and_parent_control(self):
        kernel=Kernel(standard_chemistry(3),0.)
        records={(r['source'],r['target']):r for r in rows()}
        for name,item in EXAMPLES.items():
            self.assertEqual(records[name,name]['native_first'],1)
        for source,s in EXAMPLES.items():
            for target,t in EXAMPLES.items():
                first=decode(s['p'],t['g'])
                self.assertEqual(kernel.birth_probability(s['p'],0,t['g']),records[source,target]['C'])
                if first is not None:
                    self.assertEqual(kernel.read_word(s['p'],0,t['g'],np.random.default_rng(71)),first)
                    self.assertEqual(kernel.birth_probability(first,0,t['g']),records[source,target]['S2'])
        self.assertEqual(records['A0','B']['C'],records['A1','B']['C'])
        self.assertEqual(records['A0','B']['S2'],1)
        self.assertEqual(records['A1','B']['S2'],0)
        self.assertEqual(records['A1','B']['parent_copy_control_S2'],1)

if __name__=='__main__': unittest.main()
