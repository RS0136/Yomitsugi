import unittest
import itertools
import numpy as np
from yomitsugi.core import Chemistry, Kernel, standard_chemistry, read_word_reference, NumericalPrecisionError

class KernelTests(unittest.TestCase):
    def test_exact_ambiguous_duplicates_and_absence(self):
        k = Kernel(standard_chemistry(2),0)
        self.assertEqual(k.distribution((0,0,0,1),0,0)[1].tolist(),[.5,.5])
        self.assertEqual(k.distribution((0,0,0,0),0,0)[1].tolist(),[1,0])
        self.assertEqual(k.birth_probability((0,0,0,0),0,(0,1)),0)

    def test_invariance_and_normalization(self):
        for sigma in (0,.2,.5,1,4):
            k = Kernel(standard_chemistry(),sigma)
            for p in itertools.product(range(2),repeat=4):
                for c in range(2):
                    a = k.full_distribution(p,0,c)
                    np.testing.assert_allclose(a,k.full_distribution(p[2:]+p[:2],0,c),atol=1e-15)
                    self.assertAlmostEqual(a.sum(),1)
                    self.assertTrue((a>=0).all())
        ch = Chemistry({5:0},{8:0,7:1},{8:0,7:1},{8:0,7:0})
        self.assertAlmostEqual(Kernel(standard_chemistry(),.5).birth_probability((0,1,1,0),0,(0,0,1,1)),
                               Kernel(ch,.5).birth_probability((8,7,7,8),5,(8,8,7,7)))

    def test_sampling_reference(self):
        k = Kernel(standard_chemistry(),.8)
        p,g = (0,0,0,1),(0,0,1,1)
        counts=[]
        for sampler,seed in [(k.read_word,1),(lambda p,b,g,r:read_word_reference(k,p,b,g,r),2)]:
            rng=np.random.default_rng(seed)
            data=[sampler(p,0,g,rng) for _ in range(8000)]
            counts.append(np.array([data.count(x)/len(data) for x in [None]+list(itertools.product(range(2),repeat=4))]))
        self.assertLess(abs(counts[0][0]-(1-k.birth_probability(p,0,g))),.025)
        self.assertLess(np.max(np.abs(counts[0]-counts[1])),.03)

    def test_feature_equivalence_and_underflow(self):
        k=Kernel(Chemistry({0:0},{0:0},{0:0,1:0},{0:0,1:0}),.5)
        self.assertEqual(k.distance((0,0),(1,1),[(0,0)]),1)
        self.assertEqual(k.distance((0,0),(1,1),[(0,0)],True),0)
        with self.assertRaises(NumericalPrecisionError):
            Kernel(standard_chemistry(),.001).distribution((0,0,1,1),0,0)

if __name__=='__main__': unittest.main()
