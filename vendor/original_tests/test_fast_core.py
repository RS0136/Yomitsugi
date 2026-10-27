"""Pathwise equality, including random-stream state, for optimized sampling."""
import itertools
import unittest
import numpy as np
from yomitsugi.core import Chemistry, Kernel, standard_chemistry, read_word_vector_reference


class CoreSpeedRegressionTests(unittest.TestCase):
    def assert_stream_equal(self,kernel,p,g,seed=8001,n=24):
        fast=np.random.default_rng(seed)
        reference=np.random.default_rng(seed)
        for _ in range(n):
            self.assertEqual(kernel.read_word(p,0,g,fast),read_word_vector_reference(kernel,p,0,g,reference))
        self.assertEqual(fast.bit_generator.state,reference.bit_generator.state)

    def test_all_binary_readers_and_templates(self):
        states=list(itertools.product(range(2),repeat=4))
        for sigma in (0,.2,.5,1,2,4):
            kernel=Kernel(standard_chemistry(),sigma)
            for p in states:
                for g in states: self.assert_stream_equal(kernel,p,g)

    def test_larger_catalogs_and_nonconsecutive_ids(self):
        rng=np.random.default_rng(34)
        for q in (1,3,4):
            for sigma in (0,.2,.5,2):
                kernel=Kernel(standard_chemistry(q),sigma)
                for _ in range(40):
                    p=tuple(map(int,rng.integers(q,size=2*max(q,2))))
                    g=tuple(map(int,rng.integers(q,size=len(p))))
                    self.assert_stream_equal(kernel,p,g)
        ch=Chemistry({0:0},{7:0,9:1},{3:0,11:1},{3:0,11:0})
        self.assert_stream_equal(Kernel(ch,.5),(3,11,11,3),(7,7,9,9))

    def test_cdf_boundaries_and_zero_mass_slots(self):
        class BoundaryRng:
            def __init__(self,values): self.values=np.array(values); self.calls=[]
            def random(self,n=None):
                self.calls.append(n)
                return .1 if n is None else self.values.copy()
        kernel=Kernel(standard_chemistry(3),0)
        # Duplicate exact keys give two equiprobable outputs; residue zero absent.
        p=(0,1,0,2); g=(0,0,0,0)
        values=[0.,.5,np.nextafter(.5,0.),np.nextafter(1.,0.)]
        a,b=BoundaryRng(values),BoundaryRng(values)
        self.assertEqual(kernel.read_word(p,0,g,a),read_word_vector_reference(kernel,p,0,g,b))
        self.assertEqual(a.calls,b.calls)
        self.assertEqual(kernel.read_word(p,0,g,BoundaryRng(values)),(1,2,1,2))

    def test_full_distribution_cache_readonly_and_distance_unchanged(self):
        ch=Chemistry({0:0},{0:0,1:1},{0:0,1:1,2:0},{0:0,1:0,2:0})
        kernel=Kernel(ch,.5)
        for p,q in [((0,0,1,1),(2,2,1,1)),((0,1,0,1),(0,0,1,1))]:
            values=[]
            for b,c in [(0,0),(0,1)]:
                raw=[]
                for reader in (p,q):
                    log_s,probs=kernel.distribution(reader,b,c)
                    raw.append(np.r_[np.exp(log_s)*probs,-np.expm1(log_s)])
                values.append(np.abs(raw[0]-raw[1]).sum()/2)
            self.assertAlmostEqual(kernel.distance(p,q,[(0,0),(0,1)]),sum(values)/2)
        with self.assertRaises(ValueError): kernel.full_distribution((0,0),0,0)[0]=.5
        self.assertEqual(kernel.distance((0,0),(2,2),[(0,0)],True),0)


if __name__=='__main__': unittest.main()
