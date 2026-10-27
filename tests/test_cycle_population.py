"""Independent checks for the code/population bridge and its event definitions."""
import itertools
import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from analysis.cycle_population import (build_matrix, state_annotations, key_selection, permanent,
    TARGET, CYCLE_CODES, DESCRIPTION, word_from_code, row_index, first_passage_before_loss,
    distance_matrix, population_cdf, _sample_rows, step_population, first_hit_tables)
from analysis.finite import build_matrix as baseline_matrix
from analysis.permutation_dynamics import run as enumerate_permutations

class CyclePopulationTests(unittest.TestCase):
    def test_literal_linear_kernel_agrees_with_validated_baseline(self):
        for sigma in (0.,.2,.3,.5):
            matrix,states=build_matrix(sigma)
            expected,expected_states=baseline_matrix(sigma,g=DESCRIPTION)
            np.testing.assert_array_equal(states,expected_states)
            np.testing.assert_allclose(matrix,expected,atol=4e-16,rtol=3e-13)

    def test_normalized_one_hot_preserves_binary_metric(self):
        np.testing.assert_array_equal(distance_matrix(2,'linear'),distance_matrix(2,'equidistant'))
        coordinates=np.eye(3)/np.sqrt(2)
        d=((coordinates[:,None,:]-coordinates[None,:,:])**2).sum(axis=2)
        np.testing.assert_allclose(d,distance_matrix(3,'equidistant'),atol=3e-16)

    def test_permanent_identity_for_every_complete_raw_reader(self):
        for metric in ('linear','equidistant'):
            for sigma in (0.,.2,.3,.5):
                matrix,states=build_matrix(sigma,metric=metric);m=state_annotations(states)
                self.assertEqual(int(m['complete'].sum()),36);self.assertEqual(int(m['full_repertoire'].sum()),162)
                p=permanent(key_selection(3,sigma,metric))
                np.testing.assert_allclose(matrix[np.ix_(m['complete'],m['complete'])].sum(axis=1),p*p,atol=8e-16,rtol=2e-14)
                np.testing.assert_allclose(matrix[np.ix_(m['complete'],m['full_repertoire'])].sum(axis=1),p,atol=8e-16,rtol=2e-14)

    def test_distinct_complete_exit_and_value_loss_events(self):
        matrix,states=build_matrix(.3);m=state_annotations(states)
        row=matrix[row_index(states,word_from_code(TARGET))]
        self.assertAlmostEqual(float(row[~m['complete']].sum()),.030309463282275018,places=14)
        self.assertAlmostEqual(float(row[~m['full_repertoire']].sum()),.015271338531407944,places=14)
        self.assertGreater(float(row[~m['complete']&m['full_repertoire']].sum()),0)

    def test_zero_width_native_fixed_and_alternative_cycle(self):
        matrix,states=build_matrix(0.);m=state_annotations(states)
        for code in (TARGET,*CYCLE_CODES):
            i=row_index(states,word_from_code(code));child=np.flatnonzero(matrix[i]>0)
            self.assertEqual(len(child),1);self.assertEqual(matrix[i,child[0]],1.)
            literal=tuple(code[x] for x in DESCRIPTION)
            np.testing.assert_array_equal(states[child[0]],literal)
            self.assertEqual(bool(m['native'][child[0]]),code==TARGET)

    def test_first_passage_equations_allow_incomplete_keys(self):
        matrix,states=build_matrix(.3);m=state_annotations(states)
        p,residual,condition=first_passage_before_loss(matrix,m,m['native'])
        self.assertLess(residual,1e-13);self.assertTrue(np.all((p>=-1e-14)&(p<=1+1e-12)))
        for code in CYCLE_CODES:
            i=row_index(states,word_from_code(code));direct=float(matrix[i,m['native']].sum())
            self.assertGreater(p[i],direct*1000)
        self.assertEqual(float(p[~m['full_repertoire']].sum()),0.)

    def test_vector_inverse_cdf_against_searchsorted(self):
        rng=np.random.default_rng(417);p=rng.dirichlet(np.ones(7),size=13);p[0]=[0,0,0,1,0,0,0]
        cdf=p.cumsum(axis=1);cdf[:,-1]=1.;rows=rng.integers(0,13,10000);u=rng.random(10000)
        actual=_sample_rows(cdf,rows,u);expected=np.array([np.searchsorted(cdf[r],v,side='right') for r,v in zip(rows,u)])
        np.testing.assert_array_equal(actual,expected)

    def test_independent_high_precision_slot_order_quotient(self):
        from analysis.cycle_population import high_precision_first_passage, first_passage_rows
        exact=high_precision_first_passage();full=first_passage_rows()
        indexed={(r['metric'],r['sigma'],r['start'],r['destination']):r for r in full}
        for row in exact:
            other=indexed[row['metric'],row['sigma'],row['start'],row['destination']]
            expected=float(row['probability']);observed=other['probability_before_loss_or_failure']
            self.assertLess(abs(observed-expected)/expected,2e-8)

    def test_origin_and_resource_law_in_deterministic_small_case(self):
        counts=np.array([[2,0],[0,2]])
        # Each actor always emits the other type, and all parents survive.
        cdf=population_cdf(np.array([[0.,1.],[1.,0.]]));rng=np.random.default_rng(180)
        samples=[]
        for _ in range(6000):
            child,stock,births=step_population(counts,0,cdf,rng,death=0,supply=2,capacity=2)
            self.assertEqual(stock,0);self.assertEqual(births,2);samples.append(child-counts)
        means=np.mean(samples,axis=0)
        np.testing.assert_allclose(means,np.array([[0,1],[1,0]]),atol=.035)
        frequencies=np.bincount(np.asarray(samples)[:,0,1],minlength=3)/len(samples)
        np.testing.assert_allclose(frequencies,np.array([1,4,1])/6,atol=.025)

    def test_generated_graph_two_step_formula_and_deep_returns(self):
        # Generate every target from source so a fresh checkout needs no saved results.
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root=Path(directory)
            enumerate_permutations(root=root, max_q=8)
            rows,dist,perturb=first_hit_tables(root)
        self.assertEqual(len(rows),66)
        by={(r['q'],r['cycle_type']):r for r in rows}
        special=by[8,'1-3-4'];self.assertEqual(special['native_basin'],48)
        self.assertEqual(special['centralizer_size'],12);self.assertEqual(special['centralizer_class_intersection'],4)
        hits={r['first_native_time']:r['actor_count'] for r in dist if r['q']==8 and r['cycle_type']=='1-3-4'}
        self.assertEqual({h:hits[h] for h in (0,1,2)},{0:1,1:11,2:36})
        self.assertEqual(by[8,'2-2-4']['max_native_first_hit'],7)
        self.assertEqual(by[8,'4-4']['max_native_first_hit'],6)
        self.assertEqual(by[8,'8']['max_native_first_hit'],4)

if __name__=='__main__':unittest.main()
