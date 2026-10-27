import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("population_extension",ROOT/"analysis/population_extension.py")
pe=importlib.util.module_from_spec(spec); spec.loader.exec_module(pe)

class PopulationExtensionTests(unittest.TestCase):
    def test_direct_matrix_matches_independent_original_kernel(self):
        Kernel,standard,_,_,_=pe.load_vendor(ROOT)
        states=np.array(pe.STATES)
        for sigma in (0,.2,.5,.8):
            kernel=Kernel(standard(),sigma); M=pe.transition_matrix(sigma)
            for i,p in enumerate(pe.STATES):
                laws=np.array([kernel.full_distribution(p,0,c)[:2] for c in (0,0,1,1)])
                reference=np.prod(laws[np.arange(4)[:,None],states.T],axis=0)
                np.testing.assert_allclose(M[i],reference,atol=2e-15,rtol=1e-13)

    def test_parent_child_law_deterministic_endpoints(self):
        M=pe.transition_matrix(0); counts=np.zeros((3,16),dtype=np.int64); counts[:,3]=[1,2,5]
        np.testing.assert_array_equal(pe.unrestricted_step(counts,M,0,np.random.default_rng(1)),2*counts)
        self.assertFalse(pe.unrestricted_step(counts,M,1,np.random.default_rng(1)).any())
        self.assertEqual(pe.extinction_probabilities(M,1,3,1,3),[0,1,1,1])
        self.assertEqual(pe.extinction_probabilities(M,0,3,1,3),[0,0,0,0])

    def test_independent_binomial_moments(self):
        M=pe.transition_matrix(0); counts=np.zeros((50000,16),dtype=np.int64); counts[:,3]=5
        total=pe.unrestricted_step(counts,M,.25,np.random.default_rng(97123)).sum(axis=1)
        self.assertLess(abs(total.mean()-7.5),6*np.sqrt(1.875/len(total)))
        self.assertLess(abs(total.var()-1.875),.05)
        self.assertTrue((total<=10).all())

    def test_extinction_generating_function_first_step(self):
        M=pe.transition_matrix(.5); d=.3
        for i in range(16):
            expected=(d*(1-(1-d)*M[i].sum()))**2
            self.assertAlmostEqual(pe.extinction_probabilities(M,d,1,2,i)[1],expected)

    def test_extinction_not_dropped_from_time_average(self):
        r=[dict(sweep=1,N=10,reference_count=10,births=5),dict(sweep=2,N=0,reference_count=0,births=0)]
        v=[np.eye(16)[3]*10,np.zeros(16)]
        out=pe.summarize_window(r,v,1,2)
        self.assertEqual(out["reference_fraction_zero_filled"],.5)
        self.assertEqual(out["reference_fraction_alive"],1)
        self.assertEqual(out["alive_fraction"],.5)
        self.assertEqual(out["mean_N"],5)

    def test_composition_distance_keeps_extinction_distinct(self):
        rows=[dict(sweep=1,N=10,reference_count=10,births=5),dict(sweep=2,N=0,reference_count=0,births=0)]
        vectors=[np.eye(16)[3]*10,np.zeros(16)]
        run=dict(case="test",sigma=.5,**pe.summarize_window(rows,vectors,1,2))
        summaries,_=pe.summarize_groups([run],("case",),{.5:np.eye(16)[3]})
        self.assertEqual(summaries[0]["distribution_half_L1_zero_filled"],.25)
        self.assertEqual(summaries[0]["distribution_TV_with_extinction"],.5)
        self.assertEqual(summaries[0]["distribution_TV_alive"],0)

    def test_two_value_repertoire_cannot_reappear_after_loss(self):
        for sigma in (0,.4,.5,.6,.8):
            M=pe.transition_matrix(sigma)
            self.assertFalse(M[~pe.TWO_VALUES][:,pe.TWO_VALUES].any())

    def generate_original_fixture(self, root):
        from analysis.original_population import run
        (root/"config").mkdir()
        (root/"config/original_population.json").write_text(json.dumps(dict(
            sweeps=3, repeats=1, initial_population=4, S0=4, Smax=8, J=2,
            birth_log_limit=0)))
        result=run(root,workers=1,force=True)
        plan={"original_run":result["run_id"], "original_reanalysis":{
            "expected_runs":10, "primary_window":[1,3], "sensitivity_windows":[]}}
        return root/"results/original_population",plan

    def test_generated_directory_and_explicit_archive_give_same_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,plan=self.generate_original_fixture(root)
            # A historical run ID must not constrain a newly generated study.
            historical_plan=dict(plan,original_run="unrelated_historical_run")
            generated=pe.reanalyse_original(source,root/"directory_analysis",historical_plan)
            archive_path=root/"legacy.zip"
            with zipfile.ZipFile(archive_path,"w") as archive:
                for path in source.rglob("*"):
                    if path.is_file():
                        archive.write(path,"yomitsugi_alife/results/population/"+str(path.relative_to(source)))
            legacy=pe.reanalyse_original(archive_path,root/"archive_analysis",plan)
            self.assertEqual(generated["runs"],10)
            self.assertEqual(generated["trajectory_rows"],30)
            self.assertEqual(generated["input_kind"],"generated_original_population")
            self.assertEqual(legacy["input_kind"],"posthoc_original")
            for name in ("original_group_windows.csv","original_reader_distributions.csv","original_lineage_retention.csv"):
                self.assertEqual((root/"directory_analysis"/name).read_bytes(),(root/"archive_analysis"/name).read_bytes())

    def test_generated_reanalysis_rejects_modified_or_incomplete_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,plan=self.generate_original_fixture(root)
            manifest=json.loads((source/"manifest.json").read_text())
            trajectory=source/next(iter(manifest["trajectories"]))
            trajectory.write_bytes(trajectory.read_bytes()+b"\n")
            with self.assertRaisesRegex(AssertionError,"input hash mismatch"):
                pe.reanalyse_original(source,root/"modified_analysis",plan)
            manifest["all_completed"]=False
            (source/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(AssertionError,"incomplete"):
                pe.reanalyse_original(source,root/"incomplete_analysis",plan)

    def test_generated_reanalysis_rejects_missing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan={"original_reanalysis":{"primary_window":[1,3],"sensitivity_windows":[]}}
            with self.assertRaisesRegex(FileNotFoundError,"analysis.original_population"):
                pe.reanalyse_original(root/"missing",root/"analysis",plan)

if __name__=="__main__": unittest.main()
