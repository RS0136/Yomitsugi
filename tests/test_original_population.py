"""Source-only baseline generation and verified-cache behavior."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import original_population as baseline


class OriginalPopulationTests(unittest.TestCase):
    def prepare(self, directory):
        root = Path(directory)
        config = baseline.default_config("main")
        config.update(sweeps=4, repeats=1, initial_population=4, S0=4, Smax=8,
                      J=2, birth_log_limit=0, modes=["single_a", "mixed"],
                      conditions=config["conditions"][:1])
        (root / "config").mkdir()
        (root / "config/original_population.json").write_text(json.dumps(config), encoding="utf8")
        return root

    def test_generates_without_input_archive_and_reuses_verified_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.prepare(directory)
            first = baseline.run(root, workers=1)
            self.assertTrue(first["all_completed"])
            self.assertFalse(first["cached"])
            self.assertEqual(first["task_count"], 2)
            self.assertFalse((root / "data").exists())
            manifest = json.loads((root / "results/original_population/manifest.json").read_text())
            self.assertEqual(len(manifest["trajectories"]), 2)
            with patch.object(baseline, "run_population", side_effect=AssertionError("Unneeded simulation")):
                second = baseline.run(root, workers=1)
            self.assertTrue(second["cached"])

    def test_corrupt_trajectory_and_force_rebuild_with_same_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.prepare(directory)
            first = baseline.run(root, workers=1)
            output = root / "results/original_population"
            trajectory = next((output / first["run_id"]).glob("*/trajectory.csv"))
            original = trajectory.read_bytes()
            trajectory.write_bytes(b"corrupted output\n")
            repaired = baseline.run(root, workers=1)
            self.assertFalse(repaired["cached"])
            self.assertEqual(trajectory.read_bytes(), original)
            forced = baseline.run(root, workers=1, force=True)
            self.assertFalse(forced["cached"])
            self.assertEqual(trajectory.read_bytes(), original)

    def test_changed_settings_and_runtime_do_not_reuse_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.prepare(directory)
            first = baseline.run(root, workers=1)
            path = root / "config/original_population.json"
            config = json.loads(path.read_text())
            config["base_seed"] += 1
            path.write_text(json.dumps(config), encoding="utf8")
            changed = baseline.run(root, workers=1)
            self.assertFalse(changed["cached"])
            self.assertNotEqual(changed["run_id"], first["run_id"])
            with patch.object(baseline, "source_hash", return_value="0" * 64), \
                    patch.object(baseline, "run_population", side_effect=RuntimeError("Cache invalidated")):
                with self.assertRaisesRegex(RuntimeError, "Cache invalidated"):
                    baseline.run(root, workers=1)

    def test_incomplete_run_is_not_recorded_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.prepare(directory)
            with patch.object(baseline, "run_population", return_value={
                "all_completed": False, "termination_counts": {"time_limit": 2}
            }):
                with self.assertRaisesRegex(RuntimeError, "did not complete"):
                    baseline.run(root, workers=1)
            self.assertFalse((root / "results/original_population/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
