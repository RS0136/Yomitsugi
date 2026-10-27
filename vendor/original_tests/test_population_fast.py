"""Pathwise, not just statistical, equivalence of optimized observations."""
import copy
from dataclasses import asdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from yomitsugi.core import Kernel
from yomitsugi.population import (default_config, _initial_state, _chemistry,
    sweep, save_checkpoint, load_checkpoint, source_hash)
from yomitsugi.reference_population import sweep as reference_sweep


class FastPopulationTests(unittest.TestCase):
    def test_runtime_change_invalidates_checkpoint_identity(self):
        before=source_hash()
        with patch('yomitsugi.population.np.__version__','different-test-runtime'):
            self.assertNotEqual(before,source_hash())

    def test_all_modes_exact_rows_events_state_and_rng(self):
        config = default_config('main')
        for condition in config['conditions']:
            for mode in config['modes']:
                with self.subTest(sigma=condition['sigma'], mode=mode):
                    left = _initial_state(config, condition, mode)
                    right = copy.deepcopy(left)
                    rng_left, rng_right = np.random.default_rng(42), np.random.default_rng(42)
                    kernel = Kernel(_chemistry(condition['chemistry']), condition['sigma'])
                    settings = dict(config, contact_mode=mode)
                    for _ in range(40):
                        le, re = [], []
                        lr = reference_sweep(left, kernel, rng_left, settings, le.append)
                        rr = sweep(right, kernel, rng_right, settings, re.append)
                        self.assertEqual(lr, rr)
                        self.assertEqual(le, re)
                    self.assertEqual(asdict(left), asdict(right))
                    self.assertEqual(rng_left.bit_generator.state, rng_right.bit_generator.state)

    def test_parent_copy_and_checkpoint_preserve_complete_observations(self):
        config=default_config(); condition=config['conditions'][1]
        settings=dict(config,contact_mode='pooled_separate',parent_copy=True)
        state=_initial_state(config,condition,'mixed')
        kernel=Kernel(_chemistry(condition['chemistry']),condition['sigma'])
        rng=np.random.default_rng(3); history=[]; events=[]
        for _ in range(4): history.append(sweep(state,kernel,rng,settings,events.append))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'compact.json'
            save_checkpoint(path,state,rng,'cfg','code',history,events)
            restored,rng2,h2,e2=load_checkpoint(path,'cfg','code')
            self.assertEqual(history,h2); self.assertEqual(events,e2)
            self.assertEqual(state,restored)
            self.assertEqual(sweep(state,kernel,rng,settings),reference_sweep(restored,kernel,rng2,settings))
            self.assertEqual(rng.bit_generator.state,rng2.bit_generator.state)


if __name__=='__main__': unittest.main()
