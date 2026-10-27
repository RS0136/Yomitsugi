import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from yomitsugi.core import Kernel, standard_chemistry
from yomitsugi.population import (Individual, PopulationState, sweep, save_checkpoint,
    load_checkpoint, replicate_seed, run_population, default_config)


def state(n=10, resource=100):
    return PopulationState([Individual(i,0,(0,0,1,1),(0,0,1,1)) for i in range(n)], resource, next_id=n)


class PopulationTests(unittest.TestCase):
    def test_snapshot_bookkeeping_and_parent_links(self):
        s, events = state(10), []
        row = sweep(s,Kernel(standard_chemistry(2),0),np.random.default_rng(2),dict(J=0,Smax=100,d=0),events.append)
        self.assertEqual((row['births'],row['N'],row['S'],row['attempted']),(10,20,90,10))
        self.assertTrue(all(e['reader_parent_id'] < 10 and e['template_parent_id'] < 10 for e in events))
        self.assertEqual(len({e['reader_parent_id'] for e in events}),10)
        # One individual must be its own target; child cannot act immediately.
        s, events = state(1), []
        sweep(s,Kernel(standard_chemistry(2),0),np.random.default_rng(3),dict(J=0,Smax=100,d=0),events.append)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['reader_parent_id'],events[0]['template_parent_id'])

    def test_resource_failure_death_and_empty_distinct(self):
        k = Kernel(standard_chemistry(2),0)
        s = state(10,2)
        row = sweep(s,k,np.random.default_rng(4),dict(J=0,Smax=100,d=1))
        self.assertEqual((row['births'],row['read_failed'],row['resource_blocked'],row['deaths'],row['N']),(2,0,8,12,0))
        row = sweep(s,k,np.random.default_rng(4),dict(J=30,Smax=100,d=0))
        self.assertEqual((row['births'],row['N'],row['S']),(0,0,30))
        s = PopulationState([Individual(0,0,(0,0,1,1),(0,0,0,0))],10,next_id=1)
        row = sweep(s,k,np.random.default_rng(5),dict(J=0,Smax=100,d=0))
        self.assertEqual((row['births'],row['read_failed'],row['resource_blocked'],row['S']),(0,1,0,10))

    def test_observation_labels_do_not_change_dynamics(self):
        a, b = state(10), state(10)
        b.individuals = [Individual(v.id,v.b,v.g,v.p,'renamed',1) for v in b.individuals]
        k = Kernel(standard_chemistry(2),.5)
        rng1, rng2 = np.random.default_rng(8), np.random.default_rng(8)
        for _ in range(20):
            r1 = sweep(a,k,rng1,dict(J=5,Smax=100,d=.2))
            r2 = sweep(b,k,rng2,dict(J=5,Smax=100,d=.2))
            self.assertEqual({x:r1[x] for x in ('N','S','births','read_failed','deaths')},
                             {x:r2[x] for x in ('N','S','births','read_failed','deaths')})
            self.assertEqual([(v.id,v.b,v.g,v.p) for v in a.individuals],[(v.id,v.b,v.g,v.p) for v in b.individuals])

    def test_expected_population_baselines(self):
        k = Kernel(standard_chemistry(2),0)
        for initial_resource, expected in [(100,32),(0,16)]:
            totals=[]
            for seed in range(800):
                s=state(20,initial_resource)
                sweep(s,k,np.random.default_rng(seed),dict(J=0,Smax=100,d=.2))
                totals.append(len(s.individuals))
            # >5 SE tolerance from independently initialized Bernoulli survival.
            self.assertLess(abs(np.mean(totals)-expected),.55)

    def test_checkpoint_exact_resume_and_mismatch(self):
        k=Kernel(standard_chemistry(2),.5)
        a,b=state(20),state(20)
        rng1,rng2=np.random.default_rng(10),np.random.default_rng(10)
        config=dict(J=10,Smax=100,d=.2)
        for _ in range(10):
            sweep(a,k,rng1,config)
        for _ in range(4):
            sweep(b,k,rng2,config)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.json'
            save_checkpoint(path,b,rng2,'config','code')
            with self.assertRaises(ValueError):
                load_checkpoint(path,'wrong','code')
            b,rng2,_,_=load_checkpoint(path,'config','code')
            for _ in range(6):
                sweep(b,k,rng2,config)
        self.assertEqual(a,b)
        self.assertEqual(rng1.bit_generator.state,rng2.bit_generator.state)

    def test_parallelism_reproduces_each_repeat(self):
        cfg=default_config('smoke')
        cfg.update(sweeps=4,repeats=2,modes=['mixed'],conditions=cfg['conditions'][:1],birth_log_limit=10)
        with tempfile.TemporaryDirectory() as temp:
            a=run_population(Path(temp)/'a',cfg,workers=1)
            b=run_population(Path(temp)/'b',cfg,workers=2)
            ra=sorted(Path(a['run_directory']).glob('*/trajectory.csv'))
            rb=sorted(Path(b['run_directory']).glob('*/trajectory.csv'))
            self.assertEqual([p.read_bytes() for p in ra],[p.read_bytes() for p in rb])
            again=run_population(Path(temp)/'a',cfg,workers=2,resume=True)
            self.assertEqual(again['run_id'],a['run_id'])

    def test_separate_pool_control_resource_accounting(self):
        s=state(10,10)
        s.individuals=[Individual(v.id,v.b,v.g,v.p,'A' if i<5 else 'B',int(i>=5)) for i,v in enumerate(s.individuals)]
        events=[]
        row=sweep(s,Kernel(standard_chemistry(2),0),np.random.default_rng(3),dict(J=2,Smax=100,d=0,contact_mode='pooled_separate'),events.append)
        self.assertEqual((row['births'],row['S']),(10,2))
        self.assertTrue(all(e['reader_origin']==e['template_origin'] for e in events))
        self.assertEqual(s.resource,sum(s.pool_resources.values()))


if __name__=='__main__':
    unittest.main()
