"""Full ternary coding dynamics and finite-resource populations.
All729 raw states and two immutable founder-origin labels are retained.
This is an abstract digital chemistry, with no molecular calibration.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
import zipfile
import numpy as np
from analysis.permutation_dynamics import compose, permutation_ranks
ROOT=Path(__file__).resolve().parents[1]
TARGET=(0,2,1)
DESCRIPTION=(0,0,1,2,2,1)
CYCLE_CODES=((1,0,2),(2,1,0))
METRICS=('linear','equidistant')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def atomic_npz(path,**arrays):
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as handle:
        temporary=Path(handle.name);np.savez_compressed(handle,**arrays);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)
    with zipfile.ZipFile(path) as archive:assert archive.testzip() is None
def distance_matrix(q,metric):
    if metric=='linear':return (np.arange(q)[:,None]-np.arange(q)[None,:])**2
    if metric=='equidistant':return 1-np.eye(q)
    raise ValueError('Unknown feature metric')
def key_selection(q,sigma,metric='linear'):
    if sigma<0 or not math.isfinite(sigma):raise ValueError('Finite nonnegative width required')
    if sigma==0:return np.eye(q)
    w=np.exp(-distance_matrix(q,metric)/(2*sigma*sigma));return w/w.sum(axis=1,keepdims=True)
def permanent(matrix):
    matrix=np.asarray(matrix)
    return math.fsum(float(np.prod(matrix[np.arange(len(matrix)),p])) for p in itertools.permutations(range(len(matrix))))
def build_matrix(sigma,description=DESCRIPTION,metric='linear'):
    q=max(description)+1
    states=np.asarray(list(itertools.product(range(q),repeat=len(description))),dtype=np.int16)
    keys,values=states[:,::2],states[:,1::2]
    distances=distance_matrix(q,metric)[:,keys].transpose(1,0,2);minimum=distances.min(axis=2)
    if sigma==0:
        gate=(minimum==0).astype(float);weights=(distances==minimum[:,:,None]).astype(float)
    else:
        gate=np.exp(-minimum/(2*sigma*sigma));weights=np.exp(-(distances-minimum[:,:,None])/(2*sigma*sigma))
    weights/=weights.sum(axis=2,keepdims=True);output=np.zeros((len(states),q,q))
    for a in range(q):output[:,:,a]=(weights*(values[:,None,:]==a)).sum(axis=2)
    success=np.prod(gate[:,description],axis=1)
    matrix=np.broadcast_to(success[:,None],(len(states),len(states))).copy()
    for j,symbol in enumerate(description):matrix*=output[:,symbol,states[:,j]]
    return matrix,states
def state_annotations(states,target=TARGET):
    q=len(target);keys_full=np.array([len(set(p[::2]))==q for p in states]);values_full=np.array([len(set(p[1::2]))==q for p in states])
    complete=keys_full&values_full;codes=np.full((len(states),q),-1,dtype=np.int16)
    for i in np.flatnonzero(complete):codes[i,states[i,::2]]=states[i,1::2]
    native=np.all(codes==target,axis=1);cycle=np.any(np.all(codes[:,None,:]==np.asarray(CYCLE_CODES),axis=2),axis=1)
    return dict(complete=complete,full_repertoire=values_full,native=native,cycle=cycle,codes=codes)
def word_from_code(code):return tuple(x for pair in enumerate(code) for x in pair)
def row_index(states,word):return int(np.flatnonzero(np.all(states==word,axis=1))[0])
def one_step_rows():
    rows=[]
    for metric in METRICS:
        for sigma in (0.,.2,.3,.5):
            matrix,states=build_matrix(sigma,metric=metric);masks=state_annotations(states);p=permanent(key_selection(3,sigma,metric))
            for name,code in [('native',TARGET),('cycle_phase_1',CYCLE_CODES[0]),('cycle_phase_2',CYCLE_CODES[1])]:
                row=matrix[row_index(states,word_from_code(code))]
                rows.append(dict(metric=metric,sigma=sigma,founder=name,
                    to_native=float(row[masks['native']].sum()),to_cycle=float(row[masks['cycle']].sum()),
                    to_other_complete=float(row[masks['complete']&~masks['native']&~masks['cycle']].sum()),
                    exit_complete=float(row[~masks['complete']].sum()),lose_repertoire=float(row[~masks['full_repertoire']].sum()),
                    exit_but_retain_repertoire=float(row[~masks['complete']&masks['full_repertoire']].sum()),
                    complete_probability_permanent=p*p,full_repertoire_probability_permanent=p,
                    permanent_complete_residual=float(abs(row[masks['complete']].sum()-p*p)),
                    permanent_repertoire_residual=float(abs(row[masks['full_repertoire']].sum()-p))))
    return rows

def first_passage_before_loss(matrix,masks,target_mask):
    """Serial hit before failure OR value-repertoire loss, allowing key loss."""
    transient=masks['full_repertoire']&~target_mask;sub=matrix[np.ix_(transient,transient)]
    reward=matrix[np.ix_(transient,target_mask)].sum(axis=1)
    solution=np.linalg.solve(np.eye(len(sub))-sub,reward)
    probability=np.zeros(len(matrix));probability[target_mask]=1.;probability[transient]=solution
    residual=float(np.max(np.abs(solution-sub@solution-reward)))
    return probability,residual,float(np.linalg.cond(np.eye(len(sub))-sub))

def first_passage_rows():
    rows=[]
    for metric in METRICS:
        for sigma in (.2,.3,.5):
            matrix,states=build_matrix(sigma,metric=metric);masks=state_annotations(states)
            for destination in ('native','cycle'):
                probability,residual,condition=first_passage_before_loss(matrix,masks,masks[destination])
                starts=[('native',TARGET)] if destination=='cycle' else [('cycle_phase_1',CYCLE_CODES[0]),('cycle_phase_2',CYCLE_CODES[1])]
                for name,code in starts:
                    rows.append(dict(metric=metric,sigma=sigma,start=name,destination=destination,
                        probability_before_loss_or_failure=float(probability[row_index(states,word_from_code(code))]),
                        transient_states=int(np.count_nonzero(masks['full_repertoire']&~masks[destination])),
                        equation_residual=residual,condition_number=condition))
    return rows

def high_precision_first_passage():
    """Independent27-state quotient, with60-digit literal kernel arithmetic."""
    import mpmath as mp
    rows=[];keys=list(itertools.product(range(3),repeat=3));perms=list(itertools.permutations(range(3)))
    code_key=lambda f:tuple(f.index(a) for a in range(3))
    for metric in METRICS:
        for sigma in (.2,.3):
            with mp.workdps(60):
                width=mp.mpf(str(sigma));d=distance_matrix(3,metric);kernel=mp.matrix(27,27)
                for i,key in enumerate(keys):
                    weights=[[mp.exp(-int(d[c,key[a]])/(2*width*width)) for a in range(3)] for c in range(3)]
                    gate=[max(w) for w in weights];channel=[[w/sum(weights[c]) for w in weights[c]] for c in range(3)]
                    success=mp.fprod(gate)**2
                    for j,destination_keys in enumerate(keys):
                        terms=[]
                        for ordering in perms:
                            terms.append(mp.fprod(channel[c][destination_keys[ordering[c]]]*channel[TARGET[c]][ordering[c]] for c in range(3)))
                        kernel[i,j]=success*mp.fsum(terms)
                for destination,codes in [('native',[TARGET]),('cycle',list(CYCLE_CODES))]:
                    absorbing=[keys.index(code_key(f)) for f in codes]
                    transient=[i for i in range(27) if i not in absorbing]
                    sub=mp.matrix([[int(i==j)-kernel[i,j] for j in transient] for i in transient])
                    rhs=mp.matrix([mp.fsum(kernel[i,j] for j in absorbing) for i in transient]);solution=mp.lu_solve(sub,rhs)
                    starts=[('native',TARGET)] if destination=='cycle' else [('cycle_phase_1',CYCLE_CODES[0]),('cycle_phase_2',CYCLE_CODES[1])]
                    for name,f in starts:
                        value=solution[transient.index(keys.index(code_key(f)))]
                        rows.append(dict(metric=metric,sigma=sigma,start=name,destination=destination,precision_digits=60,
                            quotient_states=27,probability=mp.nstr(value,55)))
    return rows

def tracking_rows():
    rows=[]
    for q in (3,8):
        for metric in METRICS:
            for sigma in (.15,.2,.25):
                w=key_selection(q,sigma,metric);log_step=float(2*np.log(np.diag(w)).sum())
                for h in (32,192):rows.append(dict(q=q,metric=metric,sigma_D=sigma,horizon=h,step_probability=math.exp(log_step),path_probability=math.exp(h*log_step)))
    return rows

def first_hit_tables(root):
    summaries,distributions,perturbations=[],[],[];previous=root/'results/permutation_dynamics'
    for path in sorted(previous.glob('q??_type_*.npz')):
        with np.load(path) as data:
            target=tuple(int(x) for x in data['target']);part='-'.join(map(str,data['cycle_type']))
            successor=data['next_index'];hits=data['first_native_time']
        q=len(target)
        with np.load(previous/f'actors_q{q:02d}.npz') as data:actors=data['actors']
        target_id=int(permutation_ranks([target])[0]);image=np.unique(successor)
        intersection=int(np.count_nonzero(successor[image]==target_id));centralizer=int(np.count_nonzero(successor==target_id))
        early=int(np.count_nonzero((hits>=0)&(hits<=2)));assert early==centralizer*intersection
        perturbed_hits=[]
        for a,b in itertools.combinations(range(q),2):
            transposition=list(range(q));transposition[a],transposition[b]=b,a
            hit=int(hits[int(permutation_ranks([compose(transposition,target)])[0])]);perturbed_hits.append(hit)
            perturbations.append(dict(q=q,cycle_type=part,swapped_a=a,swapped_b=b,first_native_time=hit,recovers=hit>=0))
        recovered=sum(h>=0 for h in perturbed_hits)
        summaries.append(dict(q=q,cycle_type=part,actors=len(actors),centralizer_size=centralizer,
            conjugacy_class_size=len(image),centralizer_class_intersection=intersection,
            native_basin=int(np.count_nonzero(hits>=0)),native_basin_fraction=float(np.mean(hits>=0)),
            first_hit_at_most_two_count=early,first_hit_at_most_two_fraction=early/len(actors),
            max_native_first_hit=int(hits.max()),native_basin_exhausted_by_two=bool(np.all(hits[hits>=0]<=2)),
            single_transposition_perturbations=len(perturbed_hits),single_transposition_recoveries=recovered,
            single_transposition_recovery_fraction=recovered/len(perturbed_hits) if perturbed_hits else ''))
        for horizon,count in zip(*np.unique(hits,return_counts=True)):
            distributions.append(dict(q=q,cycle_type=part,first_native_time=int(horizon),actor_count=int(count)))
    return summaries,distributions,perturbations
def _sample_rows(cdf,rows,uniform):
    lo=np.zeros(len(rows),dtype=np.int64);hi=np.full(len(rows),cdf.shape[1]-1,dtype=np.int64)
    while np.any(lo<hi):
        mid=(lo+hi)//2;right=uniform>=cdf[rows,mid]
        lo=np.where(right,mid+1,lo);hi=np.where(right,hi,mid)
    return lo

def population_cdf(matrix):
    laws=np.column_stack((matrix,np.maximum(0.,1-matrix.sum(axis=1))))
    laws/=laws.sum(axis=1,keepdims=True);cdf=laws.cumsum(axis=1);cdf[:,-1]=1.
    return cdf

def step_population(counts,resource,cdf,rng,death=.2,supply=30,capacity=120):
    types=counts.shape[1];offered=min(capacity,resource+supply);accepted=np.zeros_like(counts)
    if offered:
        parent_ids=np.repeat(np.arange(counts.size),counts.ravel())
        child_states=_sample_rows(cdf,parent_ids%types,rng.random(len(parent_ids)))
        good=np.flatnonzero(child_states<types)
        if len(good)>offered:good=rng.choice(good,offered,replace=False)
        child_ids=parent_ids[good]//types*types+child_states[good]
        accepted=np.bincount(child_ids,minlength=counts.size).reshape(counts.shape)
    births=int(accepted.sum());following=rng.binomial(counts+accepted,1-death)
    return following,offered-births,births

def initial_population(states,start):
    counts=np.zeros((2,len(states)),dtype=np.int64)
    ni=row_index(states,word_from_code(TARGET));ci=row_index(states,word_from_code(CYCLE_CODES[0]))
    if start=='native':counts[0,ni]=60
    elif start=='cycle':counts[1,ci]=60
    elif start=='mixed':counts[0,ni]=counts[1,ci]=30
    else:raise ValueError('Unknown founder condition')
    return counts

def run_case(task):
    out,case,plan_hash,source_hash,force=task;out=Path(out)
    path=out/'trajectories'/f"{case['case_id']}.npz";meta=path.with_suffix('.json')
    signature=dict(plan_sha256=plan_hash,source_sha256=source_hash,numpy_version=np.__version__,python_version=platform.python_version())
    if not force and path.exists() and meta.exists():
        old=json.loads(meta.read_text())
        if old.get('signature')==signature and old.get('trajectory_sha256')==digest(path):return dict(case_id=case['case_id'],cached=True,seconds=0.)
    start_time=time.perf_counter();matrix,states=build_matrix(case['sigma'],metric=case['metric']);cdf=population_cdf(matrix)
    horizon,replicates=case['horizon'],len(case['seeds'])
    counts=np.zeros((replicates,horizon+1,2,len(states)),dtype=np.int32)
    resources=np.zeros((replicates,horizon+1),dtype=np.int32);births=np.zeros_like(resources)
    for repeat,seed in enumerate(case['seeds']):
        rng=np.random.default_rng(seed);current=initial_population(states,case['start']);resource=60
        counts[repeat,0]=current;resources[repeat,0]=resource
        for step in range(1,horizon+1):
            if not current.any():resources[repeat,step:]=np.minimum(120,resource+30*np.arange(1,horizon-step+2));break
            current,resource,number=step_population(current,resource,cdf,rng)
            counts[repeat,step]=current;resources[repeat,step]=resource;births[repeat,step]=number
    path.parent.mkdir(exist_ok=True)
    atomic_npz(path,counts=counts,resource=resources,births=births,states=states,seeds=np.asarray(case['seeds'],dtype=np.uint64))
    elapsed=time.perf_counter()-start_time
    write_json(meta,dict(case=case,signature=signature,trajectory_sha256=digest(path),seconds=elapsed,
        trajectory_shapes={'counts':list(counts.shape),'resource':list(resources.shape)},max_total_population=int(counts.sum(axis=(2,3)).max())))
    return dict(case_id=case['case_id'],cached=False,seconds=elapsed)

def make_plan():
    conditions=[('linear',s,start) for s in (0.,.2,.3) for start in ('native','cycle','mixed')]
    conditions += [('equidistant',.3,start) for start in ('native','cycle','mixed')]
    seeds=np.random.SeedSequence(202610240901).spawn(len(conditions)*20);cases=[]
    for i,(metric,sigma,start) in enumerate(conditions):
        cases.append(dict(case_id=f'{metric}_s{sigma:g}_{start}',metric=metric,sigma=sigma,start=start,horizon=1000,
            seeds=[int(s.generate_state(1,dtype=np.uint64)[0]) for s in seeds[i*20:(i+1)*20]]))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),description=list(DESCRIPTION),target=list(TARGET),
        cycle_codes=[list(f) for f in CYCLE_CODES],state_space=729,founder_origin_labels=['native_founder','cycle_founder'],
        protocol={'initial_population':60,'initial_resource':60,'supply':30,'capacity':120,'death':.2,
                  'parents_retained':True,'newborns_act_next_sweep':True,'unit_cost_success_only':True},
        metric_normalization='linear x(c)=c; equidistant x(c)=e_c/sqrt(2), all distinct squared distances1',
        primary='Population survival and native/cycle state fractions at sweep1000, reported per condition.',
        secondary=['Zero-filled within-run state fractions averaged over sweeps501–1000',
                   'Founder-origin retention and code-state composition jointly',
                   'First full-output-repertoire loss and opposite-code appearance within founder origin'],
        uncertainty='Wilson95 for binary endpoints; run-level means, medians and quartiles for continuous endpoints.',
        scope='Prospectively fixed internal design, not external preregistration; abstract nonliving digital system.',cases=cases)

def wilson(successes,trials):
    z=1.959963984540054;p=successes/trials;den=1+z*z/trials;center=(p+z*z/(2*trials))/den
    radius=z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/den
    return max(0.,center-radius),min(1.,center+radius)
def summarize(out,plan):
    individual,summaries,time_rows=[],[],[]
    validation=dict(cases=0,trajectories=0,transitions=0,all_distinct_seeds=True,resource_balance=True,
                    population_death_bound=True,extinction_absorbing=True,origin_not_created=True)
    all_seeds=[]
    for case in plan['cases']:
        with np.load(out/'trajectories'/f"{case['case_id']}.npz") as data:
            counts,resources,births=data['counts'],data['resource'],data['births'];states,seeds=data['states'],data['seeds']
        masks=state_annotations(states);total=counts.sum(axis=(2,3));origins=counts.sum(axis=3)
        native=counts[:,:,:,masks['native']].sum(axis=(2,3));cycle=counts[:,:,:,masks['cycle']].sum(axis=(2,3))
        full=counts[:,:,:,masks['full_repertoire']].sum(axis=(2,3));complete=counts[:,:,:,masks['complete']].sum(axis=(2,3))
        origin_native=counts[:,:,:,masks['native']].sum(axis=3);origin_cycle=counts[:,:,:,masks['cycle']].sum(axis=3)
        offered=np.minimum(120,resources[:,:-1]+30)
        assert np.array_equal(resources[:,1:],offered-births[:,1:])
        assert np.all((births[:,1:]>=0)&(births[:,1:]<=offered))
        assert np.all(total[:,1:]<=total[:,:-1]+births[:,1:]);assert np.all((total[:,:-1]!=0)|(total[:,1:]==0))
        assert np.all((origins[:,:-1]!=0)|(origins[:,1:]==0));all_seeds.extend(int(x) for x in seeds)
        for rep,seed in enumerate(seeds):
            first=lambda a:int(np.flatnonzero(a)[0]) if np.any(a) else -1
            row=dict(case_id=case['case_id'],metric=case['metric'],sigma=case['sigma'],start=case['start'],replicate=rep,seed=int(seed),
                survives1000=int(total[rep,-1]>0),population1000=int(total[rep,-1]),native1000=int(native[rep,-1]),cycle1000=int(cycle[rep,-1]),
                full_repertoire1000=int(full[rep,-1]),native_founder_descendants1000=int(origins[rep,-1,0]),cycle_founder_descendants1000=int(origins[rep,-1,1]),
                first_extinction=first(total[rep]==0),first_full_repertoire_loss=first(full[rep]==0),
                first_native_code_in_cycle_origin=first(origin_native[rep,:,1]>0),first_cycle_code_in_native_origin=first(origin_cycle[rep,:,0]>0))
            for name,values in [('native',native),('cycle',cycle),('full_repertoire',full),('complete',complete)]:
                fraction=np.divide(values[rep],total[rep],out=np.zeros(total.shape[1]),where=total[rep]>0)
                row[f'{name}_fraction1000']=float(fraction[-1]);row[f'{name}_fraction_window501_1000']=float(fraction[501:].mean())
            individual.append(row)
        selected=individual[-len(seeds):]
        result=dict(case_id=case['case_id'],metric=case['metric'],sigma=case['sigma'],start=case['start'],replicates=len(seeds))
        for endpoint,values in [('survival',total[:,-1]>0),('cycle_present',cycle[:,-1]>0),('native_present',native[:,-1]>0),
                                ('full_repertoire_present',full[:,-1]>0),('both_origins_present',np.all(origins[:,-1]>0,axis=1))]:
            successes=int(values.sum());low,high=wilson(successes,len(seeds))
            result.update({f'{endpoint}_count':successes,f'{endpoint}_fraction':successes/len(seeds),
                           f'{endpoint}_wilson95_low':low,f'{endpoint}_wilson95_high':high})
        for name in ('native','cycle','full_repertoire','complete'):
            values=np.array([r[f'{name}_fraction_window501_1000'] for r in selected])
            result.update({f'{name}_window_mean':float(values.mean()),f'{name}_window_median':float(np.median(values)),
                           f'{name}_window_q25':float(np.quantile(values,.25)),f'{name}_window_q75':float(np.quantile(values,.75))})
        result['mean_population1000']=float(total[:,-1].mean())
        result['cycle_origin_native_appearance_count']=sum(r['first_native_code_in_cycle_origin']>=0 for r in selected)
        result['native_origin_cycle_appearance_count']=sum(r['first_cycle_code_in_native_origin']>=0 for r in selected)
        summaries.append(result)
        for h in range(total.shape[1]):
            row=dict(case_id=case['case_id'],metric=case['metric'],sigma=case['sigma'],start=case['start'],sweep=h,
                     alive_fraction=float(np.mean(total[:,h]>0)),mean_population=float(total[:,h].mean()),mean_native=float(native[:,h].mean()),
                     mean_cycle=float(cycle[:,h].mean()),mean_full_repertoire=float(full[:,h].mean()),mean_complete=float(complete[:,h].mean()),
                     mean_native_origin=float(origins[:,h,0].mean()),mean_cycle_origin=float(origins[:,h,1].mean()))
            for name,values in [('native',native),('cycle',cycle),('full_repertoire',full),('complete',complete)]:
                fractions=np.divide(values[:,h],total[:,h],out=np.zeros(len(seeds)),where=total[:,h]>0);row[f'mean_{name}_fraction']=float(fractions.mean())
            time_rows.append(row)
        validation['cases']+=1;validation['trajectories']+=len(seeds);validation['transitions']+=len(seeds)*case['horizon']
    assert len(all_seeds)==len(set(all_seeds));write_json(out/'trajectory_validation.json',validation)
    write_csv(out/'population_replicates.csv',individual);write_csv(out/'population_summary.csv',summaries);write_csv(out/'population_timecourse.csv',time_rows)
    return summaries,validation

def run(root=ROOT,workers=2,force=False):
    root=Path(root);out=root/'results/cycle_population';out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();plan_path=out/'plan.json'
    if plan_path.exists():plan=json.loads(plan_path.read_text())
    else:plan=make_plan();write_json(plan_path,plan)
    plan_hash=digest(plan_path);source_hash=digest(__file__)
    write_csv(out/'one_step_rates.csv',one_step_rows());write_csv(out/'first_passage.csv',first_passage_rows());write_csv(out/'first_passage_high_precision.csv',high_precision_first_passage());write_csv(out/'metric_tracking.csv',tracking_rows())
    hits,distributions,perturbations=first_hit_tables(root)
    write_csv(out/'first_hit_summary.csv',hits);write_csv(out/'first_hit_distributions.csv',distributions);write_csv(out/'single_transposition_perturbations.csv',perturbations)
    tasks=[(str(out),c,plan_hash,source_hash,force) for c in plan['cases']]
    if workers==1:completed=[run_case(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(2,workers)) as executor:completed=list(executor.map(run_case,tasks))
    summaries,validation=summarize(out,plan)
    execution=dict(plan_sha256=plan_hash,source_sha256=source_hash,python_version=platform.python_version(),numpy_version=np.__version__,
        workers=min(2,workers),seconds=time.perf_counter()-started,cases=completed,validation=validation,first_hit_target_types=len(hits),
        single_transposition_cases=len(perturbations),outputs='All raw729-state count trajectories retain two founder-origin labels; no population cap.')
    write_json(out/'execution.json',execution);print(json.dumps(execution,indent=2));return execution

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--force',action='store_true')
    arguments=parser.parse_args();run(arguments.root,arguments.workers,arguments.force)
