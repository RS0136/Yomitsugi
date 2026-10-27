"""Prespecified one-axis comparisons; exact first-step laws, bounded serial trials."""
from __future__ import annotations
import itertools, json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .core import Chemistry, Kernel, standard_chemistry, NumericalPrecisionError

HORIZONS = (1,2,4,8,16,32)
EPSILONS = (0,.01,.05,.1)

def system(name,b,g,p):
    return dict(name=name,b=b,g=list(g),p=list(p))

def conditions():
    records=[]
    identity=system('identity',0,(0,0,1,1),(0,0,1,1))
    def add(name,axis,sigma,ch,a,b,**extra):
        records.append(dict(name=name,axis=axis,sigma=sigma,chemistry=ch.to_dict(),types=[a,b],**extra))
    for sig in (0,.2,.5,1):
        for delta in (0,.25,.5,1):
            ch=Chemistry({0:0,1:delta},{0:0,1:1},{0:0,1:1,2:0,3:1},{0:0,1:0,2:delta,3:delta})
            for compensated in (False,True):
                b=system('changed',1,(0,0,1,1),(2,2,3,3) if compensated else (0,0,1,1))
                add(f'backbone_d{delta}_comp{int(compensated)}_s{sig}','backbone',sig,ch,identity,b,delta=delta,compensated=compensated)
        for kind in ('unused','alias','shift025','shift05','shift1'):
            delta={'unused':2,'alias':0,'shift025':.25,'shift05':.5,'shift1':1}[kind]
            ch=Chemistry({0:0},{0:0,1:1,2:delta},{0:0,1:1},{0:0,1:0})
            b=system('changed',0,(0,0,1,1) if kind=='unused' else (2,2,1,1),(0,0,1,1))
            add(f'alphabet_{kind}_s{sig}','alphabet',sig,ch,identity,b,delta=delta,control=kind)
        for kind in ('unused','alias','shift025','shift05','shift1'):
            delta={'unused':2,'alias':0,'shift025':.25,'shift05':.5,'shift1':1}[kind]
            ch=Chemistry({0:0},{0:0,1:1},{0:0,1:1,2:delta,3:1+delta},{i:0 for i in range(4)})
            b=system('changed',0,(0,0,1,1),(0,0,1,1) if kind=='unused' else (2,2,3,3))
            add(f'residue_{kind}_s{sig}','residue',sig,ch,identity,b,delta=delta,control=kind)
        add(f'code_swap_s{sig}','code',sig,standard_chemistry(),identity,system('swap',0,(1,0,0,1),(0,1,1,0)))
        for duplicate in (0,1):
            p=(0,0,1,1,duplicate,duplicate)
            # Shared four available composite IDs over the same base alphabet U={u0,u1}.
            ch=Chemistry({0:0},{i:float(i) for i in range(4)},{i:float(i) for i in range(3)},{i:0 for i in range(3)})
            add(f'code_capacity_dup{duplicate}_s{sig}','code',sig,ch,
                system('two_values',0,p,p),system('three_values',0,(0,0,1,1,2,2),(0,0,1,1,2,2)),
                base_alphabet_size=2,available_composite_ids=4,duplicate=duplicate)
    return records

def offspring_law(k,p,b,g):
    values=tuple(sorted(set(p[1::2])))
    states=list(itertools.product(values,repeat=len(g)))
    c=k.birth_probability(p,b,g)
    if c==0:
        return states,np.zeros(len(states))
    lookup={a:i for i,a in enumerate(k.residues)}
    probabilities=[k.distribution(tuple(p),b,x)[1] for x in g]
    weights=[]
    log_c=k.log_success(p,b,g)
    for s in states:
        factors=[float(probabilities[r][lookup[a]]) for r,a in enumerate(s)]
        if any(factor==0 for factor in factors):
            weights.append(0.)
            continue
        log_weight=log_c+math.fsum(math.log(factor) for factor in factors)
        weight=math.exp(log_weight)
        if weight==0:
            raise NumericalPrecisionError('Positive offspring probability underflow; high precision required')
        weights.append(weight)
    weights=np.array(weights)
    return states,weights

def serial(k,p,b,g,states,horizons,rng,trials):
    if len(states)<=64:
        matrix=np.zeros((len(states),len(states)))
        indices={s:i for i,s in enumerate(states)}
        for i,s in enumerate(states):
            children,weights=offspring_law(k,s,b,g)
            for child,w in zip(children,weights): matrix[i,indices[child]]=w
        children,weights=offspring_law(k,p,b,g)
        row=np.zeros(len(states))
        for child,w in zip(children,weights): row[indices[child]]=w
        values={1:float(row.sum())}
        for h in range(2,max(horizons)+1):
            row=row@matrix
            if h in horizons: values[h]=float(row.sum())
        return values,'finite_matrix',None
    counts={h:0 for h in horizons}
    for _ in range(trials):
        reader=tuple(p)
        for h in range(1,max(horizons)+1):
            reader=k.read_word(reader,b,g,rng)
            if reader is None: break
            if h in counts: counts[h]+=1
    return {h:counts[h]/trials for h in horizons},'independent_serial_trials',counts

def run_screen(output_dir:Path,seed=31001,trials=256):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    scenarios=conditions()
    (output_dir/'conditions.json').write_text(json.dumps(scenarios,indent=2),encoding='utf8')
    rows=[]; distance_rows=[]
    for condition_index,cond in enumerate(scenarios):
        k=Kernel(Chemistry(**cond['chemistry']),cond['sigma'])
        panel=list(itertools.product(sorted(k.chemistry.backbone_features),sorted(k.chemistry.symbol_features)))
        for i,a in enumerate(cond['types']):
            for j,t in enumerate(cond['types']):
                p=tuple(a['p']); g=tuple(t['g']); b=t['b']; native=tuple(t['p'])
                sub_seed=seed+condition_index*4+i*2+j
                states,weights=offspring_law(k,p,b,g)
                c=k.birth_probability(p,b,g)
                distances=np.array([k.distance(s,native,panel) for s in states])
                effective=np.array([k.distance(s,native,panel,True) for s in states])
                continuation,method,counts=serial(k,p,b,g,states,HORIZONS,np.random.default_rng(sub_seed),trials)
                common=dict(condition=cond['name'],axis=cond['axis'],sigma=cond['sigma'],actor=i,target=j,
                            L=len(g),m=len(p)//2,C_catalog=len(k.chemistry.symbol_features),A_catalog=len(k.residues),
                            used_symbols=len(set(g)),used_values=len(set(p[1::2])),used_keys=len(set(p[::2])),
                            seed=sub_seed,replicate_id=0,termination_reason='normal',probe_panel=json.dumps(panel),
                            C=c,log_C=k.log_success(p,b,g),method=method)
                recovery={f'R_{eps}':float(weights[distances<=eps].sum()/c) if c else None for eps in EPSILONS}
                recovery.update({f'CR_{eps}':float(weights[distances<=eps].sum()) for eps in EPSILONS})
                mean_d=float(weights@distances/c) if c else None
                mean_effective=float(weights@effective/c) if c else None
                for h,sh in continuation.items():
                    rows.append(common|recovery|dict(h=h,S_h=sh,conditional_after_first=sh/c if c else None,
                                distance_mean=mean_d,effective_distance_mean=mean_effective,
                                serial_trials=trials if counts else None,serial_successes=counts[h] if counts else None))
                for s,w,d,e in zip(states,weights,distances,effective):
                    distance_rows.append(common|dict(h=1,child=json.dumps(s),probability=float(w),D=float(d),D_features=float(e)))
    pd.DataFrame(rows).to_csv(output_dir/'comparisons.csv',index=False)
    pd.DataFrame(distance_rows).to_csv(output_dir/'distance_distribution.csv',index=False)
    result=dict(conditions=len(scenarios),directions=len(scenarios)*4,rows=len(rows),seed=seed,trials=trials,
                status='complete',selection='All fixed one-axis cases retained; no outcome-driven filtering')
    (output_dir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result

def run_confirm(output_dir:Path,seed=71003):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    plan=dict(claims=['H1_birth_vs_native_recovery','H2_equal_C_different_S2'],
              conditions=['q3_all_permutation_pairs','m3_identity_and_3cycle_to_two_keys'],
              horizons=list(HORIZONS),epsilons=list(EPSILONS),seed=seed,independent_repeats=2000,
              controls=['parent_copy','slot_pair_permutation','global_ID_relabel'],
              statistics='Exact enumeration has no sampling CI; simulation verification only, no selection')
    raw=json.dumps(plan,sort_keys=True,indent=2)
    plan_path=output_dir/'confirmation_plan.json'
    if plan_path.exists() and plan_path.read_text(encoding='utf8')!=raw:
        raise ValueError('Frozen confirmation plan differs; select a new run directory')
    plan_path.write_text(raw,encoding='utf8')
    digest=hashlib.sha256(raw.encode()).hexdigest()
    (output_dir/'confirmation_plan.sha256').write_text(digest+'\n')
    k=Kernel(standard_chemistry(3),0)
    g=(0,0,1,1,0,0)
    rows=[]
    for ai,p in enumerate([(0,0,1,1,2,2),(0,1,1,2,2,0)]):
        rng=np.random.default_rng(seed+ai)
        for copy in (False,True):
            counts={h:0 for h in HORIZONS}
            for _ in range(plan['independent_repeats']):
                reader=p
                for h in range(1,33):
                    child=k.read_word(reader,0,g,rng)
                    if child is None: break
                    reader=reader if copy else child
                    if h in counts: counts[h]+=1
            for h,n in counts.items():
                expected=1. if (copy or ai==0 or h==1) else 0.
                if n/plan['independent_repeats']!=expected: raise AssertionError('H2 confirmation mismatch')
                rows.append(dict(condition='equal_first_birth',actor=ai,parent_copy=copy,h=h,C=1,
                            S_h=expected,observed=n/plan['independent_repeats'],replicate_id=0,seed=seed+ai,
                            trials=plan['independent_repeats'],plan_hash=digest,termination_reason='normal'))
    pd.DataFrame(rows).to_csv(output_dir/'confirmation.csv',index=False)
    result=dict(status='complete',plan_hash=digest,H2_exact_passed=True,
                note='H1 uses independently reproduced exact enumeration; no new significance test')
    (output_dir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result
