#!/usr/bin/env python3
"""Analyse generated populations and run bounded, purely digital confirmations."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import argparse, csv, gzip, hashlib, io, itertools, json, math, os
from pathlib import Path
import platform, sys, time, zipfile
import importlib.metadata
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
STATES = tuple(itertools.product((0, 1), repeat=4))
STATE_KEYS = tuple(",".join(map(str, p)) for p in STATES)
STATE_INDEX = {key: i for i, key in enumerate(STATE_KEYS)}
NATIVE_KEYS = {"0,0,1,1": {"0,0,1,1", "1,1,0,0"}, "1,0,0,1": {"0,1,1,0", "1,0,0,1"}}
NATIVE_IDENTITY = np.array([key in NATIVE_KEYS["0,0,1,1"] for key in STATE_KEYS])
MONOMORPHIC = np.array([len(set(p)) == 1 for p in STATES])
TWO_VALUES = np.array([p[1] != p[3] for p in STATES])
WINDOWS = ((501, 1000), (251, 1000), (751, 1000))

def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()

def implementation_signature(root):
    """Include the original rules and numerical runtime in resume identity."""
    paths=[Path(__file__)]+sorted((Path(root)/"vendor/yomitsugi").glob("*.py"))
    hasher=hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode()); hasher.update(path.read_bytes())
    hasher.update(json.dumps(dict(python=sys.version,numpy=np.__version__,scipy=importlib.metadata.version("scipy")),sort_keys=True).encode())
    return hasher.hexdigest()

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf8")
    os.replace(temporary, path)

def write_rows(path, rows):
    rows=list(rows); path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8"); return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf8") as stream:
        writer=csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def interval(values):
    values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]
    if not len(values): return dict(n=0,mean=None,ci95_low=None,ci95_high=None,sd=None)
    mean=float(values.mean())
    if len(values)==1: return dict(n=1,mean=mean,ci95_low=None,ci95_high=None,sd=None)
    sd=float(values.std(ddof=1)); radius=float(student_t.ppf(.975,len(values)-1)*sd/math.sqrt(len(values)))
    return dict(n=len(values),mean=mean,ci95_low=mean-radius,ci95_high=mean+radius,sd=sd)

def wilson(successes,n,z=1.959963984540054):
    if not n: return None,None
    p=successes/n; denominator=1+z*z/n; center=(p+z*z/(2*n))/denominator
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return center-radius,center+radius

def transition_matrix(sigma,g=(0,0,1,1)):
    """Independent direct four-position formula, not the original Kernel."""
    if sigma<0 or not math.isfinite(sigma): raise ValueError("Finite nonnegative sigma required")
    M=np.zeros((16,16)); state_array=np.array(STATES,dtype=int)
    for i,p in enumerate(STATES):
        laws=[]
        for c in g:
            d2=np.array([(c-p[0])**2,(c-p[2])**2],dtype=float)
            weights=(d2==0).astype(float) if sigma==0 else np.exp(-d2/(2*sigma*sigma))
            if not weights.any(): laws.append(np.zeros(2)); continue
            success=float(weights.max()); law=np.zeros(2)
            for j,weight in enumerate(weights): law[p[2*j+1]]+=success*weight/weights.sum()
            laws.append(law)
        M[i]=np.prod(np.array(laws)[np.arange(4)[:,None],state_array.T],axis=0)
    if M.min()<0 or M.sum(axis=1).max()>1+1e-13: raise ArithmeticError("Invalid substochastic matrix")
    return M

def dominant_composition(M):
    values,vectors=np.linalg.eig(M.T); i=int(np.argmax(values.real))
    if abs(values[i].imag)>1e-10: raise ArithmeticError("Nonreal dominant eigenvalue")
    vector=vectors[:,i].real; vector/=vector.sum()
    if vector.min()<-1e-10: raise ArithmeticError("Dominant vector is not nonnegative")
    vector[np.abs(vector)<1e-15]=0; vector/=vector.sum()
    residual=float(np.max(np.abs(vector@M-values[i].real*vector)))
    return float(values[i].real),vector,residual

def row_metrics(row):
    n=int(row["N"]); counts=json.loads(row["reader_counts"])
    vec=np.array([counts.get(k,0) for k in STATE_KEYS],dtype=float)
    if int(vec.sum())!=n: raise AssertionError("Reader counts do not match population")
    native=0
    for key,count in json.loads(row["template_reader_counts"]).items():
        template,reader=key.split("|"); backbone,g=template.split(":",1)
        if backbone!="0" or g not in NATIVE_KEYS: raise ValueError("Unexpected template")
        if reader in NATIVE_KEYS[g]: native+=int(count)
    origins=json.loads(row["origin_counts"])
    return dict(sweep=int(row["sweep"]),N=n,S=int(row["S"]),births=int(row["births"]),
                deaths=int(row["deaths"]),read_failed=int(row["read_failed"]),attempted=int(row["attempted"]),
                resource_blocked=int(row["resource_blocked"]),reference_count=native,
                monomorphic_count=int(vec[MONOMORPHIC].sum()),two_value_count=int(vec[TWO_VALUES].sum()),
                alive=int(n>0),both_origins=int(sum(int(v)>0 for v in origins.values())==2)),vec

def summarize_window(rows,vectors,low,high):
    selected=[(r,v) for r,v in zip(rows,vectors) if low<=r["sweep"]<=high]
    if not selected: raise ValueError("Empty analysis window")
    rr,vv=zip(*selected); n=np.array([r["N"] for r in rr]); occupied=n>0
    fractions=np.divide(np.array(vv),n[:,None],out=np.zeros((len(rr),16)),where=n[:,None]>0)
    reference=np.array([r["reference_count"] for r in rr]); ref_fraction=np.divide(reference,n,out=np.zeros(len(rr)),where=n>0)
    average=fractions.mean(axis=0); average_alive=fractions[occupied].mean(axis=0) if occupied.any() else None
    result=dict(window_start=low,window_end=high,observations=len(rr),alive_fraction=float(occupied.mean()),
                mean_N=float(n.mean()),reference_fraction_zero_filled=float(ref_fraction.mean()),
                reference_fraction_alive=float(ref_fraction[occupied].mean()) if occupied.any() else None,
                reference_count_mean=float(reference.mean()),mean_births=float(np.mean([r["births"] for r in rr])),
                monomorphic_fraction_zero_filled=float(average[MONOMORPHIC].sum()),
                two_value_fraction_zero_filled=float(average[TWO_VALUES].sum()))
    for i,key in enumerate(STATE_KEYS):
        result["p"+key.replace(",","")]=float(average[i])
        result["alive_p"+key.replace(",","")]=float(average_alive[i]) if occupied.any() else None
    return result

def summarize_groups(run_windows,group_keys,predictions=None):
    groups={}
    for row in run_windows:
        key=tuple(row[k] for k in (*group_keys,"window_start","window_end")); groups.setdefault(key,[]).append(row)
    summaries=[]; distributions=[]
    for key,rows in groups.items():
        metadata=dict(zip((*group_keys,"window_start","window_end"),key)); summary=dict(metadata,independent_replicates=len(rows))
        for metric in ("mean_N","reference_fraction_zero_filled","reference_fraction_alive","reference_count_mean",
                       "monomorphic_fraction_zero_filled","two_value_fraction_zero_filled","alive_fraction","mean_births"):
            stat=interval([r[metric] for r in rows if r[metric] is not None]); summary.update({metric+"_"+k:v for k,v in stat.items()})
        predicted=predictions.get(float(rows[0]["sigma"])) if predictions is not None else None; means=[]; alive_means=[]
        for i,state in enumerate(STATE_KEYS):
            stat=interval([r["p"+state.replace(",","")] for r in rows]); means.append(stat["mean"])
            alive_stat=interval([r["alive_p"+state.replace(",","")] for r in rows if r["alive_p"+state.replace(",","")] is not None])
            alive_means.append(alive_stat["mean"])
            distributions.append(dict(metadata,reader=state,**stat,**{"alive_"+k:v for k,v in alive_stat.items()},
                                      prediction=float(predicted[i]) if predicted is not None else ""))
        if predicted is not None:
            summary["reference_prediction"]=float(predicted[NATIVE_IDENTITY].sum())
            half_l1=float(np.abs(np.array(means)-predicted).sum()/2)
            summary["distribution_half_L1_zero_filled"]=half_l1
            summary["distribution_TV_with_extinction"]=half_l1+abs(1-sum(means))/2
            summary["distribution_TV_alive"]=float(np.abs(np.array(alive_means)-predicted).sum()/2) if all(v is not None for v in alive_means) else None
        summaries.append(summary)
    return summaries,distributions

@contextmanager
def original_population_input(evidence,plan):
    """Read a verified generated population, or an explicitly selected legacy ZIP."""
    evidence=Path(evidence)
    if evidence.is_dir():
        manifest=json.loads((evidence/"manifest.json").read_text())
        if not manifest.get("all_completed"):
            raise AssertionError("Generated original population is incomplete")
        artifacts=manifest["artifacts"]
        def read(relative):
            path=(evidence/relative).resolve()
            if not path.is_relative_to(evidence.resolve()):
                raise ValueError("Population input path leaves its output directory")
            raw=path.read_bytes()
            if digest_bytes(raw)!=artifacts.get(relative):
                raise AssertionError("Generated population input hash mismatch: "+relative)
            return raw
        latest=json.loads(read("latest.json"))
        run_id=manifest["run_id"]
        if not latest.get("all_completed") or latest["run_id"]!=run_id:
            raise AssertionError("Generated population completion metadata mismatch")
        recorded=json.loads(read(run_id+"/plan.json"))
        if any(recorded[key]!=manifest["signature"][key] or latest[key]!=manifest["signature"][key]
               for key in ("config_hash","code_hash")):
            raise AssertionError("Generated population configuration/source mismatch")
        settings=recorded["settings"]
        expected={(condition["name"],mode,repeat)
                  for condition in settings["conditions"] for mode in settings["modes"]
                  for repeat in range(settings["repeats"])}
        yield read,run_id,"results/original_population/","generated_original_population",settings,expected
    elif evidence.is_file():
        with zipfile.ZipFile(evidence) as archive:
            def read(relative):
                return archive.read("yomitsugi_alife/results/population/"+relative)
            yield read,plan["original_run"],"results/population/","posthoc_original",None,None
    else:
        raise FileNotFoundError("Original population input is missing. Run python -m analysis.original_population first.")


def reanalyse_original(evidence,output,plan):
    original=plan["original_reanalysis"]; all_windows=[]; final_rows=[]; source_hashes={}
    windows=[tuple(original["primary_window"]),*[tuple(w) for w in original["sensitivity_windows"]]]
    horizon=max(high for _,high in windows)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with original_population_input(evidence,plan) as source, gzip.open(output/"original_trajectory_metrics.csv.gz","wt",newline="",encoding="utf8") as trajectory_stream:
        read,run_id,source_prefix,analysis,settings,expected=source
        runs_raw=read("runs.csv")
        source_hashes[source_prefix+"runs.csv"]=digest_bytes(runs_raw)
        runs=list(csv.DictReader(io.StringIO(runs_raw.decode())))
        if len(runs)!=original["expected_runs"]: raise AssertionError("Unexpected original run count")
        identities=[(run["condition"],run["mode"],int(run["repeat_id"])) for run in runs]
        if len(set(identities))!=len(identities) or (expected is not None and set(identities)!=expected):
            raise AssertionError("Original population conditions or replicates differ from its plan")
        if settings is not None and settings["sweeps"]!=horizon:
            raise AssertionError("Population horizon differs from the analysis windows")
        writer=None
        for run in runs:
            if run["run_id"]!=run_id or run["termination_reason"]!="normal_completion":
                raise AssertionError("Original provenance or completion mismatch")
            relative=f"{run_id}/{run['condition']}__{run['mode']}__r{int(run['repeat_id']):03d}"
            if settings is not None:
                expected_seed=int(np.random.SeedSequence([int(settings["base_seed"]),int(run["repeat_id"])]).generate_state(1,dtype=np.uint64)[0])
                summary=json.loads(read(relative+"/summary.json"))
                if (int(run["seed"])!=expected_seed or summary["seed"]!=expected_seed
                        or int(run["completed_sweeps"])!=horizon or summary["completed_sweeps"]!=horizon
                        or summary["termination_reason"]!="normal_completion"
                        or any(str(summary[key])!=run[key] for key in ("run_id","config_hash","code_hash","condition","mode","repeat_id"))):
                    raise AssertionError("Generated population run metadata mismatch")
            sigma=float(run["condition"].rsplit("_",1)[1])
            metadata=dict(condition=run["condition"],sigma=sigma,mode=run["mode"],repeat_id=int(run["repeat_id"]),
                          seed=run["seed"],source_run=run["run_id"],analysis=analysis)
            member=relative+"/trajectory.csv"
            raw=read(member); source_hashes[source_prefix+member]=digest_bytes(raw); rows=[]; vectors=[]
            for raw_row in csv.DictReader(io.StringIO(raw.decode())):
                if any(raw_row[key]!=run[key] for key in ("run_id","config_hash","code_hash","condition","mode","repeat_id","seed")) or raw_row["termination_reason"]!="normal_completion":
                    raise AssertionError("Trajectory provenance differs from its run record")
                row,vector=row_metrics(raw_row); rows.append(row); vectors.append(vector); out_row=dict(metadata,**row)
                if writer is None:
                    writer=csv.DictWriter(trajectory_stream,fieldnames=list(out_row)); writer.writeheader()
                writer.writerow(out_row)
            if [row["sweep"] for row in rows]!=list(range(1,horizon+1)): raise AssertionError("Incomplete original trajectory")
            for lo,hi in windows: all_windows.append(dict(metadata,**summarize_window(rows,vectors,lo,hi)))
            initially_mixed=run["mode"] in ("mixed","pooled_shared","pooled_separate")
            loss=next((r["sweep"] for r in rows if not r["both_origins"]),None) if initially_mixed else None
            final_rows.append(dict(metadata,final_N=rows[-1]["N"],final_S=rows[-1]["S"],cumulative_births=int(run["births"]),
                                   cumulative_overflow=int(run["overflow"]),initially_two_origins=initially_mixed,
                                   both_origins_final=rows[-1]["both_origins"] if initially_mixed else "",
                                   first_origin_loss=loss if loss is not None else "",
                                   origin_loss_right_censored=bool(initially_mixed and loss is None)))
    prediction=dominant_composition(transition_matrix(.5))[1]
    summaries,distributions=summarize_groups(all_windows,("condition","sigma","mode"))
    for row in distributions:
        if row["sigma"]==.5 and row["mode"]=="single_a": row["prediction"]=float(prediction[STATE_INDEX[row["reader"]]])
    write_rows(output/"original_run_windows.csv",all_windows)
    write_rows(output/"original_group_windows.csv",summaries)
    write_rows(output/"original_reader_distributions.csv",distributions)
    write_rows(output/"original_final_outcomes.csv",final_rows)
    lineage=[]
    for sigma in (0.,.5):
        for mode in ("mixed","pooled_shared","pooled_separate"):
            selected=[r for r in final_rows if r["sigma"]==sigma and r["mode"]==mode]
            count=sum(r["both_origins_final"] for r in selected); low,high=wilson(count,len(selected))
            lineage.append(dict(sigma=sigma,mode=mode,n=len(selected),both_origins=count,fraction=count/len(selected),
                                 wilson95_low=low,wilson95_high=high,time=horizon,
                                 interpretation="finite_time_retention_not_permanent_coexistence"))
    write_rows(output/"original_lineage_retention.csv",lineage); write_json(output/"original_source_hashes.json",source_hashes)
    return dict(runs=len(final_rows),trajectory_rows=len(final_rows)*horizon,analysis_windows=len(all_windows),
                source_run=run_id,input_kind=analysis)

def extinction_probabilities(M,death,horizon,initial_population=60,initial_index=3):
    """Exact finite-time extinction via the parent/child generating function."""
    q=np.zeros(len(M)); survival=1-death; results=[0.0]; C=M.sum(axis=1)
    for _ in range(horizon):
        q=(death+survival*q)*(1-survival*C+survival*(M@q))
        if q.min()<-1e-13 or q.max()>1+1e-13: raise ArithmeticError("Invalid extinction generating function")
        results.append(float(q[initial_index]**initial_population))
    return results

def unrestricted_step(counts,M,death,rng):
    """Exact aggregate law, vectorized over independent replicate populations."""
    survival=1-death; next_counts=rng.binomial(counts,survival)
    for i in range(len(M)):
        probabilities=np.r_[survival*M[i],1-survival*M[i].sum()]
        if probabilities.min()<-1e-14: raise ArithmeticError("Invalid offspring probabilities")
        children=rng.multinomial(counts[:,i],probabilities)
        next_counts+=children[:,:-1]
    return next_counts

def run_unlimited(output,plan,plan_hash):
    cfg=plan["unlimited"]; M=transition_matrix(cfg["sigma"]); rho,_,_=dominant_composition(M)
    death_values=[.30,rho/(1+rho),.45]; all_rows=[]; end_rows=[]; (output/"unlimited").mkdir(exist_ok=True)
    for case_index,death in enumerate(death_values):
        label=("d030","dcritical","d045")[case_index]
        seed=int(np.random.SeedSequence([cfg["base_seed"],case_index]).generate_state(1,dtype=np.uint64)[0])
        rng=np.random.default_rng(seed); count=np.zeros((cfg["replicates"],16),dtype=np.int64)
        initial_index=STATES.index(tuple(cfg["initial_reader"])); count[:,initial_index]=cfg["initial_population"]
        trajectory=np.empty((cfg["horizon"]+1,cfg["replicates"],16),dtype=np.int64); trajectory[0]=count
        for time_step in range(1,cfg["horizon"]+1):
            count=unrestricted_step(count,M,death,rng); trajectory[time_step]=count
        B=(1-death)*(np.eye(16)+M); expected=trajectory[0,0].astype(float)
        extinction=extinction_probabilities(M,death,cfg["horizon"],cfg["initial_population"],initial_index)
        for time_step,observed in enumerate(trajectory):
            total=observed.sum(axis=1); ref=observed[:,NATIVE_IDENTITY].sum(axis=1)
            nstat,rstat=interval(total),interval(ref); extinct=int(np.count_nonzero(total==0)); wil_low,wil_high=wilson(extinct,len(total))
            row=dict(case=label,sigma=cfg["sigma"],death=death,time=time_step,replicates=len(total),seed=seed,
                     rho_M=rho,rho_B=(1-death)*(1+rho),expected_N=float(expected.sum()),
                     expected_reference_count=float(expected[NATIVE_IDENTITY].sum()),extinct_replicates=extinct,
                     extinction_frequency=extinct/len(total),exact_extinction_probability=extinction[time_step],
                     extinction_wilson95_low=wil_low,extinction_wilson95_high=wil_high)
            row.update({"N_"+k:v for k,v in nstat.items()}); row.update({"reference_count_"+k:v for k,v in rstat.items()})
            all_rows.append(row)
            if time_step==cfg["horizon"]: end_rows.append(row)
            expected=expected@B
        np.savez_compressed(output/"unlimited"/(label+".npz"),counts=trajectory,states=np.array(STATES),
                            sigma=cfg["sigma"],death=death,seed=np.array(seed,dtype=np.uint64),M=M,B=B)
    write_rows(output/"unlimited_expectations.csv",all_rows); write_rows(output/"unlimited_endpoints.csv",end_rows)
    return dict(cases=len(death_values),replicates_per_case=cfg["replicates"],horizon=cfg["horizon"],
                rho_M=rho,critical_death=rho/(1+rho),plan_hash=plan_hash)

def load_vendor(root):
    vendor=Path(root)/"vendor"
    if not (vendor/"yomitsugi"/"population.py").exists(): raise FileNotFoundError("Expected original package at vendor/yomitsugi")
    if str(vendor) not in sys.path: sys.path.insert(0,str(vendor))
    from yomitsugi.core import Kernel, standard_chemistry
    from yomitsugi.population import Individual, PopulationState, sweep
    return Kernel,standard_chemistry,Individual,PopulationState,sweep

def finite_task(task):
    root,output,plan,plan_hash,case_index,repeat,reuse=task; cfg=plan["finite_resource"]; case=cfg["cases"][case_index]
    sigma,scale=case["sigma"],case["scale"]; case_name=f"sigma{sigma:g}_scale{scale}"
    folder=Path(output)/"finite"/(case_name+f"__r{repeat:03d}"); folder.mkdir(parents=True,exist_ok=True)
    trajectory_path=folder/"trajectory.csv.gz"; summary_path=folder/"summary.json"; code_hash=implementation_signature(root)
    if reuse and summary_path.exists() and trajectory_path.exists():
        old=json.loads(summary_path.read_text())
        if (old["plan_hash"]==plan_hash and old["code_hash"]==code_hash and old["status"]=="complete"
                and old.get("trajectory_sha256")==digest_bytes(trajectory_path.read_bytes())): return old
    Kernel,standard_chemistry,Individual,PopulationState,sweep=load_vendor(root)
    seed=int(np.random.SeedSequence([cfg["base_seed"],case_index,repeat]).generate_state(1,dtype=np.uint64)[0])
    rng=np.random.default_rng(seed); kernel=Kernel(standard_chemistry(),sigma); n=cfg["initial_population_per_scale"]*scale
    g,p=tuple(cfg["template"]),tuple(cfg["initial_reader"])
    state=PopulationState([Individual(i,0,g,p,"identity",0) for i in range(n)],cfg["initial_resource_per_scale"]*scale,next_id=n)
    settings=dict(J=cfg["resource_supply_per_scale"]*scale,Smax=cfg["resource_capacity_per_scale"]*scale,
                  d=cfg["death"],contact_mode="single_a",parent_copy=False)
    started=time.monotonic(); rows=[]; vectors=[]
    with gzip.open(trajectory_path.with_suffix(".gz.tmp"),"wt",newline="",encoding="utf8") as stream:
        writer=None
        for _ in range(cfg["sweeps"]):
            raw=sweep(state,kernel,rng,settings); row,vec=row_metrics(raw); rows.append(row); vectors.append(vec)
            saved=dict(row,**{"p"+key.replace(",",""):int(vec[i]) for i,key in enumerate(STATE_KEYS)})
            if writer is None: writer=csv.DictWriter(stream,fieldnames=list(saved)); writer.writeheader()
            writer.writerow(saved)
    os.replace(trajectory_path.with_suffix(".gz.tmp"),trajectory_path)
    metadata=dict(case=case_name,sigma=sigma,scale=scale,repeat_id=repeat,seed=seed,
                  analysis="fresh_finite_resource_confirmation",death=cfg["death"])
    windows=[dict(metadata,**summarize_window(rows,vectors,lo,hi)) for lo,hi in WINDOWS]
    repertoire_loss=next((r for r in rows if r["two_value_count"]==0),None)
    if repertoire_loss is not None and any(r["two_value_count"] for r in rows if r["sweep"]>=repertoire_loss["sweep"]):
        raise AssertionError("A lost output value repertoire was recreated without an innovation rule")
    result=dict(metadata,plan_hash=plan_hash,code_hash=code_hash,status="complete",sweeps=state.sweep,
                elapsed_seconds=time.monotonic()-started,final_N=len(state.individuals),final_S=state.resource,
                extinct=not bool(state.individuals),extinction_time=state.extinction_time,cumulative=state.cumulative,windows=windows,
                first_repertoire_loss_time=repertoire_loss["sweep"] if repertoire_loss is not None else None,
                population_at_repertoire_loss=repertoire_loss["N"] if repertoire_loss is not None else None,
                trajectory_sha256=digest_bytes(trajectory_path.read_bytes()))
    write_json(summary_path,result); return result

def run_finite(root,output,plan,plan_hash,workers=2,resume=True):
    cfg=plan["finite_resource"]
    tasks=[(str(root),str(output),plan,plan_hash,i,r,resume) for i in range(len(cfg["cases"])) for r in range(cfg["replicates"])]
    if workers==1: results=[finite_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool: results=list(pool.map(finite_task,tasks))
    all_windows=[w for result in results for w in result["windows"]]
    predictions={float(c["sigma"]):dominant_composition(transition_matrix(c["sigma"]))[1] for c in cfg["cases"]}
    summaries,distributions=summarize_groups(all_windows,("case","sigma","scale"),predictions)
    write_rows(output/"finite_run_windows.csv",all_windows); write_rows(output/"finite_group_windows.csv",summaries)
    write_rows(output/"finite_reader_distributions.csv",distributions)
    write_rows(output/"finite_run_endpoints.csv",[{k:v for k,v in r.items() if k not in ("windows","cumulative")}|r["cumulative"] for r in results])
    loss_rows=[]; loss_groups=[]
    for result in results:
        first_loss=result["first_repertoire_loss_time"]; extinction=result["extinction_time"]
        loss_rows.append({k:result[k] for k in ("case","sigma","scale","repeat_id","seed","first_repertoire_loss_time","population_at_repertoire_loss","extinction_time")}|
                         dict(analysis="posthoc_fresh_trajectory_diagnostic",repertoire_loss_right_censored=first_loss is None,
                              extinction_delay_after_repertoire_loss=extinction-first_loss if first_loss is not None and extinction is not None else None))
    for case in dict.fromkeys(r["case"] for r in results):
        selected=[r for r in results if r["case"]==case]; loss_count=sum(r["first_repertoire_loss_time"] is not None for r in selected)
        ext_count=sum(r["extinct"] for r in selected); lo,hi=wilson(ext_count,len(selected)); sigma=selected[0]["sigma"]
        loss_groups.append(dict(case=case,sigma=sigma,scale=selected[0]["scale"],n=len(selected),repertoire_losses=loss_count,
                               extinct_replicates=ext_count,extinction_frequency=ext_count/len(selected),
                               extinction_wilson95_low=lo,extinction_wilson95_high=hi,
                               post_loss_unlimited_spectral_factor=(1-cfg["death"])*(1+math.exp(-1/(sigma*sigma)))))
    write_rows(output/"finite_repertoire_loss.csv",loss_rows); write_rows(output/"finite_repertoire_loss_groups.csv",loss_groups)
    return dict(runs=len(results),all_complete=all(r["status"]=="complete" for r in results),
                extinct_runs=sum(r["extinct"] for r in results),workers=workers,replicates_per_condition=cfg["replicates"])

def run(root=ROOT,evidence=None,workers=2,resume=True,stages=("original","unlimited","finite")):
    root=Path(root); plan_raw=(root/"config/population_plan.json").read_bytes(); plan,plan_hash=json.loads(plan_raw),digest_bytes(plan_raw)
    output=root/"results/population_extension"; output.mkdir(parents=True,exist_ok=True)
    evidence=Path(evidence) if evidence else root/"results/original_population"
    if (output/"plan_executed.json").exists():
        previous=json.loads((output/"plan_executed.json").read_text())
        if previous["plan_hash"]!=plan_hash: raise ValueError("Plan changed: use a fresh output directory")
    write_json(output/"plan_executed.json",dict(plan_hash=plan_hash,plan=plan,status="internal_plan_before_new_runs_not_external_preregistration"))
    start=time.monotonic()
    metadata=dict(plan_hash=plan_hash,code_hash=implementation_signature(root),python=sys.version,numpy=np.__version__,
                  platform=platform.platform(),supplied_evidence=str(evidence),random_generator="numpy.random.Generator/PCG64",stages=list(stages))
    if "original" in stages:
        metadata["original"]=reanalyse_original(evidence,output,plan)
        if evidence.is_dir(): metadata["input_manifest_sha256"]=digest_bytes((evidence/"manifest.json").read_bytes())
        else: metadata["evidence_sha256"]=digest_bytes(evidence.read_bytes())
    if "unlimited" in stages: metadata["unlimited"]=run_unlimited(output,plan,plan_hash)
    if "finite" in stages: metadata["finite"]=run_finite(root,output,plan,plan_hash,workers,resume)
    metadata["elapsed_seconds"]=time.monotonic()-start; metadata["status"]="complete"
    write_json(output/"execution.json",metadata); return metadata

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--evidence",type=Path,help="Optional generated population directory or historical evidence ZIP; defaults to results/original_population")
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    parser.add_argument("--no-resume",action="store_true")
    parser.add_argument("--stages",nargs="+",choices=("original","unlimited","finite"),default=("original","unlimited","finite"))
    args=parser.parse_args()
    print(json.dumps(run(args.root,args.evidence,args.workers,not args.no_resume,tuple(args.stages)),indent=2))
