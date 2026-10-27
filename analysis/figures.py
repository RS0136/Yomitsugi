"""Publication figures from recorded outputs. Run: python -m analysis.figures --root .

Intervals use independent replicates. Exact zero probabilities are never
replaced with positive plotting constants. Each plot is exported as PDF, SVG,
and PNG; a source-hash manifest records which analysis tables were plotted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "yomitsugi_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from .finite import lifetime_closed, spectral_quantities

BLUE="#0072B2"; ORANGE="#D55E00"; GREEN="#009E73"; PURPLE="#CC79A7"
INK="#202A34"; GRAY="#6A737D"; PALE="#F1F5F8"
PALETTE=[INK,BLUE,GREEN,ORANGE,PURPLE]
CASES=[(.4,1),(.5,1),(.5,4),(.6,1),(.8,1),(.8,4)]

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.labelsize":10.5,"axes.titlesize":11,"xtick.labelsize":9,
        "ytick.labelsize":9,"legend.fontsize":9,"legend.frameon":False,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.edgecolor":"#737C84","axes.labelcolor":INK,"text.color":INK,
        "xtick.color":INK,"ytick.color":INK,"axes.linewidth":.7,
        "lines.linewidth":1.8,"pdf.fonttype":42,"ps.fonttype":42,
        "svg.fonttype":"none","savefig.facecolor":"white",
        "figure.facecolor":"white","mathtext.default":"regular"})

def panel(ax,letter,title):
    ax.set_title(title,loc="left",pad=10)
    ax.text(-.13,1.04,letter,transform=ax.transAxes,fontweight="bold",fontsize=12,va="bottom",ha="left")

def grid(ax,axis="both"):
    ax.grid(True,which="major",axis=axis,linewidth=.55,color="#DBE0E5",zorder=0)
    ax.set_axisbelow(True)

def sigma_axis(ax):
    ax.set_xscale("log");ax.set_xlim(.15,8)
    ax.xaxis.set_major_locator(FixedLocator([.2,.5,1,2,4,8]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:g}"))
    ax.tick_params(axis="x",which="minor",bottom=False)
    ax.set_xlabel(r"Common width $\sigma$ ($\sigma_R=\sigma_D$)")

def save(fig,out,name,sources,registry):
    for ext in ("pdf","svg","png"):
        kw={"bbox_inches":"tight","pad_inches":.07}
        if ext=="png":kw["dpi"]=220
        if ext=="pdf":kw["metadata"]={"Creator":"Yomitsugi reproducible figure script"}
        with tempfile.NamedTemporaryFile(dir=out, suffix="."+ext, delete=False) as stream:
            temporary=Path(stream.name)
            fig.savefig(stream,format=ext,**kw)
            stream.flush();os.fsync(stream.fileno())
        payload=temporary.read_bytes()
        if not payload or (ext=="pdf" and not payload.rstrip().endswith(b"%%EOF")):
            raise IOError("Incomplete figure export: "+name+"."+ext)
        os.replace(temporary,out/f"{name}.{ext}")
    registry[name]={"sources":sources,"artifacts":[f"figures/{name}.{e}" for e in ("pdf","svg","png")],
                    "artifact_sha256":{f"figures/{name}.{e}":hashlib.sha256((out/f"{name}.{e}").read_bytes()).hexdigest() for e in ("pdf","svg","png")}}
    plt.close(fig)

def rounded(ax,xy,width,height,color="white",edge="#C5CDD4"):
    ax.add_patch(FancyBboxPatch(xy,width,height,boxstyle="round,pad=0.008,rounding_size=0.018",facecolor=color,edgecolor=edge,linewidth=1))

def arrow(ax,start,end,color=GRAY):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=13,linewidth=1.4,color=color))

def tokens(ax,values,x,y,width=.043,height=.095):
    for j,value in enumerate(values):
        color=BLUE if value==0 else ORANGE
        ax.add_patch(Rectangle((x+j*width,y),width-.003,height,facecolor=color,edgecolor="white",linewidth=.8))
        ax.text(x+(j+.5)*width-.0015,y+height/2,str(value),color="white",va="center",ha="center",fontsize=11,fontweight="bold")

def fig_model(root,out,registry):
    fig=plt.figure(figsize=(8.4,4.7));gs=fig.add_gridspec(2,1,height_ratios=[2.6,1],hspace=.12)
    ax=fig.add_subplot(gs[0]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    ax.text(0,.98,"A",fontweight="bold",fontsize=12,va="top")
    ax.text(.04,.98,"A reader reconstructs a reader from a fixed description",fontsize=12,va="top")
    rounded(ax,(.035,.50),.24,.33,PALE)
    ax.text(.155,.78,r"Reader $p=(0,0\mid1,1)$",ha="center",fontsize=10.5)
    ax.text(.07,.675,"key",fontsize=9.5);ax.text(.07,.565,"value",fontsize=9.5)
    tokens(ax,[0,1],.16,.635,.046,.085);tokens(ax,[0,1],.16,.525,.046,.085)
    ax.text(.155,.405,r"Target description $g$",ha="center",fontsize=10.5)
    tokens(ax,[0,0,1,1],.07,.27,.045,.10)
    rounded(ax,(.38,.45),.28,.34,"#F4F8FB",BLUE)
    ax.text(.52,.715,"Local reading kernel",ha="center",fontsize=11,fontweight="bold")
    ax.text(.52,.625,r"Word success: $C(p,g)$",ha="center",fontsize=10)
    ax.text(.52,.54,r"Output: $Q_p(a\mid c)$",ha="center",fontsize=10)
    ax.text(.52,.385,r"Widths $\sigma_R$ and $\sigma_D$",ha="center",fontsize=10)
    arrow(ax,(.28,.67),(.375,.67),BLUE);arrow(ax,(.255,.315),(.38,.51))
    rounded(ax,(.77,.58),.205,.22,"#EDF7F2",GREEN)
    ax.text(.8725,.725,r"Child $(g,p')$",ha="center",fontsize=11,fontweight="bold")
    ax.text(.8725,.635,r"New reader $p'$",ha="center",fontsize=10)
    arrow(ax,(.665,.67),(.765,.67),GREEN);ax.text(.713,.74,r"$C(p,g)$",ha="center",fontsize=9)
    rounded(ax,(.785,.27),.175,.12,"#F8F1F0",ORANGE)
    ax.text(.8725,.33,"Failed attempt",ha="center",va="center",fontsize=10)
    arrow(ax,(.665,.49),(.78,.33),ORANGE);ax.text(.735,.425,r"$1-C(p,g)$",ha="center",fontsize=9)
    arrow(ax,(.245,.235),(.87,.235));ax.text(.56,.155,r"The target description $g$ is copied unchanged",ha="center",fontsize=10)
    ax.text(.50,.035,"Reading constructs the next reader, which reads the unchanged description.",ha="center",fontsize=9.5,color=GRAY)
    ax=fig.add_subplot(gs[1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    ax.text(0,1,"B",fontweight="bold",fontsize=12,va="top")
    boxes=[(.04,BLUE,"Encounter",r"$C(p,g)=P(\mathrm{output})$","One attempted\nreconstruction"),
           (.365,ORANGE,"Serial lineage",r"$S_h$, $P(\mathrm{native}\mid\mathrm{live})$","One descendant\nper step"),
           (.69,GREEN,"Population",r"$E[N_t]$, $P(N_t=0)$","Parents retained\nbetween attempts")]
    for x,color,title,formula,subtitle in boxes:
        rounded(ax,(x,.04),.275,.83,"white",color)
        ax.text(x+.1375,.68,title,ha="center",fontsize=11,fontweight="bold",color=color)
        ax.text(x+.1375,.425,formula,ha="center",fontsize=10.5)
        ax.text(x+.1375,.14,subtitle,ha="center",fontsize=8.8,linespacing=1.15)
    save(fig,out,"fig1_model",["analysis/finite.py","analysis/population_extension.py"],registry)

def fig_serial(root,out,registry):
    source="results/finite/sigma_scan.csv";df=pd.read_csv(root/source)
    df=df[df.sigma_r>0].sort_values("sigma_r");s=df.sigma_r.to_numpy()
    crossing=json.loads((root/"results/finite/summary.json").read_text())["diagonal_crossover_sigma"]
    opt=minimize_scalar(lifetime_closed,bounds=(.4,1.2),method="bounded",options={"xatol":1e-12})
    fig,axes=plt.subplots(2,2,figsize=(8.4,6.4),layout="constrained")
    ax=axes[0,0];panel(ax,"A","Continuation and native reconstruction")
    ax.text(.98,.97,"First-attempt success = 1",transform=ax.transAxes,ha="right",va="top",fontsize=8.5,color=GRAY)
    ax.plot(s,df.S_h,color=BLUE,label=r"Survival $S_{32}$")
    ax.plot(s,df.native_joint,color=ORANGE,ls="--",label=r"Native $R_{32}$")
    ax.set_yscale("log");ax.set_ylim(1e-15,2);ax.set_ylabel("Probability");ax.legend(loc="lower left",fontsize=8.8)
    ax=axes[0,1];panel(ax,"B","Composition of surviving lineages")
    ax.plot(s,df.native_conditional,color=ORANGE,label="Native reader")
    ax.plot(s,df.homogeneous_conditional,color=GREEN,ls="--",label="Homogeneous reader")
    ax.set_ylim(-.03,1.05);ax.set_ylabel(r"Conditional probability at $h=32$");ax.legend(loc="center right",fontsize=9)
    u=df.u.to_numpy();a=df.a.to_numpy() if 'a' in df else np.array([2*np.exp(-1/(2*x*x))/(1+np.exp(-1/(2*x*x)))**2 for x in s])
    retained=(4-u+4*a-4*a*a)/(a*(8-4*a-3*u+2*a*u))
    depleted=1/(1-u)
    ax=axes[1,0];panel(ax,"C","Where the serial lifetime is spent")
    ax.plot(s,df.expected_trials,color=INK,label=r"All attempts: $E[\tau]$")
    ax.plot(s,retained,color=BLUE,ls="--",label=r"Two-value classes: $E[T_{\mathcal{H}}]$")
    ax.plot(s,depleted,color=ORANGE,ls=":",label=r"Depleted classes: $1/(1-u)$")
    ax.scatter([opt.x],[opt.fun],color=PURPLE,s=30,zorder=6)
    ax.set_yscale("log");ax.set_ylim(.8,3e9);ax.set_ylabel("Expected attempts");ax.legend(loc="upper right",fontsize=8.3)
    ax=axes[1,1];panel(ax,"D","Repertoire and demographic thresholds")
    lam=df.lambda_repertoire.to_numpy()
    ax.plot(s,lam/(1+lam),color=BLUE,label=r"Two-value threshold $d_H$")
    ax.plot(s,u/(1+u),color=GREEN,ls="--",label=r"Depleted threshold $d_D$")
    ax.set_ylim(-.0125,.525);ax.set_ylabel("Critical death probability");ax.legend(loc="lower right",fontsize=8.5)
    critical_u=math.exp(-1/crossing**2)
    ax.annotate(f"$\\sigma_c={crossing:.4f}$",xy=(crossing,critical_u/(1+critical_u)),xytext=(.34,.40),fontsize=9,
                bbox={"facecolor":"white","edgecolor":"none","alpha":.95,"pad":2},
                arrowprops={"arrowstyle":"-","lw":.8,"color":GRAY})
    for ax in axes.flat:
        sigma_axis(ax);grid(ax);ax.axvline(crossing,color=GRAY,lw=.9,ls=(0,(3,3)),alpha=.7,zorder=0)
    save(fig,out,"fig2_serial",[source,"results/finite/summary.json","analysis/finite.py"],registry)

def ci_error(ax,x,mean,lower,upper,color,marker="o",label=None,**kwargs):
    mean=np.asarray(mean,dtype=float);lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float)
    return ax.errorbar(x,mean,yerr=np.vstack([np.maximum(0,mean-lower),np.maximum(0,upper-mean)]),fmt=marker,
                      ms=4.5,color=color,capsize=3,elinewidth=1.15,label=label,zorder=4,**kwargs)

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,mid-half),min(1,mid+half)

def finite_order(df):
    return pd.concat([df[np.isclose(df.sigma,s)&(df.scale==scale)] for s,scale in CASES],ignore_index=True)

def case_labels():return [f"{s:g}\n×{scale}" for s,scale in CASES]

def bootstrap_groups(root, groups, cohort):
    """Replace composition intervals with independent-run percentile intervals."""
    boot=pd.read_csv(root/'results/editorial_summaries/composition_bootstrap.csv')
    result=groups.copy()
    for index,row in result.iterrows():
        selected=boot[(boot.cohort==cohort)&np.isclose(boot.sigma,row.sigma)&
                      (boot.scale==int(row.get('scale',1)))&(boot.window_start==row.window_start)]
        for metric in ['reference_fraction_zero_filled','reference_fraction_alive']:
            value=selected[selected.metric==metric].iloc[0]
            result.loc[index,metric+'_ci95_low']=value.bootstrap95_low
            result.loc[index,metric+'_ci95_high']=value.bootstrap95_high
    return result

def fig_population(root,out,registry):
    base=root/"results/population_extension";readers=pd.read_csv(base/"original_reader_distributions.csv")
    readers=readers[np.isclose(readers.sigma,.5)&(readers["mode"]=="single_a")&(readers.window_start==501)].sort_values("reader")
    assert len(readers)==16 and np.isclose(readers["mean"].sum(),1)
    original=bootstrap_groups(root,pd.read_csv(base/"original_group_windows.csv").query("mode == 'single_a'"),'cohort')
    original=original[np.isclose(original.sigma,.5)&(original["mode"]=="single_a")&(original.window_start==501)].iloc[0]
    groups=bootstrap_groups(root,pd.read_csv(base/"finite_group_windows.csv"),'validation');groups=finite_order(groups[groups.window_start==501])
    endpoints=pd.read_csv(base/"finite_run_endpoints.csv")
    fig=plt.figure(figsize=(8.5,7.2),layout="constrained");gs=fig.add_gridspec(2,2,height_ratios=[1.05,1])
    ax=fig.add_subplot(gs[0,:]);panel(ax,"A",r"Finite-resource cohort: all 16 readers at $\sigma=0.5$")
    x=np.arange(16);native=readers.reader.isin(["0,0,1,1","1,1,0,0"])
    ax.bar(x,100*readers["mean"],width=.68,color=[BLUE if flag else "#A4BFCE" for flag in native],zorder=2)
    ci_error(ax,x,100*readers["mean"],100*readers.ci95_low,100*readers.ci95_high,INK,"none","Cohort mean ± 95% CI")
    ax.scatter(x,100*readers.prediction,marker="D",s=19,color=ORANGE,label="Fluid fixed point",zorder=5)
    ax.set_xticks(x,[r.replace(",","") for r in readers.reader],rotation=45,ha="right",fontfamily="DejaVu Sans Mono",fontsize=8.8)
    ax.set_ylabel("Reader fraction (%)");ax.set_xlabel("Reader word (alternating key and value)");ax.set_ylim(0,50);grid(ax,"y")
    ax.legend(loc="upper right",fontsize=9);tv=.5*np.abs(readers["mean"].to_numpy()-readers.prediction.to_numpy()).sum()
    ax.text(.985,.62,"Native fraction = "+f"{100*original.reference_fraction_zero_filled_mean:.2f}%\n"
            +f"95% CI [{100*original.reference_fraction_zero_filled_ci95_low:.2f}, {100*original.reference_fraction_zero_filled_ci95_high:.2f}]%\n"
            +f"Full-distribution TV = {tv:.5f}",transform=ax.transAxes,ha="right",va="top",fontsize=9.2,
            bbox={"facecolor":"white","edgecolor":"none","alpha":.92,"pad":4})
    ax=fig.add_subplot(gs[1,0]);panel(ax,"B","Finite-resource validation")
    x=np.arange(6);pred=groups.reference_prediction.to_numpy()
    for shift,prefix,color,marker,label in [(-.10,"reference_fraction_zero_filled",BLUE,"o","Extinction assigned zero"),
                                           (.10,"reference_fraction_alive",ORANGE,"s","Occupied times only")]:
        ci_error(ax,x+shift,100*(groups[prefix+"_mean"]-pred),100*(groups[prefix+"_ci95_low"]-pred),
                 100*(groups[prefix+"_ci95_high"]-pred),color,marker,label)
    ax.axhline(0,color=INK,lw=1,ls="--");ax.set_xticks(x,case_labels());ax.set_xlabel(r"Width $\sigma$ / resource scale")
    ax.set_ylabel("Native fraction − fluid fixed point\n(percentage points)");ax.set_ylim(-6.5,1.9);grid(ax,"y");ax.legend(loc="lower left",fontsize=8.6)
    ax=fig.add_subplot(gs[1,1]);panel(ax,"C","Extinction by sweep 1000")
    ext=[];ns=[];lo=[];hi=[]
    for s,scale in CASES:
        e=endpoints[np.isclose(endpoints.sigma,s)&(endpoints.scale==scale)]
        k=int(e.extinct.astype(bool).sum());n=len(e);l,u=wilson(k,n)
        ext.append(k/n);ns.append((k,n));lo.append(l);hi.append(u)
    ci_error(ax,x,100*np.array(ext),100*np.array(lo),100*np.array(hi),PURPLE,"o","Endpoint frequency ± Wilson 95% CI")
    for i,(k,n) in enumerate(ns):ax.text(i,100*hi[i]+3,f"{k}/{n}",ha="center",fontsize=8.8)
    ax.set_xticks(x,case_labels());ax.set_xlabel(r"Width $\sigma$ / resource scale");ax.set_ylabel("Extinct replicates (%)")
    ax.set_ylim(-3,87);grid(ax,"y");ax.legend(loc="upper left",fontsize=8.4)
    save(fig,out,"fig3_population",["results/population_extension/"+s for s in
         ["original_reader_distributions.csv","original_group_windows.csv","finite_group_windows.csv","finite_run_endpoints.csv"]]
         +['results/editorial_summaries/composition_bootstrap.csv'],registry)

def fig_branching(root,out,registry):
    source="results/population_extension/unlimited_expectations.csv";df=pd.read_csv(root/source)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.8),layout="constrained")
    panel(axes[0],"A","Unconditional population size");panel(axes[1],"B","Finite-time extinction probability")
    for case,color in {"d030":BLUE,"dcritical":GREEN,"d045":ORANGE}.items():
        d=df[df["case"]==case].sort_values("time");end=d.iloc[-1]
        label=(r"$d=d_c\approx0.38475$" if case=="dcritical" else f"$d={end.death:.2f}$")
        t=d.time.to_numpy();sel=d[d.time%5==0]
        axes[0].plot(t,d.expected_N,color=color,label=label)
        ci_error(axes[0],sel.time,sel.N_mean,sel.N_ci95_low,sel.N_ci95_high,color,"o")
        axes[1].plot(t,d.exact_extinction_probability,color=color,label=label)
        offset={"d030":-.16,"dcritical":0.,"d045":.16}[case]
        ci_error(axes[1],sel.time+offset,sel.extinction_frequency,sel.extinction_wilson95_low,sel.extinction_wilson95_high,color,"o")
    axes[0].set_yscale("log");axes[0].set_ylim(5,5000);axes[0].set_ylabel(r"$E[N_t]$ and replicate mean")
    axes[1].set_ylim(-.007,.215);axes[1].set_ylabel(r"$P(N_t=0)$ and replicate frequency")
    for ax in axes:
        ax.set_xlim(-.7,25.7);ax.set_xticks([0,5,10,15,20,25]);ax.set_xlabel("Population sweep t");grid(ax)
    axes[0].legend(loc="upper left",fontsize=8.8)
    axes[1].text(.035,.96,"Lines: exact expectation / PGF\nPoints: 2,000 replicate means / frequencies\nBars: 95% intervals",transform=axes[1].transAxes,va="top",fontsize=8.6)
    endpoint=df[df["case"]=="d045"].sort_values("time").iloc[-1]
    axes[1].text(.035,.62,f"$d={endpoint.death:.2f}$: {int(endpoint.extinct_replicates):,} / {int(endpoint.replicates):,} extinct\n"
                +f"Exact $P(N_{{{int(endpoint.time)}}}=0)={endpoint.exact_extinction_probability:.5f}$",
                transform=axes[1].transAxes,ha="left",va="top",fontsize=8.7)
    save(fig,out,"fig4_branching",[source,"results/population_extension/plan_executed.json"],registry)

def edges(values):
    values=np.asarray(values);mid=np.sqrt(values[:-1]*values[1:])
    return np.r_[values[0]**2/mid[0],mid,values[-1]**2/mid[-1]]

def fig_two_widths(root,out,registry):
    source="results/finite/width_grid.csv";df=pd.read_csv(root/source)
    df["information_conditional"]=df.decoding_information_joint_bits/df.S_h
    r=np.sort(df.sigma_r.unique());d=np.sort(df.sigma_d.unique())
    fig,axes=plt.subplots(1,3,figsize=(9.,3.8),layout="constrained")
    plots=[("S_h",r"A  Survival $S_{32}$",1e-19,[1e-18,1e-12,1e-6,1],"Probability"),
           ("native_conditional","B  Native given survival",1e-10,[1e-9,1e-6,1e-3,1],"Conditional probability"),
           ("information_conditional","C  Decoding information",1e-14,[1e-12,1e-8,1e-4,1],"Conditional mean (bits)")]
    dd=np.geomspace(.2,4,500);td=np.exp(-1/(2*dd**2));a=2*td/(1+td)**2
    uc=(1-a)*(3-2*a)/3;rr=1/np.sqrt(-np.log(uc))
    for ax,(col,title,vmin,ticks,label) in zip(axes,plots):
        z=df.pivot(index="sigma_r",columns="sigma_d",values=col).reindex(index=r,columns=d).to_numpy()
        if np.any(z<=0):raise ValueError(f"Nonpositive value in {col}; no zero clipping permitted")
        pcm=ax.pcolormesh(edges(d),edges(r),z,norm=LogNorm(vmin=vmin,vmax=1),cmap="cividis",shading="flat",rasterized=False)
        ax.plot(dd,dd,color="white",lw=1.3,ls=(0,(4,3)))
        valid=(rr>=r.min())&(rr<=r.max());ax.plot(dd[valid],rr[valid],color="#F2735C",lw=1.7)
        ax.set(xscale="log",yscale="log",xlim=(.2,4),ylim=(.2,4));ax.set_aspect("equal")
        tick=[.2,.5,1,2,4];ax.set_xticks(tick,[f"{v:g}" for v in tick]);ax.set_yticks(tick,[f"{v:g}" for v in tick])
        ax.tick_params(which="minor",bottom=False,left=False)
        ax.set_xlabel(r"Decoding width $\sigma_D$");ax.set_title(title,loc="left",fontsize=10.5,pad=8)
        cb=fig.colorbar(pcm,ax=ax,orientation="horizontal",pad=.12,fraction=.065,ticks=ticks)
        cb.set_label(label,fontsize=9);cb.ax.tick_params(labelsize=8)
        if cb.solids is not None:cb.solids.set_rasterized(False)
    axes[0].set_ylabel(r"Recognition width $\sigma_R$")
    for ax in axes[1:]:ax.set_yticklabels([])
    handles=[Line2D([0],[0],color=GRAY,lw=1.3,ls=(0,(4,3)),label=r"One-width model: $\sigma_R=\sigma_D$"),
             Line2D([0],[0],color="#F2735C",lw=1.7,label=r"Decay-mode crossover: $3u=(1-a)(3-2a)$")]
    fig.legend(handles=handles,loc="outside upper center",ncol=2,fontsize=9)
    save(fig,out,"fig5_two_widths",[source,"analysis/finite.py"],registry)

def fig_q3(root,out,registry):
    source="results/finite/q3_extension.csv";df=pd.read_csv(root/source)
    fig,axes=plt.subplots(2,2,figsize=(8.4,5.9),layout="constrained")
    for col,start in enumerate(["native","non_native_cycle"]):
        a=df[df.start==start]
        for sigma,color in zip([0,.2,.5,1,2],PALETTE):
            d=a[np.isclose(a.sigma,sigma)].sort_values("h")
            axes[0,col].plot(d.h,d.S_h,"o-",color=color,ms=3.5,label=f"$\\sigma={sigma:g}$")
            axes[1,col].plot(d.h,d.native_conditional,"o-",color=color,ms=3.5)
        axes[0,col].set_yscale("log");axes[0,col].set_ylim(1e-23,3)
        axes[1,col].set_ylim(-.035,1.06)
        for row in [0,1]:
            axes[row,col].set_xscale("log",base=2)
            axes[row,col].set_xticks([1,2,4,8,16,32],["1","2","4","8","16","32"])
            axes[row,col].set_xlabel("Successful serial reads h");grid(axes[row,col])
    panel(axes[0,0],"A","Native transposition start");panel(axes[0,1],"B","Non-native cycle start")
    panel(axes[1,0],"C","Native identity among survivors");panel(axes[1,1],"D","Native identity among survivors")
    axes[0,0].set_ylabel(r"Survival $S_h$");axes[1,0].set_ylabel("Conditional native probability")
    axes[0,0].legend(loc="lower left",fontsize=8.6,ncol=2)
    axes[1,1].text(.97,.92,r"At $\sigma=0$: survival = 1"+"\nNative probability = 0",transform=axes[1,1].transAxes,ha="right",va="top",fontsize=9)
    save(fig,out,"figS1_q3",[source],registry)

def fig_sensitivity(root,out,registry):
    base=root/"results/population_extension"
    original=bootstrap_groups(root,pd.read_csv(base/"original_group_windows.csv").query("mode == 'single_a'"),'cohort')
    finite=bootstrap_groups(root,pd.read_csv(base/"finite_group_windows.csv"),'validation');ret=pd.read_csv(base/"original_lineage_retention.csv")
    fig,axes=plt.subplots(2,1,figsize=(8.4,6.3),layout="constrained")
    ax=axes[0];panel(ax,"A","Sensitivity to the averaging window (extinction assigned zero)")
    labels=["Cohort\n0.5 / ×1"]+[f"Validation\n{s:g} / ×{scale}" for s,scale in CASES]
    for start,color,shift,marker in [(251,BLUE,-.18,"o"),(501,ORANGE,0,"s"),(751,GREEN,.18,"^")]:
        o=original[np.isclose(original.sigma,.5)&(original["mode"]=="single_a")&(original.window_start==start)].iloc[0]
        f=finite_order(finite[finite.window_start==start]);p=spectral_quantities(.5)["native_spectral_fraction"]
        mean=np.r_[o.reference_fraction_zero_filled_mean,f.reference_fraction_zero_filled_mean]
        lower=np.r_[o.reference_fraction_zero_filled_ci95_low,f.reference_fraction_zero_filled_ci95_low]
        upper=np.r_[o.reference_fraction_zero_filled_ci95_high,f.reference_fraction_zero_filled_ci95_high]
        pred=np.r_[p,f.reference_prediction]
        ci_error(ax,np.arange(7)+shift,100*(mean-pred),100*(lower-pred),100*(upper-pred),color,marker,f"Sweeps {start}–1000")
    ax.axhline(0,color=INK,lw=1,ls="--");ax.set_xticks(np.arange(7),labels,fontsize=8.8)
    ax.set_ylabel("Native fraction − fluid fixed point (pp)");ax.set_ylim(-7,2);grid(ax,"y")
    ax.legend(loc="lower left",ncol=3,fontsize=8.8)
    ax=axes[1];panel(ax,"B","Two-origin retention at sweep 1000")
    modes=["mixed","pooled_shared","pooled_separate"]
    for sigma,color,shift,marker in [(0,BLUE,-.10,"o"),(.5,ORANGE,.10,"s")]:
        d=pd.concat([ret[np.isclose(ret.sigma,sigma)&(ret["mode"]==mode)] for mode in modes]);xx=np.arange(3)+shift
        ci_error(ax,xx,d.fraction,d.wilson95_low,d.wilson95_high,color,marker,f"$\\sigma={sigma:g}$")
        for x,(_,row) in zip(xx,d.iterrows()):
            yy=row.fraction-.12 if row.fraction>.9 else row.wilson95_high+.06
            ax.text(x,yy,f"{int(row.both_origins)}/{int(row.n)}",ha="center",fontsize=9,color=color)
    ax.set_xticks(np.arange(3),["Mixed population","Pooled, shared resources","Pooled, separate resources"])
    ax.set_ylabel("Fraction retaining both origins");ax.set_ylim(-.04,1.09);ax.set_xlim(-.45,2.45)
    grid(ax,"y");ax.legend(loc="upper left",fontsize=9)
    save(fig,out,"figS2_sensitivity",["results/population_extension/"+s for s in
         ["original_group_windows.csv","finite_group_windows.csv","original_lineage_retention.csv"]]
         +['results/editorial_summaries/composition_bootstrap.csv'],registry)

def fig_crossing(root,out,registry):
    source="results/crossing/same_features.csv";df=pd.read_csv(root/source);order=["A0","A1","B"]
    fig,axes=plt.subplots(1,3,figsize=(8.4,3.4),layout="constrained")
    cmap=ListedColormap(["#E7EEF3",BLUE])
    for ax,(column,title) in zip(axes,[("C","A  Encounter success"),("native_first","B  Native first child"),("S2","C  Two-step continuation")]):
        mat=df.pivot(index="source",columns="target",values=column).reindex(index=order,columns=order).to_numpy()
        ax.pcolormesh(np.arange(4)-.5,np.arange(4)-.5,mat,vmin=0,vmax=1,cmap=cmap,
                      edgecolors="white",linewidth=1.2,rasterized=False,shading="flat")
        ax.set_xlim(-.5,2.5);ax.set_ylim(2.5,-.5);ax.set_aspect("equal")
        for i in range(3):
            for j in range(3):ax.text(j,i,str(int(mat[i,j])),ha="center",va="center",fontsize=16,color="white" if mat[i,j] else INK)
        ax.set_xticks(range(3),[r"$A_0$",r"$A_1$","B"]);ax.set_yticks(range(3),[r"$A_0$",r"$A_1$","B"])
        ax.set_xlabel("Target description");ax.set_title(title,loc="left",fontsize=10.5)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.set_xticks(np.arange(-.5,3,.5)[::2],minor=True)
        ax.set_yticks(np.arange(-.5,3,.5)[::2],minor=True)
        ax.grid(which="minor",color="white",linewidth=2);ax.tick_params(which="both",length=0)
    axes[0].set_ylabel("Actor reader")
    for ax in axes[1:]:ax.set_yticklabels([])
    for ax in axes:
        ax.add_patch(Rectangle((1.52,.52),.96,.96,fill=False,edgecolor=ORANGE,linewidth=2.8,zorder=5))
    fig.suptitle(r"Same standard features, $\sigma=0$: $A_1\!\to B$ succeeds once but not twice",fontsize=11)
    save(fig,out,"figS3_crossing",[source,"results/crossing/definition.json"],registry)

def run(root,only=None):
    root=Path(root).resolve();out=root/"figures";out.mkdir(parents=True,exist_ok=True);style();registry={}
    functions=[fig_model,fig_serial,fig_population,fig_branching,fig_two_widths,fig_q3,fig_sensitivity,fig_crossing]
    selected=range(1,9) if only is None else only
    for number in selected:functions[number-1](root,out,registry)
    for item in registry.values():
        item["source_sha256"]={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in item["sources"]}
        item["generator"]="analysis/figures.py"
        item["generator_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest=out/"figure_manifest.json";previous=json.loads(manifest.read_text()) if manifest.exists() else {}
    previous.update(registry);manifest.write_text(json.dumps(previous,indent=2)+"\n")
    print(json.dumps({"figures":list(registry),"formats":["pdf","svg","png"]},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--only",type=int,nargs="+",choices=range(1,9),help="Render selected numbers; 6–8 are supplementary")
    args=parser.parse_args();run(args.root,args.only)
