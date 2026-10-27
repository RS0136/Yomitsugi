"""Figures connecting coding cycles, population persistence, and innovation."""
from pathlib import Path
import hashlib,json
from .figures import (plt,np,pd,BLUE,ORANGE,GREEN,PURPLE,INK,GRAY,
                      style,panel,grid,save,rounded,arrow,tokens)

ROOT=Path(__file__).resolve().parents[1]


def cycles(root,out,registry):
    base='results/cycle_population/'
    rates=pd.read_csv(root/(base+'one_step_rates.csv'))
    passage=pd.read_csv(root/(base+'first_passage.csv'))
    time=pd.read_csv(root/(base+'population_timecourse.csv'))
    runs=pd.read_csv(root/(base+'population_replicates.csv'))
    fig,axes=plt.subplots(2,2,figsize=(8.8,7.1),layout='constrained')
    ax=axes[0,0];panel(ax,'A','One-birth changes from complete readers')
    configs=[('linear',.2),('linear',.3),('equidistant',.3)]
    measures=[('to_cycle',BLUE,'Native to cycle'),('to_native',ORANGE,'Cycle to native'),
              ('exit_complete',PURPLE,'Leave complete sector'),('lose_repertoire',GREEN,'Lose a value')]
    for k,(measure,color,label) in enumerate(measures):
        values=[]
        for metric,sigma in configs:
            d=rates[(rates.metric==metric)&np.isclose(rates.sigma,sigma)]
            if k==0:value=d[d.founder=='native'][measure].iloc[0]
            elif k==1:value=d[d.founder.str.startswith('cycle')][measure].mean()
            else:value=d[d.founder=='native'][measure].iloc[0]
            values.append(value)
        ax.bar(np.arange(3)+(k-1.5)*.19,values,.18,color=color,label=label,zorder=3)
    ax.set_yscale('log');ax.set_ylim(1e-24,1)
    ax.set_xticks(range(3),['Linear\n0.2','Linear\n0.3','Equal distance\n0.3'])
    ax.set_xlabel('Feature metric / width');ax.set_ylabel('Probability in one birth')
    ax.legend(loc='lower right',fontsize=7.3,frameon=True,facecolor='white',edgecolor='none',framealpha=.96);grid(ax,'y')
    ax=axes[0,1];panel(ax,'B','Recovery before repertoire loss or failure')
    for metric,color,marker,label in [('linear',BLUE,'o','Linear features'),('equidistant',ORANGE,'s','Equal distances')]:
        d=passage[(passage.metric==metric)&passage.start.str.startswith('cycle')]
        d=d.groupby('sigma',as_index=False).probability_before_loss_or_failure.mean()
        ax.plot(d.sigma,d.probability_before_loss_or_failure,marker=marker,color=color,label=label)
    ax.set_yscale('log');ax.set_xlabel('Common width');ax.set_ylabel('Native first-hit probability')
    ax.text(.04,.96,'All intermediate full-value readers included',transform=ax.transAxes,va='top',fontsize=8)
    ax.legend(loc='lower right',fontsize=8);grid(ax)
    ax=axes[1,0];panel(ax,'C','Alternative codes inside a population')
    d=time[time.case_id=='linear_s0.3_cycle']
    for col,color,ls,label in [('mean_population',GRAY,'-','All individuals'),
                              ('mean_full_repertoire',GREEN,'--','Full-value readers'),
                              ('mean_cycle',ORANGE,'-','Cycle-code readers'),
                              ('mean_native',BLUE,'-','Native-code readers')]:
        ax.plot(d.sweep,d[col],color=color,ls=ls,label=label)
    ax.set(xlim=(0,1000),ylim=(-3,140),xlabel='Population sweep',ylabel='Mean count across 20 paths')
    ax.legend(loc='center right',fontsize=8)
    ax.text(.03,.97,r'Cycle founders; linear $\sigma=0.3$',transform=ax.transAxes,va='top',fontsize=8.5)
    grid(ax)
    ax=axes[1,1];panel(ax,'D','Joint population and code persistence')
    for case,color,marker,label in [('linear_s0_cycle',GRAY,'o','Exact matching'),
                                   ('linear_s0.2_cycle',BLUE,'s','Linear, 0.2'),
                                   ('linear_s0.3_cycle',ORANGE,'^','Linear, 0.3'),
                                   ('equidistant_s0.3_cycle',PURPLE,'D','Equal, 0.3')]:
        d=runs[runs.case_id==case]
        ax.scatter(d.population1000,d.cycle_fraction1000,color=color,marker=marker,
                   s=26,alpha=.7,edgecolor='white',linewidth=.3,label=label,zorder=3)
    ax.set(xlabel='Population at sweep 1000',ylabel='Cycle-code fraction at sweep 1000',ylim=(.6,1.025))
    ax.legend(loc='lower right',fontsize=7.7);grid(ax)
    save(fig,out,'fig9_cycle_population',[base+n for n in ['one_step_rates.csv','first_passage.csv','population_timecourse.csv','population_replicates.csv']],registry)


def intervention(root,out,registry):
    base='results/intervention_revision/'
    dose=pd.read_csv(root/(base+'dose_response.csv'))
    contrast=pd.read_csv(root/(base+'resource_contrasts.csv'))
    curves=pd.read_csv(root/(base+'reference_first_hit_curves.csv'))
    exact=pd.read_csv(root/(base+'unrestricted_curves.csv'))
    fig,axes=plt.subplots(2,2,figsize=(8.8,7.2),layout='constrained')
    ax=axes[0,0];panel(ax,'A','Reorganize existing values')
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    for x,word,y in [(.06,[0,0,0,0],.63),(.06,[1,1,1,1],.42),
                     (.65,[0,0,0,1],.63),(.65,[1,1,1,0],.42)]:
        tokens(ax,word,x,y,.055,.12)
    arrow(ax,(.32,.59),(.61,.59),ORANGE)
    ax.text(.17,.84,'Single-value pair',ha='center',fontsize=10)
    ax.text(.77,.84,'Two-value pair',ha='center',fontsize=10)
    ax.text(.47,.68,'One value\nexchanged',ha='center',fontsize=9,color=ORANGE)
    ax.text(.5,.27,'Same keys, positional symbol totals, population,\nresource, and first-sweep demographic law',ha='center',va='top',fontsize=9)
    ax.text(.5,.08,'Sham: exchange both values; repertoire stays single-value',ha='center',fontsize=8.2,color=GRAY)
    ax=axes[0,1];panel(ax,'B','Dose controls later persistence')
    d=dose[dose.arm.isin(['unchanged','pairs1','pairs2','pairs5','pairs10','pairs20'])].sort_values('pairs')
    for prefix,color,marker,label in [('N500',BLUE,'o','Any population'),('H500',GREEN,'s','Two-value readers')]:
        y=d[prefix+'_fraction'].to_numpy();lo=d[prefix+'_hoeffding95_low'].to_numpy();hi=d[prefix+'_hoeffding95_high'].to_numpy()
        ax.errorbar(d.pairs,y,yerr=np.vstack([y-lo,hi-y]),color=color,marker=marker,ms=4,capsize=2,label=label)
    ax.plot(d.pairs,d.unrestricted_N500,color=GRAY,ls='--',label='Unrestricted: exact N survival')
    ax.set(xlim=(-.7,20.7),ylim=(-.03,1.04),xticks=[0,1,2,5,10,20],xlabel='Exchanged pairs',ylabel='Probability at sweep 500')
    ax.legend(loc='lower right',fontsize=7.5);grid(ax)
    ax=axes[1,0];panel(ax,'C','Reconstruction and retention (10 pairs)')
    d=curves[curves.arm=='pairs10'];e=exact[exact.arm=='pairs10']
    ax.plot(d.sweep,d.reference_ever,color=PURPLE,label='Native ever reconstructed')
    ax.plot(d.sweep,d.reference_present,color=BLUE,label='Native present at census')
    ax.plot(e.sweep,e.reference_ever,color=GRAY,ls='--',label='Unrestricted: exact first hit')
    ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,10,100,500],['0','1','10','100','500'])
    ax.set(xlim=(0,500),ylim=(-.03,1.03),xlabel='Population sweep after exchange',ylabel='Fraction of continuations')
    ax.legend(loc='lower right',fontsize=7.6);grid(ax)
    ax=axes[1,1];panel(ax,'D','Resource restriction affects maintenance')
    d=contrast[contrast.arm=='pairs10'].set_index('metric').loc[['N500','H500','reference_ever500']]
    y=100*d.unrestricted_minus_finite.to_numpy();lo=100*d.conditional_hoeffding95_low.to_numpy();hi=100*d.conditional_hoeffding95_high.to_numpy()
    colors=[BLUE,GREEN,PURPLE]
    ax.barh(range(3),y,color=colors,height=.55,zorder=2)
    ax.errorbar(y,range(3),xerr=np.vstack([y-lo,hi-y]),fmt='none',color=INK,capsize=3,zorder=3)
    ax.set_yticks(range(3),['Population\nat 500','Two-value readers\nat 500','Native first hit\nby 500'])
    ax.invert_yaxis();ax.axvline(0,color=INK,lw=.8)
    ax.set(xlim=(-7,39),xlabel='Unrestricted minus finite resource\n(percentage points)')
    grid(ax,'x')
    ax.text(.98,.03,'Same initial states and event horizons',transform=ax.transAxes,ha='right',fontsize=7.5)
    save(fig,out,'fig10_intervention',[base+n for n in ['dose_response.csv','resource_contrasts.csv','reference_first_hit_curves.csv','unrestricted_curves.csv']],registry)


def innovation(root,out,registry):
    base='results/intervention_revision/'
    data=pd.read_csv(root/(base+'mutation_thresholds.csv'))
    gaps=pd.read_csv(root/(base+'mutation_crossover.csv'))
    fig,axes=plt.subplots(2,2,figsize=(8.7,6.5),layout='constrained')
    choices=[(0.,GRAY,'0'),(1e-4,BLUE,r'$10^{-4}$'),(.01,ORANGE,r'$10^{-2}$')]
    ax=axes[0,0];panel(ax,'A','Leading class eigenvalues near the crossing')
    for mu,color,label in choices:
        d=data[np.isclose(data.mu,mu,rtol=1e-10,atol=1e-15)].sort_values('sigma')
        ax.plot(d.sigma,d.rho,color=color,label=r'$\mu=$'+label)
        ax.plot(d.sigma,d.second_eigenvalue_real,color=color,ls='--',lw=1.2)
    ax.set(xlim=(.88,1.1),ylim=(.26,.50),xlabel='Common width',ylabel='Eigenvalue')
    ax.legend(fontsize=8);grid(ax)
    ax=axes[0,1];panel(ax,'B','Square-root opening of the class gap')
    d=gaps[gaps.mu>0];x=np.geomspace(d.mu.min(),d.mu.max(),200)
    ax.loglog(x,.9515708050*np.sqrt(x),color=GRAY,ls='--',label=r'$0.951570805\sqrt{\mu}$')
    ax.loglog(d.mu,d.gap_at_zero_mutation_crossing,'o',color=BLUE,label='Exact class eigenvalues')
    ax.set(xlabel=r'Output mutation probability $\mu$',ylabel=r'Class eigenvalue gap at $\sigma_c$')
    ax.legend(fontsize=8);grid(ax)
    ax=axes[1,0];panel(ax,'C','A single demographic threshold')
    for mu,color,label in choices:
        d=data[np.isclose(data.mu,mu,rtol=1e-10,atol=1e-15)].sort_values('sigma')
        ax.plot(d.sigma,d.demographic_death_threshold,color=color,label=r'$\mu=$'+label)
    ax.set(xlabel='Common width',ylabel=r'$d_c=\rho/(1+\rho)$',xlim=(.4,2))
    ax.legend(fontsize=8);grid(ax)
    ax=axes[1,1];panel(ax,'D','Two-value contribution to the leading mode')
    for mu,color,label in choices:
        d=data[np.isclose(data.mu,mu,rtol=1e-10,atol=1e-15)].sort_values('sigma')
        ax.plot(d.sigma,d.perron_H_fraction,color=color,label=r'$\mu=$'+label)
    ax.set(xlabel='Common width',ylabel='Two-value fraction in Perron vector',xlim=(.4,2),ylim=(-.02,1))
    ax.legend(fontsize=8);grid(ax)
    save(fig,out,'figS4_innovation',[base+'mutation_thresholds.csv',base+'mutation_crossover.csv'],registry)


def finite_size(root,out,registry):
    base='results/intervention_revision/'
    runs=pd.read_csv(root/(base+'finite_size_run_distribution.csv'))
    summary=pd.read_csv(root/(base+'finite_size_mixture_summary.csv'))
    errors=pd.read_csv(root/(base+'fluid_conditional_errors.csv'))
    fig,axes=plt.subplots(1,2,figsize=(8.8,4.2),layout='constrained')
    ax=axes[0];panel(ax,'A','Run distributions reveal extinction mixtures')
    for x,scale in enumerate([1,4,16]):
        d=runs[runs.scale==scale].sort_values('value');offset=np.linspace(-.18,.18,len(d))
        colors=[ORANGE if value else BLUE for value in d.extinct1000]
        ax.scatter(x+offset,d.value,color=colors,s=24,edgecolor='white',linewidth=.4,zorder=3)
        row=summary[summary.scale==scale].iloc[0]
        ax.errorbar([x],[row['mean']],yerr=[[row['mean']-row.bootstrap95_low],[row.bootstrap95_high-row['mean']]],
                    fmt='D',color=INK,ms=5,capsize=4,zorder=5)
    target=float(summary.fixed_point_prediction.iloc[0]);ax.axhline(target,color=GREEN,ls='--',label='Fluid fixed point')
    ax.scatter([],[],color=BLUE,s=24,label='Occupied at sweep 1000');ax.scatter([],[],color=ORANGE,s=24,label='Extinct by sweep 1000')
    ax.plot([],[],'D',color=INK,label='Mean and bootstrap 95% interval')
    ax.set(xticks=[0,1,2],xticklabels=['1 (20 runs)','4 (20 runs)','16 (40 runs)'],
           xlabel='Population and resource scale',ylabel='Native fraction averaged over sweeps 501–1000',ylim=(-.004,.10))
    ax.legend(loc='lower right',fontsize=7.5);grid(ax,'y')
    ax=axes[1];panel(ax,'B','Conditioning separates routes to large error')
    d=errors[np.isclose(errors.sigma,.8)&(errors.scale==1)].sort_values('sweep')
    columns=[('mse_unconditional',BLUE,'All paths'),('mse_alive_conditional',ORANGE,'Occupied paths'),('mse_H_conditional',GREEN,'Two-value paths')]
    # Column names are checked against the saved analysis schema below.
    for k,(column,color,label) in enumerate(columns):
        ax.bar(np.arange(len(d))+(k-1)*.22,d[column],width=.21,color=color,label=label,zorder=3)
    ax.set_xticks(range(len(d)),[str(int(x)) for x in d.sweep])
    ax.set(xlabel='Evaluation sweep (width 0.8, scale 1)',ylabel='Mean squared class-density error')
    ax.legend(fontsize=8);grid(ax,'y')
    save(fig,out,'figS5_finite_size',[base+n for n in ['finite_size_run_distribution.csv','finite_size_mixture_summary.csv','fluid_conditional_errors.csv']],registry)


def run():
    root=ROOT;out=root/'figures';style();registry={}
    for function in [cycles,intervention,innovation,finite_size]:function(root,out,registry)
    for record in registry.values():
        record['sources']+=['analysis/figures.py']
        record['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in record['sources']}
        record['generator']='analysis/response_figures.py'
        record['generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path=out/'figure_manifest.json';previous=json.loads(path.read_text());previous.update(registry)
    path.write_text(json.dumps(previous,indent=2)+'\n')
    print(json.dumps({'figures':list(registry),'formats':['pdf','svg','png']}))


if __name__=='__main__':run()
