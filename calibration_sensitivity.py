import argparse
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from simulation_study import scenario_study, SCENARIOS, tex_table

SEED = 20261107
DESIGNS = ((49,50),(99,50),(199,50),(399,50),(199,0),(199,10),(199,100))


def dependence_constants():
    lt=.45+.5*math.exp(-1.5)
    lv=math.sqrt(.12)
    rho=math.sqrt(lt**2+lv**2)
    moment=.70/(1-.45**2-.12)
    rows=[]
    for law,af,bf in [('Gaussian',norm.pdf(0),2*norm.pdf(1)),
                       ('Student',8/(3*math.pi*math.sqrt(3)),
                        2*math.sqrt(.6)*8/(3*math.pi*math.sqrt(3))/(1.2**3))]:
        smoothing=(af*lt+bf*lv)/math.sqrt(.70)
        c=2*smoothing*math.sqrt(moment)/rho
        for gap in (0,10,25,50,100):
            beta=min(1.,c*rho**gap) if gap else 1.


            rows.append(dict(law=law,gap=gap,rho=rho,C=c,smoothing=smoothing,
                             stationary_second_moment_bound=moment,beta_bound=beta,
                             total_allowance=min(1.,200*beta),
                             burn_in_500=smoothing*math.sqrt(moment)*rho**499))

    f0=8/(3*math.pi*math.sqrt(3))
    f=lambda x:f0*(1+x*x/3)**-3
    af=quad(lambda x:abs(-6*x/(3+x*x)*f(x)), -np.inf,np.inf)[0]/2
    bf=quad(lambda x:abs(1-6*x*x/(3+x*x))*f(x),-np.inf,np.inf,epsabs=1e-11)[0]/2
    assert abs(af-f0)<1e-9
    assert abs(bf-2*math.sqrt(.6)*f(math.sqrt(.6)))<1e-8
    assert rho<1
    return pd.DataFrame(rows)


def randomized_rank_results(details, scenario, rng):


    cal,mon=details['calibration'],details['monitored']
    truth=details['truth']; k=cal.shape[1]
    uc=rng.random(cal.shape[:2]); um=rng.random(mon.shape[:2])
    out=[]
    for j,method in enumerate(('Count','Block sum','Energy','Capped energy','Interval scan','Run statistic')):
        ranks=np.ones(mon.shape[:2],dtype=int)
        for i in range(k):
            less=(cal[:,i,j,None]<mon[:,:,j])
            tied=(cal[:,i,j,None]==mon[:,:,j]) & (uc[:,i,None]<um)
            ranks+=less|tied
        reject=ranks>math.ceil((k+1)*.99-1e-12)
        row={'scenario':scenario.name,'method':method,'replications':len(mon),'K':k,'target':.01}
        for lab,name in ((0,'null'),(1,'short'),(2,'long')):
            rate=reject[:,truth==lab].mean(axis=1)
            row[name]=float(rate.mean())
            row[name+'_se']=float(rate.std(ddof=1)/math.sqrt(len(rate)))
        out.append(row)
    return out


def run(reps,root):
    tables=root/'tables';tables.mkdir(parents=True,exist_ok=True)
    rows=[];matched=[]; start=time.time()

    for si in (0,2):
        sc=SCENARIOS[si]
        for di,(k,g) in enumerate(DESIGNS):
            seed=SEED+10000*si+di
            perf,abl,det=scenario_study(sc,reps,seed,k_cal=k,block_gap=g,return_details=True)
            for pr in perf[:2]:
                row=dict(pr,K=k,gap=g,seed=seed)
                for n in ('a','k_direct','k_record','point_threshold'):
                    v=det[n]; finite=v[np.isfinite(v)]
                    row['mean_'+n]=float(finite.mean()) if len(finite) else np.inf
                    row['sd_'+n]=float(finite.std(ddof=1)) if len(finite)>1 else np.nan
                row['infeasible_fraction']=abl['infeasible_fraction']
                rows.append(row)
            if (k,g)==(199,50):
                matched.extend(randomized_rank_results(det,sc,np.random.default_rng(seed+500000)))
            print(f'{sc.name}: K={k}, gap={g}, seconds={time.time()-start:.1f}',flush=True)

    for si in (1,3,4,5,6):
        sc=SCENARIOS[si];seed=SEED+10000*si+2
        _,_,det=scenario_study(sc,reps,seed,return_details=True)
        matched.extend(randomized_rank_results(det,sc,np.random.default_rng(seed+500000)))
        print(f'Matched ranks: {sc.name}, seconds={time.time()-start:.1f}',flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(tables/'calibration_sensitivity.csv',index=False)
    mf=pd.DataFrame(matched);mf.to_csv(tables/'fitted_matched_rank.csv',index=False)
    dep=dependence_constants();dep.to_csv(tables/'charn_dependence_allowances.csv',index=False)
    for law in ('Gaussian','Student'):
        show=frame[(frame.scenario==law+' mean 0.40') & (frame.method=='Record count')].copy()
        show['design']=[f'{int(k)} / {int(g)}' for k,g in zip(show.K,show.gap)]
        show['upper']=[r'$\infty$' if not np.isfinite(v) else f'{v:.2f}' for v in show.mean_k_record]
        show['a_display']=[f'{v:.2f} ({sd:.2f})' for v,sd in zip(show.mean_a,show.sd_a)]
        tex_table(show,['design','a_display','upper','H0_warning','H0_persistent','H1_correct','H2_correct'],
                  [r'$K/g$',r'$\overline a$ (s.d.)',r'$\overline k$',r'$0\to+$',r'$0\to2$',r'$1\to1$',r'$2\to2$'],
                  tables/('sensitivity_'+law.lower()+'.tex'))
    summary=mf[mf.method.isin(('Count','Block sum','Energy'))].copy()
    summary['display']=summary.scenario+' / '+summary.method
    summary['size_display']=[f'{v:.4f} ({se:.4f})' for v,se in zip(summary.null,summary.null_se)]
    tex_table(summary,['display','size_display','short','long'],
              ['Scenario / statistic','Null (MC s.e.)','Short','Long'],tables/'fitted_matched_rank.tex')
    dp=dep[dep.gap.isin((10,25,50,100))].copy()
    dp['display']=[f'{law} / {int(g)}' for law,g in zip(dp.law,dp.gap)]
    tex_table(dp,['display','beta_bound','total_allowance'],
              ['Innovation / gap',r'$\overline\beta(g)$',r'$\min\{1,200\overline\beta(g)\}$'],tables/'charn_dependence_allowances.tex',
              {'beta_bound':'.3g','total_allowance':'.3g'})
    meta={'master_seed':SEED,'replications':reps,'designs':DESIGNS,'python':platform.python_version(),
          'numpy':np.__version__,'pandas':pd.__version__,
          'max_sensitivity_mc_se':float(frame.filter(like='_se').max().max()),
          'max_matched_mc_se':float(mf.filter(like='_se').max().max()),
          'elapsed_seconds':time.time()-start,
          'interpretation':'Independent whole-series repetitions; empirical sizes reported, not forced equal. Randomized ties remove discreteness; no record or middle-label floor in matched binary tests.'}
    (root/'sensitivity_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replications',type=int,default=1000)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    a=ap.parse_args()
    if a.replications<2:ap.error('at least two replications required')
    run(a.replications,a.output)
