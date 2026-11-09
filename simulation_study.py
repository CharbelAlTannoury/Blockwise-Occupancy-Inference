from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binom, chi2, ncx2, norm

from count_distributions import (
    poisson_binomial_pmf, iid_record_threshold, yang_record_probabilities,
    upper_tail_threshold_from_pmf, size_matched_gaussian_benchmark,
    markov_binomial_pmf, duration_strength_benchmark,
)

SEED = 20260919
M = 20
R = 10
P0 = .05
WARNING = .05
PERSISTENCE = .01
K_CAL = 199
FIT_LENGTH = 2000
POINT_SCORES = 200
GAP = 50
METHODS = ('Direct count', 'Record count', 'Block sum', 'Energy',
           'Capped energy', 'Interval scan', 'Run statistic')


@dataclass(frozen=True)
class Scenario:
    name: str
    law: str
    family: str
    effect: float


SCENARIOS = (
    Scenario('Gaussian mean 0.40', 'gaussian', 'mean', .40),
    Scenario('Gaussian mean 0.55', 'gaussian', 'mean', .55),
    Scenario('Student mean 0.40', 'student', 'mean', .40),
    Scenario('Student mean 0.55', 'student', 'mean', .55),
    Scenario('Gaussian scale 1.50', 'gaussian', 'scale', 1.50),
    Scenario('Student scale 1.50', 'student', 'scale', 1.50),
    Scenario('Gaussian alternating 0.55', 'gaussian', 'alternating', .55),
)


def noise(rng, size, law):
    if law == 'gaussian':
        return rng.standard_normal(size)
    return rng.standard_t(5, size=size) / math.sqrt(5 / 3)


def true_mean(x):
    return .45*x - .25*x*np.exp(-.5*x*x)


def advance(rng, state, steps, law, keep=False):
    out = np.empty((len(state), steps+1)) if keep else None
    if keep:
        out[:, 0] = state
    for j in range(steps):
        state = true_mean(state) + np.sqrt(.70 + .12*state*state)*noise(rng, len(state), law)
        if keep:
            out[:, j+1] = state
    return state, out


def fit_charn(x, law):
    lag, response = x[:, :-1], x[:, 1:]
    design = np.stack((np.ones_like(lag), lag, lag*np.exp(-.5*lag*lag)), axis=2)
    xtx = np.einsum('rni,rnj->rij', design, design)
    xty = np.einsum('rni,rn->ri', design, response)
    b = np.linalg.solve(xtx, xty[..., None])[..., 0]
    residual = response - np.einsum('rni,ri->rn', design, b)
    vdesign = np.stack((np.ones_like(lag), lag*lag), axis=2)
    vtx = np.einsum('rni,rnj->rij', vdesign, vdesign)
    vty = np.einsum('rni,rn->ri', vdesign, residual*residual)
    v = np.linalg.solve(vtx, vty[..., None])[..., 0]
    v[:, 0] = np.maximum(v[:, 0], .10)
    v[:, 1] = np.maximum(v[:, 1], 0.)
    fitted_var = v[:, :1] + v[:, 1:]*lag*lag
    fisher = 1. if law == 'gaussian' else 1.25
    information = fisher*np.mean(1/fitted_var, axis=1)
    return b, v, information


def score_stream(rng, state, nscore, law, fit, mu=None, scale=None):
    b, v, info = fit
    reps = len(state)
    scores = np.zeros((reps, nscore))
    for j in range(nscore*R):
        mean_shift = 0. if mu is None else float(mu[j])
        multiplier = 1. if scale is None else float(scale[j])
        lag = state
        state = (mean_shift + true_mean(lag)
                 + multiplier*np.sqrt(.70+.12*lag*lag)*noise(rng, reps, law))
        fitted_mean = b[:, 0] + b[:, 1]*lag + b[:, 2]*lag*np.exp(-.5*lag*lag)
        fitted_sd = np.sqrt(v[:, 0]+v[:, 1]*lag*lag)
        residual = (state-fitted_mean)/fitted_sd
        psi = residual if law == 'gaussian' else 6*residual/(3+residual*residual)
        scores[:, j//R] += psi/fitted_sd
    return state, scores/np.sqrt(R*info[:, None])


def statistics(scores, point_threshold):

    absolute = np.abs(scores)
    count = np.sum(absolute > np.asarray(point_threshold)[..., None], axis=-1)
    block_sum = np.abs(np.sum(scores, axis=-1))/math.sqrt(scores.shape[-1])
    energy = np.sum(scores*scores, axis=-1)
    capped = np.sum(np.minimum(scores*scores, np.asarray(point_threshold)[..., None]**2), axis=-1)
    run = np.max(np.minimum(np.minimum(absolute[..., :-2], absolute[..., 1:-1]),
                            absolute[..., 2:]), axis=-1)
    cumulative = np.concatenate((np.zeros(scores.shape[:-1]+(1,)),
                                 np.cumsum(scores, axis=-1)), axis=-1)
    scan = np.zeros(scores.shape[:-1])
    for length in range(3, scores.shape[-1]+1):
        value = np.max(np.abs(cumulative[..., length:]-cumulative[..., :-length]), axis=-1)/math.sqrt(length)
        scan = np.maximum(scan, value)
    return np.stack((count, block_sum, energy, capped, scan, run), axis=-1)


def rank_cutoff(values, alpha):

    k = values.shape[1]
    order = int(math.ceil((k+1)*(1-alpha)-1e-12))
    if order > k:
        return np.full((values.shape[0], values.shape[2]), np.inf)
    return np.sort(values, axis=1)[:, order-1, :]


def signal_design(scenario):
    n = 4000
    mu, scale = np.zeros(n), np.ones(n)
    labels = np.zeros(n, dtype=np.int8)
    for lo, hi, sign in ((1350, 2050, 1), (2500, 3200, -1)):
        labels[lo:hi] = 2
        if scenario.family == 'scale':
            scale[lo:hi] = scenario.effect
        elif scenario.family == 'alternating':
            mu[lo:hi] = scenario.effect*np.where((np.arange(lo, hi)//R) % 2 == 0, 1., -1.)
        else:
            mu[lo:hi] = sign*scenario.effect
    for lo, hi, sign in ((1150, 1190, 1), (2250, 2290, -1), (3500, 3540, 1)):
        mu[lo:hi] = sign*1.20
        labels[lo:hi] = 1
    micro = labels.reshape(-1, R)
    micro_label = np.where(np.mean(micro == 2, axis=1) >= .5, 2,
                           np.where(np.any(micro == 1, axis=1), 1, 0))
    starts = np.arange(0, len(micro_label)-M+1, 5)
    truth = []
    for start in starts:
        lab = micro_label[start:start+M]
        frac = np.mean(lab == 2)
        truth.append(2 if frac >= .80 else (1 if frac == 0 and np.any(lab == 1)
                                          else (0 if frac == 0 else 3)))
    return mu, scale, starts, np.array(truth)


def metrics(decisions, truth):
    masks = {i: truth == i for i in (0, 1, 2, 3)}
    return {
        'H0_warning': np.mean(decisions[:, masks[0]] > 0, axis=1),
        'H0_persistent': np.mean(decisions[:, masks[0]] == 2, axis=1),
        'H1_correct': np.mean(decisions[:, masks[1]] == 1, axis=1),
        'H1_persistent': np.mean(decisions[:, masks[1]] == 2, axis=1),
        'H2_correct': np.mean(decisions[:, masks[2]] == 2, axis=1),
        'boundary_signal': np.mean(decisions[:, masks[3]] > 0, axis=1),
    }


def scenario_study(scenario, reps, seed, k_cal=K_CAL, block_gap=GAP, return_details=False):
    rng = np.random.default_rng(seed)
    state = np.zeros(reps)
    state, _ = advance(rng, state, 500, scenario.law)
    state, fit_data = advance(rng, state, FIT_LENGTH, scenario.law, keep=True)
    fit = fit_charn(fit_data, scenario.law)
    del fit_data
    state, _ = advance(rng, state, GAP, scenario.law)
    state, point_scores = score_stream(rng, state, POINT_SCORES, scenario.law, fit)
    c0 = np.quantile(np.abs(point_scores), 1-P0, axis=1)
    calibration = np.empty((reps, k_cal, 6))
    for j in range(k_cal):
        state, _ = advance(rng, state, block_gap, scenario.law)
        state, block = score_stream(rng, state, M, scenario.law, fit)
        calibration[:, j, :] = statistics(block, c0)
    lower = rank_cutoff(calibration, WARNING)
    upper = rank_cutoff(calibration, PERSISTENCE)

    a = lower[:, 0]
    k_direct = np.maximum(upper[:, 0]+1, a+2)
    k_record = np.maximum(k_direct, iid_record_threshold(M, .10))
    state, _ = advance(rng, state, 2*GAP, scenario.law)
    mu, scale, starts, truth = signal_design(scenario)
    state, monitored = score_stream(rng, state, len(mu)//R, scenario.law, fit, mu, scale)
    blocks = np.stack([monitored[:, j:j+M] for j in starts], axis=1)
    monitored_stats = statistics(blocks, c0[:, None])
    decisions = {}
    e = monitored_stats[..., 0]
    for method, k in (('Direct count', k_direct), ('Record count', k_record)):
        decisions[method] = np.where(e >= k[:, None], 2, np.where(e > a[:, None], 1, 0))
    for col, method in enumerate(METHODS[2:], start=1):
        value = monitored_stats[..., col]
        decisions[method] = np.where(value > upper[:, col, None], 2,
                                      np.where(value > lower[:, col, None], 1, 0))
    rows = []
    for method, decision in decisions.items():
        row = {'scenario': scenario.name, 'method': method, 'replications': reps}
        for name, values in metrics(decision, truth).items():
            row[name] = float(values.mean())
            row[name+'_se'] = float(values.std(ddof=1)/math.sqrt(reps))
        rows.append(row)
    ablation = {
        'scenario': scenario.name, 'mean_a': float(a.mean()),
        'mean_k_direct': float(k_direct.mean()), 'mean_k_record': float(k_record.mean()),
        'record_binding_fraction': float(np.mean(k_record > k_direct)),
        'decision_difference': float(np.mean(decisions['Direct count'] != decisions['Record count'])),
        'infeasible_fraction': float(np.mean(k_record > M)),
    }
    if return_details:
        return rows, ablation, {
            'calibration': calibration, 'monitored': monitored_stats, 'truth': truth,
            'a': a, 'k_direct': k_direct, 'k_record': k_record, 'point_threshold': c0,
        }
    return rows, ablation


def exact_benchmarks():
    power, calibration = size_matched_gaussian_benchmark()
    alpha = float(binom.sf(5, 20, .05))
    critical = float(chi2.isf(alpha, M))
    power['energy'] = [alpha,
                       float(ncx2.sf(critical, M, M*.6**2)),
                       float(ncx2.sf(critical, M, M)),
                       float(chi2.sf(critical/1.25**2, M)),
                       float(chi2.sf(critical/1.5**2, M))]
    calibration = pd.concat((calibration, pd.DataFrame([{
        'method': 'Gaussian energy', 'pointwise_tail': np.nan,
        'critical_value': critical, 'exact_null_size': chi2.sf(critical, M)}])), ignore_index=True)
    consistency = []
    for m in (20, 100, 1000, 10000, 100000):
        fixed = int(binom.ppf(.95, m, .05))
        vanishing = math.ceil(m*(.05+math.sqrt(math.log(m)/m)))
        consistency.append({'m': m, 'fixed_a_rate': fixed/m,
                            'fixed_warning': binom.sf(fixed, m, .05),
                            'vanishing_a_rate': vanishing/m,
                            'vanishing_warning': binom.sf(vanishing, m, .05)})
    return power, calibration, pd.DataFrame(consistency)


def robustness_benchmark(replications=200000):


    rng = np.random.default_rng(SEED+880)
    alpha = float(binom.sf(5, M, P0))
    energy_cut = chi2.isf(alpha, M)
    sum_cut = norm.isf(alpha/2)
    point_cut = norm.isf(P0/2)
    rows = []
    for law, mu, sigma in (('Null', 0., 1.), ('Mean 1.00', 1., 1.), ('Scale 1.50', 0., 1.5)):
        clean = rng.normal(mu, sigma, size=(replications, M))
        for contamination in ('Clean', 'One value set to +100'):
            x = clean.copy()
            if contamination != 'Clean':
                x[:, 0] = 100.
            e = np.sum(np.abs(x) > point_cut, axis=1)
            squared = x*x
            events = {
                'Nominal count k=6': e >= 6,
                'Certified count k=7': e >= 7,
                'Block sum': np.abs(np.sum(x, axis=1))/math.sqrt(M) > sum_cut,
                'Energy': np.sum(squared, axis=1) > energy_cut,
                'Certified trimmed energy': np.sum(squared, axis=1)-np.max(squared, axis=1) > energy_cut,
            }
            for method, event in events.items():
                rate = float(event.mean())
                rows.append({'law': law, 'contamination': contamination, 'method': method,
                             'probability': rate, 'mc_se': math.sqrt(rate*(1-rate)/replications),
                             'replications': replications})
    return pd.DataFrame(rows)


def tex_table(frame, columns, headers, path, formats=None):
    formats = formats or {}
    lines = [r'\begin{tabular}{'+'l'+'r'*(len(columns)-1)+'}', r'\toprule',
             ' & '.join(headers)+r' \\', r'\midrule']
    for _, row in frame.iterrows():
        vals = []
        for c in columns:
            value = row[c]
            vals.append(format(float(value), formats.get(c, '.3f')) if isinstance(value, (int, float, np.integer, np.floating)) else str(value))
        lines.append(' & '.join(vals)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    path.write_text('\n'.join(lines)+'\n')


def write_outputs(root, performance, ablation, power, calibration, consistency, robust, reps):
    tables, figures = root/'tables', root/'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, frame in [('fitted_performance', performance), ('record_ablation', ablation),
                        ('size_matched_power', power), ('size_matched_calibration', calibration),
                        ('consistency_calibration', consistency), ('contamination', robust)]:
        frame.to_csv(tables/(name+'.csv'), index=False)
    display_power = power.rename(columns={'alternative': 'law'})
    tex_table(display_power, ['law','record_count','calibrated_run3','calibrated_maximum','gaussian_block_sum','energy'],
              ['Score law','Count','Run','Maximum','Sum','Energy'], tables/'size_matched_power.tex',
              {c: '.6f' for c in display_power.columns if c != 'law'})
    tex_table(consistency, ['m','fixed_warning','vanishing_warning'],
              ['$m$','Fixed warning level','Vanishing warning level'], tables/'consistency_calibration.tex',
              {'m': '.0f', 'fixed_warning': '.5f', 'vanishing_warning': '.3g'})
    tex_table(ablation, ['scenario','mean_a','mean_k_direct','mean_k_record','record_binding_fraction','decision_difference'],
              ['Scenario',r'$\overline a$',r'$\overline k_{\rm dir}$',r'$\overline k_{\rm rec}$','Record active','Labels differ'], tables/'record_ablation.tex')
    for filename, names in [('fitted_means', [s.name for s in SCENARIOS[:4]]),
                             ('fitted_other', [s.name for s in SCENARIOS[4:]])]:
        selected = performance[performance.scenario.isin(names) & performance.method.isin(METHODS[:4])].copy()
        selected['display'] = selected['scenario']+' / '+selected['method']
        tex_table(selected, ['display','H0_warning','H0_persistent','H1_correct','H1_persistent','H2_correct'],
                  ['Scenario / method',r'$0\to +$',r'$0\to2$',r'$1\to1$',r'$1\to2$',r'$2\to2$'], tables/(filename+'.tex'))
    all_perf = performance.copy()
    all_perf['display'] = all_perf['scenario']+' / '+all_perf['method']
    tex_table(all_perf, ['display','H0_warning','H0_persistent','H1_correct','H1_persistent','H2_correct'],
              ['Scenario / method',r'$0\to +$',r'$0\to2$',r'$1\to1$',r'$1\to2$',r'$2\to2$'], tables/'fitted_all_methods.tex')
    extra = all_perf[~all_perf.method.isin(METHODS[:4])]
    tex_table(extra, ['display','H0_warning','H0_persistent','H1_correct','H1_persistent','H2_correct'],
              ['Scenario / method',r'$0\to +$',r'$0\to2$',r'$1\to1$',r'$1\to2$',r'$2\to2$'], tables/'fitted_additional.tex')
    selected = robust[robust.contamination != 'Clean'].copy()
    selected['display'] = selected.law+' / '+selected.method
    tex_table(selected, ['display','probability','mc_se'], ['Law / method','Rejection','MC s.e.'], tables/'contamination.tex',
              {'probability': '.5f','mc_se': '.5f'})
    exact_markov = []
    for q in (1, 2, 3, 5, 8, 12):
        pmf = markov_binomial_pmf(P0, .70**q, M)
        tail = float(pmf[6:].sum())
        eps = (M-1)*2*P0*(1-P0)*.70**q
        exact_markov.append({'spacing': q, 'tail': tail, 'binomial': binom.sf(5,M,P0), 'coupling_bound': eps})
    pd.DataFrame(exact_markov).to_csv(tables/'dependence_exact.csv', index=False)
    tex_table(pd.DataFrame(exact_markov), ['spacing','tail','binomial','coupling_bound'],
              ['$q$','Exact Markov tail','Binomial tail','Coupling allowance'], tables/'dependence_exact.tex',
              {'spacing': '.0f','tail': '.6f','binomial': '.6f','coupling_bound': '.6f'})
    duration_strength_benchmark().to_csv(tables/'duration_strength.csv', index=False)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.3))
    markov = pd.DataFrame(exact_markov)
    axs[0].semilogy(markov.spacing, markov['tail'], 'o-', label='Exact dependent tail')
    axs[0].axhline(markov.binomial.iloc[0], color='#b95b32', ls='--', label='Independent tail')
    axs[0].set(xlabel='Diagnostic spacing', ylabel='False-persistent probability')
    axs[0].legend(frameon=False, fontsize=8)
    mean_rows = performance[performance.scenario == 'Gaussian mean 0.40'].set_index('method')
    for i, method in enumerate(METHODS[:4]):
        axs[1].bar(i, mean_rows.loc[method,'H2_correct'], color=['#969caa','#24648c','#c17b42','#658b60'][i])
        axs[1].errorbar(i, mean_rows.loc[method,'H2_correct'], yerr=1.96*mean_rows.loc[method,'H2_correct_se'], color='black', capsize=3)
    axs[1].set_xticks(range(4), ['Direct\ncount','Record\ncount','Block\nsum','Energy'])
    axs[1].set(ylabel='Persistent-block sensitivity', ylim=(0,1), title='Fitted CHARN: Gaussian mean 0.40')
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures/'revised_summary.pdf', bbox_inches='tight')
    fig.savefig(figures/'revised_summary.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    summary = {'seed':SEED,'replications':reps,'m':M,'score_length':R,
               'fit_length':FIT_LENGTH,'point_calibration_scores':POINT_SCORES,
               'count_calibration_blocks':K_CAL,'gap':GAP,
               'warning_budget':WARNING,'persistence_budget':PERSISTENCE,
               'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,
               'matplotlib':matplotlib.__version__,
               'python':platform.python_version(),
               'maximum_mc_se':float(performance.filter(like='_se').max().max()),
               'record_reference_persistence_range':[float(performance.loc[performance.method=='Record count','H0_persistent'].min()),
                                                     float(performance.loc[performance.method=='Record count','H0_persistent'].max())]}
    (root/'experiment_metadata.json').write_text(json.dumps(summary,indent=2)+'\n')
    macros = [r'\newcommand{\SimulationReplications}{'+str(reps)+'}',
              r'\newcommand{\LargestMCSE}{'+f"{summary['maximum_mc_se']:.4f}"+'}']
    (tables/'simulation_macros.tex').write_text('\n'.join(macros)+'\n')


def validate_math():
    pmf = poisson_binomial_pmf(1/np.arange(1,21))
    assert np.isclose(pmf.sum(),1)
    assert iid_record_threshold(20,.10) == 6
    assert iid_record_threshold(1,.10) == 2
    assert [upper_tail_threshold_from_pmf(poisson_binomial_pmf(
        yang_record_probabilities(20,gamma)), .10)
        for gamma in (1.02,1.05,1.10,1.25)] == [7,7,8,9]
    assert np.isclose(pmf[6:].sum(),.09436515188515888)
    assert np.isclose(markov_binomial_pmf(.05,0,20),binom.pmf(np.arange(21),20,.05)).all()

    values = np.zeros((1,19,1))
    assert rank_cutoff(values,.05)[0,0] == 0
    assert np.isinf(rank_cutoff(values,.01)[0,0])


    import itertools
    for p in (.1, .5, .9):
        probability = 0.
        for bits in itertools.product((0,1), repeat=5):
            cutoff = rank_cutoff(np.array(bits[:4])[None,:,None], .4)[0,0]
            weight = p**sum(bits)*(1-p)**(5-sum(bits))
            probability += weight*(bits[4] > cutoff)
        assert probability <= .4+1e-12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replications',type=int,default=1000)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.replications < 2:
        parser.error('--replications must be at least 2 to compute Monte Carlo standard errors')
    args.output.mkdir(parents=True,exist_ok=True)
    validate_math()
    start = time.time()
    rows, abl = [], []
    for index, scenario in enumerate(SCENARIOS):
        print('Running '+scenario.name,flush=True)
        r,a = scenario_study(scenario,args.replications,SEED+100*index)
        rows.extend(r); abl.append(a)
        print('Completed in %.1f seconds; record persistence sensitivity %.4f' %
              (time.time()-start,r[1]['H2_correct']),flush=True)
    power, calibration, consistency = exact_benchmarks()
    robust = robustness_benchmark()
    performance = pd.DataFrame(rows)
    ablation = pd.DataFrame(abl)
    write_outputs(args.output, performance, ablation, power, calibration, consistency, robust, args.replications)
    print(performance[performance.method.isin(METHODS[:4])].to_string(index=False),flush=True)
    print(ablation.to_string(index=False),flush=True)
    print('All outputs written. Seconds: %.1f' % (time.time()-start),flush=True)


if __name__ == '__main__':
    main()
