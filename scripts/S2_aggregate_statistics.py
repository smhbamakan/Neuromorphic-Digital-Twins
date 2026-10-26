"""
S2 -- Aggregates seed-level results of S1 into the quantities reported in Tables 7-9 and the
supplementary tables. Usage:  python S2_aggregate_statistics.py results/sim

Statistical procedure
  * 95% confidence intervals: mean +/- t_{0.975, n-1} * SD / sqrt(n) across independent seeds.
  * Randomised across seeds: telegraph trajectory, evidence events, network initial state,
    bias currents, input/recurrent weights and delays, rate-baseline weights, and event-loss draws.
  * Comparisons: two-sided paired t-tests over seeds, effect size d_z = mean(diff)/SD(diff).
  * Multiple comparisons: Holm-Bonferroni within each family of tests (reported as p_holm).
"""
import csv, glob, json, os, sys
import numpy as np
from scipy import stats

D = sys.argv[1]
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, 'seed_*.json')),
                                        key=lambda s: int(s.split('_')[-1].split('.')[0]))]
n = len(R)
MET = ['rmse_full', 'rmse_obs', 'brier', 'logloss', 'ece', 'accuracy', 'frac_clipped',
       't_adapt_median', 't_adapt_censored']


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean()), float(h)]


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj.tolist()


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    return dict(diff=ci(d), t=float(t), p=float(p), dz=float(d.mean() / d.std(ddof=1)))


def family(tests):
    adj = holm([t['p'] for t in tests.values()])
    for (k, t), pa in zip(tests.items(), adj):
        t['p_holm'] = pa
    return tests


S = dict(n_seeds=n)
g = lambda m, k: [r[m][k] for r in R]

# ---------------- Table 7: predictive quality
S['table7'] = {m: {k: ci(g(m, k)) for k in MET} for m in ['NDT', 'RATE', 'BAYES']}
S['switches'] = {m: dict(evaluated=int(sum(g(m, 'n_switch'))), censored=int(sum(g(m, 'n_censored'))))
                 for m in ['NDT', 'RATE', 'BAYES', 'NDT_nodelay', 'NDT_norec']}
S['tests_ndt_vs_rate'] = family({k: paired(g('NDT', k), g('RATE', k)) for k in
                                 ['rmse_full', 'rmse_obs', 'brier', 'logloss', 'ece', 'accuracy',
                                  't_adapt_censored']})
S['rate_hz'] = ci(g('NDT', 'rate_hz'))

# ---------------- operation accounting (held-out window and full trial)
se = int(round(R[0]['cfg']['dt_eval'] / R[0]['cfg']['dt_sim']))
N = R[0]['cfg']['N']
acc = {}
for win in ['counts_test', 'counts_full']:
    T = (R[0]['cfg']['T_total'] - R[0]['cfg']['T_train']) if win == 'counts_test' else R[0]['cfg']['T_total']
    frames = int(round(T / (R[0]['cfg']['dt_eval'] * 1e-3)))
    readout = frames * (N + 1)
    rows = {}
    nd = lambda k: [r['NDT'][win][k] for r in R]
    rt = lambda k: [r['RATE'][win][k] for r in R]
    rows['NDT_spikes'] = ci(nd('spikes'))
    rows['NDT_syn_recurrent'] = ci(nd('syn_rec'))
    rows['NDT_syn_input'] = ci(nd('syn_in'))
    rows['NDT_neuron_updates'] = ci(nd('neuron_updates'))
    rows['NDT_trace_updates'] = ci(nd('trace_updates'))
    rows['readout_macs'] = readout
    ex = np.array(nd('syn_rec')) + np.array(nd('syn_in')) + np.array(nd('neuron_updates')) \
        + np.array(nd('trace_updates')) + readout
    # event-driven projection: one state update per delivered synaptic event (exact integration between
    # events), one trace increment per spike, N trace evaluations per output sample, and the readout
    ev = np.array(nd('syn_rec')) + np.array(nd('syn_in')) + np.array(nd('spikes')) + frames * N + readout
    rows['NDT_trace_evaluations_event_driven'] = frames * N
    syn = np.array(nd('syn_rec')) + np.array(nd('syn_in'))
    rows['NDT_total_executed'] = ci(ex)
    rows['NDT_total_event_driven_projection'] = ci(ev)
    rows['RATE_mac_recurrent'] = ci(rt('mac_rec'))
    rows['RATE_mac_recurrent_dense'] = ci(rt('mac_rec_dense'))
    rows['RATE_mac_input'] = ci(rt('mac_in'))
    rows['RATE_unit_updates'] = ci(rt('unit_updates'))
    rows['RATE_encoding'] = ci(rt('encoding_ops'))
    rt_tot = np.array(rt('mac_rec')) + np.array(rt('mac_in')) + np.array(rt('unit_updates')) \
        + np.array(rt('encoding_ops')) + readout
    rows['RATE_total'] = ci(rt_tot)
    rows['ratio_synaptic_events_vs_macs'] = ci((np.array(rt('mac_rec')) + np.array(rt('mac_in'))) / syn)
    rows['ratio_total_rate_vs_ndt_executed'] = ci(rt_tot / ex)
    rows['ratio_total_rate_vs_ndt_event_driven'] = ci(rt_tot / ev)
    acc[win] = rows
S['operations'] = acc

# ---------------- Table 8: ablations
S['table8'] = {m: {k: ci(g(m, k)) for k in ['rmse_full', 'brier', 'logloss', 't_adapt_censored']}
               for m in ['NDT', 'NDT_nodelay', 'NDT_norec']}
for m in ['NDT', 'NDT_nodelay', 'NDT_norec']:
    S['table8'][m]['spikes_full'] = ci([r[m]['counts_full']['spikes'] for r in R])
    S['table8'][m]['syn_full'] = ci([r[m]['counts_full']['syn_rec'] + r[m]['counts_full']['syn_in'] for r in R])
S['tests_ablation'] = family({f'{a}_{k}': paired(g(a, k), g('NDT', k))
                              for a in ['NDT_nodelay', 'NDT_norec'] for k in ['rmse_full', 't_adapt_censored']})

# ---------------- Table 9: robustness
drops = R[0]['cfg']['drops']
rb = lambda m, p, k='rmse_full': [r['robustness'][f'{m}_{p}'][k] for r in R]
S['table9'] = {m: {str(p): ci(rb(m, p)) for p in drops} for m in ['NDT', 'RATE', 'BAYES']}
S['table9_accuracy'] = {m: {str(p): ci(rb(m, p, 'accuracy')) for p in drops} for m in ['NDT', 'RATE', 'BAYES']}
inc = {m: np.array(rb(m, drops[-1])) - np.array(rb(m, drops[0])) for m in ['NDT', 'RATE', 'BAYES']}
rel = {m: inc[m] / np.array(rb(m, drops[0])) for m in ['NDT', 'RATE', 'BAYES']}
S['robust_abs_increase'] = {m: ci(inc[m]) for m in inc}
S['robust_rel_increase'] = {m: ci(rel[m]) for m in rel}
tests = {}
for p in drops[1:]:
    a = np.array(rb('NDT', p)) - np.array(rb('NDT', 0.0))
    b = np.array(rb('RATE', p)) - np.array(rb('RATE', 0.0))
    tests[f'abs_increase_NDT_vs_RATE_p{p}'] = paired(a, b)
S['tests_robustness'] = family(tests)
S['gap_rate_minus_ndt_by_level'] = {str(p): ci(np.array(rb('RATE', p)) - np.array(rb('NDT', p))) for p in drops}

# ---------------- output-interval sensitivity
iv = R[0]['cfg']['intervals']
S['interval_sensitivity'] = {str(i): dict(
    ndt_rmse=ci([r['interval_sensitivity'][str(i)]['ndt_rmse'] for r in R]),
    rate_rmse=ci([r['interval_sensitivity'][str(i)]['rate_rmse'] for r in R]),
    rate_tuned_rmse=ci([r['interval_sensitivity'][str(i)]['rate_tuned_rmse'] for r in R]),
    rate_tuned_scales=[r['interval_sensitivity'][str(i)]['rate_tuned_scale'] for r in R],
    paired_tuned=paired([r['interval_sensitivity'][str(i)]['ndt_rmse'] for r in R],
                        [r['interval_sensitivity'][str(i)]['rate_tuned_rmse'] for r in R]),
    rate_macs_test=ci([r['interval_sensitivity'][str(i)]['rate_macs_test'] for r in R]),
    ndt_readout_macs=R[0]['interval_sensitivity'][str(i)]['ndt_readout_macs'],
    rate_wall=ci([r['interval_sensitivity'][str(i)]['rate_wall'] for r in R]),
    paired=paired([r['interval_sensitivity'][str(i)]['ndt_rmse'] for r in R],
                  [r['interval_sensitivity'][str(i)]['rate_rmse'] for r in R])) for i in iv}
family({i: S['interval_sensitivity'][i]['paired'] for i in S['interval_sensitivity']})
family({i: S['interval_sensitivity'][i]['paired_tuned'] for i in S['interval_sensitivity']})

# ---------------- reliability curves (pooled over seeds, held-out window)
S['reliability'] = {m: [r[m]['reliability'] for r in R] for m in ['NDT', 'RATE', 'BAYES']}

json.dump(S, open(os.path.join(D, 'summary.json'), 'w'), indent=1)

# seed-level CSV
with open(os.path.join(D, 'seed_level.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['seed', 'model'] + MET)
    for r in R:
        for m in ['NDT', 'RATE', 'BAYES', 'NDT_nodelay', 'NDT_norec']:
            w.writerow([r['seed'], m] + [r[m][k] for k in MET])


def show(d, ind=0):
    for k, v in d.items():
        if k == 'reliability':
            continue
        if isinstance(v, dict):
            print(' ' * ind + str(k)); show(v, ind + 2)
        else:
            print(' ' * ind + f'{k}: {v}')


show(S)
