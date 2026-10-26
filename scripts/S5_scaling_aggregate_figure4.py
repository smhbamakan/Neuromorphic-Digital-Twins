"""
S5 -- Aggregates S4 outputs into Table 10, the fitted exponents, and Figure 4.
Usage: python S5_scaling_aggregate_figure4.py results/scale

Definitions
  comm. ratio (ratio of means) = mean_seeds(inter-agent events) / mean_seeds(intra-agent events), same regime
  comm. ratio (mean of ratios) = mean_seeds(inter/intra) with 95% t-interval
  expected structural ratio    = K_inter * d_out / (M * p_intra * (M - 1)), i.e. inter/intra out-degree per
                                  neuron, exact when all neurons fire at the same rate
  exponent alpha               = slope of an ordinary least-squares fit of log(y) on log(N), fitted separately
                                  for each seed; reported as mean +/- 95% t-interval across seeds
  rate-normalised exponent     = same fit applied to y / (total spikes), which removes the contribution of
                                  changes in firing rate with N
"""
import glob, json, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = sys.argv[1]
rows, cfg = [], None
for f in glob.glob(os.path.join(D, 'scale_seeds_*.json')):
    J = json.load(open(f)); rows += J['rows']; cfg = J['cfg']
seeds = sorted({r['seed'] for r in rows}); Ns = cfg['Ns']
get = lambda kind, N, k: np.array([r[k] for s in seeds for r in rows if r['seed'] == s and r['N'] == N and r['kind'] == kind], float)


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [float(v.mean()), float('nan')]
    return [float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))]


def slope_per_seed(kind, key, Nset, norm=None):
    out = []
    for s in seeds:
        y = []
        for N in Nset:
            r = [r for r in rows if r['seed'] == s and r['N'] == N and r['kind'] == kind][0]
            v = r[key] / (r[norm] if norm else 1.0)
            y.append(v)
        out.append(np.polyfit(np.log(Nset), np.log(y), 1)[0])
    return ci(out)


S = dict(seeds=seeds, cfg=cfg, table10=[], exponents={})
for N in Ns:
    row = dict(N=N)
    for kind in ['dense', 'sparse']:
        intra, inter = get(kind, N, 'n_intra'), get(kind, N, 'n_inter')
        row[kind] = dict(intra=ci(intra), inter=ci(inter), inputs=ci(get(kind, N, 'n_input')),
                         ratio_of_means=float(inter.mean() / intra.mean()),
                         mean_of_ratios=ci(inter / intra),
                         expected_ratio=float(cfg['K_inter'] * get(kind, N, 'out_degree')[0] /
                                              (cfg['M'] * cfg['p_intra'] * (cfg['M'] - 1))),
                         out_degree=float(get(kind, N, 'out_degree')[0]),
                         rate_hz=ci(get(kind, N, 'rate_hz')), rmse=ci(get(kind, N, 'rmse')))
    S['table10'].append(row)
for rng_name, Nset in [('all', Ns), ('N>=8', [n for n in Ns if n >= 8])]:
    S['exponents'][rng_name] = {
        'intra_dense': slope_per_seed('dense', 'n_intra', Nset),
        'intra_sparse': slope_per_seed('sparse', 'n_intra', Nset),
        'inter_dense': slope_per_seed('dense', 'n_inter', Nset),
        'inter_sparse': slope_per_seed('sparse', 'n_inter', Nset),
        'inter_dense_rate_normalised': slope_per_seed('dense', 'n_inter', Nset, 'n_spk'),
        'inter_sparse_rate_normalised': slope_per_seed('sparse', 'n_inter', Nset, 'n_spk'),
        'spikes_dense': slope_per_seed('dense', 'n_spk', Nset),
        'rate_dense': slope_per_seed('dense', 'rate_hz', Nset),
        'rate_sparse': slope_per_seed('sparse', 'rate_hz', Nset)}
# RMSE trend test: slope of RMSE on log N per seed
S['rmse_trend'] = {k: slope_per_seed(k, 'rmse', Ns) for k in ['dense', 'sparse']}
Nmax = max(Ns)
S['crossing'] = {}
for kind in ['sparse', 'community', 'community_rewired']:
    S['crossing'][kind] = {mp: {P: ci(get(kind, Nmax, f'cross_{mp}_P{P}')) for P in cfg['partitions']}
                           for mp in ('locality', 'roundrobin')}
    S['crossing'][kind]['out_degree'] = float(get(kind, Nmax, 'out_degree')[0])
json.dump(S, open(os.path.join(D, 'summary_scale.json'), 'w'), indent=1)

# ---------------------------------------------------------------- printout
print('seeds', seeds)
for r in S['table10']:
    d, s = r['dense'], r['sparse']
    print(f"N={r['N']:3d} d_out={d['out_degree']:.0f}/{s['out_degree']:.0f} "
          f"intra={d['intra'][0]:.2e}/{s['intra'][0]:.2e} inter={d['inter'][0]:.2e}/{s['inter'][0]:.2e} "
          f"RoM={d['ratio_of_means']:.3f}/{s['ratio_of_means']:.3f} "
          f"MoR={d['mean_of_ratios'][0]:.3f}±{d['mean_of_ratios'][1]:.3f}/{s['mean_of_ratios'][0]:.3f}±{s['mean_of_ratios'][1]:.3f} "
          f"exp={d['expected_ratio']:.3f}/{s['expected_ratio']:.3f} rate={d['rate_hz'][0]:.1f}/{s['rate_hz'][0]:.1f} "
          f"rmse={d['rmse'][0]:.3f}±{d['rmse'][1]:.3f}/{s['rmse'][0]:.3f}±{s['rmse'][1]:.3f}")
print(json.dumps(S['exponents'], indent=1)); print('rmse trend', S['rmse_trend'])
print(json.dumps(S['crossing'], indent=1))

# ---------------------------------------------------------------- Figure 4
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8, 'axes.linewidth': 0.7})
C_D, C_S, C_I = '#c0392b', '#1b6ca8', '#555555'
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0)); plt.subplots_adjust(hspace=0.5, wspace=0.32)
T = S['table10']; NN = np.array(Ns)
E = S['exponents']['all']
ax = axs[0, 0]
for key, kind, c, mk, ls, lab in [
        ('intra', 'dense', C_I, '^', '--', f"Intra-agent, dense ($\\alpha$={E['intra_dense'][0]:.2f})"),
        ('intra', 'sparse', '#999999', 'v', ':', f"Intra-agent, sparse ($\\alpha$={E['intra_sparse'][0]:.2f})"),
        ('inter', 'dense', C_D, 'o', '-', f"Inter-agent, dense ($\\alpha$={E['inter_dense'][0]:.2f})"),
        ('inter', 'sparse', C_S, 's', '-', f"Inter-agent, sparse ($\\alpha$={E['inter_sparse'][0]:.2f})")]:
    m = np.array([r[kind][key][0] for r in T]); h = np.array([r[kind][key][1] for r in T])
    ax.errorbar(NN, m, yerr=h, color=c, marker=mk, ms=4, ls=ls, lw=1.1, capsize=2, label=lab)
ax.set_xscale('log'); ax.set_yscale('log'); ax.minorticks_off()
ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
ax.set_xlabel('Number of agents $N$'); ax.set_ylabel('Synaptic events per trial')
ax.legend(frameon=False, fontsize=6.6, loc='upper left'); ax.set_title('(a) Event counts', loc='left', fontsize=8)

ax = axs[0, 1]
for kind, c, mk, lab in [('dense', C_D, 'o', 'Dense'), ('sparse', C_S, 's', 'Sparse ($d$ = 4)')]:
    m = np.array([r[kind]['mean_of_ratios'][0] for r in T]); h = np.array([r[kind]['mean_of_ratios'][1] for r in T])
    ax.errorbar(NN, m, yerr=h, color=c, marker=mk, ms=4, lw=1.1, capsize=2, label=lab + ', measured')
    ax.plot(NN, [r[kind]['expected_ratio'] for r in T], color=c, lw=3.5, alpha=0.25, zorder=0, label=lab + ', structural')
ax.axhline(1, color='0.6', lw=0.7, ls='--')
ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
ax.set_xlabel('Number of agents $N$'); ax.set_ylabel('Inter- / intra-agent events')
ax.legend(frameon=False, fontsize=6.6, loc='upper left'); ax.set_title('(b) Event ratio', loc='left', fontsize=8)

ax = axs[1, 0]
for kind, c, mk, lab, off in [('dense', C_D, 'o', 'Dense', 0.97), ('sparse', C_S, 's', 'Sparse', 1.03)]:
    m = np.array([r[kind]['rmse'][0] for r in T]); h = np.array([r[kind]['rmse'][1] for r in T])
    ax.errorbar(NN * off, m, yerr=h, color=c, marker=mk, ms=4, lw=1.1, capsize=2, label=lab)
ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
ax.set_ylim(0.10, 0.30)
ax.set_xlabel('Number of agents $N$'); ax.set_ylabel('RMSE of agent 0 vs. $x_B^*$')
ax.legend(frameon=False, fontsize=6.6, loc='upper left'); ax.set_title('(c) Readout error (shared input)', loc='left', fontsize=8)

ax = axs[1, 1]
P = cfg['partitions']
for kind, c, lab in [('sparse', '#555555', 'Random'), ('community', '#1e8449', 'Community'),
                     ('community_rewired', '#b7950b', 'Community, 25% rewired')]:
    for mp, ls, mk in [('locality', '-', 'o'), ('roundrobin', '--', 'v')]:
        m = [S['crossing'][kind][mp][p][0] for p in P]
        ax.plot(P, m, color=c, ls=ls, marker=mk, ms=3.5, lw=1.1,
                label=f"{lab}, {'locality-aware' if mp == 'locality' else 'round-robin'}")
ax.set_xscale('log', base=2); ax.minorticks_off(); ax.set_xticks(P); ax.set_xticklabels([str(p) for p in P])
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel('Number of partitions $P$'); ax.set_ylabel('Fraction of inter-agent events crossing')
ax.legend(frameon=False, fontsize=5.8, loc='lower right')
ax.set_title(f'(d) Partition boundary traffic ($N$ = {Nmax})', loc='left', fontsize=8)
for a in axs.flat:
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
out = os.path.join(D, 'Figure4_measured_scaling.png')
fig.savefig(out, dpi=400, bbox_inches='tight', facecolor='white'); print('saved', out)
