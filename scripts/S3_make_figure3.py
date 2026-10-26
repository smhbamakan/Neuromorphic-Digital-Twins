"""
S3 -- Figure 3 from the outputs of S1/S2.   Usage: python S3_make_figure3.py results/sim
Panels: (a) held-out trajectories (seed 0); (b) reliability against the binary state; (c) robustness;
(d) output-interval sensitivity; (e) mechanism ablations. Error bars: 95% CI over seeds.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = sys.argv[1]
S = json.load(open(os.path.join(D, 'summary.json')))
tr = np.load(os.path.join(D, 'traces_seed0.npz'))
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8, 'axes.linewidth': 0.7,
                     'xtick.major.width': 0.7, 'ytick.major.width': 0.7})
C_NDT, C_RATE, C_BAY, C_REF = '#1b6ca8', '#c0392b', '#7f8c8d', '#111111'

fig = plt.figure(figsize=(7.2, 8.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.1, 1.1], hspace=0.55, wspace=0.32)

# (a) trajectories
ax = fig.add_subplot(gs[0, :])
dt = float(tr['dt']); t0 = float(tr['t0'])
w0, w1 = 260, 260 + int(8 / dt)
t = t0 + np.arange(w0, w1) * dt
ax.plot(t, tr['x_full'][w0:w1], color=C_REF, lw=1.5, label=r'Reference posterior $x_B^*(t)$')
ax.plot(t, tr['x_obs'][w0:w1], color=C_BAY, lw=1.0, ls=':', label=r'Observer posterior $x_{obs}(t)$')
ax.plot(t, tr['rate'][w0:w1], color=C_RATE, lw=0.9, alpha=0.85, label='Rate baseline')
ax.plot(t, tr['ndt'][w0:w1], color=C_NDT, lw=1.0, label='NDT (spiking)')
z = tr['z'][w0:w1]
for s in np.flatnonzero(np.diff(z) != 0):
    ax.axvline(t[s], color='0.78', lw=0.6, zorder=0)
ax.set_xlabel('Time (s)'); ax.set_ylabel('P(z = +1)')
ax.set_ylim(-0.05, 1.05); ax.set_xlim(t[0], t[-1])
ax.legend(ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7,
          handlelength=1.6, columnspacing=1.0)
ax.set_title('(a) Held-out estimates, seed 0; grey lines mark switches of the binary state',
             loc='left', fontsize=8, pad=22)


def errbar(ax, x, mh, **kw):
    m = np.array([v[0] for v in mh]); h = np.array([v[1] for v in mh])
    ax.errorbar(x, m, yerr=h, capsize=2, ms=3.5, lw=1.1, **kw)


# (b) reliability (pooled bins over seeds)
ax = fig.add_subplot(gs[1, 0])
ax.plot([0, 1], [0, 1], color='0.6', lw=0.7, ls='--')
for m, c, mk, lab in [('NDT', C_NDT, 'o', 'NDT'), ('RATE', C_RATE, 's', 'Rate'), ('BAYES', C_BAY, '^', 'Observer')]:
    curves = np.array(S['reliability'][m], dtype=float)          # seeds x bins x (p, freq, n)
    n = np.nansum(curves[:, :, 2], axis=0)
    p = np.nansum(np.nan_to_num(curves[:, :, 0]) * curves[:, :, 2], axis=0) / np.where(n > 0, n, 1)
    f = np.nansum(np.nan_to_num(curves[:, :, 1]) * curves[:, :, 2], axis=0) / np.where(n > 0, n, 1)
    k = n > 50
    ax.plot(p[k], f[k], color=c, marker=mk, ms=3.5, lw=1.0, label=lab)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel('Predicted P(z = +1)'); ax.set_ylabel('Observed frequency')
ax.legend(frameon=False, fontsize=6.5, handlelength=1.3, loc='upper left')
ax.set_title('(b) Reliability', loc='left', fontsize=8, pad=4)

# (c) robustness
ax = fig.add_subplot(gs[1, 1])
drops = [0.0, 0.1, 0.2, 0.4]
for m, c, mk, lab in [('NDT', C_NDT, 'o', 'NDT'), ('RATE', C_RATE, 's', 'Rate'), ('BAYES', C_BAY, '^', 'Observer')]:
    errbar(ax, drops, [S['table9'][m][str(p)] for p in drops], color=c, marker=mk, label=lab)
ax.set_xlabel('Event-loss probability'); ax.set_ylabel('RMSE vs. $x_B^*$')
ax.set_xticks(drops)
ax.legend(frameon=False, fontsize=6.5, handlelength=1.3, loc='upper left')
ax.set_title('(c) Event loss', loc='left', fontsize=8, pad=4)

# (d) interval sensitivity
ax = fig.add_subplot(gs[2, 0])
iv = [0.5, 1.0, 2.0, 5.0, 10.0]
I = S['interval_sensitivity']
errbar(ax, iv, [I[str(i)]['ndt_rmse'] for i in iv], color=C_NDT, marker='o', label='NDT')
errbar(ax, iv, [I[str(i)]['rate_rmse'] for i in iv], color=C_RATE, marker='s', label='Rate (fixed)')
errbar(ax, iv, [I[str(i)]['rate_tuned_rmse'] for i in iv], color=C_RATE, marker='D', ls='--',
       mfc='white', label='Rate (re-tuned)')
ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(iv); ax.set_xticklabels(['0.5', '1', '2', '5', '10'])
ax.set_xlabel('Output interval (ms)'); ax.set_ylabel('RMSE vs. $x_B^*$')
ax.legend(frameon=False, fontsize=6.5, handlelength=1.6, loc='lower left')
ax.set_title('(d) Output interval', loc='left', fontsize=8, pad=4)

# (e) ablations
ax = fig.add_subplot(gs[2, 1])
keys = ['NDT', 'NDT_nodelay', 'NDT_norec']
m = [S['table8'][k]['rmse_full'][0] for k in keys]; h = [S['table8'][k]['rmse_full'][1] for k in keys]
ax.bar(range(3), m, yerr=h, capsize=3, width=0.62, color=[C_NDT, '#8fb8d4', '#8fb8d4'],
       edgecolor='black', lw=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(['Full', 'Uniform\n1 ms delay', 'No\nrecurrence'], fontsize=6.8)
ax.set_ylabel('RMSE vs. $x_B^*$'); ax.set_ylim(0, 0.28)
ax.set_title('(e) Ablations', loc='left', fontsize=8, pad=4)

for a in fig.axes:
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
out = os.path.join(D, 'Figure3_minimal_simulation.png')
fig.savefig(out, dpi=400, bbox_inches='tight', facecolor='white')
print('saved', out)
