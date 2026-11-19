"""
S3 -- Figure 3 from the outputs of S1/S2.   Usage: python S3_make_figure3.py results/sim
Panels: (a) held-out belief trajectories (seed 0); (b) belief RMSE under event loss (Table 9);
(c) belief RMSE against primitive operations per held-out window (Table 7): filled markers are executed
software counts, the open marker is the projected event-driven count of the spiking model;
(d) mechanism ablations (Table 8). Error bars: 95% confidence intervals over seeds.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

D = sys.argv[1] if len(sys.argv) > 1 else 'results/sim'
S = json.load(open(os.path.join(D, 'summary.json')))
tr = np.load(os.path.join(D, 'traces_seed0.npz'))
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'mathtext.fontset': 'dejavuserif',
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, RED, GRAY, LBLUE, REF = '#1b6ca8', '#c9574b', '#7f8c8d', '#8fb8d6', '#111111'

fig = plt.figure(figsize=(8.4, 6.3))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.55, wspace=0.42,
                      left=0.075, right=0.985, top=0.9, bottom=0.11)

# (a) held-out trajectories, first 8 s of the held-out window of seed 0
ax = fig.add_subplot(gs[0, :])
dt = float(tr['dt']); t0 = float(tr['t0'])
w0, w1 = 260, 260 + int(8 / dt)
t = t0 + np.arange(w0, w1) * dt
ax.plot(t, tr['x_full'][w0:w1], color=REF, lw=1.8, label=r'Reference belief $x_B^*(t)$')
ax.plot(t, tr['x_obs'][w0:w1], color=GRAY, lw=1.2, ls=':', label='Bayesian tracker')
ax.plot(t, tr['rate'][w0:w1], color=RED, lw=1.0, label='Rate baseline')
ax.plot(t, tr['ndt'][w0:w1], color=BLUE, lw=1.2, label='NDT (spiking)')
z = tr['z'][w0:w1]
for s in np.flatnonzero(np.diff(z) != 0):
    ax.axvline(t[s], color='0.75', lw=0.8, zorder=0)
ax.set_xlim(t[0], t[-1]); ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Belief')
ax.legend(ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.99), fontsize=9,
          handlelength=1.8, columnspacing=1.2)
ax.set_title('(a) Held-out belief tracking; vertical lines mark latent switches', loc='left',
             fontsize=10.5, pad=26)


def eb(ax, x, mh, **kw):
    m = np.array([v[0] for v in mh]); h = np.array([v[1] for v in mh])
    ax.errorbar(x, m, yerr=h, capsize=2.5, lw=1.4, ms=4, **kw)


# (b) robustness
ax = fig.add_subplot(gs[1, 0])
drops = [0.0, 0.1, 0.2, 0.4]
for m, c, lab in [('NDT', BLUE, 'NDT'), ('RATE', RED, 'Rate'), ('BAYES', GRAY, 'Bayes')]:
    eb(ax, drops, [S['table9'][m][str(p)] for p in drops], color=c, marker='o', label=lab)
ax.set_xlabel('Event-drop probability'); ax.set_ylabel('Belief RMSE')
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4]); ax.set_ylim(0.08, 0.30)
ax.legend(frameon=False, fontsize=8, loc='upper left', handlelength=1.6)
ax.set_title('(b) Robustness', loc='left', fontsize=10.5)

# (c) accuracy against primitive operations (held-out window)
ax = fig.add_subplot(gs[1, 1])
op = S['operations']['counts_test']
nd, ra = S['table7']['NDT']['rmse_full'], S['table7']['RATE']['rmse_full']
x_ex, x_ev, x_ra = op['NDT_total_executed'][0], op['NDT_total_event_driven_projection'][0], op['RATE_total'][0]
ax.errorbar([x_ex], [nd[0]], yerr=[nd[1]], fmt='o', color=BLUE, ms=6, capsize=3, label='NDT, executed')
ax.errorbar([x_ev], [nd[0]], yerr=[nd[1]], fmt='o', mfc='white', mec=BLUE, color=BLUE, ms=6, capsize=3,
            label='NDT, projected event-driven')
ax.errorbar([x_ra], [ra[0]], yerr=[ra[1]], fmt='s', color=RED, ms=6, capsize=3, label='Rate, executed')
ax.annotate('', xy=(x_ev * 1.15, nd[0]), xytext=(x_ex * 0.89, nd[0]),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.9, ls='--'))
ax.set_xscale('log'); ax.set_xlim(3e6, 4e7); ax.set_ylim(0.10, 0.25)
ax.xaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_major_locator(FixedLocator([5e6, 1e7, 2e7]))
ax.set_xticklabels([r'$5\times10^{6}$', r'$10^{7}$', r'$2\times10^{7}$'])
ax.set_xlabel('Primitive operations per trial'); ax.set_ylabel('Belief RMSE')
ax.legend(frameon=False, fontsize=7.5, loc='lower left', handlelength=1.2)
ax.set_title('(c) Accuracy vs. operation count', loc='left', fontsize=10.5)

# (d) ablations
ax = fig.add_subplot(gs[1, 2])
keys = ['NDT', 'NDT_nodelay', 'NDT_norec']
m = [S['table8'][k]['rmse_full'][0] for k in keys]; h = [S['table8'][k]['rmse_full'][1] for k in keys]
ax.bar(range(3), m, yerr=h, color=[BLUE, LBLUE, LBLUE], edgecolor='black', lw=0.8, capsize=4, width=0.62,
       error_kw={'lw': 1.2})
ax.set_xticks(range(3)); ax.set_xticklabels(['Full\nNDT', 'No\ndelays', 'No\nrecurrence'], fontsize=8.5)
ax.set_ylabel('Belief RMSE'); ax.set_ylim(0, 0.28)
ax.set_title('(d) Mechanism ablations', loc='left', fontsize=10.5)

out = os.path.join(D, 'Figure3.png')
fig.savefig(out, dpi=300, facecolor='white')
print('saved', out)
