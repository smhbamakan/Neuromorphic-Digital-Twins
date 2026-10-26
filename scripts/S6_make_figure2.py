"""
S6 -- Figure 2. (a) Conceptual latency-power plane relative to a matched conventional baseline.
(b) Measured cost-accuracy trade-off of Section 6.4 (values read from results/sim/summary.json).
Usage: python S6_make_figure2.py results/sim
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

D = sys.argv[1] if len(sys.argv) > 1 else 'results/sim'
S = json.load(open(os.path.join(D, 'summary.json')))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.linewidth': 0.7,
                     'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'mathtext.fontset': 'dejavusans'})
C_NDT, C_RATE, C_OBS, C_DOM = '#1b6ca8', '#c0392b', '#6b7280', '#2e8b57'

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.15), gridspec_kw=dict(wspace=0.32))

# ---------------- (a) conceptual plane, log axes without numerical ticks
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
bx0, by0 = 6.2, 6.0                                     # matched conventional baseline
ax.add_patch(Rectangle((0, 0), bx0, by0, fc=C_DOM, alpha=0.10, ec='none'))
ax.plot([bx0, bx0], [0, by0], color=C_DOM, lw=0.8, ls='--')
ax.plot([0, bx0], [by0, by0], color=C_DOM, lw=0.8, ls='--')
ax.text(0.35, 0.45, 'Pareto improvement over the baseline\n(lower latency and power at equal accuracy)',
        color=C_DOM, fontsize=6.8, va='bottom')
ax.add_patch(Ellipse((7.1, 6.8), 3.6, 2.8, fc=C_RATE, alpha=0.12, ec=C_RATE, lw=0.9))
ax.text(7.3, 8.35, 'Frame-synchronous CPU/GPU', ha='center', color=C_RATE, fontsize=7)
ax.plot(bx0, by0, 'o', color=C_RATE, ms=6, mec='white', mew=1.0, zorder=5)
ax.text(bx0 + 0.3, by0 + 0.05, 'matched\nbaseline', fontsize=6.8, color=C_RATE, va='center')
ax.add_patch(Ellipse((3.1, 3.4), 3.6, 3.0, fc=C_NDT, alpha=0.10, ec=C_NDT, lw=0.9, ls=(0, (4, 2))))
ax.text(3.1, 3.55, 'Event-driven\nneuromorphic\n(hypothesized)', ha='center', color=C_NDT, fontsize=7)
ax.annotate('', xy=(4.4, 4.55), xytext=(5.85, 5.7),
            arrowprops=dict(arrowstyle='-|>', color='0.25', lw=0.9, shrinkA=0, shrinkB=0))
ax.text(0.35, 6.55, 'hypothesized shift: requires matched\nmeasurement and is task-dependent', fontsize=6.3, color='0.25', ha='left')
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('End-to-end latency (log scale)')
ax.set_ylabel('Power (log scale)')
ax.text(0.35, 9.75, 'Accuracy is the third objective (panel b)', ha='left', va='top', fontsize=6.5, color='0.35')
ax.set_title('(a) Conceptual design space', loc='left', fontsize=8.5)

# ---------------- (b) measured cost-accuracy trade-off
op = S['operations']['counts_test']
r_ndt = S['table7']['NDT']['rmse_full']; r_rate = S['table7']['RATE']['rmse_full']
r_obs = S['table7']['BAYES']['rmse_full']
x_rate = op['RATE_total'][0]; x_ex = op['NDT_total_executed'][0]; x_ev = op['NDT_total_event_driven_projection'][0]
bx.set_xscale('log')
bx.add_patch(Rectangle((1e6, 0.05), x_rate - 1e6, r_rate[0] - 0.05, fc=C_DOM, alpha=0.10, ec='none'))
bx.axvline(x_rate, color=C_DOM, lw=0.8, ls='--', ymax=(r_rate[0] - 0.05) / (0.25 - 0.05))
bx.text(1.25e6, 0.058, 'would dominate\nthe rate baseline', color=C_DOM, fontsize=6.8)
bx.axhline(r_obs[0], color=C_OBS, lw=0.9, ls=':')
bx.text(1.25e6, r_obs[0] + 0.004, 'observer-posterior bound', color=C_OBS, fontsize=6.8)
bx.errorbar([x_rate], [r_rate[0]], yerr=[[r_rate[1]], [r_rate[1]]], fmt='s', color=C_RATE, ms=6,
            capsize=2.5, mec='white', mew=0.8, zorder=5)
bx.errorbar([x_ex], [r_ndt[0]], yerr=[[r_ndt[1]], [r_ndt[1]]], fmt='o', color=C_NDT, ms=6,
            capsize=2.5, mec='white', mew=0.8, zorder=5)
bx.errorbar([x_ev], [r_ndt[0]], yerr=[[r_ndt[1]], [r_ndt[1]]], fmt='o', color=C_NDT, ms=6, mfc='white',
            capsize=2.5, zorder=5)
bx.annotate('', xy=(x_ev * 1.12, r_ndt[0]), xytext=(x_ex * 0.88, r_ndt[0]),
            arrowprops=dict(arrowstyle='-|>', color=C_NDT, lw=0.8, ls='--'))
bx.text(x_rate * 1.15, r_rate[0] - 0.004, 'Rate baseline\n(executed)', color=C_RATE, fontsize=6.8, va='top')
bx.text(x_ex * 1.15, r_ndt[0] + 0.006, 'NDT\n(executed)', color=C_NDT, fontsize=6.8)
bx.text(x_ev, r_ndt[0] + 0.012, 'NDT, event-driven\n(projected, not executed)', color=C_NDT, fontsize=6.8,
        ha='center')
bx.set_xlim(1e6, 1e8); bx.set_ylim(0.05, 0.25)
bx.set_xlabel('Total operations in held-out window')
bx.set_ylabel(r'RMSE vs. $x_B^{*}$ (lower is better)')
bx.set_title('(b) Measured trade-off (Section 6.4)', loc='left', fontsize=8.5)
for a in (ax, bx):
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
out = 'Figure2_design_space.png'
fig.savefig(out, dpi=400, bbox_inches='tight', facecolor='white')
print('saved', out)
