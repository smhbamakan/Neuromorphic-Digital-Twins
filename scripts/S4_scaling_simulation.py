"""
S4 -- Measured scaling of the multi-agent NDT simulation (Section 6.5.5), revision 3.

Usage:   python S4_scaling_simulation.py <comma-separated seeds> <out_dir>
Example: python S4_scaling_simulation.py 0,1,2,3,4 results/scale

Each agent is a recurrent LIF population of M neurons (same neuron model and parameters as S1).
All agents receive the SAME delayed, lossy evidence stream directly; one designated agent (agent 0)
carries the trained linear readout. Inter-agent coupling therefore does not carry task information,
and the readout error characterises a shared-input simulation only.

Social graphs (directed, no self-loops, no repeated agent pairs):
  dense      every agent projects to every other agent: out-degree N-1
  sparse     random d-out-regular graph: each agent projects to min(d, N-1) distinct agents drawn
             uniformly without replacement. For N <= d+1 this graph is identical to the dense graph.
  community  agents in contiguous communities of size C; each agent projects to the next
             min(d, C-1) agents of its own community (ring order). With rewiring probability q,
             each edge is independently redirected to a uniformly drawn agent outside the community.
Each directed agent pair carries K distinct synapses (source and target neurons drawn without replacement
from the M x M possible pairs); every inter-agent synapse has the 20 ms transmission delay.

Partitioning of agents over P partitions (software proxy for core/chip placement):
  locality-aware  contiguous blocks of N/P agents (aligned with communities when C divides N/P)
  round-robin     agent a -> partition a mod P
A crossing event is an inter-agent synaptic event whose source and target agents lie in different partitions.

Counting: synaptic events = sum over neurons of (spike count x out-degree), computed separately for
intra-agent and inter-agent synapses; input synaptic events are counted separately.
"""

import json, os, sys, time
import numpy as np
from brian2 import (prefs, start_scope, NeuronGroup, Synapses, SpikeGeneratorGroup,
                    SpikeMonitor, StateMonitor, run, defaultclock, ms, second)
from brian2 import seed as b2seed

prefs.codegen.target = 'cython'

CFG = dict(
    T_total=30.0, T_train=20.0, dt_sim=0.5, dt_out=5.0,
    hazard=0.5, r_hi=40.0, r_lo=10.0, delay_ab=20.0, p_drop=0.10,
    Ns=[2, 4, 8, 16, 32, 64],
    M=100, p_intra=0.10, K_inter=50, d_degree=4, community=8, rewire=0.25,
    g=1.6, g_inter=0.9, tau_m=20.0, tau_d=50.0, t_ref=2.0,
    d_min=1.0, d_max=15.0, w_in=1.30, I_mu=0.55, I_sd=0.10,
    ridge=1e-3, partitions=[2, 4, 8, 16],
)


def generate_task(cfg, rng):
    dt = cfg['dt_sim'] * 1e-3
    n = int(round(cfg['T_total'] / dt))
    z = np.empty(n, np.int8); cur = rng.choice([-1, 1])
    sw = rng.random(n) < cfg['hazard'] * dt
    for k in range(n):
        if sw[k]:
            cur = -cur
        z[k] = cur
    r0 = np.where(z > 0, cfg['r_hi'], cfg['r_lo']); r1 = np.where(z > 0, cfg['r_lo'], cfg['r_hi'])
    ev0 = rng.random(n) < r0 * dt; ev1 = rng.random(n) < r1 * dt
    gain = np.log(cfg['r_hi'] / cfg['r_lo']); h2 = 2 * cfg['hazard'] * dt
    L = np.zeros(n); x = 0.0
    for k in range(n):
        x -= h2 * np.sinh(np.clip(x, -20, 20))
        if ev0[k]: x += gain
        if ev1[k]: x -= gain
        L[k] = x
    return dict(dt=dt, n=n, z=z, ev0=ev0, ev1=ev1, x_ref=1 / (1 + np.exp(-np.clip(L, -30, 30))))


def social_graph(N, cfg, grng, kind):
    d = min(cfg['d_degree'], N - 1)
    E = set()
    if kind == 'dense':
        E = {(a, b) for a in range(N) for b in range(N) if a != b}
    elif kind == 'sparse':
        for a in range(N):
            for b in grng.choice([b for b in range(N) if b != a], size=d, replace=False):
                E.add((a, int(b)))
    else:                                    # community, optionally rewired
        C = min(cfg['community'], N)
        q = cfg['rewire'] if kind == 'community_rewired' else 0.0
        dc = min(cfg['d_degree'], C - 1)
        for a in range(N):
            base = (a // C) * C
            for k in range(1, dc + 1):
                b = base + ((a - base + k) % C)
                if q > 0 and grng.random() < q and N > C:
                    outside = [x for x in range(N) if x // C != a // C and (a, x) not in E]
                    b = int(grng.choice(outside))
                E.add((a, b))
    return sorted(E)


def run_network(N, cfg, seed, kind, task):
    rng = np.random.default_rng(seed + 1000)          # neuronal parameters and weights
    grng = np.random.default_rng(seed + 5000)         # social graph and synapse placement
    M = cfg['M']
    start_scope(); b2seed(seed + 1000); defaultclock.dt = cfg['dt_sim'] * ms
    shift = int(round(cfg['delay_ab'] * 1e-3 / task['dt']))
    drng = np.random.default_rng(seed + 2000)
    i0 = np.flatnonzero(task['ev0']); i1 = np.flatnonzero(task['ev1'])
    i0 = i0[drng.random(i0.size) >= cfg['p_drop']] + shift; i1 = i1[drng.random(i1.size) >= cfg['p_drop']] + shift
    i0 = i0[i0 < task['n']]; i1 = i1[i1 < task['n']]
    times = np.concatenate([i0, i1]) * task['dt']
    ids = np.concatenate([np.zeros(i0.size, int), np.ones(i1.size, int)])
    o = np.argsort(times, kind='stable')
    inp = SpikeGeneratorGroup(2, ids[o], times[o] * second)
    eqs = '''
    dv/dt = (-v + I)/tau : 1 (unless refractory)
    dr/dt = -r/tau_r     : 1
    I : 1
    tau : second
    tau_r : second
    '''
    G = NeuronGroup(N * M, eqs, threshold='v>1', reset='v=0; r += 1',
                    refractory=cfg['t_ref'] * ms, method='exact')
    G.tau = cfg['tau_m'] * ms; G.tau_r = cfg['tau_d'] * ms
    G.v = rng.random(N * M); G.I = cfg['I_mu'] + cfg['I_sd'] * rng.standard_normal(N * M)
    Sin = Synapses(inp, G, 'w:1', on_pre='v_post += w'); Sin.connect(p=1.0)
    Sin.w = cfg['w_in'] * rng.standard_normal(2 * N * M)
    Sr = Synapses(G, G, 'w:1', on_pre='v_post += w')
    Sr.connect(condition='i!=j and int(i/M_) == int(j/M_)', p=cfg['p_intra'], namespace={'M_': M})
    k = len(Sr.w)
    Sr.w = cfg['g'] / np.sqrt(cfg['p_intra'] * M) * rng.standard_normal(k)
    Sr.delay = (cfg['d_min'] + (cfg['d_max'] - cfg['d_min']) * rng.random(k)) * ms
    edges = social_graph(N, cfg, grng, kind)
    src, tgt, eid = [], [], []
    for e, (a, b) in enumerate(edges):
        # placement and weights depend only on (seed, a, b): identical graphs give identical networks
        pairs = np.random.default_rng([seed, 7, a, b]).choice(M * M, size=cfg['K_inter'], replace=False)
        src.append(a * M + pairs // M); tgt.append(b * M + pairs % M); eid.append(np.full(cfg['K_inter'], e))
    src = np.concatenate(src); tgt = np.concatenate(tgt)
    Se = Synapses(G, G, 'w:1', on_pre='v_post += w'); Se.connect(i=src, j=tgt)
    Se.w = cfg['g_inter'] / np.sqrt(cfg['K_inter']) * np.concatenate(
        [np.random.default_rng([seed, 13, a, b]).standard_normal(cfg['K_inter']) for (a, b) in edges])
    Se.delay = cfg['delay_ab'] * ms
    spk = SpikeMonitor(G)
    mon = StateMonitor(G, 'r', record=np.arange(M), dt=cfg['dt_out'] * ms)
    t0 = time.time(); run(cfg['T_total'] * second); wall = time.time() - t0
    counts = np.bincount(np.asarray(spk.i), minlength=N * M)
    n_intra = int(counts @ np.bincount(np.asarray(Sr.i), minlength=N * M))
    n_inter = int(counts @ np.bincount(src, minlength=N * M))
    return dict(R=np.asarray(mon.r).T, wall=wall, counts=counts, src=src, tgt=tgt,
                n_intra=n_intra, n_inter=n_inter, n_input=int((i0.size + i1.size) * N * M),
                n_spk=int(counts.sum()), rate_hz=float(counts.sum() / (N * M * cfg['T_total'])),
                n_edges=len(edges), out_degree=len(edges) / N, n_syn_inter=int(src.size), n_syn_intra=int(k))


def readout_rmse(R, x_out, cfg):
    m = min(R.shape[0], len(x_out)); R = R[:m]; y = x_out[:m]
    n_tr = int(round(cfg['T_train'] / cfg['T_total'] * m))
    Phi = np.hstack([R, np.ones((m, 1))]); A = Phi[:n_tr]
    W = np.linalg.solve(A.T @ A + cfg['ridge'] * n_tr * np.eye(A.shape[1]), A.T @ y[:n_tr])
    p = np.clip(Phi @ W, 1e-4, 1 - 1e-4)
    return float(np.sqrt(np.mean((p[n_tr:] - y[n_tr:]) ** 2)))


def crossing(res, N, M, P, mapping):
    agent = np.arange(N * M) // M
    part = (np.arange(N) * P // N) if mapping == 'locality' else (np.arange(N) % P)
    ev = res['counts'][res['src']].astype(float)
    cross = part[agent[res['src']]] != part[agent[res['tgt']]]
    return float(ev[cross].sum() / ev.sum()) if ev.sum() else 0.0


def main(seeds, out):
    cfg = CFG; rows = []
    os.makedirs(out, exist_ok=True)
    for seed in seeds:
        task = generate_task(cfg, np.random.default_rng(seed))
        x_out = task['x_ref'][::int(round(cfg['dt_out'] / cfg['dt_sim']))]
        for N in cfg['Ns']:
            kinds = ['dense', 'sparse'] + (['community', 'community_rewired'] if N == max(cfg['Ns']) else [])
            for kind in kinds:
                t0 = time.time()
                r = run_network(N, cfg, seed, kind, task)
                rec = {k: r[k] for k in ('n_intra', 'n_inter', 'n_input', 'n_spk', 'rate_hz', 'n_edges',
                                         'out_degree', 'n_syn_inter', 'n_syn_intra', 'wall')}
                rec.update(seed=seed, N=N, kind=kind, rmse=readout_rmse(r['R'], x_out, cfg),
                           comm_ratio=r['n_inter'] / max(r['n_intra'], 1))
                if N == max(cfg['Ns']) and kind != 'dense':
                    for P in cfg['partitions']:
                        for mp in ('locality', 'roundrobin'):
                            rec[f'cross_{mp}_P{P}'] = crossing(r, N, cfg['M'], P, mp)
                rows.append(rec)
                print(f"seed{seed} N={N:3d} {kind:17s} intra={r['n_intra']:.3e} inter={r['n_inter']:.3e} "
                      f"rate={r['rate_hz']:.1f}Hz rmse={rec['rmse']:.3f} ({time.time()-t0:.0f}s)", flush=True)
                json.dump(dict(cfg=cfg, rows=rows), open(os.path.join(out, f'scale_seeds_{"_".join(map(str, seeds))}.json'), 'w'))
    print('done')


if __name__ == '__main__':
    main([int(s) for s in sys.argv[1].split(',')], sys.argv[2])
