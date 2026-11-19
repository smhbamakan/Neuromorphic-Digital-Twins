"""
S1 -- Two-agent belief-tracking experiment: proof-of-concept instantiation of the evaluation
      protocol (Section 6.4; Tables 7-9 and Figure 3).

Usage:   python S1_ndt_simulation.py <first_seed> <last_seed_exclusive> <out_dir>
Example: python S1_ndt_simulation.py 0 10 results/sim

Task (Section 6.3.7)
  z(t)        symmetric telegraph process in {-1,+1}, switching hazard h            (binary environmental state)
  events      two Poisson channels, rates r_hi / r_lo depending on z(t)
  x_full(t)   exact posterior P(z=+1 | all events up to t)  -- agent B's belief    (full-information reference)
  x_obs(t)    Bayesian belief tracker: exact filter applied to the events received by the observer
              (20 ms delay; i.i.d. losses in the robustness analysis)              (normative reference)
  Primary estimation target of every model: x_full(t).

Models (identical degraded input stream, identical targets, identical data split, identical ridge readout)
  NDT    recurrent LIF network (Brian2), heterogeneous synaptic delays, fixed weights, linear readout
  RATE   frame-synchronous leaky-tanh recurrent network (echo-state type), fixed weights, linear readout
  BAYES  exact observer posterior x_obs(t) (normative reference, not trained)
All recurrent weights are fixed after initialisation; only the linear readout is fitted (ridge regression,
first 40 s). Predictions in the held-out 20 s are produced with frozen readout weights; no online learning.

Ablation 'NDT_nodelay' replaces the heterogeneous 1-15 ms synaptic delays by a uniform 1 ms delay;
'NDT_norec' removes the recurrent synapses.

Outputs per seed: seed_<s>.json (all metrics, operation counts, censoring info) and, for seed 0,
traces_seed0.npz (held-out trajectories used in Figure 3a).
"""

import json, os, sys, time
import numpy as np
from brian2 import (prefs, start_scope, NeuronGroup, Synapses, SpikeGeneratorGroup,
                    SpikeMonitor, StateMonitor, run, defaultclock, ms, second)
from brian2 import seed as b2seed

prefs.codegen.target = 'cython'

CFG = dict(
    T_total=60.0, T_train=40.0,   # s
    dt_sim=0.5,                   # ms, integration step of the spiking simulator
    dt_eval=5.0,                  # ms, common evaluation grid and rate-model frame period
    hazard=0.5, r_hi=40.0, r_lo=10.0,
    delay_ab=20.0, p_drop=0.0,
    drops=[0.0, 0.1, 0.2, 0.4],
    N=200, p_conn=0.10, g=1.6, tau_m=20.0, tau_d=50.0,
    d_min=1.0, d_max=15.0, w_in=1.30, I_mu=0.55, I_sd=0.10, t_ref=2.0,
    # rate baseline
    rate_win=0.8, rate_rho=0.92, rate_tau=-5.0 / np.log(1 - 0.3),  # ms; equals leak 0.3 at a 5 ms frame
    ridge=1e-3, p_clip=1e-4,
    eps0=0.10, T_hold=100.0, T_max_adapt=3.0,   # adaptation criterion, Eq. (20)
)


# ------------------------------------------------------------------ task
def generate_task(cfg, rng):
    dt = cfg['dt_sim'] * 1e-3
    n = int(round(cfg['T_total'] / dt))
    z = np.empty(n, np.int8)
    cur = rng.choice([-1, 1])
    sw = rng.random(n) < cfg['hazard'] * dt
    for k in range(n):
        if sw[k]:
            cur = -cur
        z[k] = cur
    r0 = np.where(z > 0, cfg['r_hi'], cfg['r_lo'])
    r1 = np.where(z > 0, cfg['r_lo'], cfg['r_hi'])
    ev0 = rng.random(n) < r0 * dt
    ev1 = rng.random(n) < r1 * dt
    x_full = posterior(ev0, ev1, cfg, dt)
    return dict(dt=dt, n=n, z=z, ev0=ev0, ev1=ev1, x_full=x_full)


def posterior(e0, e1, cfg, dt):
    """Exact filter (Wonham) in log-odds form. For symmetric channels the drift terms cancel, and i.i.d.
    thinning with probability p multiplies both rates by (1-p), leaving the jump size log(r_hi/r_lo)
    unchanged; the same recursion is therefore exact for a lossy stream."""
    gain = np.log(cfg['r_hi'] / cfg['r_lo'])
    h2 = 2 * cfg['hazard'] * dt
    L = np.zeros(e0.size); x = 0.0
    for k in range(e0.size):
        x -= h2 * np.sinh(np.clip(x, -20, 20))
        if e0[k]:
            x += gain
        if e1[k]:
            x -= gain
        L[k] = x
    return 1 / (1 + np.exp(-np.clip(L, -30, 30)))


def degrade(task, cfg, rng, p_drop):
    shift = int(round(cfg['delay_ab'] * 1e-3 / task['dt']))
    out = []
    for ev in (task['ev0'], task['ev1']):
        idx = np.flatnonzero(ev)
        idx = idx[rng.random(idx.size) >= p_drop] + shift
        out.append(idx[idx < task['n']])
    return out


def observer_posterior(task, cfg, i0, i1):
    """Continuous-time Bayesian belief tracker: the exact (Wonham-type) filter applied to the event
    stream received by the observer, i.e. the posterior computable from the events received so far."""
    e0 = np.zeros(task['n'], bool); e0[i0] = True
    e1 = np.zeros(task['n'], bool); e1[i1] = True
    return posterior(e0, e1, cfg, task['dt'])


# ------------------------------------------------------------------ NDT
def run_ndt(task, cfg, rng, i0, i1, use_delays=True, use_recurrence=True, net_seed=0):
    start_scope(); b2seed(net_seed)
    defaultclock.dt = cfg['dt_sim'] * ms
    N = cfg['N']
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
    G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0; r += 1',
                    refractory=cfg['t_ref'] * ms, method='exact')
    G.tau = cfg['tau_m'] * ms; G.tau_r = cfg['tau_d'] * ms
    G.v = rng.random(N)
    G.I = cfg['I_mu'] + cfg['I_sd'] * rng.standard_normal(N)
    Sin = Synapses(inp, G, 'w:1', on_pre='v_post += w'); Sin.connect(p=1.0)
    Sin.w = cfg['w_in'] * rng.standard_normal(2 * N)
    outdeg = np.zeros(N)
    if use_recurrence:
        S = Synapses(G, G, 'w:1', on_pre='v_post += w')
        S.connect(p=cfg['p_conn'], condition='i!=j')
        k = len(S.w)
        S.w = cfg['g'] / np.sqrt(cfg['p_conn'] * N) * rng.standard_normal(k)
        S.delay = ((cfg['d_min'] + (cfg['d_max'] - cfg['d_min']) * rng.random(k)) if use_delays
                   else np.full(k, cfg['d_min'])) * ms
        outdeg = np.bincount(np.asarray(S.i), minlength=N).astype(float)
    spk = SpikeMonitor(G)
    mon = StateMonitor(G, 'r', record=True, dt=cfg['dt_sim'] * ms)
    t0 = time.time(); run(cfg['T_total'] * second); wall = time.time() - t0
    si = np.asarray(spk.i); st = np.asarray(spk.t / second)
    return dict(R=np.asarray(mon.r).T, wall=wall, spk_i=si, spk_t=st, outdeg=outdeg, N=N)


def ndt_counts(nd, i0, i1, task, cfg, t_from, t_to):
    """Event counts of the executed spiking model inside [t_from, t_to) seconds."""
    N = nd['N']; dt = task['dt']
    m = (nd['spk_t'] >= t_from) & (nd['spk_t'] < t_to)
    spikes = int(m.sum())
    rec = int(nd['outdeg'][nd['spk_i'][m]].sum())                 # recurrent synaptic events (exact)
    n_in = int(sum(((idx * dt >= t_from) & (idx * dt < t_to)).sum() for idx in (i0, i1)))
    steps = int(round((t_to - t_from) / dt))
    has_rec = nd['outdeg'].sum() > 0
    return dict(spikes=spikes, syn_rec=rec, syn_in=n_in * N, input_events=n_in,
                syn_rec_meanfield=float(spikes * (N - 1) * cfg['p_conn']) if has_rec else 0.0,
                neuron_updates=N * steps,      # clock-driven membrane updates
                trace_updates=N * steps)       # clock-driven readout-trace updates


# ------------------------------------------------------------------ rate baseline
def rate_weights(cfg, rng):
    N = cfg['N']
    Win = cfg['rate_win'] * rng.standard_normal((N, 2))
    W = rng.standard_normal((N, N)) * (rng.random((N, N)) < cfg['p_conn'])
    W /= np.max(np.abs(np.linalg.eigvals(W))) + 1e-9
    W *= cfg['rate_rho']
    return Win, W


def run_rate(task, cfg, Win, W, i0, i1, frame_ms):
    N = cfg['N']
    step = int(round(frame_ms / cfg['dt_sim']))
    nf = task['n'] // step
    u = np.zeros((nf, 2))
    np.add.at(u[:, 0], np.clip(i0 // step, 0, nf - 1), 1.0)
    np.add.at(u[:, 1], np.clip(i1 // step, 0, nf - 1), 1.0)
    leak = 1 - np.exp(-frame_ms / cfg['rate_tau'])
    X = np.zeros((nf, N)); s = np.zeros(N)
    t0 = time.time()
    for k in range(nf):
        s = (1 - leak) * s + leak * np.tanh(W @ s + Win @ u[k])
        X[k] = s
    return dict(R=X, wall=time.time() - t0, step=step, nnz=int(np.count_nonzero(W)), nf=nf)


def rate_counts(nnz, N, frames, input_events):
    return dict(mac_rec=frames * nnz, mac_rec_dense=frames * N * N, mac_in=frames * 2 * N,
                unit_updates=frames * N, encoding_ops=input_events)


# ------------------------------------------------------------------ readout and metrics
def fit_ridge(R, y, n_tr, cfg):
    Phi = np.hstack([R[:n_tr], np.ones((n_tr, 1))])
    return np.linalg.solve(Phi.T @ Phi + cfg['ridge'] * n_tr * np.eye(Phi.shape[1]), Phi.T @ y[:n_tr])


def predict(R, W):
    """Linear readout. Returns the raw output and the probability obtained by clipping to
    [p_clip, 1 - p_clip]; the clipped value is used for every probabilistic metric."""
    raw = np.hstack([R, np.ones((R.shape[0], 1))]) @ W
    return raw


def clip(raw, cfg):
    return np.clip(raw, cfg['p_clip'], 1 - cfg['p_clip'])


def adaptation(err, z, dt_s, cfg):
    hold = max(1, int(round(cfg['T_hold'] * 1e-3 / dt_s)))
    lim = int(round(cfg['T_max_adapt'] / dt_s))
    ok = err <= cfg['eps0']
    sw = np.flatnonzero(np.diff(z) != 0) + 1
    times, cens = [], 0
    for s in sw:
        if s + lim + hold > len(ok):
            continue                      # switch too close to the end of the window: not evaluable
        hit = None
        for k in range(s, s + lim):
            if ok[k:k + hold].all():
                hit = (k - s) * dt_s; break
        if hit is None:
            cens += 1
        else:
            times.append(hit)
    n = len(times) + cens
    return dict(t_adapt=float(np.median(times)) if times else np.nan,   # median over re-acquired switches
                n_switch=n, n_censored=cens)


def metrics(raw, x_full, x_obs, z, dt_s, cfg):
    """All metrics on the held-out window, computed against the reference posterior x_full(t).
    raw: readout output; targets on the same grid."""
    p = clip(raw, cfg)
    out = dict(
        rmse_full=float(np.sqrt(np.mean((p - x_full) ** 2))),
        rmse_obs=float(np.sqrt(np.mean((p - x_obs) ** 2))),
        brier=float(np.mean((p - x_full) ** 2)),
        logloss=float(-np.mean(x_full * np.log(p) + (1 - x_full) * np.log(1 - p))),
        accuracy=float(np.mean((p > 0.5) == (x_full > 0.5))),
        frac_clipped=float(np.mean((raw < cfg['p_clip']) | (raw > 1 - cfg['p_clip']))),
    )
    out.update(adaptation(np.abs(p - x_full), z, dt_s, cfg))
    return out


def on_grid(a, step):
    return a[::step]


# ------------------------------------------------------------------ one seed
def one_seed(seed, cfg, out_dir):
    rng = np.random.default_rng(seed)
    task = generate_task(cfg, rng)
    n, dt = task['n'], task['dt']
    n_tr_f = int(round(cfg['T_train'] / dt))
    i0, i1 = degrade(task, cfg, rng, cfg['p_drop'])
    x_obs = observer_posterior(task, cfg, i0, i1)
    NSEED = seed + 1000
    net = lambda: np.random.default_rng(NSEED)
    se = int(round(cfg['dt_eval'] / cfg['dt_sim']))
    res = dict(seed=seed, cfg={k: v for k, v in cfg.items()})

    def eval_on(step, raw):
        sl = slice(n_tr_f // step, None)
        return metrics(raw[sl], on_grid(task['x_full'], step)[sl], on_grid(x_obs, step)[sl],
                       on_grid(task['z'], step)[sl], step * dt, cfg)

    # ---- NDT, full model
    nd = run_ndt(task, cfg, net(), i0, i1, net_seed=NSEED)
    Rn = nd['R']
    Wn = fit_ridge(on_grid(Rn, se), on_grid(task['x_full'], se), n_tr_f // se, cfg)
    res['NDT'] = eval_on(se, predict(on_grid(Rn, se), Wn))
    res['NDT']['counts_full'] = ndt_counts(nd, i0, i1, task, cfg, 0, cfg['T_total'])
    res['NDT']['counts_test'] = ndt_counts(nd, i0, i1, task, cfg, cfg['T_train'], cfg['T_total'])
    res['NDT']['wall'] = nd['wall']
    res['NDT']['rate_hz'] = nd['spk_i'].size / (cfg['N'] * cfg['T_total'])

    # ---- RATE baseline at the matched interval
    Win, Wr = rate_weights(cfg, net())
    rr = run_rate(task, cfg, Win, Wr, i0, i1, cfg['dt_eval'])
    We = fit_ridge(rr['R'], on_grid(task['x_full'], se), n_tr_f // se, cfg)
    res['RATE'] = eval_on(se, predict(rr['R'], We))
    nev = lambda a, b: int(sum(((idx * dt >= a) & (idx * dt < b)).sum() for idx in (i0, i1)))
    res['RATE']['counts_full'] = rate_counts(rr['nnz'], cfg['N'], rr['nf'], nev(0, cfg['T_total']))
    nf_te = rr['nf'] - n_tr_f // se
    res['RATE']['counts_test'] = rate_counts(rr['nnz'], cfg['N'], nf_te, nev(cfg['T_train'], cfg['T_total']))
    res['RATE']['wall'] = rr['wall']

    # ---- Bayesian observer posterior
    res['BAYES'] = eval_on(se, on_grid(x_obs, se))

    # ---- ablations (same task, same weights where applicable, readout refitted)
    for tag, kw in [('NDT_nodelay', dict(use_delays=False)), ('NDT_norec', dict(use_recurrence=False))]:
        a = run_ndt(task, cfg, net(), i0, i1, net_seed=NSEED, **kw)
        Wa = fit_ridge(on_grid(a['R'], se), on_grid(task['x_full'], se), n_tr_f // se, cfg)
        res[tag] = eval_on(se, predict(on_grid(a['R'], se), Wa))
        res[tag]['counts_full'] = ndt_counts(a, i0, i1, task, cfg, 0, cfg['T_total'])
        res[tag]['counts_test'] = ndt_counts(a, i0, i1, task, cfg, cfg['T_train'], cfg['T_total'])

    # ---- robustness: readouts frozen at the nominal condition; nested drop sets (common random numbers)
    rob = {}
    for pd in cfg['drops']:
        j0, j1 = degrade(task, cfg, np.random.default_rng(seed + 777), pd)
        xo = observer_posterior(task, cfg, j0, j1)
        ndd = run_ndt(task, cfg, net(), j0, j1, net_seed=NSEED)
        r1 = eval_on(se, predict(on_grid(ndd['R'], se), Wn))
        r2 = eval_on(se, predict(run_rate(task, cfg, Win, Wr, j0, j1, cfg['dt_eval'])['R'], We))
        r3 = eval_on(se, on_grid(xo, se))
        for nm, r in (('NDT', r1), ('RATE', r2), ('BAYES', r3)):
            rob[f'{nm}_{pd}'] = {k: r[k] for k in ('rmse_full', 'rmse_obs', 'brier', 'logloss', 'accuracy')}
    res['robustness'] = rob

    # ---- held-out traces for Figure 3a (seed 0 only)
    if seed == 0:
        sl = slice(n_tr_f // se, None)
        np.savez_compressed(os.path.join(out_dir, 'traces_seed0.npz'),
                            dt=se * dt, t0=cfg['T_train'],
                            x_full=on_grid(task['x_full'], se)[sl], x_obs=on_grid(x_obs, se)[sl],
                            z=on_grid(task['z'], se)[sl],
                            ndt=clip(predict(on_grid(Rn, se), Wn), cfg)[sl],
                            rate=clip(predict(rr['R'], We), cfg)[sl])
    return res


if __name__ == '__main__':
    lo, hi, out_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    for s in range(lo, hi):
        t0 = time.time()
        r = one_seed(s, CFG, out_dir)
        with open(os.path.join(out_dir, f'seed_{s}.json'), 'w') as f:
            json.dump(r, f, default=float)
        print(f'seed {s} done in {time.time() - t0:.0f}s', flush=True)
