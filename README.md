# Neuromorphic Digital Twins: supplementary code

Code and results for the manuscript *Neuromorphic Digital Twins of Social Cognition: A Multiscale, Falsifiable
Framework for Event-Driven Cognitive–Behavioral Modeling* (S. M. Hosseini Bamakan, S. Banaeian Far,
A. Zareravasan).

The scripts reproduce every value in Tables 7–9 and Figure 3 (Section 6.4, proof-of-concept instantiation of
the evaluation protocol). Run them from this folder.

Tested with Python 3.13, Brian2 2.11.0 (Cython code generation), NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2.

    pip install -r requirements.txt
    python scripts/S1_ndt_simulation.py 0 10 results/sim      # seeds 0-9, about 100 s per seed
    python scripts/S2_aggregate_statistics.py results/sim     # summary.json, seed_level.csv, prints Tables 7-9
    python scripts/S3_make_figure3.py results/sim             # Figure3.png

S2 and S3 run on the included seed-level results, so the tables and the figure can be checked without
re-running the Brian2 simulation.

## What each script does

| Script | Output | Manuscript |
| --- | --- | --- |
| `S1_ndt_simulation.py` | `seed_<s>.json` per seed, `traces_seed0.npz` | Section 6.4.1 (task, models, readout, robustness sweep, ablations) |
| `S2_aggregate_statistics.py` | `summary.json`, `seed_level.csv`, printed tables | Tables 7, 8 and 9 |
| `S3_make_figure3.py` | `Figure3.png` | Figure 3 |

The model equations, parameters, weight and delay distributions, readout and regularisation settings, metric
definitions and the adaptation criterion are defined in the `CFG` dictionary and the functions of S1.
S2 documents the confidence intervals, paired tests and Holm–Bonferroni families.

## Conventions

* Belief RMSE is computed against the reference posterior x_B*(t). Brier score, negative log-likelihood and
  decision accuracy are computed against the realised binary state. All metrics use the held-out 20 s window
  on a common 5 ms evaluation grid.
* Operation counts in Table 7 refer to the held-out window. Executed counts include membrane, trace, synaptic
  and readout operations. The projected event-driven count omits updates of inactive neurons; it is a
  projection, not a hardware measurement.
* Ablation event counts in Table 8 refer to the full 60 s trial. The "uniform delay" ablation replaces the
  heterogeneous 1–15 ms synaptic delays with a uniform 1 ms delay.

## Archive

`archive/unreported_analyses/` holds analyses from an earlier revision that the current manuscript does not
report: the multi-agent scaling simulation and its results, and an earlier Figure 2 script. They are kept for
transparency only and are not needed to reproduce the manuscript. An output-interval sensitivity analysis and
reliability curves from the same revision were removed from S1/S2; they remain available in commit `cba8c96`.
