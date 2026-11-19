# Unreported analyses (archived)

These files come from an earlier revision of the manuscript and are not reported in the current version.
They are kept for transparency and are not needed to reproduce Tables 7–9 or Figure 3.

* `scripts/S4_scaling_simulation.py`, `scripts/S5_scaling_aggregate_figure4.py`, `results/scale/`:
  multi-agent scaling simulation (up to 64 agents) and its aggregated results. The current manuscript treats
  scaling analytically (Section 6.5).
* `scripts/S6_make_figure2.py`: an earlier two-panel Figure 2 with a measured cost–accuracy panel. The current
  Figure 2 is a qualitative schematic.

The scripts were moved without modification and expect to be run from the repository root with paths
adjusted, e.g. `python archive/unreported_analyses/scripts/S4_scaling_simulation.py 0,1,2,3,4 archive/unreported_analyses/results/scale`.
