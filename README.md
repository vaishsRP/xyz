# Dynamic Discounting and Self-Control Simulation

A small research-style behavioral economics project modeling intertemporal choice under:

- Exponential discounting
- Quasi-hyperbolic (β–δ) discounting

It demonstrates preference reversals and evaluates commitment devices.

## Project structure

- `data/intertemporal_choices.csv` — intertemporal choice menu with immediate and future scenarios.
- `src/dynamic_discounting_simulation.py` — modular simulation, analysis, and plotting pipeline.
- `outputs/` — generated report + metrics (+ plots if matplotlib is available).

## Run

```bash
python src/dynamic_discounting_simulation.py
```

## Notes on dependencies

- The simulation and CSV outputs run with Python standard library only.
- Plot generation uses `matplotlib` when available.
- In restricted environments without `matplotlib`, the script still completes and writes all analysis CSVs and report.

## Expected outputs

- `outputs/beta_sweep_metrics.csv`
- `outputs/reversal_region_metrics.csv`
- `outputs/welfare_comparison.csv`
- `outputs/report.md`
- Plot files (if matplotlib is installed):
  - `outputs/discount_curves.png`
  - `outputs/beta_sweep_outcomes.png`
  - `outputs/reversal_region_heatmap.png`
