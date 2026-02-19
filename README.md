# Dynamic Discounting and Self-Control Simulation

A lightweight behavioral economics project that simulates intertemporal choice under:

- Exponential discounting
- Quasi-hyperbolic (β–δ) discounting

It demonstrates preference reversals and evaluates commitment devices.

## Project structure

- `data/intertemporal_choices.csv`: choice menu with SS/LL rewards and delays (immediate and future scenarios).
- `src/dynamic_discounting_simulation.py`: modular simulation + analysis pipeline.
- `outputs/`: generated figures, metrics, and markdown report.

## Run

```bash
python src/dynamic_discounting_simulation.py
```

## Outputs

After running, inspect:

- `outputs/discount_curves.png`
- `outputs/beta_sweep_outcomes.png`
- `outputs/reversal_region_heatmap.png`
- `outputs/report.md`

