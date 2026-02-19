# Dynamic Discounting and Self-Control: Structured Report

## 1) Dynamic inconsistency
Exponential discounting implies **time-consistent** preferences: if LL is preferred to SS today, it remains preferred when both options are shifted forward equally. In contrast, quasi-hyperbolic discounting introduces present bias via \(\beta < 1\), which disproportionately penalizes delayed payoffs at the moment of choice.

## 2) Why preference reversals emerge under present bias
Preference reversals appear when an agent plans to choose LL in advance but later switches to SS once SS becomes immediate. The simulation explicitly checks this by evaluating each item at:

1. planning date \(t=0\), and
2. a later date when SS delay reaches zero.

Under β–δ discounting, the "now" premium can overturn the earlier LL plan, generating dynamic inconsistency.

## 3) Commitment devices: economic + psychological interpretation
The commitment condition prevents switching after the initial LL plan. This captures classic pre-commitment logic:

- **Economic interpretation:** commitment can raise realized utility by preventing temptation-driven deviations.
- **Psychological interpretation:** commitment acts as a self-control scaffold, aligning short-run actions with long-run goals.

The project computes welfare under both conditions and reports per-item and average utility differences.

## 4) Real-world parallels
- Retirement auto-enrollment and lock-in savings accounts.
- Subscription prepayment (e.g., classes, gyms) to encourage future adherence.
- Penalty-based commitment contracts (late fees, cancellation costs).
- App blockers or spending limits as digital commitment tools.

## 5) Generated artifacts (after running the script)
- `outputs/discount_curves.png`
- `outputs/beta_sweep_outcomes.png`
- `outputs/reversal_region_heatmap.png`
- `outputs/beta_sweep_metrics.csv`
- `outputs/reversal_region_metrics.csv`
- `outputs/welfare_comparison.csv`
