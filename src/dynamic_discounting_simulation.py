"""Dynamic discounting simulation for present bias and commitment.

This implementation is intentionally lightweight and dependency-minimal:
- Core simulation/analysis uses Python standard library only.
- Plotting uses matplotlib if available; otherwise plot generation is skipped gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import random
from statistics import mean

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # matplotlib not available in restricted environments
    plt = None


@dataclass(frozen=True)
class ChoiceItem:
    """A single SS-LL intertemporal choice item."""

    item_id: int
    ss_amount: float
    ll_amount: float
    ss_delay_days: int
    ll_delay_days: int
    scenario: str
    source: str


@dataclass(frozen=True)
class ModelParams:
    """Discounting parameters for the agent."""

    beta: float
    delta: float
    risk_aversion: float = 1.0


def utility(amount: float, gamma: float = 1.0) -> float:
    """CRRA utility; gamma=1.0 yields linear utility."""
    if gamma == 1.0:
        return amount
    return (amount ** (1 - gamma)) / (1 - gamma)


def exp_discount(delay_days: int, delta: float) -> float:
    """Exponential discount factor."""
    return delta**delay_days


def qh_discount(delay_days: int, beta: float, delta: float) -> float:
    """Quasi-hyperbolic (beta-delta) discount factor."""
    if delay_days == 0:
        return 1.0
    return beta * (delta**delay_days)


def discounted_utility(amount: float, delay_days: int, params: ModelParams, model: str) -> float:
    """Discounted utility under selected model."""
    base_u = utility(amount, params.risk_aversion)
    if model == "exponential":
        return exp_discount(delay_days, params.delta) * base_u
    if model == "beta_delta":
        return qh_discount(delay_days, params.beta, params.delta) * base_u
    raise ValueError("model must be 'exponential' or 'beta_delta'")


def read_dataset(path: Path) -> list[ChoiceItem]:
    """Read the intertemporal choice dataset from CSV."""
    items: list[ChoiceItem] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(
                ChoiceItem(
                    item_id=int(row["item_id"]),
                    ss_amount=float(row["ss_amount"]),
                    ll_amount=float(row["ll_amount"]),
                    ss_delay_days=int(row["ss_delay_days"]),
                    ll_delay_days=int(row["ll_delay_days"]),
                    scenario=row["scenario"],
                    source=row["source"],
                )
            )
    return items


def evaluate_choice(item: ChoiceItem, params: ModelParams, model: str, decision_time: int) -> dict:
    """Evaluate one item's SS-vs-LL choice at a given decision time."""
    ss_remaining = max(item.ss_delay_days - decision_time, 0)
    ll_remaining = max(item.ll_delay_days - decision_time, 0)

    ss_v = discounted_utility(item.ss_amount, ss_remaining, params, model)
    ll_v = discounted_utility(item.ll_amount, ll_remaining, params, model)

    return {
        "item_id": item.item_id,
        "scenario": item.scenario,
        "decision_time": decision_time,
        "ss_value": ss_v,
        "ll_value": ll_v,
        "choice": "LL" if ll_v >= ss_v else "SS",
    }


def detect_preference_reversal(items: list[ChoiceItem], params: ModelParams, model: str) -> list[dict]:
    """Detect reversals: LL at t=0 then SS when SS becomes immediate."""
    results: list[dict] = []
    for item in items:
        early = evaluate_choice(item, params, model, decision_time=0)
        late = evaluate_choice(item, params, model, decision_time=item.ss_delay_days)
        reversal = early["choice"] == "LL" and late["choice"] == "SS"
        results.append(
            {
                "item_id": item.item_id,
                "scenario": item.scenario,
                "initial_choice": early["choice"],
                "later_choice": late["choice"],
                "reversal": reversal,
            }
        )
    return results


def commitment_welfare(items: list[ChoiceItem], params: ModelParams) -> list[dict]:
    """Compare realized utility with and without commitment (beta-delta behavior)."""
    out: list[dict] = []
    for item in items:
        initial = evaluate_choice(item, params, "beta_delta", decision_time=0)
        late = evaluate_choice(item, params, "beta_delta", decision_time=item.ss_delay_days)

        no_commit_choice = late["choice"]
        with_commit_choice = initial["choice"]

        no_commit_amount = item.ll_amount if no_commit_choice == "LL" else item.ss_amount
        with_commit_amount = item.ll_amount if with_commit_choice == "LL" else item.ss_amount

        no_commit_u = utility(no_commit_amount, params.risk_aversion)
        with_commit_u = utility(with_commit_amount, params.risk_aversion)

        out.append(
            {
                "item_id": item.item_id,
                "scenario": item.scenario,
                "no_commit_choice": no_commit_choice,
                "with_commit_choice": with_commit_choice,
                "no_commit_realized_u": no_commit_u,
                "commit_realized_u": with_commit_u,
                "welfare_gain_commitment": with_commit_u - no_commit_u,
            }
        )
    return out


def sweep_beta(items: list[ChoiceItem], delta: float, beta_grid: list[float]) -> list[dict]:
    """Sweep beta while holding delta fixed."""
    rows: list[dict] = []
    for beta in beta_grid:
        params = ModelParams(beta=beta, delta=delta)
        rev = detect_preference_reversal(items, params, "beta_delta")
        welfare = commitment_welfare(items, params)
        rows.append(
            {
                "beta": beta,
                "reversal_rate": mean(1.0 if r["reversal"] else 0.0 for r in rev),
                "avg_welfare_gain": mean(w["welfare_gain_commitment"] for w in welfare),
            }
        )
    return rows


def reversal_region(items: list[ChoiceItem], beta_grid: list[float], delta_grid: list[float]) -> list[dict]:
    """Compute reversal region in beta-delta parameter space."""
    rows: list[dict] = []
    for beta in beta_grid:
        for delta in delta_grid:
            params = ModelParams(beta=beta, delta=delta)
            rev = detect_preference_reversal(items, params, "beta_delta")
            share = mean(1.0 if r["reversal"] else 0.0 for r in rev)
            rows.append(
                {
                    "beta": beta,
                    "delta": delta,
                    "has_reversal": int(share > 0),
                    "reversal_share": share,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_discount_curves(output_dir: Path, beta: float, delta: float) -> bool:
    """Create discount curve plot if matplotlib exists."""
    if plt is None:
        return False
    t = list(range(0, 181))
    exp_curve = [exp_discount(x, delta) for x in t]
    qh_curve = [qh_discount(x, beta, delta) for x in t]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(t, exp_curve, label=f"Exponential (δ={delta:.3f})", linewidth=2)
    ax.plot(t, qh_curve, label=f"β–δ (β={beta:.2f}, δ={delta:.3f})", linewidth=2)
    ax.set_xlabel("Delay (days)")
    ax.set_ylabel("Discount factor")
    ax.set_title("Discount Curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "discount_curves.png", dpi=160)
    plt.close(fig)
    return True


def plot_beta_sweep(output_dir: Path, sweep_rows: list[dict]) -> bool:
    """Plot reversal rate and welfare gain vs beta."""
    if plt is None:
        return False
    betas = [r["beta"] for r in sweep_rows]
    reversal_rates = [r["reversal_rate"] for r in sweep_rows]
    welfare_gains = [r["avg_welfare_gain"] for r in sweep_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(betas, reversal_rates, color="tab:red", linewidth=2)
    axes[0].set_title("Preference Reversals vs β")
    axes[0].set_xlabel("β")
    axes[0].set_ylabel("Reversal rate")
    axes[0].grid(alpha=0.25)

    axes[1].plot(betas, welfare_gains, color="tab:blue", linewidth=2)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Commitment Welfare Gain vs β")
    axes[1].set_xlabel("β")
    axes[1].set_ylabel("Average utility gain")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "beta_sweep_outcomes.png", dpi=160)
    plt.close(fig)
    return True


def plot_reversal_region(output_dir: Path, region_rows: list[dict], beta_grid: list[float], delta_grid: list[float]) -> bool:
    """Plot heatmap of reversal share in parameter space."""
    if plt is None:
        return False

    matrix: list[list[float]] = []
    for delta in sorted(delta_grid, reverse=True):
        row = []
        for beta in beta_grid:
            match = next(r for r in region_rows if math.isclose(r["beta"], beta) and math.isclose(r["delta"], delta))
            row.append(match["reversal_share"])
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(beta_grid)))
    ax.set_xticklabels([f"{b:.2f}" for b in beta_grid], rotation=45)
    ax.set_yticks(range(len(delta_grid)))
    ax.set_yticklabels([f"{d:.3f}" for d in sorted(delta_grid, reverse=True)])
    ax.set_xlabel("β")
    ax.set_ylabel("δ")
    ax.set_title("Reversal Share in Parameter Space")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Share of items with reversal")
    fig.tight_layout()
    fig.savefig(output_dir / "reversal_region_heatmap.png", dpi=160)
    plt.close(fig)
    return True


def generate_report(output_dir: Path, params: ModelParams, sweep_rows: list[dict], welfare_rows: list[dict], plots_generated: bool) -> None:
    """Write structured markdown interpretation report."""
    avg_gain = mean(r["welfare_gain_commitment"] for r in welfare_rows)
    gain_share = mean(1.0 if r["welfare_gain_commitment"] > 0 else 0.0 for r in welfare_rows)
    peak = max(sweep_rows, key=lambda r: r["reversal_rate"])

    plot_status = "generated" if plots_generated else "not generated (matplotlib unavailable in this environment)"

    report = f"""# Dynamic Discounting and Self-Control: A Simulation of Present Bias and Commitment

## 1) Dynamic inconsistency
Under exponential discounting, relative preferences are time-consistent when both options are shifted forward together.
Under quasi-hyperbolic discounting, \\(\\beta < 1\\) creates present bias: delayed outcomes get an extra short-run penalty, so planned choices can differ from later choices.

## 2) Why preference reversals emerge
Preference reversals occur when LL is chosen at planning (t=0) but SS is chosen later when SS becomes immediate.
This happens because the immediate period receives special weight under β–δ preferences.
In this run (δ={params.delta:.3f}), the highest reversal rate appears around β={peak['beta']:.2f} with reversal rate {peak['reversal_rate']:.2%}.

## 3) Commitment devices
A commitment device forces execution of the initial plan and blocks later switching.
Average realized utility gain from commitment is {avg_gain:.2f}, with positive gains in {gain_share:.1%} of items.
Economically, commitment protects long-run welfare from short-run temptation.
Psychologically, it externalizes self-control.

## 4) Real-world parallels
- Savings and retirement lock-ins.
- Subscription prepayment and cancellation frictions.
- Penalty-backed commitment contracts.
- App blockers or spending controls for digital self-regulation.

## 5) Output artifacts
- `outputs/beta_sweep_metrics.csv`
- `outputs/reversal_region_metrics.csv`
- `outputs/welfare_comparison.csv`
- Plots status: **{plot_status}**
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Run full simulation pipeline."""
    random.seed(42)

    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "intertemporal_choices.csv"
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    items = read_dataset(data_path)
    baseline_params = ModelParams(beta=0.72, delta=0.995)

    beta_grid = [0.40 + 0.02 * i for i in range(31)]
    delta_grid = [0.985 + 0.001 * i for i in range(15)]

    sweep_rows = sweep_beta(items, delta=baseline_params.delta, beta_grid=beta_grid)
    region_rows = reversal_region(items, beta_grid=beta_grid, delta_grid=delta_grid)
    welfare_rows = commitment_welfare(items, baseline_params)

    write_csv(output_dir / "beta_sweep_metrics.csv", sweep_rows)
    write_csv(output_dir / "reversal_region_metrics.csv", region_rows)
    write_csv(output_dir / "welfare_comparison.csv", welfare_rows)

    generated = []
    generated.append(plot_discount_curves(output_dir, baseline_params.beta, baseline_params.delta))
    generated.append(plot_beta_sweep(output_dir, sweep_rows))
    generated.append(plot_reversal_region(output_dir, region_rows, beta_grid, delta_grid))

    generate_report(output_dir, baseline_params, sweep_rows, welfare_rows, plots_generated=all(generated))


if __name__ == "__main__":
    main()
