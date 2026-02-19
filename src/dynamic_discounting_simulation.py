"""Simulation project for dynamic discounting, present bias, and commitment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelParams:
    """Container for discounting parameters."""

    beta: float
    delta: float
    risk_aversion: float = 1.0


def utility(amount: np.ndarray | float, gamma: float = 1.0) -> np.ndarray | float:
    """CRRA utility with gamma=1 interpreted as linear utility."""
    amount = np.asarray(amount)
    if gamma == 1.0:
        return amount
    return np.power(amount, 1 - gamma) / (1 - gamma)


def exp_discount(delay_days: np.ndarray | float, delta: float) -> np.ndarray | float:
    """Exponential discount factor using daily delay and daily delta."""
    return np.power(delta, np.asarray(delay_days))


def qh_discount(delay_days: np.ndarray | float, beta: float, delta: float) -> np.ndarray | float:
    """Quasi-hyperbolic (beta-delta) discount factor."""
    delay = np.asarray(delay_days)
    return np.where(delay == 0, 1.0, beta * np.power(delta, delay))


def discounted_utility(amount: np.ndarray, delay: np.ndarray, params: ModelParams, model: str) -> np.ndarray:
    """Compute discounted utility under exponential or quasi-hyperbolic discounting."""
    base_u = utility(amount, params.risk_aversion)
    if model == "exponential":
        discount = exp_discount(delay, params.delta)
    elif model == "beta_delta":
        discount = qh_discount(delay, params.beta, params.delta)
    else:
        raise ValueError("model must be 'exponential' or 'beta_delta'")
    return discount * base_u


def evaluate_choices(df: pd.DataFrame, params: ModelParams, model: str, decision_time: int) -> pd.DataFrame:
    """Evaluate SS-vs-LL choices at a specific decision time."""
    out = df.copy()
    ss_remaining = np.maximum(out["ss_delay_days"].to_numpy() - decision_time, 0)
    ll_remaining = np.maximum(out["ll_delay_days"].to_numpy() - decision_time, 0)

    ss_v = discounted_utility(out["ss_amount"].to_numpy(), ss_remaining, params, model)
    ll_v = discounted_utility(out["ll_amount"].to_numpy(), ll_remaining, params, model)

    out["decision_time"] = decision_time
    out["ss_value"] = ss_v
    out["ll_value"] = ll_v
    out["choice"] = np.where(ll_v >= ss_v, "LL", "SS")
    return out


def detect_preference_reversal(df: pd.DataFrame, params: ModelParams, model: str) -> pd.DataFrame:
    """Detect reversal comparing choices at t=0 and when SS becomes immediate."""
    early = evaluate_choices(df, params, model, decision_time=0).set_index("item_id")
    rows = []
    for _, row in df.iterrows():
        t_switch = int(row["ss_delay_days"])
        late = evaluate_choices(pd.DataFrame([row]), params, model, decision_time=t_switch).iloc[0]
        initial_choice = early.loc[row["item_id"], "choice"]
        reversal = (initial_choice == "LL") and (late["choice"] == "SS")
        rows.append(
            {
                "item_id": row["item_id"],
                "scenario": row["scenario"],
                "initial_choice": initial_choice,
                "later_choice": late["choice"],
                "reversal": reversal,
            }
        )
    return pd.DataFrame(rows)


def commitment_welfare(df: pd.DataFrame, params: ModelParams) -> pd.DataFrame:
    """Compare realized utility with and without a commitment device under beta-delta preferences."""
    records = []
    initial = evaluate_choices(df, params, model="beta_delta", decision_time=0).set_index("item_id")
    for _, row in df.iterrows():
        item = int(row["item_id"])
        t_switch = int(row["ss_delay_days"])
        later = evaluate_choices(pd.DataFrame([row]), params, model="beta_delta", decision_time=t_switch).iloc[0]

        no_commit_choice = later["choice"]
        with_commit_choice = initial.loc[item, "choice"]

        no_commit_payoff = row["ll_amount"] if no_commit_choice == "LL" else row["ss_amount"]
        commit_payoff = row["ll_amount"] if with_commit_choice == "LL" else row["ss_amount"]

        records.append(
            {
                "item_id": item,
                "scenario": row["scenario"],
                "no_commit_choice": no_commit_choice,
                "with_commit_choice": with_commit_choice,
                "no_commit_realized_u": utility(no_commit_payoff, params.risk_aversion),
                "commit_realized_u": utility(commit_payoff, params.risk_aversion),
            }
        )
    out = pd.DataFrame(records)
    out["welfare_gain_commitment"] = out["commit_realized_u"] - out["no_commit_realized_u"]
    return out


def sweep_beta(df: pd.DataFrame, delta: float, beta_grid: np.ndarray) -> pd.DataFrame:
    """Sweep beta and track reversal rates and average welfare gains from commitment."""
    rows = []
    for beta in beta_grid:
        params = ModelParams(beta=beta, delta=delta)
        rev = detect_preference_reversal(df, params, "beta_delta")
        welfare = commitment_welfare(df, params)
        rows.append(
            {
                "beta": beta,
                "reversal_rate": rev["reversal"].mean(),
                "avg_welfare_gain": welfare["welfare_gain_commitment"].mean(),
            }
        )
    return pd.DataFrame(rows)


def reversal_region(df: pd.DataFrame, beta_grid: np.ndarray, delta_grid: np.ndarray) -> pd.DataFrame:
    """Map parameter pairs where at least one reversal occurs."""
    rows = []
    for beta in beta_grid:
        for delta in delta_grid:
            params = ModelParams(beta=beta, delta=delta)
            rev = detect_preference_reversal(df, params, "beta_delta")
            rows.append(
                {
                    "beta": beta,
                    "delta": delta,
                    "has_reversal": int(rev["reversal"].any()),
                    "reversal_share": rev["reversal"].mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_discount_curves(output_dir: Path, beta: float, delta: float) -> None:
    """Plot discount-factor curves."""
    t = np.arange(0, 181)
    exp_curve = exp_discount(t, delta)
    qh_curve = qh_discount(t, beta, delta)

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


def plot_beta_sweep(output_dir: Path, sweep_df: pd.DataFrame) -> None:
    """Plot reversal rates and welfare effects versus beta."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].plot(sweep_df["beta"], sweep_df["reversal_rate"], color="tab:red", linewidth=2)
    axes[0].set_title("Preference Reversals vs β")
    axes[0].set_xlabel("β")
    axes[0].set_ylabel("Reversal rate")
    axes[0].grid(alpha=0.25)

    axes[1].plot(sweep_df["beta"], sweep_df["avg_welfare_gain"], color="tab:blue", linewidth=2)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Commitment Welfare Gain vs β")
    axes[1].set_xlabel("β")
    axes[1].set_ylabel("Average utility gain")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "beta_sweep_outcomes.png", dpi=160)
    plt.close(fig)


def plot_reversal_region(output_dir: Path, region_df: pd.DataFrame, beta_grid: np.ndarray, delta_grid: np.ndarray) -> None:
    """Plot parameter region where preference reversals occur."""
    pivot = region_df.pivot(index="delta", columns="beta", values="reversal_share").sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(beta_grid)))
    ax.set_xticklabels([f"{b:.2f}" for b in beta_grid], rotation=45)
    ax.set_yticks(np.arange(len(delta_grid)))
    ax.set_yticklabels([f"{d:.3f}" for d in delta_grid[::-1]])
    ax.set_xlabel("β")
    ax.set_ylabel("δ")
    ax.set_title("Reversal Share in Parameter Space")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Share of items with reversal")
    fig.tight_layout()
    fig.savefig(output_dir / "reversal_region_heatmap.png", dpi=160)
    plt.close(fig)


def generate_report(output_dir: Path, params: ModelParams, sweep_df: pd.DataFrame, welfare_df: pd.DataFrame) -> None:
    """Create structured markdown report with interpretation."""
    avg_gain = welfare_df["welfare_gain_commitment"].mean()
    gain_share = (welfare_df["welfare_gain_commitment"] > 0).mean()
    peak_rev = sweep_df.loc[sweep_df["reversal_rate"].idxmax(), ["beta", "reversal_rate"]]

    report = f"""# Dynamic Discounting and Self-Control

## 1) Dynamic inconsistency
Under exponential discounting, relative preferences between two dated options depend only on the time gap, so plans are time-consistent. Under quasi-hyperbolic discounting, \(\beta < 1\) creates an extra penalty on all delayed outcomes, making the immediate period special and generating a mismatch between earlier plans and later actions.

## 2) Why preference reversals emerge with present bias
In the simulation, many agents choose Larger–Later (LL) at the planning date but switch to Smaller–Sooner (SS) when SS becomes immediate. The reversal mechanism is the present-bias wedge: once "now" arrives, delayed utility is multiplied by \(\beta\), reducing LL's attractiveness. Reversal prevalence peaks at β={peak_rev['beta']:.2f} with a reversal rate of {peak_rev['reversal_rate']:.2%} (δ fixed at {params.delta:.3f}).

## 3) Commitment devices: economic and psychological interpretation
A commitment device locks in the ex-ante LL plan and blocks later switching. In this run, average realized utility gain from commitment is {avg_gain:.2f}, and commitment strictly improves outcomes in {gain_share:.1%} of choice items. Economically, commitment mitigates self-control costs. Psychologically, it acts as a pre-commitment strategy that protects long-run goals from short-run temptation.

## 4) Real-world parallels
- **Savings and retirement:** automatic enrollment and withdrawal penalties prevent impulsive present spending.
- **Subscriptions and prepayment:** paying upfront for a gym or class raises follow-through.
- **Deadlines and penalties:** commitment contracts convert intentions into enforceable actions.
- **Digital self-control:** app blockers and spending locks reduce immediate temptation.

## 5) Artifacts
- `outputs/discount_curves.png`
- `outputs/beta_sweep_outcomes.png`
- `outputs/reversal_region_heatmap.png`
- `outputs/beta_sweep_metrics.csv`
- `outputs/welfare_comparison.csv`
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    """Run project pipeline end-to-end."""
    np.random.seed(42)

    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "intertemporal_choices.csv"
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    baseline_params = ModelParams(beta=0.72, delta=0.995)

    beta_grid = np.linspace(0.4, 1.0, 31)
    delta_grid = np.linspace(0.985, 0.999, 15)

    sweep_df = sweep_beta(df, delta=baseline_params.delta, beta_grid=beta_grid)
    region_df = reversal_region(df, beta_grid=beta_grid, delta_grid=delta_grid)
    welfare_df = commitment_welfare(df, baseline_params)

    sweep_df.to_csv(output_dir / "beta_sweep_metrics.csv", index=False)
    region_df.to_csv(output_dir / "reversal_region_metrics.csv", index=False)
    welfare_df.to_csv(output_dir / "welfare_comparison.csv", index=False)

    plot_discount_curves(output_dir, baseline_params.beta, baseline_params.delta)
    plot_beta_sweep(output_dir, sweep_df)
    plot_reversal_region(output_dir, region_df, beta_grid, delta_grid)
    generate_report(output_dir, baseline_params, sweep_df, welfare_df)


if __name__ == "__main__":
    main()
