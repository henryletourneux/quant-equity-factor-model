"""Chart the rolling rebalanced equity curve vs. benchmark, and each factor's IC."""

import pandas as pd
import matplotlib.pyplot as plt

curve = pd.read_csv("rolling_equity_curve.csv", index_col=0, parse_dates=True)
ic = pd.read_csv("ic_by_period.csv")
ic_summary = pd.read_csv("ic_summary.csv", index_col=0).sort_values("mean_ic")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.bar(
    ic_summary.index,
    ic_summary["mean_ic"],
    yerr=ic_summary["std_ic"] / (ic_summary["n_periods"] ** 0.5),
    color=["tab:red" if v < 0 else "tab:green" for v in ic_summary["mean_ic"]],
    capsize=4,
)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Mean Information Coefficient by Factor\n(monthly rank IC, ±1 SE)")
ax.set_ylabel("Mean IC")
ax.tick_params(axis="x", rotation=30)
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ax.plot(curve.index, curve["rolling_factor_portfolio"], label="Rolling Factor Portfolio", linewidth=2)
ax.plot(curve.index, curve["benchmark"], label="S&P 500 (Benchmark)", linewidth=2, linestyle="--")
ax.set_title("Monthly-Rebalanced Backtest vs. S&P 500")
ax.set_ylabel("Growth of $1")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("rolling_analysis.png", dpi=150)
print("Saved rolling_analysis.png")
