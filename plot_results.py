"""Generate an equity-curve chart comparing the factor portfolio to the benchmark."""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("equity_curve.csv", index_col=0, parse_dates=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df.index, df["factor_portfolio"], label="Factor Portfolio", linewidth=2)
ax.plot(df.index, df["benchmark"], label="S&P 500 (Benchmark)", linewidth=2, linestyle="--")
ax.set_title("Quantitative Equity Factor Model vs. S&P 500")
ax.set_ylabel("Growth of $1")
ax.set_xlabel("Date")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("equity_curve.png", dpi=150)
print("Saved equity_curve.png")
