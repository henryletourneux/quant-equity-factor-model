# Quantitative Equity Factor Model

A multi-factor stock ranking and portfolio construction model built in Python.
The model scores a universe of large-cap equities on four classic quantitative
factors, combines the scores into a single composite ranking, constructs a
long-only equal-weight portfolio from the top-ranked names, and backtests the
result against the S&P 500.

## Factors

- **Value** — earnings yield (1/P-E) and book-to-price, z-scored and combined.
- **Momentum** — 12-1 month cumulative return (excludes the most recent month
  to control for short-term reversal).
- **Quality** — return on equity, gross margin, and (inverse) debt-to-equity.
- **Low Volatility** — trailing 6-month realized volatility, inverted so
  lower risk scores higher.

Each factor is standardized (z-scored) across the universe before being
combined into a composite score using configurable weights.

## Pipeline

1. Pull adjusted price history and fundamental data for the universe via
   `yfinance`.
2. Compute the four factor scores per ticker.
3. Combine into a composite score and rank the universe.
4. Build an equal-weight portfolio from the top 30% of ranked names.
5. Backtest the portfolio against the S&P 500 (`^GSPC`) and report total
   return, annualized volatility, Sharpe ratio, and max drawdown.

## Sample Results

Backtest over a trailing 2-year window on a 40-stock large-cap universe:

| Metric | Factor Portfolio | S&P 500 |
|---|---|---|
| Total Return | 44.5% | 33.8% |
| Annualized Volatility | 12.6% | 16.1% |
| Sharpe Ratio | 1.54 | 0.99 |
| Max Drawdown | -13.6% | -18.9% |

![Equity curve](equity_curve.png)

Results are generated live from current market data and will differ on each
run — see [Notes](#notes).

## Usage

```bash
pip install -r requirements.txt
python factor_model.py     # runs the full pipeline, saves CSVs
python plot_results.py     # generates equity_curve.png
```

## Notes

- This is a research/educational backtest, not investment advice. It does not
  account for transaction costs, slippage, taxes, or survivorship bias in the
  universe selection.
- The backtest is a single buy-and-hold period (no periodic rebalancing).
  Extending to rolling rebalances is a natural next step.
- Fundamental data availability varies by ticker; names missing sufficient
  data are dropped from the ranking.
