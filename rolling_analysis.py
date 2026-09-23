"""
Rolling Rebalancing & Factor Information Coefficient Analysis
----------------------------------------------------------------
Extends factor_model.py's single buy-and-hold backtest into a monthly
rebalanced backtest, and measures each factor's predictive power directly
via the Information Coefficient (IC): the cross-sectional rank correlation
between a factor's score at time t and each stock's forward return to t+1.

Momentum and Low-Volatility are recomputed at every rebalance date from
price history available up to that date only (no look-ahead). Value and
Quality are computed once from current fundamentals and held static across
the backtest window, since point-in-time historical fundamentals aren't
available from this data source -- see the README for why that's a
reasonable simplification here and what it costs in realism.
"""

import numpy as np
import pandas as pd

from factor_model import (
    UNIVERSE, BENCHMARK, fetch_price_history, fetch_fundamentals,
    zscore, momentum_factor, volatility_factor, value_factor, quality_factor,
)

ROLLING_LOOKBACK_PERIOD = "4y"
MOMENTUM_LOOKBACK_MONTHS = 12
VOL_WINDOW_DAYS = 126
TOP_DECILE = 0.3
FACTOR_WEIGHTS = {"value": 0.25, "momentum": 0.25, "quality": 0.25, "low_vol": 0.25}


def rebalance_dates(prices, momentum_lookback=MOMENTUM_LOOKBACK_MONTHS):
    """Actual last-trading-day-of-month dates (not calendar month-end, which may
    not be a trading day), starting once enough history exists for the
    momentum lookback."""
    last_trading_day_per_month = prices.index.to_series().resample("ME").last()
    return pd.DatetimeIndex(last_trading_day_per_month.values[momentum_lookback + 1:])


def composite_at_date(price_history, static_value, static_quality, weights=FACTOR_WEIGHTS):
    mom = momentum_factor(price_history, skip_months=1, lookback_months=MOMENTUM_LOOKBACK_MONTHS)
    vol = volatility_factor(price_history, window_days=VOL_WINDOW_DAYS)

    common = (
        mom.dropna().index
        .intersection(vol.dropna().index)
        .intersection(static_value.dropna().index)
        .intersection(static_quality.dropna().index)
    )
    if len(common) < 5:
        return pd.DataFrame()

    scores = pd.DataFrame({
        "value_z": zscore(static_value.loc[common]),
        "momentum_z": zscore(mom.loc[common]),
        "quality_z": zscore(static_quality.loc[common]),
        "low_vol_z": zscore(-vol.loc[common]),
    })
    scores["composite"] = (
        weights["value"] * scores["value_z"]
        + weights["momentum"] * scores["momentum_z"]
        + weights["quality"] * scores["quality_z"]
        + weights["low_vol"] * scores["low_vol_z"]
    )
    return scores


def rolling_backtest_and_ic(prices, fundamentals, top_decile=TOP_DECILE):
    static_value = value_factor(fundamentals)
    static_quality = quality_factor(fundamentals)

    dates = rebalance_dates(prices)
    portfolio_returns, ic_records, holdings_log = [], [], []

    for t0, t1 in zip(dates[:-1], dates[1:]):
        price_hist = prices.loc[:t0]
        scores = composite_at_date(price_hist, static_value, static_quality)
        if scores.empty:
            continue

        tickers = scores.index
        fwd_ret = prices.loc[t1, tickers] / prices.loc[t0, tickers] - 1

        for factor in ["value_z", "momentum_z", "quality_z", "low_vol_z", "composite"]:
            ic = scores[factor].corr(fwd_ret, method="spearman")
            ic_records.append({"date": t0, "factor": factor, "ic": ic})

        n = max(1, int(len(scores) * top_decile))
        top = scores.sort_values("composite", ascending=False).head(n).index
        port_ret = fwd_ret.loc[top].mean()
        portfolio_returns.append({"date": t1, "return": port_ret})
        holdings_log.append({"date": t0, "holdings": ",".join(top)})

    return (
        pd.DataFrame(portfolio_returns).set_index("date"),
        pd.DataFrame(ic_records),
        pd.DataFrame(holdings_log),
    )


def build_equity_curves(portfolio_returns, holdings_log, benchmark_prices):
    """Chain per-period portfolio returns into a cumulative curve, and compare
    against the benchmark's price ratio over the same rebalance dates."""
    start_date = pd.to_datetime(holdings_log["date"].iloc[0])
    dates = [start_date] + list(portfolio_returns.index)

    port_curve = pd.concat([
        pd.Series([1.0], index=[start_date]),
        (1 + portfolio_returns["return"]).cumprod(),
    ])

    bench_at_dates = benchmark_prices.reindex(dates, method="ffill")
    bench_curve = bench_at_dates / bench_at_dates.iloc[0]

    return pd.DataFrame({"rolling_factor_portfolio": port_curve, "benchmark": bench_curve})


def summarize_ic(ic_df):
    grouped = ic_df.groupby("factor")["ic"]
    summary = pd.DataFrame({
        "mean_ic": grouped.mean(),
        "std_ic": grouped.std(),
        "hit_rate": grouped.apply(lambda s: (s > 0).mean()),
        "n_periods": grouped.count(),
    })
    summary["information_ratio"] = summary["mean_ic"] / summary["std_ic"]
    return summary.sort_values("mean_ic", ascending=False)


def performance_metrics(curve_df):
    total_return = curve_df.iloc[-1] / curve_df.iloc[0] - 1
    period_ret = curve_df.pct_change().dropna()
    # ~monthly rebalancing -> annualize with 12 periods/year
    ann_vol = period_ret.std() * np.sqrt(12)
    sharpe = (period_ret.mean() * 12) / ann_vol
    max_dd = (curve_df / curve_df.cummax() - 1).min()
    return pd.DataFrame({
        "Total Return": total_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    })


def run(universe=UNIVERSE, benchmark=BENCHMARK):
    print(f"Fetching {ROLLING_LOOKBACK_PERIOD} of price history for {len(universe)} tickers...")
    prices = fetch_price_history(universe, period=ROLLING_LOOKBACK_PERIOD)
    valid = list(prices.columns)

    print("Fetching fundamentals...")
    fundamentals = fetch_fundamentals(valid).dropna(thresh=3)
    common = [t for t in valid if t in fundamentals.index]
    prices = prices[common]

    print("Running monthly rolling rebalance + IC analysis...")
    portfolio_returns, ic_df, holdings_log = rolling_backtest_and_ic(prices, fundamentals.loc[common])
    print(f"Completed {len(portfolio_returns)} rebalance periods.")

    ic_summary = summarize_ic(ic_df)
    print("\nFactor Information Coefficient summary (monthly rebalance):")
    print(ic_summary.round(4))

    bench_prices = fetch_price_history([benchmark], period=ROLLING_LOOKBACK_PERIOD)[benchmark]
    curve = build_equity_curves(portfolio_returns, holdings_log, bench_prices)
    metrics = performance_metrics(curve)

    print("\nRolling backtest performance vs. buy-and-hold benchmark:")
    print(metrics.round(4))

    ic_df.to_csv("ic_by_period.csv", index=False)
    ic_summary.round(4).to_csv("ic_summary.csv")
    curve.to_csv("rolling_equity_curve.csv")
    metrics.round(4).to_csv("rolling_performance_metrics.csv")
    holdings_log.to_csv("rolling_holdings_log.csv", index=False)

    print("\nSaved: ic_by_period.csv, ic_summary.csv, rolling_equity_curve.csv, "
          "rolling_performance_metrics.csv, rolling_holdings_log.csv")

    return ic_summary, curve, metrics


if __name__ == "__main__":
    run()
