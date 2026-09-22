"""
Quantitative Equity Factor Model
---------------------------------
Ranks a universe of equities on four classic factors (Value, Momentum,
Quality, Low Volatility), combines them into a single composite score,
builds a long-only top-decile portfolio, and backtests it against a
benchmark index.

Data source: Yahoo Finance (via yfinance).
"""

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V",
    "UNH", "HD", "PG", "MA", "XOM", "CVX", "KO", "PEP", "MRK", "ABBV",
    "COST", "WMT", "BAC", "DIS", "ADBE", "CRM", "NFLX", "INTC", "CSCO",
    "T", "VZ", "PFE", "NKE", "MCD", "ORCL", "IBM", "GE", "CAT", "BA",
    "GS", "LMT",
]
BENCHMARK = "^GSPC"
LOOKBACK_PERIOD = "2y"


def fetch_price_history(tickers, period=LOOKBACK_PERIOD):
    """Download adjusted close prices for a list of tickers."""
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return prices.dropna(axis=1, thresh=int(len(prices) * 0.9))


def fetch_fundamentals(tickers):
    """Pull the trailing fundamentals needed for the Value and Quality factors."""
    rows = {}
    for t in tickers:
        info = yf.Ticker(t).info
        rows[t] = {
            "trailingPE": info.get("trailingPE"),
            "priceToBook": info.get("priceToBook"),
            "returnOnEquity": info.get("returnOnEquity"),
            "debtToEquity": info.get("debtToEquity"),
            "grossMargins": info.get("grossMargins"),
        }
    return pd.DataFrame(rows).T


def zscore(series):
    return (series - series.mean()) / series.std(ddof=0)


def momentum_factor(prices, skip_months=1, lookback_months=12):
    """12-1 momentum: cumulative return over the lookback window, excluding
    the most recent month to avoid short-term reversal effects."""
    monthly = prices.resample("ME").last()
    end = monthly.iloc[-(skip_months + 1)]
    start = monthly.iloc[-(skip_months + lookback_months)]
    return (end / start) - 1


def volatility_factor(prices, window_days=126):
    """Realized annualized volatility over the trailing window (lower is better)."""
    daily_ret = prices.pct_change().dropna()
    return daily_ret.tail(window_days).std() * np.sqrt(252)


def value_factor(fundamentals):
    """Composite of earnings yield and book-to-price (higher is cheaper/better)."""
    earnings_yield = 1 / fundamentals["trailingPE"].replace(0, np.nan)
    book_to_price = 1 / fundamentals["priceToBook"].replace(0, np.nan)
    return zscore(earnings_yield) + zscore(book_to_price)


def quality_factor(fundamentals):
    """Composite of ROE, gross margin, and (inverse) leverage."""
    roe = fundamentals["returnOnEquity"]
    margin = fundamentals["grossMargins"]
    leverage = fundamentals["debtToEquity"]
    return zscore(roe) + zscore(margin) - zscore(leverage)


def build_composite_score(prices, fundamentals, weights=None):
    weights = weights or {"value": 0.25, "momentum": 0.25, "quality": 0.25, "low_vol": 0.25}

    mom = momentum_factor(prices)
    vol = volatility_factor(prices)
    val = value_factor(fundamentals)
    qual = quality_factor(fundamentals)

    scores = pd.DataFrame({
        "value_z": zscore(val),
        "momentum_z": zscore(mom),
        "quality_z": zscore(qual),
        "low_vol_z": zscore(-vol),  # negate so higher score = lower volatility
    })

    scores["composite"] = (
        weights["value"] * scores["value_z"]
        + weights["momentum"] * scores["momentum_z"]
        + weights["quality"] * scores["quality_z"]
        + weights["low_vol"] * scores["low_vol_z"]
    )
    return scores.sort_values("composite", ascending=False)


def build_top_decile_portfolio(scores, decile=0.3):
    """Equal-weight the top `decile` fraction of names by composite score."""
    n = max(1, int(len(scores) * decile))
    holdings = scores.head(n).index.tolist()
    weight = 1.0 / len(holdings)
    return {ticker: weight for ticker in holdings}


def backtest(prices, portfolio, benchmark_prices):
    """Rebalance-once, buy-and-hold backtest of the portfolio vs. the benchmark."""
    tickers = list(portfolio.keys())
    weights = np.array([portfolio[t] for t in tickers])

    port_prices = prices[tickers].dropna()
    normalized = port_prices / port_prices.iloc[0]
    port_value = (normalized * weights).sum(axis=1)

    bench = benchmark_prices.dropna()
    bench_normalized = bench / bench.iloc[0]

    df = pd.DataFrame({"factor_portfolio": port_value, "benchmark": bench_normalized}).dropna()

    total_return = df.iloc[-1] / df.iloc[0] - 1
    daily_ret = df.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / ann_vol

    running_max = df.cummax()
    drawdown = df / running_max - 1
    max_drawdown = drawdown.min()

    metrics = pd.DataFrame({
        "Total Return": total_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    })

    return df, metrics


def run(universe=UNIVERSE, benchmark=BENCHMARK):
    print(f"Fetching price history for {len(universe)} tickers...")
    prices = fetch_price_history(universe)
    valid_tickers = list(prices.columns)

    print("Fetching fundamentals...")
    fundamentals = fetch_fundamentals(valid_tickers)
    fundamentals = fundamentals.dropna(thresh=3)
    common = [t for t in valid_tickers if t in fundamentals.index]

    print("Scoring factors...")
    scores = build_composite_score(prices[common], fundamentals.loc[common])

    print("\nTop 10 ranked equities:")
    print(scores.head(10).round(3))

    portfolio = build_top_decile_portfolio(scores)
    print(f"\nPortfolio ({len(portfolio)} holdings, equal-weighted):")
    print(list(portfolio.keys()))

    print(f"\nBacktesting against {benchmark}...")
    bench_prices = fetch_price_history([benchmark])[benchmark]
    equity_curve, metrics = backtest(prices, portfolio, bench_prices)

    print("\nPerformance summary:")
    print(metrics.round(4))

    equity_curve.to_csv("equity_curve.csv")
    scores.round(4).to_csv("factor_scores.csv")
    metrics.round(4).to_csv("performance_metrics.csv")
    print("\nSaved: equity_curve.csv, factor_scores.csv, performance_metrics.csv")

    return scores, portfolio, equity_curve, metrics


if __name__ == "__main__":
    run()
