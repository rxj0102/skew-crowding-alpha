"""
factor_engine.py
================
Equity factor score computation from price data only.

All factors are implemented as pure pandas/numpy transformations — no
external factor libraries (e.g. ``alphalens``, ``riskfolio``) are required.

Factors implemented
-------------------
* **Momentum (12-1)**   : 12-month trailing return skipping the most recent month.
* **Low Volatility**    : Negative of 63-day realised volatility.
* **Quality (proxy)**   : Inverse of trailing 252-day maximum drawdown magnitude.

Notes on quality proxy
~~~~~~~~~~~~~~~~~~~~~~
In a rigorous implementation, quality uses accounting fundamentals: ROE,
debt-to-equity ratio, earnings stability (Piotroski F-score, MSCI Quality
composite).  Because this repository uses only price data, quality is proxied
by price stability — the inverse of trailing maximum drawdown depth.  This
captures the "stable compounder" characteristic: high-quality companies tend to
have smoother price paths and shallower drawdowns relative to cyclicals.  The
correlation with fundamental quality is imperfect, so treat factor-model results
with appropriate scepticism.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import compute_drawdown, bps_to_fraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Momentum Factor
# ---------------------------------------------------------------------------

def compute_momentum_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional 12-1 month momentum scores.

    Parameters
    ----------
    price_df : pd.DataFrame
        Wide-format adjusted close prices.  Index = trading dates,
        columns = ticker symbols.

    Returns
    -------
    pd.DataFrame
        Same shape as ``price_df``.  Values = 12-1 momentum score
        (trailing 252-day return, skipping the most recent 21 days).
        NaN where insufficient history.

    Notes
    -----
    The standard momentum anomaly (Jegadeesh & Titman 1993) uses the return
    from t-252 to t-21, skipping t-21 to t to avoid the 1-month reversal effect.
    Scores are cross-sectionally normalised (ranked 0–1) downstream in
    ``rank_factor_universe``.
    """
    logger.info("Computing 12-1 momentum factor …")
    lookback = config.MOMENTUM_LOOKBACK   # 252 trading days
    skip = config.MOMENTUM_SKIP           # 21 trading days

    # Forward-fill up to 5 days for minor gaps, then leave as NaN
    prices = price_df.ffill(limit=5)

    momentum = (prices.shift(skip) / prices.shift(lookback)) - 1.0
    logger.debug("Momentum factor: non-NaN entries = %d", momentum.count().sum())
    return momentum


# ---------------------------------------------------------------------------
# 2. Low Volatility Factor
# ---------------------------------------------------------------------------

def compute_low_vol_factor(
    price_df: pd.DataFrame,
    window: int = config.LOW_VOL_WINDOW,
) -> pd.DataFrame:
    """Compute low-volatility factor scores from realised vol.

    Parameters
    ----------
    price_df : pd.DataFrame
        Adjusted close prices (wide format).
    window : int, optional
        Rolling window for realised volatility (default 63 days ≈ 1 quarter).

    Returns
    -------
    pd.DataFrame
        Low-vol factor scores.  Higher score = lower realised vol.
        Specifically: ``score = -realised_vol`` so that standard top-quintile
        selection picks the *least volatile* names.

    Notes
    -----
    The low-volatility anomaly (Ang et al. 2006, Baker et al. 2011) documents
    that low-beta / low-vol stocks outperform on a risk-adjusted basis — the
    so-called "low-vol puzzle."  This score is inversed so that factor selection
    (top quintile = high score) consistently means *high quality on that factor*.
    """
    logger.info("Computing low-volatility factor (window=%d) …", window)
    returns = price_df.ffill(limit=5).pct_change()
    realised_vol = returns.rolling(window=window, min_periods=window // 2).std()
    # Annualise for interpretability (optional — doesn't affect rankings)
    realised_vol_ann = realised_vol * np.sqrt(252)
    # Invert: high score = low vol
    low_vol_score = -realised_vol_ann
    return low_vol_score


# ---------------------------------------------------------------------------
# 3. Quality Factor (Price-Based Proxy)
# ---------------------------------------------------------------------------

def compute_quality_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute quality factor scores via trailing maximum drawdown proxy.

    Parameters
    ----------
    price_df : pd.DataFrame
        Adjusted close prices (wide format).

    Returns
    -------
    pd.DataFrame
        Quality proxy scores.  Higher score = shallower trailing drawdown
        = more "stable compounder" characteristic.
        Formula: ``score = 1 / (1 + |max_drawdown_252d|)``

    Notes
    -----
    **Important caveat**: This is a price-based proxy, not true fundamental
    quality.  In a production-grade factor model, quality would be computed from:

    * Return on equity (ROE)
    * Earnings stability (low variance of annual EPS growth)
    * Balance-sheet strength (low debt/equity, high current ratio)
    * Accruals (Sloan 1996 accrual anomaly)

    The price-proxy used here correlates with fundamental quality because
    high-quality businesses compound more steadily and suffer shallower
    drawdowns, but the relationship is noisy.  For best results, augment with
    accounting data via Compustat or Sharadar.

    The formula ``1 / (1 + |max_dd|)`` maps max drawdown from (0, ∞) to (0, 1):
    * Max DD = 0   → score = 1.0 (perfect quality — never drew down)
    * Max DD = 50% → score = 0.67
    * Max DD = 100% → score = 0.50
    """
    logger.info("Computing quality factor (price proxy, 252d window) …")
    prices = price_df.ffill(limit=5)
    returns = prices.pct_change()

    # Compute rolling 252-day maximum drawdown for each stock
    def _rolling_max_dd(returns_col: pd.Series, window: int = 252) -> pd.Series:
        """Compute rolling maximum drawdown over a trailing window."""
        scores = []
        for t in range(len(returns_col)):
            start = max(0, t - window + 1)
            window_rets = returns_col.iloc[start : t + 1].dropna()
            if len(window_rets) < 20:  # Insufficient history
                scores.append(np.nan)
                continue
            cum_ret = (1 + window_rets).cumprod()
            roll_max = cum_ret.expanding().max()
            dd = (cum_ret / roll_max) - 1.0
            max_dd = float(dd.min())  # Negative value
            quality_score = 1.0 / (1.0 + abs(max_dd))
            scores.append(quality_score)
        return pd.Series(scores, index=returns_col.index)

    # Vectorised approach (much faster than per-column loop above)
    cum_ret = (1 + returns.fillna(0)).cumprod()

    quality_scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for col in prices.columns:
        cr = (1 + returns[col].ffill(limit=5)).cumprod()
        # Expanding max over rolling 252-day window
        rolling_max = cr.rolling(window=config.QUALITY_WINDOW, min_periods=21).apply(
            lambda x: x.max(), raw=True
        )
        rolling_max = rolling_max.fillna(method="bfill")
        dd_series = (cr / rolling_max.where(rolling_max != 0, np.nan)) - 1.0
        max_dd_rolling = dd_series.rolling(window=config.QUALITY_WINDOW, min_periods=21).min()
        quality_scores[col] = 1.0 / (1.0 + max_dd_rolling.abs())

    logger.debug("Quality factor: non-NaN entries = %d", quality_scores.count().sum())
    return quality_scores


# ---------------------------------------------------------------------------
# 4. Cross-Sectional Ranking
# ---------------------------------------------------------------------------

def rank_factor_universe(
    factor_df: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    """Cross-sectionally rank factor scores for a single date.

    Parameters
    ----------
    factor_df : pd.DataFrame
        Wide-format factor scores (index = dates, columns = tickers).
    date : str
        Date to extract and rank.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:
        ``ticker``, ``factor_score``, ``factor_rank``, ``top_quintile``.
        ``factor_rank`` is a percentile rank in [0, 1].
        ``top_quintile`` is True if ``factor_rank`` > ``config.FACTOR_QUINTILE``.

    Notes
    -----
    Ranks are computed cross-sectionally (across tickers) on a single date.
    NaN scores are dropped before ranking so the percentile calculation is
    unaffected by missing data.
    """
    if date not in factor_df.index:
        # Find nearest available date
        available = factor_df.index[factor_df.index <= pd.Timestamp(date)]
        if available.empty:
            logger.warning("No factor data available on or before %s", date)
            return pd.DataFrame(columns=["ticker", "factor_score", "factor_rank", "top_quintile"])
        date = available[-1]

    scores = factor_df.loc[date].dropna()
    ranked = scores.rank(pct=True)

    result = pd.DataFrame(
        {
            "ticker": scores.index,
            "factor_score": scores.values,
            "factor_rank": ranked.values,
            "top_quintile": ranked.values > config.FACTOR_QUINTILE,
        }
    )
    result.set_index("ticker", inplace=True)
    return result


# ---------------------------------------------------------------------------
# 5. Factor Portfolio Construction
# ---------------------------------------------------------------------------

def build_factor_portfolio(
    factor_df: pd.DataFrame,
    factor_name: str,
    price_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build a monthly-rebalanced top-quintile equal-weight factor portfolio.

    Parameters
    ----------
    factor_df : pd.DataFrame
        Factor scores (index = dates, columns = tickers).
    factor_name : str
        One of ``"momentum"``, ``"low_vol"``, ``"quality"``.
    price_df : pd.DataFrame, optional
        Not used for portfolio construction, kept for API consistency.

    Returns
    -------
    pd.DataFrame
        Long-format holdings:
        columns = ``date``, ``ticker``, ``factor_score``, ``factor_rank``,
        ``in_top_quintile``, ``portfolio_weight``.
        Only top-quintile names are included.

    Notes
    -----
    Rebalancing occurs on the first trading day of each calendar month.
    Equal weighting is used within the top quintile.  For a universe of 100
    names, the top quintile has approximately 20 stocks.

    The portfolio is saved to ``config.factor_portfolio_path(factor_name)``.
    """
    logger.info("Building %s factor portfolio (monthly rebalance, top quintile) …", factor_name)

    # Identify monthly rebalance dates (first trading day of each month)
    all_dates = factor_df.index
    rebalance_dates = all_dates[
        all_dates.to_series().diff().dt.days.fillna(30) > 15
        | (all_dates.month != all_dates.to_series().shift(1).dt.month.fillna(-1))
    ]

    records: list[dict] = []

    for rb_date in rebalance_dates:
        date_scores = factor_df.loc[rb_date].dropna()
        if date_scores.empty:
            continue

        ranked = date_scores.rank(pct=True)
        top_q = ranked[ranked > config.FACTOR_QUINTILE]
        if top_q.empty:
            continue

        n = len(top_q)
        weight = 1.0 / n

        for ticker in top_q.index:
            records.append(
                {
                    "date": rb_date,
                    "ticker": ticker,
                    "factor_score": float(date_scores[ticker]),
                    "factor_rank": float(ranked[ticker]),
                    "in_top_quintile": True,
                    "portfolio_weight": weight,
                }
            )

    portfolio_df = pd.DataFrame(records)
    if portfolio_df.empty:
        logger.warning("Factor portfolio for %s is empty!", factor_name)
        return portfolio_df

    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])

    out_path = config.factor_portfolio_path(factor_name)
    portfolio_df.to_csv(out_path, index=False)
    logger.info(
        "Saved %s portfolio (%d rebalance dates, avg %d holdings) to %s",
        factor_name,
        portfolio_df["date"].nunique(),
        portfolio_df.groupby("date").size().mean(),
        out_path,
    )
    return portfolio_df


# ---------------------------------------------------------------------------
# 6. Factor Returns
# ---------------------------------------------------------------------------

def compute_factor_returns(
    factor_portfolio_df: pd.DataFrame,
    price_df: pd.DataFrame,
    transaction_cost_bps: int = config.TRANSACTION_COST_BPS,
) -> pd.Series:
    """Compute monthly equal-weight factor portfolio returns with turnover costs.

    Parameters
    ----------
    factor_portfolio_df : pd.DataFrame
        Output of ``build_factor_portfolio``.
    price_df : pd.DataFrame
        Adjusted close prices (wide format).
    transaction_cost_bps : int, optional
        One-way transaction cost in basis points.

    Returns
    -------
    pd.Series
        Monthly portfolio returns indexed by rebalance date.
        Transaction costs are deducted on turnover.

    Notes
    -----
    Turnover is estimated as the fraction of portfolio names that change
    between consecutive rebalance dates.  The cost is applied one-way
    (charged only on the shares traded, not the entire portfolio).

    Monthly returns are computed as the equal-weight average of monthly price
    returns for the constituents held over the period.
    """
    logger.info("Computing factor portfolio returns …")

    # Get the rebalance dates
    rebalance_dates = sorted(factor_portfolio_df["date"].unique())
    if len(rebalance_dates) < 2:
        logger.warning("Fewer than 2 rebalance dates — cannot compute returns.")
        return pd.Series(dtype=float)

    monthly_returns: list[tuple] = []
    tc_fraction = bps_to_fraction(transaction_cost_bps)

    prev_tickers: set[str] = set()

    for i in range(len(rebalance_dates) - 1):
        t0 = rebalance_dates[i]
        t1 = rebalance_dates[i + 1]

        holdings = factor_portfolio_df[factor_portfolio_df["date"] == t0]["ticker"].tolist()
        if not holdings:
            continue

        # Get price data between t0 and t1
        mask = (price_df.index > t0) & (price_df.index <= t1)
        period_prices = price_df.loc[mask, [h for h in holdings if h in price_df.columns]]

        if period_prices.empty:
            continue

        # Compute total return over the holding period
        valid_cols = period_prices.dropna(axis=1, how="all").columns.tolist()
        if not valid_cols:
            continue

        period_ret = (period_prices[valid_cols].iloc[-1] / period_prices[valid_cols].iloc[0]) - 1.0
        ew_return = float(period_ret.mean())

        # Turnover cost
        current_tickers = set(holdings)
        new_names = current_tickers - prev_tickers
        turnover = len(new_names) / len(current_tickers) if current_tickers else 0.0
        cost = turnover * tc_fraction
        net_return = ew_return - cost

        monthly_returns.append((t1, net_return))
        prev_tickers = current_tickers

    if not monthly_returns:
        return pd.Series(dtype=float)

    return_series = pd.Series(
        [r for _, r in monthly_returns],
        index=pd.DatetimeIndex([d for d, _ in monthly_returns]),
        name="factor_return",
    )
    logger.info(
        "Factor returns: %d periods, mean = %.2f%%, std = %.2f%%",
        len(return_series),
        return_series.mean() * 100,
        return_series.std() * 100,
    )
    return return_series
