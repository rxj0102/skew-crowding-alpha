"""
utils.py
========
Shared utility functions for the skew-crowding-alpha research framework.

Key concern: look-ahead bias prevention.  Every module that produces a signal
must consume it through ``lag_signal`` before comparing against forward returns.
Centralising this here makes accidental look-ahead impossible to miss in code
review.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Look-ahead bias prevention
# ---------------------------------------------------------------------------

def lag_signal(
    series: pd.Series,
    periods: int = 1,
) -> pd.Series:
    """Lag a signal series by ``periods`` to prevent look-ahead bias.

    Parameters
    ----------
    series : pd.Series
        Time-indexed signal (e.g., regime classification, crowding index).
    periods : int, optional
        Number of periods to shift forward (default 1 day).

    Returns
    -------
    pd.Series
        Shifted series.  The first ``periods`` values will be NaN.

    Notes
    -----
    **IMPORTANT**: All regime-based position changes must use the *prior* day's
    classification to avoid look-ahead bias.  Call this function on any signal
    before it is used to determine portfolio weights.

    Example
    -------
    >>> regime_today = lag_signal(regime_series, periods=1)
    >>> weights = weights.where(regime_today != "WARNING", other=0.5)
    """
    return series.shift(periods)


def lag_frame(
    df: pd.DataFrame,
    periods: int = 1,
) -> pd.DataFrame:
    """Lag every column in a DataFrame by ``periods`` trading days.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame.
    periods : int, optional
        Number of periods to shift (default 1).

    Returns
    -------
    pd.DataFrame
        Shifted DataFrame.
    """
    return df.shift(periods)


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------

def compute_drawdown(cumulative_returns: pd.Series) -> pd.Series:
    """Compute the running drawdown series from a cumulative return series.

    Parameters
    ----------
    cumulative_returns : pd.Series
        Cumulative (gross) returns starting at 1.0.

    Returns
    -------
    pd.Series
        Drawdown at each date (values ≤ 0, expressed as fractions).
    """
    rolling_max = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / rolling_max) - 1.0
    return drawdown


def compute_max_drawdown(
    returns: pd.Series,
) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Return maximum drawdown and peak/trough dates.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (not cumulative).

    Returns
    -------
    tuple[float, pd.Timestamp | None, pd.Timestamp | None]
        (max_drawdown, peak_date, trough_date).  max_drawdown is negative.
    """
    cum_ret = (1 + returns).cumprod()
    drawdown = compute_drawdown(cum_ret)
    max_dd = float(drawdown.min())
    if drawdown.empty or np.isnan(max_dd):
        return (np.nan, None, None)

    trough_date = drawdown.idxmin()
    peak_date = cum_ret.loc[:trough_date].idxmax()
    return (max_dd, peak_date, trough_date)


def annualised_return(daily_returns: pd.Series, trading_days: int = 252) -> float:
    """Compound annualised return from a daily return series.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns (as fractions, not percentages).
    trading_days : int, optional
        Number of trading days per year (default 252).

    Returns
    -------
    float
        Annualised return.
    """
    total = float((1 + daily_returns.dropna()).prod())
    n = len(daily_returns.dropna())
    if n == 0:
        return np.nan
    return total ** (trading_days / n) - 1.0


def annualised_vol(daily_returns: pd.Series, trading_days: int = 252) -> float:
    """Annualised volatility of daily returns.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns.
    trading_days : int, optional
        Trading days per year.

    Returns
    -------
    float
        Annualised volatility.
    """
    return float(daily_returns.dropna().std() * np.sqrt(trading_days))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns.
    risk_free_rate : float, optional
        Annual risk-free rate (default 2%).
    trading_days : int, optional
        Trading days per year.

    Returns
    -------
    float
        Sharpe ratio.
    """
    excess = annualised_return(daily_returns, trading_days) - risk_free_rate
    vol = annualised_vol(daily_returns, trading_days)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return excess / vol


def sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> float:
    """Annualised Sortino ratio (downside deviation denominator).

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns.
    risk_free_rate : float, optional
        Annual risk-free rate.
    trading_days : int, optional
        Trading days per year.

    Returns
    -------
    float
        Sortino ratio.
    """
    downside = daily_returns[daily_returns < 0].dropna()
    downside_std = float(downside.std() * np.sqrt(trading_days))
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    excess = annualised_return(daily_returns, trading_days) - risk_free_rate
    return excess / downside_std


def calmar_ratio(daily_returns: pd.Series, trading_days: int = 252) -> float:
    """Calmar ratio: annualised return / |max drawdown|.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns.
    trading_days : int, optional
        Trading days per year.

    Returns
    -------
    float
        Calmar ratio.
    """
    ann_ret = annualised_return(daily_returns, trading_days)
    max_dd, _, _ = compute_max_drawdown(daily_returns)
    if max_dd == 0 or np.isnan(max_dd):
        return np.nan
    return ann_ret / abs(max_dd)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def rolling_zscore(
    series: pd.Series,
    window: int,
    winsorise_clip: float | None = None,
) -> pd.Series:
    """Compute a rolling z-score with optional winsorisation.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    window : int
        Rolling window for mean and standard deviation.
    winsorise_clip : float or None, optional
        If provided, clip z-scores to ±``winsorise_clip`` to handle outliers.

    Returns
    -------
    pd.Series
        Rolling z-score series (same index as ``series``).
    """
    roll_mean = series.rolling(window=window, min_periods=window // 2).mean()
    roll_std = series.rolling(window=window, min_periods=window // 2).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    if winsorise_clip is not None:
        z = z.clip(-winsorise_clip, winsorise_clip)
    return z


def compute_turnover(
    weights_t: pd.Series,
    weights_tm1: pd.Series,
) -> float:
    """Compute portfolio turnover between two weight vectors.

    Parameters
    ----------
    weights_t : pd.Series
        Current period weights (indexed by ticker).
    weights_tm1 : pd.Series
        Previous period weights (indexed by ticker).

    Returns
    -------
    float
        One-way turnover as a fraction (e.g. 0.30 = 30% turnover).
    """
    combined = weights_t.align(weights_tm1, fill_value=0.0)
    return float(abs(combined[0] - combined[1]).sum() / 2.0)


def bps_to_fraction(bps: float) -> float:
    """Convert basis points to a fractional cost (10 bps = 0.001).

    Parameters
    ----------
    bps : float
        Cost in basis points.

    Returns
    -------
    float
        Fractional equivalent.
    """
    return bps / 10_000.0
