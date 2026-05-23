"""
backtest.py
===========
Portfolio performance evaluation and historical event study.

This module provides:

1. **Baseline backtest**: Unfiltered factor portfolio at full exposure.
2. **Regime-filtered backtest**: Exposure scaled by crowding regime.
3. **Performance metrics**: Sharpe, Sortino, Calmar, max drawdown, win rate.
4. **Crash-date event study**: How much did the filter protect during known crashes?
5. **Strategy comparison**: Side-by-side metrics table.

Design notes
------------
* All returns are assumed to be computed from ``factor_engine.compute_factor_returns``
  (monthly frequency) or ``regime_filter.apply_regime_filter`` (daily/monthly).
* Transaction costs are applied via ``src.utils.bps_to_fraction`` — hardcoded
  to ``config.TRANSACTION_COST_BPS`` (10 bps one-way).
* The event study is the most important deliverable: it directly tests whether
  the filter protected capital during the crashes that matter.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import (
    annualised_return,
    annualised_vol,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    compute_max_drawdown,
    compute_drawdown,
    bps_to_fraction,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Baseline Factor Backtest
# ---------------------------------------------------------------------------

def run_baseline_factor_backtest(
    factor_returns: pd.Series,
    transaction_cost_bps: int = config.TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Run unfiltered factor portfolio backtest.

    Parameters
    ----------
    factor_returns : pd.Series
        Raw factor portfolio returns (daily or monthly, as fractions).
    transaction_cost_bps : int, optional
        One-way transaction cost.  Default from ``config``.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``daily_return``, ``cumulative_return``, ``drawdown``.
        Drawdown is expressed as a fraction (e.g. -0.10 = 10% underwater).

    Notes
    -----
    Transaction costs are already embedded in the factor returns from
    ``compute_factor_returns``.  The additional cost parameter here is
    for any incremental costs not yet applied.  Set to 0 if costs are
    already in the input.
    """
    logger.info("Running baseline factor backtest …")
    returns = factor_returns.dropna().copy()

    cum_ret = (1 + returns).cumprod()
    drawdown = compute_drawdown(cum_ret)

    result = pd.DataFrame(
        {
            "daily_return": returns,
            "cumulative_return": cum_ret,
            "drawdown": drawdown,
        }
    )
    result.index.name = "date"
    return result


# ---------------------------------------------------------------------------
# 2. Regime-Filtered Backtest
# ---------------------------------------------------------------------------

def run_regime_filtered_backtest(
    filtered_returns_df: pd.DataFrame,
    transaction_cost_bps: int = config.TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Run regime-filtered portfolio backtest.

    Parameters
    ----------
    filtered_returns_df : pd.DataFrame
        Output of ``regime_filter.apply_regime_filter``.
        Must have column ``filtered_return``.
    transaction_cost_bps : int, optional
        Additional one-way cost for regime transitions.

    Returns
    -------
    pd.DataFrame
        Same structure as ``run_baseline_factor_backtest`` output, plus
        columns: ``regime_label``, ``factor_weight``.

    Notes
    -----
    **Regime-transition transaction costs**

    When the portfolio shifts between NORMAL (100% factor) and WARNING (50%
    factor), approximately 50% of the portfolio is traded.  At 10 bps one-way,
    this costs 5 bps per regime transition.  These costs are applied here
    based on day-to-day changes in ``factor_weight``.
    """
    logger.info("Running regime-filtered backtest …")
    df = filtered_returns_df.copy().dropna(subset=["filtered_return"])
    returns = df["filtered_return"].copy()

    # Apply transition costs: |Δweight| × tc per day
    tc_fraction = bps_to_fraction(transaction_cost_bps)
    weight_changes = df["factor_weight"].diff().abs().fillna(0)
    tc_daily = weight_changes * tc_fraction
    net_returns = returns - tc_daily

    cum_ret = (1 + net_returns).cumprod()
    drawdown = compute_drawdown(cum_ret)

    result = df.copy()
    result["daily_return"] = net_returns
    result["cumulative_return"] = cum_ret
    result["drawdown"] = drawdown

    return result


# ---------------------------------------------------------------------------
# 3. Performance Metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(
    returns_df: pd.DataFrame,
    label: str,
    return_col: str = "daily_return",
    risk_free_rate: float = config.RISK_FREE_RATE,
) -> dict:
    """Compute a comprehensive set of portfolio performance metrics.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Backtest output.  Must have a ``return_col`` column.
    label : str
        Strategy label for display purposes.
    return_col : str, optional
        Column name for the return series.  Default ``"daily_return"``.
    risk_free_rate : float, optional
        Annual risk-free rate.  Default from ``config``.

    Returns
    -------
    dict
        Keys: ``label``, ``annualized_return``, ``annualized_vol``,
        ``sharpe``, ``sortino``, ``calmar``, ``max_drawdown``,
        ``peak_date``, ``trough_date``, ``win_rate``,
        ``best_period``, ``worst_period``, ``total_return``.

    Notes
    -----
    Win rate is computed on a monthly basis regardless of the input frequency.
    If monthly returns are already provided, this is the fraction of months
    with positive returns.
    """
    rets = returns_df[return_col].dropna()

    ann_ret = annualised_return(rets)
    ann_vol = annualised_vol(rets)
    sh = sharpe_ratio(rets, risk_free_rate)
    so = sortino_ratio(rets, risk_free_rate)
    ca = calmar_ratio(rets)
    max_dd, peak_dt, trough_dt = compute_max_drawdown(rets)
    total_ret = float((1 + rets).prod() - 1)

    # Win rate — resample to monthly
    monthly_rets = rets.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    win_rate = float((monthly_rets > 0).mean())
    best_month = float(monthly_rets.max())
    worst_month = float(monthly_rets.min())

    metrics = {
        "label": label,
        "annualized_return": round(ann_ret * 100, 2),          # % pa
        "annualized_vol": round(ann_vol * 100, 2),             # % pa
        "sharpe": round(sh, 3),
        "sortino": round(so, 3),
        "calmar": round(ca, 3),
        "max_drawdown": round(max_dd * 100, 2),                # % (negative)
        "peak_date": str(peak_dt.date()) if peak_dt else "N/A",
        "trough_date": str(trough_dt.date()) if trough_dt else "N/A",
        "win_rate": round(win_rate * 100, 1),                  # % months
        "best_period": round(best_month * 100, 2),             # % best month
        "worst_period": round(worst_month * 100, 2),           # % worst month
        "total_return": round(total_ret * 100, 2),             # % total
    }

    # Pretty-print table
    print(f"\n{'=' * 50}")
    print(f"  PERFORMANCE METRICS: {label}")
    print(f"{'=' * 50}")
    print(f"  Annualized Return  : {metrics['annualized_return']:>8.2f}%")
    print(f"  Annualized Vol     : {metrics['annualized_vol']:>8.2f}%")
    print(f"  Sharpe Ratio       : {metrics['sharpe']:>8.3f}")
    print(f"  Sortino Ratio      : {metrics['sortino']:>8.3f}")
    print(f"  Calmar Ratio       : {metrics['calmar']:>8.3f}")
    print(f"  Max Drawdown       : {metrics['max_drawdown']:>8.2f}%")
    print(f"    Peak Date        : {metrics['peak_date']}")
    print(f"    Trough Date      : {metrics['trough_date']}")
    print(f"  Win Rate (monthly) : {metrics['win_rate']:>8.1f}%")
    print(f"  Best Month         : {metrics['best_period']:>8.2f}%")
    print(f"  Worst Month        : {metrics['worst_period']:>8.2f}%")
    print(f"  Total Return       : {metrics['total_return']:>8.2f}%")
    print(f"{'=' * 50}\n")

    return metrics


# ---------------------------------------------------------------------------
# 4. Crash-Date Event Study
# ---------------------------------------------------------------------------

def run_crash_date_event_study(
    baseline_returns: pd.DataFrame,
    filtered_returns: pd.DataFrame,
    crash_dates_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute strategy drawdowns during each known factor crash.

    Parameters
    ----------
    baseline_returns : pd.DataFrame
        Output of ``run_baseline_factor_backtest``.
    filtered_returns : pd.DataFrame
        Output of ``run_regime_filtered_backtest``.
    crash_dates_df : pd.DataFrame
        Reference data from ``factor_crash_dates.csv``.
        Must have: ``event_name``, ``start_date``, ``trough_date``.

    Returns
    -------
    pd.DataFrame
        Columns:
        * ``event_name``
        * ``baseline_crash_return`` : Cumulative return over crash window (%)
        * ``filtered_crash_return``  : Same, for regime-filtered strategy (%)
        * ``protection_pct``         : filtered - baseline (positive = filtered protected)
        * ``regime_was_warning_before_crash`` : True if WARNING fired in 30-day pre-window

    Notes
    -----
    **This is the most important table in the research.**

    The event study directly answers: "Given a known crash, was the filter
    actually invested at reduced exposure, and did that reduce the drawdown?"

    A strategy can generate a WARNING signal before the crash (good signal)
    but still suffer a full drawdown if the warning is not actionable (e.g.,
    if the crash happens within 1 day of the warning).  The event study
    distinguishes between signal quality and protection effectiveness.

    **Window convention**

    The crash window is ``[start_date, trough_date]``.  Returns are computed
    as the cumulative product over daily returns in this window.
    For events outside the backtest date range, the row is included with NaN.
    """
    logger.info("Running crash-date event study for %d events …", len(crash_dates_df))

    baseline_col = "daily_return"
    filtered_col = "daily_return"

    records: list[dict] = []

    for _, event in crash_dates_df.iterrows():
        event_name = event["event_name"]
        try:
            start = pd.Timestamp(event["start_date"])
            trough = pd.Timestamp(event["trough_date"])
        except Exception:
            logger.warning("Cannot parse dates for event %s", event_name)
            continue

        # Check if event falls within backtest range
        bt_start = baseline_returns.index.min()
        bt_end = baseline_returns.index.max()

        if trough < bt_start or start > bt_end:
            records.append(
                {
                    "event_name": event_name,
                    "baseline_crash_return": np.nan,
                    "filtered_crash_return": np.nan,
                    "protection_pct": np.nan,
                    "regime_was_warning_before_crash": "Out of backtest range",
                }
            )
            continue

        # Clip window to backtest range
        start_clip = max(start, bt_start)
        trough_clip = min(trough, bt_end)

        # Compute cumulative returns over crash window
        baseline_window = baseline_returns.loc[start_clip:trough_clip, baseline_col].dropna()
        filtered_window = filtered_returns.loc[start_clip:trough_clip, filtered_col].dropna()

        baseline_cum = float((1 + baseline_window).prod() - 1) * 100 if len(baseline_window) > 0 else np.nan
        filtered_cum = float((1 + filtered_window).prod() - 1) * 100 if len(filtered_window) > 0 else np.nan

        protection = (filtered_cum - baseline_cum) if not (np.isnan(baseline_cum) or np.isnan(filtered_cum)) else np.nan

        # Check if WARNING was triggered in the 30-day pre-crash window
        if "regime_label" in filtered_returns.columns:
            pre_start = start - pd.Timedelta(days=30)
            pre_window = filtered_returns.loc[pre_start:start, "regime_label"]
            was_warning = bool((pre_window == config.REGIME_WARNING).any())
        else:
            was_warning = False

        records.append(
            {
                "event_name": event_name,
                "baseline_crash_return": round(baseline_cum, 2) if not np.isnan(baseline_cum) else np.nan,
                "filtered_crash_return": round(filtered_cum, 2) if not np.isnan(filtered_cum) else np.nan,
                "protection_pct": round(protection, 2) if not np.isnan(protection) else np.nan,
                "regime_was_warning_before_crash": was_warning,
            }
        )

    event_df = pd.DataFrame(records)
    logger.info("Event study complete:\n%s", event_df.to_string())
    return event_df


# ---------------------------------------------------------------------------
# 5. Strategy Comparison
# ---------------------------------------------------------------------------

def compare_strategies(
    baseline_metrics: dict,
    filtered_metrics: dict,
) -> pd.DataFrame:
    """Generate a side-by-side strategy comparison table.

    Parameters
    ----------
    baseline_metrics : dict
        Output of ``compute_performance_metrics`` for the baseline strategy.
    filtered_metrics : dict
        Output of ``compute_performance_metrics`` for the filtered strategy.

    Returns
    -------
    pd.DataFrame
        Comparison table with three columns:
        ``Baseline``, ``Regime-Filtered``, ``Improvement``.
        Improvement shows the ratio (filtered / baseline) for key metrics,
        and the difference (filtered - baseline) for absolute metrics.

    Notes
    -----
    **Interpreting improvements**

    * Sharpe improvement > 1.0 means regime filtering improved risk-adjusted
      returns relative to the unfiltered baseline.
    * Max drawdown improvement > 1.0 means the filtered strategy had a
      shallower maximum drawdown (positive sign due to sign convention).
    * If improvements are near 1.0 across the board, the crowding signal
      is adding noise rather than value.
    """
    metrics_to_compare = [
        ("Annualized Return (%)", "annualized_return", "diff"),
        ("Annualized Vol (%)", "annualized_vol", "diff"),
        ("Sharpe Ratio", "sharpe", "ratio"),
        ("Sortino Ratio", "sortino", "ratio"),
        ("Calmar Ratio", "calmar", "ratio"),
        ("Max Drawdown (%)", "max_drawdown", "diff"),
        ("Win Rate (%)", "win_rate", "diff"),
        ("Total Return (%)", "total_return", "diff"),
    ]

    rows: list[dict] = []
    for display_name, key, improvement_type in metrics_to_compare:
        base_val = baseline_metrics.get(key, np.nan)
        filt_val = filtered_metrics.get(key, np.nan)

        if improvement_type == "ratio":
            if base_val and abs(base_val) > 0.001:
                improvement = round(filt_val / base_val, 3)
                impr_str = f"{improvement:.3f}×"
            else:
                improvement = np.nan
                impr_str = "N/A"
        else:  # diff
            improvement = filt_val - base_val if not (np.isnan(base_val) or np.isnan(filt_val)) else np.nan
            impr_str = f"{improvement:+.2f}" if not np.isnan(improvement) else "N/A"

        rows.append(
            {
                "Metric": display_name,
                "Baseline": base_val,
                "Regime-Filtered": filt_val,
                "Improvement": impr_str,
            }
        )

    comparison_df = pd.DataFrame(rows).set_index("Metric")

    print(f"\n{'=' * 70}")
    print("  STRATEGY COMPARISON: Baseline vs Regime-Filtered")
    print(f"{'=' * 70}")
    print(comparison_df.to_string())
    print(f"{'=' * 70}\n")

    return comparison_df
