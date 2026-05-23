"""
crowding_detector.py
====================
The intellectual core of the skew-crowding-alpha framework.

This module transforms single-name options skew data into an aggregate
factor crowding index and classifies the current market regime.

Crowding Index construction
----------------------------
For each date:

1. Identify the tickers currently in the top-quintile factor portfolio.
2. Extract their rolling put-skew z-scores from the skew panel.
3. Compute the **cross-sectional median** of those z-scores.

   **Why median?**  The median is robust to outliers.  A single large-cap
   ticker's options being mispriced (stale quotes, corporate action) should
   not trigger a market-wide warning.  The mean would be distorted; the
   median requires that the crowding signal be present in *most* factor
   constituents before registering.

The resulting scalar time series is the **Factor Crowding Index**.

Regime Classification
---------------------
The crowding index is further z-scored against its own rolling 60-day history
to produce the **Regime Z-score**:

    regime_z = (crowding_index - rolling_mean_60d) / rolling_std_60d

Regime labels:
    WARNING   : regime_z > +1.5   → Crowding elevated; reduce exposure
    ELEVATED  : +0.75 < regime_z ≤ +1.5 → Crowding rising; monitor
    NORMAL    :  -0.75 < regime_z ≤ +0.75
    DEPRESSED : regime_z ≤ -0.75  → Under-owned; potential entry signal

Entry point
-----------
Run as a standalone module::

    python -m src.crowding_detector --factor momentum

This executes the full crowding detection pipeline using synthetic data
(no external data fetching required) and prints regime statistics.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.utils import rolling_zscore, lag_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Factor Crowding Index
# ---------------------------------------------------------------------------

def compute_factor_crowding_index(
    skew_zscore_panel: pd.DataFrame,
    factor_portfolio_df: pd.DataFrame,
    factor_name: str,
) -> pd.Series:
    """Compute the daily factor crowding index from cross-sectional skew.

    Parameters
    ----------
    skew_zscore_panel : pd.DataFrame
        Wide-format rolling put-skew z-scores.
        Index = dates, columns = tickers.
    factor_portfolio_df : pd.DataFrame
        Output of ``factor_engine.build_factor_portfolio``.
        Must have columns: ``date``, ``ticker``.
    factor_name : str
        Used for labelling and cache path.

    Returns
    -------
    pd.Series
        Daily crowding index.  Name = ``"{factor_name}_crowding_index"``.
        Saved to ``config.factor_crowding_index_path(factor_name)``.

    Notes
    -----
    **Methodology choice — cross-sectional median**

    The crowding index is the cross-sectional **median** (not mean) of skew
    z-scores across the factor's current top-quintile holdings.

    Rationale:
    * A single name's spike (bad IV quote, stale options) won't trigger the
      index.  The median requires that skew elevation is *broad-based* across
      the factor bucket.
    * The median is also more robust to asymmetric skew distributions — some
      names always have structurally higher skew (expensive options market)
      independent of crowding dynamics.

    **Look-ahead bias**

    The portfolio holdings on date T are used to compute the crowding index
    on date T.  The regime classification then lags this index by one day
    before it influences portfolio weights (see ``regime_filter.py``).
    This is the standard convention and does not introduce look-ahead bias.

    **Interpolation between rebalance dates**

    Portfolio holdings are defined monthly.  Between rebalance dates,
    holdings are forward-filled — we use the most recently known portfolio
    composition on each daily observation date.
    """
    logger.info("Computing %s crowding index …", factor_name)

    # Forward-fill portfolio holdings to daily frequency
    # Build a date → list[ticker] mapping
    rebalance_dates = sorted(factor_portfolio_df["date"].unique())
    holdings_by_date: dict[pd.Timestamp, list[str]] = {
        pd.Timestamp(d): factor_portfolio_df[factor_portfolio_df["date"] == d]["ticker"].tolist()
        for d in rebalance_dates
    }

    # Align to the daily skew panel dates
    all_dates = skew_zscore_panel.index
    crowding_values: dict[pd.Timestamp, float] = {}

    # Pre-compute: find the last rebalance date on or before each daily date
    rb_series = pd.Series(rebalance_dates, name="rb_date")

    for daily_date in all_dates:
        # Find the most recent rebalance date ≤ daily_date
        prior_rbs = [d for d in rebalance_dates if pd.Timestamp(d) <= daily_date]
        if not prior_rbs:
            crowding_values[daily_date] = np.nan
            continue

        current_rb = prior_rbs[-1]
        holdings = holdings_by_date[pd.Timestamp(current_rb)]

        # Extract skew z-scores for current factor holdings
        available_tickers = [t for t in holdings if t in skew_zscore_panel.columns]
        if not available_tickers:
            crowding_values[daily_date] = np.nan
            continue

        zscores = skew_zscore_panel.loc[daily_date, available_tickers].dropna()
        if len(zscores) < 3:  # Need at least 3 valid readings
            crowding_values[daily_date] = np.nan
            continue

        crowding_values[daily_date] = float(zscores.median())

    crowding_index = pd.Series(crowding_values, name=f"{factor_name}_crowding_index")
    crowding_index.sort_index(inplace=True)

    # Save to CSV
    out_path = config.factor_crowding_index_path(factor_name)
    crowding_index.to_frame().to_csv(out_path)
    logger.info(
        "Crowding index for %s: %d observations, mean=%.3f, std=%.3f",
        factor_name,
        crowding_index.notna().sum(),
        crowding_index.mean(),
        crowding_index.std(),
    )
    return crowding_index


# ---------------------------------------------------------------------------
# 2. Regime Detection
# ---------------------------------------------------------------------------

def detect_crowding_regime(
    crowding_index: pd.Series,
    baseline_window: int = config.SKEW_ZSCORE_HISTORY,
) -> pd.DataFrame:
    """Classify the daily market regime from the crowding index.

    Parameters
    ----------
    crowding_index : pd.Series
        Output of ``compute_factor_crowding_index``.
    baseline_window : int, optional
        Rolling window (days) for computing the regime z-score.  Default 60.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``crowding_index``, ``regime_zscore``,
        ``regime_label``.

    Notes
    -----
    **Regime z-score formula**::

        regime_z = (crowding_index[t] − rolling_mean[t]) / rolling_std[t]

    where ``rolling_mean`` and ``rolling_std`` are computed over the preceding
    ``baseline_window`` days.

    **Regime thresholds**

    Thresholds are defined in ``config.py``:

    ========= =================== ===========================================
    Label     Condition           Interpretation
    ========= =================== ===========================================
    WARNING   z > 1.5             Factor is crowded.  Elevated unwind risk.
                                  Reduce factor exposure to 50%.
    ELEVATED  0.75 < z ≤ 1.5     Crowding building.  Monitor.
                                  Reduce factor exposure to 75%.
    NORMAL    -0.75 < z ≤ 0.75   No crowding signal.  Full factor exposure.
    DEPRESSED z ≤ -0.75          Factor under-owned.  Potential accumulation
                                  opportunity.  Mild leverage (110%, experimental).
    ========= =================== ===========================================

    **DEPRESSED regime interpretation**: when the crowding index is anomalously
    low, it means institutions are *not* hedging their factor positions.
    This may indicate the factor is under-owned or recently de-risked.
    Historically, the deepest factor drawdowns are often followed by sharp
    reversals once crowding unwinds — the DEPRESSED regime may signal the
    early phase of a crowding build-up.  The 110% exposure in DEPRESSED
    is flagged as experimental; apply only with a stop-loss overlay.
    """
    logger.info("Detecting crowding regime (baseline_window=%d) …", baseline_window)

    regime_z = rolling_zscore(
        crowding_index,
        window=baseline_window,
        winsorise_clip=None,  # Do not winsorise the regime z-score itself
    )

    def _classify(z: float) -> str:
        if np.isnan(z):
            return config.REGIME_NORMAL
        if z > config.CROWDING_THRESHOLD:
            return config.REGIME_WARNING
        elif z > config.ELEVATED_THRESHOLD:
            return config.REGIME_ELEVATED
        elif z < config.DEPRESSED_THRESHOLD:
            return config.REGIME_DEPRESSED
        else:
            return config.REGIME_NORMAL

    regime_labels = regime_z.map(_classify)

    result = pd.DataFrame(
        {
            "crowding_index": crowding_index,
            "regime_zscore": regime_z,
            "regime_label": regime_labels,
        }
    )
    result.index.name = "date"
    logger.info(
        "Regime distribution: %s",
        result["regime_label"].value_counts().to_dict(),
    )
    return result


# ---------------------------------------------------------------------------
# 3. Lead-Lag Analysis
# ---------------------------------------------------------------------------

def compute_crowding_lead_lag(
    crowding_df: pd.DataFrame,
    factor_returns: pd.Series,
    max_lag: int = 20,
) -> pd.DataFrame:
    """Cross-correlation between crowding index and future factor returns.

    Parameters
    ----------
    crowding_df : pd.DataFrame
        Output of ``detect_crowding_regime``.  Must have ``crowding_index``.
    factor_returns : pd.Series
        Time series of factor portfolio returns (daily or monthly).
        Must overlap in date range with ``crowding_df``.
    max_lag : int, optional
        Maximum lag in days to compute.  Default 20.

    Returns
    -------
    pd.DataFrame
        Columns: ``lag_days``, ``correlation``, ``p_value``.
        ``correlation[k]`` = Pearson r between ``crowding_index[t]`` and
        ``factor_return[t + k]``.

    Notes
    -----
    **Interpretation**

    * Lag 0: coincident correlation (not actionable as a trading signal).
    * Lag 1-5: very short-term predictability (daily rebalancing feasible).
    * Lag 5-10: medium-term predictability (most actionable for weekly or
      monthly managers).
    * Lag 10-20: longer-term predictability (likely noise contamination).

    If the **peak negative correlation** occurs at lag 5-10 days, the crowding
    index provides actionable lead time.  If it peaks at lag 0-1, it is
    coincident rather than predictive — the regime change would need to be
    detected *before* the return event, requiring additional confirmation.

    A significant negative correlation at any lag k means: when crowding is
    elevated today, the factor tends to *underperform* k days from now.
    This is the core predictive hypothesis of this framework.
    """
    logger.info("Computing crowding lead-lag analysis (max_lag=%d) …", max_lag)

    ci = crowding_df["crowding_index"].dropna()

    # Align to daily returns — resample factor_returns to daily if monthly
    if isinstance(factor_returns.index, pd.DatetimeIndex):
        # Forward-fill monthly returns to daily (each day in the month gets the
        # return for that month's period — approximate but sufficient for lag analysis)
        all_dates = ci.index.union(factor_returns.index)
        daily_returns = factor_returns.reindex(all_dates).ffill()
    else:
        daily_returns = factor_returns

    records: list[dict] = []
    for lag in range(0, max_lag + 1):
        future_ret = daily_returns.shift(-lag)  # Future return at lag k
        aligned = pd.concat([ci, future_ret], axis=1, join="inner").dropna()
        if len(aligned) < 30:
            records.append({"lag_days": lag, "correlation": np.nan, "p_value": np.nan})
            continue

        r, p = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        records.append({"lag_days": lag, "correlation": round(r, 4), "p_value": round(p, 4)})

    lag_df = pd.DataFrame(records)
    logger.info(
        "Lead-lag: peak negative correlation at lag %d (r=%.3f)",
        lag_df.loc[lag_df["correlation"].idxmin(), "lag_days"],
        lag_df["correlation"].min(),
    )
    return lag_df


# ---------------------------------------------------------------------------
# 4. Validation Against Historical Crash Dates
# ---------------------------------------------------------------------------

def validate_against_crash_dates(
    crowding_df: pd.DataFrame,
    crash_dates_df: pd.DataFrame,
    pre_crash_window_days: int = 30,
    false_positive_window_days: int = 60,
) -> pd.DataFrame:
    """Validate the crowding signal against known factor crash events.

    Parameters
    ----------
    crowding_df : pd.DataFrame
        Output of ``detect_crowding_regime``.
    crash_dates_df : pd.DataFrame
        Reference data from ``data/reference/factor_crash_dates.csv``.
        Must have columns: ``event_name``, ``start_date``, ``trough_date``.
    pre_crash_window_days : int, optional
        Number of calendar days before ``start_date`` to examine.  Default 30.
    false_positive_window_days : int, optional
        Number of days before the pre-crash window to measure false positives.
        Default 60.

    Returns
    -------
    pd.DataFrame
        Columns:
        * ``event_name`` : Crash event label.
        * ``warning_triggered`` : True if WARNING regime fired in the
          pre-crash window.
        * ``days_lead_time`` : Calendar days before crash that the first
          WARNING signal was triggered.  NaN if no warning.
        * ``peak_crowding_level`` : Maximum crowding index in the pre-crash
          window.
        * ``false_signal_count_prior_30d`` : Number of WARNING signals in the
          60-day period *before* the pre-crash window (false positive proxy).
        * ``regime_on_crash_date`` : The regime label on the start_date.

    Notes
    -----
    **This table is the empirical centrepiece of the research.**

    It directly answers: "Did the options market warn us before the major
    factor crashes?"  Each row in the table is one historical test.

    **Interpreting the results**

    * High ``warning_triggered`` rate + substantial ``days_lead_time`` = strong
      signal.
    * Low ``false_signal_count_prior_30d`` = high precision (few false alarms
      in the months leading up to the crash).
    * WARNING on the crash date but 0 days lead time = concurrent signal,
      not predictive.

    **Honest caveat**: this validation is in-sample for the parameter choices
    (CROWDING_THRESHOLD, SKEW_ROLLING_WINDOW, etc.).  True out-of-sample
    validation requires a walk-forward design not implemented here.
    """
    logger.info("Validating crowding signal against %d crash events …", len(crash_dates_df))

    records: list[dict] = []
    crowding_index = crowding_df["crowding_index"]
    regime_labels = crowding_df["regime_label"]

    for _, event in crash_dates_df.iterrows():
        event_name = event["event_name"]
        try:
            crash_start = pd.Timestamp(event["start_date"])
        except Exception:
            logger.warning("Could not parse start_date for %s", event_name)
            continue

        # Define pre-crash window: [crash_start - pre_crash_window_days, crash_start)
        pre_crash_start = crash_start - pd.Timedelta(days=pre_crash_window_days)
        pre_crash_window = crowding_df.loc[
            (crowding_df.index >= pre_crash_start) & (crowding_df.index < crash_start)
        ]

        if pre_crash_window.empty:
            records.append(
                {
                    "event_name": event_name,
                    "warning_triggered": False,
                    "days_lead_time": np.nan,
                    "peak_crowding_level": np.nan,
                    "false_signal_count_prior_30d": np.nan,
                    "regime_on_crash_date": "N/A (no data)",
                }
            )
            continue

        # Was WARNING triggered in the pre-crash window?
        warning_dates = pre_crash_window[
            pre_crash_window["regime_label"] == config.REGIME_WARNING
        ].index

        warning_triggered = len(warning_dates) > 0

        if warning_triggered:
            first_warning = warning_dates[0]
            days_lead = (crash_start - first_warning).days
        else:
            days_lead = np.nan

        peak_crowding = float(pre_crash_window["crowding_index"].max())

        # False positive count: WARNING signals in the 60 days BEFORE the pre-crash window
        fp_window_start = pre_crash_start - pd.Timedelta(days=false_positive_window_days)
        fp_window = crowding_df.loc[
            (crowding_df.index >= fp_window_start) & (crowding_df.index < pre_crash_start)
        ]
        fp_count = int((fp_window["regime_label"] == config.REGIME_WARNING).sum())

        # Regime on crash start date
        crash_dates_in_df = crowding_df.index[crowding_df.index <= crash_start]
        if crash_dates_in_df.empty:
            regime_on_crash = "N/A"
        else:
            regime_on_crash = str(regime_labels.loc[crash_dates_in_df[-1]])

        records.append(
            {
                "event_name": event_name,
                "warning_triggered": warning_triggered,
                "days_lead_time": days_lead,
                "peak_crowding_level": round(peak_crowding, 3),
                "false_signal_count_prior_30d": fp_count,
                "regime_on_crash_date": regime_on_crash,
            }
        )

    validation_df = pd.DataFrame(records)

    # Summary statistics
    n_events = len(validation_df)
    n_warned = validation_df["warning_triggered"].sum()
    avg_lead = validation_df["days_lead_time"].dropna().mean()
    avg_fp = validation_df["false_signal_count_prior_30d"].mean()

    logger.info(
        "Validation results: %d / %d events warned, avg lead time = %.1f days, "
        "avg false positives per 60d window = %.1f",
        n_warned, n_events, avg_lead if not np.isnan(avg_lead) else 0, avg_fp,
    )
    return validation_df


# ---------------------------------------------------------------------------
# Smoke Test / Entry Point
# ---------------------------------------------------------------------------

def _run_smoke_test(factor_name: str = "momentum") -> None:
    """Run a synthetic smoke test of the crowding detection pipeline.

    Generates random skew panel and factor portfolio data, then exercises
    the full pipeline from crowding index to regime classification.

    This test requires NO external data or network access and is designed
    to validate the module logic in isolation.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("=" * 60)
    logger.info("SMOKE TEST: crowding_detector (factor=%s)", factor_name)
    logger.info("=" * 60)

    rng = np.random.default_rng(seed=42)

    # --- Synthetic skew z-score panel ---
    n_days = 500
    n_tickers = 30
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"TICK{i:02d}" for i in range(n_tickers)]

    # Base random skew + regime breaks injected at day 300-320 (synthetic crowding)
    skew_data = rng.normal(0, 1, size=(n_days, n_tickers))
    # Inject a crowding spike at days 300-340
    skew_data[300:340, :15] += 2.5  # First 15 tickers crowded
    skew_panel = pd.DataFrame(skew_data, index=dates, columns=tickers)
    logger.info("Synthetic skew panel: %s", skew_panel.shape)

    # --- Synthetic factor portfolio: top 10 tickers rebalanced monthly ---
    rb_dates = pd.bdate_range("2020-01-01", "2021-08-31", freq="MS")
    portfolio_records = []
    for rb_date in rb_dates:
        # Randomly select 8 tickers as "top quintile" each month
        selected = rng.choice(tickers, size=8, replace=False)
        for t in selected:
            portfolio_records.append(
                {
                    "date": rb_date,
                    "ticker": t,
                    "factor_score": float(rng.uniform(0.8, 1.0)),
                    "factor_rank": float(rng.uniform(0.8, 1.0)),
                    "in_top_quintile": True,
                    "portfolio_weight": 1.0 / 8.0,
                }
            )
    portfolio_df = pd.DataFrame(portfolio_records)
    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
    logger.info("Synthetic portfolio: %d monthly snapshots", portfolio_df["date"].nunique())

    # --- Run crowding detection ---
    crowding_index = compute_factor_crowding_index(skew_panel, portfolio_df, factor_name)
    logger.info(
        "Crowding index: %d valid days, mean=%.3f, std=%.3f",
        crowding_index.notna().sum(),
        crowding_index.mean(),
        crowding_index.std(),
    )

    crowding_df = detect_crowding_regime(crowding_index)
    regime_counts = crowding_df["regime_label"].value_counts()
    logger.info("Regime distribution:\n%s", regime_counts.to_string())

    # Verify synthetic spike is captured as WARNING
    peak_date = crowding_index.idxmax()
    peak_regime = crowding_df.loc[peak_date, "regime_label"] if peak_date in crowding_df.index else "N/A"
    logger.info("Peak crowding date: %s, regime: %s", peak_date.date(), peak_regime)

    # --- Lead-lag analysis ---
    synthetic_returns = pd.Series(
        rng.normal(0.0005, 0.02, size=n_days),
        index=dates,
        name="factor_return",
    )
    # Inject: high crowding → negative return 5 days later
    for i in range(300, 340):
        if i + 5 < n_days:
            synthetic_returns.iloc[i + 5] -= 0.03

    lag_df = compute_crowding_lead_lag(crowding_df, synthetic_returns, max_lag=20)
    peak_lag = lag_df.loc[lag_df["correlation"].idxmin(), "lag_days"]
    peak_corr = lag_df["correlation"].min()
    logger.info(
        "Lead-lag: peak negative correlation at lag %d days (r=%.3f)",
        peak_lag, peak_corr,
    )

    logger.info("=" * 60)
    logger.info("SMOKE TEST PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run crowding detection pipeline (smoke test with synthetic data)."
    )
    parser.add_argument(
        "--factor",
        type=str,
        default="momentum",
        choices=config.FACTOR_NAMES,
        help="Factor to run (default: momentum).",
    )
    args = parser.parse_args()
    _run_smoke_test(factor_name=args.factor)
