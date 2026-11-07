"""
skew_engine.py
==============
Options put-skew computation, rolling z-score construction, and VIX overlay.

The put skew is the core signal in this framework.  It measures the excess
implied volatility of out-of-the-money (25-delta) puts relative to at-the-money
puts on the same expiry.  When large institutional investors hedge concentrated
factor positions, they buy OTM puts, driving up the OTM put IV relative to ATM.

Skew construction pipeline
---------------------------
1. Fetch raw options data for each ticker/date (with cache).
2. Compute ``put_skew = IV(25d put) - IV(ATM put)`` per ticker per date.
3. Build a wide panel: dates × tickers.
4. Compute rolling z-score with winsorisation.
5. Merge VIX term structure as a macro overlay.

Notes on data quality
~~~~~~~~~~~~~~~~~~~~~~
Single-name options data quality varies significantly:

* **Large caps** (top 50 S&P 500): liquid options markets, tight bid-ask,
  reliable IV surface — skew signal is high quality.
* **Smaller names**: wide bid-ask spreads inflate IV noise; skew z-score
  will exhibit more mean-reversion noise.  For production, filter to the top
  50 names by options open interest before computing the crowding index.
* **Time coverage**: full options history (with delta data) is expensive.
  This module caches each ticker/date as a JSON file to minimise re-fetching.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src.data_loader import fetch_options_skew_barchart, fetch_cboe_vix_term_structure
from src.utils import rolling_zscore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Single-ticker skew computation
# ---------------------------------------------------------------------------

def compute_put_skew_single(
    ticker: str,
    date: str,
    options_data: dict,
) -> float:
    """Extract put skew from a pre-fetched options data dict.

    Parameters
    ----------
    ticker : str
        Equity ticker (used for logging only).
    date : str
        Reference date (used for logging only).
    options_data : dict
        Output of ``fetch_options_skew_barchart``.  Must have keys:
        ``put_iv_25d``, ``atm_iv``.

    Returns
    -------
    float
        ``put_skew = IV(25d put) - IV(ATM put)``.
        Returns ``np.nan`` if either leg is missing.

    Notes
    -----
    A positive put skew is the normal state: OTM puts trade at a premium to
    ATM because of the volatility risk premium and demand for crash protection.
    An unusually high skew (high z-score) signals elevated hedging demand —
    the crowding signal.

    Confidence check: if ``put_iv_25d`` or ``atm_iv`` is NaN, or if either is
    outside a reasonable range (0.01 to 5.0 in annualised vol terms), the
    function returns NaN.
    """
    put_25d = options_data.get("put_iv_25d", np.nan)
    atm = options_data.get("atm_iv", np.nan)

    if np.isnan(put_25d) or np.isnan(atm):
        return np.nan

    # Sanity check: IV must be in (1%, 500%) range
    if not (0.01 < put_25d < 5.0) or not (0.01 < atm < 5.0):
        logger.debug(
            "Implausible IV for %s @ %s: put_25d=%.3f, atm=%.3f — returning NaN",
            ticker, date, put_25d, atm,
        )
        return np.nan

    return float(put_25d - atm)


# ---------------------------------------------------------------------------
# 2. Skew time-series for a single ticker
# ---------------------------------------------------------------------------

def build_skew_timeseries(
    ticker: str,
    date_range: pd.DatetimeIndex,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build a daily put-skew time series for one ticker.

    Parameters
    ----------
    ticker : str
        Equity ticker.
    date_range : pd.DatetimeIndex
        Trading dates to process.
    use_cache : bool, optional
        Use JSON cache if available.  Default True.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``ticker``, ``put_skew``, ``put_iv_25d``,
        ``atm_iv``, ``data_source``.
        Rows for which valid skew data could not be obtained are included
        with NaN in the skew columns.

    Notes
    -----
    Data coverage is logged at the end: the fraction of dates with valid skew
    readings.  Coverage below 50% suggests options data quality issues for this
    ticker (e.g., illiquid options, no listed puts).
    """
    records: list[dict] = []
    date_strs = [d.strftime("%Y-%m-%d") for d in date_range]

    for date_str in date_strs:
        raw = fetch_options_skew_barchart(ticker, date_str, use_cache=use_cache)
        put_skew = compute_put_skew_single(ticker, date_str, raw)
        records.append(
            {
                "date": pd.Timestamp(date_str),
                "ticker": ticker,
                "put_skew": put_skew,
                "put_iv_25d": raw.get("put_iv_25d", np.nan),
                "atm_iv": raw.get("atm_iv", np.nan),
                "data_source": raw.get("data_source", "unknown"),
            }
        )

    df = pd.DataFrame(records)
    valid_count = df["put_skew"].notna().sum()
    total_count = len(df)
    logger.info(
        "  %s: skew coverage = %d / %d (%.1f%%)",
        ticker, valid_count, total_count, 100 * valid_count / max(total_count, 1),
    )
    return df


# ---------------------------------------------------------------------------
# 3. Universe-wide skew panel
# ---------------------------------------------------------------------------

def build_universe_skew_panel(
    tickers: list[str],
    date_range: pd.DatetimeIndex,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build the full cross-sectional skew panel for all tickers.

    Parameters
    ----------
    tickers : list[str]
        Universe of equity tickers.
    date_range : pd.DatetimeIndex
        Trading dates to process.
    use_cache : bool, optional
        Use JSON cache.  Default True.

    Returns
    -------
    pd.DataFrame
        Wide-format skew panel: index = dates, columns = tickers,
        values = put_skew.
        Saved to ``config.SKEW_PANEL_PATH``.

    Notes
    -----
    This function can take several minutes for a large universe because options
    data is fetched one ticker at a time.  The per-ticker JSON cache ensures
    that only missing data points are re-fetched on subsequent runs.

    For production use, parallelize with ``concurrent.futures.ThreadPoolExecutor``
    (the rate-limit sleep in ``fetch_options_skew_barchart`` limits the speedup
    to ~3-5×, but it avoids getting blocked).
    """
    if config.SKEW_PANEL_PATH.exists() and use_cache:
        logger.info("Loading skew panel from cache: %s", config.SKEW_PANEL_PATH)
        return pd.read_parquet(config.SKEW_PANEL_PATH)

    logger.info(
        "Building skew panel for %d tickers over %d dates …",
        len(tickers), len(date_range),
    )

    all_frames: list[pd.DataFrame] = []
    for ticker in tqdm(tickers, desc="Fetching skew data"):
        ts = build_skew_timeseries(ticker, date_range, use_cache=use_cache)
        all_frames.append(ts[["date", "ticker", "put_skew"]])

    if not all_frames:
        raise RuntimeError("No skew data fetched.")

    long_df = pd.concat(all_frames, ignore_index=True)

    # Pivot to wide format
    wide_df = long_df.pivot(index="date", columns="ticker", values="put_skew")
    wide_df.sort_index(inplace=True)

    wide_df.to_parquet(config.SKEW_PANEL_PATH)
    logger.info("Skew panel saved to %s (%s)", config.SKEW_PANEL_PATH, wide_df.shape)
    return wide_df


# ---------------------------------------------------------------------------
# 4. Rolling skew z-score
# ---------------------------------------------------------------------------

def compute_rolling_skew_zscore(
    skew_panel: pd.DataFrame,
    window: int = config.SKEW_ROLLING_WINDOW,
    winsorise_clip: float = config.SKEW_WINSOR_CLIP,
) -> pd.DataFrame:
    """Compute rolling z-score for each ticker's put-skew time series.

    Parameters
    ----------
    skew_panel : pd.DataFrame
        Wide skew panel (index = dates, columns = tickers).
    window : int, optional
        Rolling window (days) for z-score normalisation.  Default 20.
    winsorise_clip : float, optional
        Clip z-scores to ±``winsorise_clip``.  Default ±4.

    Returns
    -------
    pd.DataFrame
        Same structure as ``skew_panel`` but values are rolling z-scores.
        Saved to ``config.SKEW_ZSCORE_PANEL_PATH``.

    Notes
    -----
    Winsorisation at ±4 clips extreme readings that are almost certainly data
    errors (e.g., a stale bid on a wide-market option briefly quoted at 300%
    IV).  Without winsorisation, a single bad data point would dominate the
    cross-sectional median and generate a false WARNING regime.

    The rolling window of 20 days captures approximately one month of trading.
    A shorter window (10 days) is more reactive but noisier; a longer window
    (40 days) is smoother but slower to adapt to regime changes.  The 20-day
    default was chosen to balance timeliness and noise.
    """
    logger.info(
        "Computing rolling skew z-score (window=%d, winsorise=±%.1f) …",
        window, winsorise_clip,
    )

    zscore_panel = pd.DataFrame(index=skew_panel.index, columns=skew_panel.columns, dtype=float)

    for col in skew_panel.columns:
        zscore_panel[col] = rolling_zscore(
            skew_panel[col],
            window=window,
            winsorise_clip=winsorise_clip,
        )

    zscore_panel.to_parquet(config.SKEW_ZSCORE_PANEL_PATH)
    logger.info("Skew z-score panel saved to %s", config.SKEW_ZSCORE_PANEL_PATH)
    return zscore_panel


# ---------------------------------------------------------------------------
# 5. VIX overlay
# ---------------------------------------------------------------------------

def add_vix_overlay(
    skew_zscore_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge VIX term-structure data onto the skew z-score panel.

    Parameters
    ----------
    skew_zscore_df : pd.DataFrame
        Skew z-score panel (index = dates).
    vix_df : pd.DataFrame, optional
        VIX term-structure data from ``fetch_cboe_vix_term_structure``.
        If None, the function fetches it automatically.

    Returns
    -------
    pd.DataFrame
        ``skew_zscore_df`` with three additional columns appended:
        ``vix_spot``, ``vix3m``, ``vix_term_spread``.

    Notes
    -----
    The VIX term spread (VIX3M − VIX spot) serves as a macro conditioning
    variable.  When the term structure is in backwardation (spread < 0),
    near-term fear exceeds medium-term fear — consistent with a crowded
    unwind in progress.  Cross-conditioning the crowding index on term
    structure backwardation can reduce false positives.

    Usage example::

        if regime == WARNING and vix_term_spread < 0:
            # Higher conviction: skew-based signal is confirmed by VIX structure
            exposure = 0.30   # Reduce further
        elif regime == WARNING and vix_term_spread > 0:
            # Lower conviction: skew signal may be a false positive
            exposure = 0.60   # Moderate reduction

    This interaction is not implemented in the default regime_filter but is
    documented here as an extension direction.
    """
    if vix_df is None:
        logger.info("Fetching VIX term structure for overlay …")
        vix_df = fetch_cboe_vix_term_structure()

    vix_aligned = vix_df[["vix_spot", "vix3m", "vix_term_spread"]].reindex(
        skew_zscore_df.index, method="ffill"
    )

    enriched = skew_zscore_df.copy()
    enriched["vix_spot"] = vix_aligned["vix_spot"]
    enriched["vix3m"] = vix_aligned["vix3m"]
    enriched["vix_term_spread"] = vix_aligned["vix_term_spread"]

    logger.info("VIX overlay merged. VIX columns added to skew z-score panel.")
    return enriched
