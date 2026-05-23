"""
regime_filter.py
================
Regime-conditioned exposure scaling for factor portfolios.

The regime filter translates the crowding index regime classification into
portfolio weight adjustments:

+-------------+---------+---------------------------------------------------+
| Regime      | Weight  | Action                                            |
+=============+=========+===================================================+
| WARNING     |  50%    | Reduce factor; allocate remainder to defensives.  |
| ELEVATED    |  75%    | Reduce factor; retain partial defensive overlay.  |
| NORMAL      | 100%    | Full factor exposure.                             |
| DEPRESSED   | 110%    | Mild factor lever (experimental, use with care).  |
+-------------+---------+---------------------------------------------------+

Look-ahead bias prevention
--------------------------
**All regime classifications are lagged by one trading day** before
influencing portfolio weights.  This is enforced via ``src.utils.lag_signal``
and explicitly documented in every function.  The pattern is:

    regime_for_trading = lag_signal(regime_from_today, periods=1)

Without this lag, the strategy would "know" tonight's regime and trade on it
today — an unrealisable edge in live markets.

Defensive basket
----------------
When factor exposure is below 100%, the residual allocation goes to the
defensive ETF basket (XLU, XLP, SPLV, VYM).  These are equal-weighted.
The defensive basket provides capital preservation during crowding unwinds
while maintaining market participation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import lag_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Defensive Basket Returns
# ---------------------------------------------------------------------------

def compute_defensive_basket_return(
    price_df: pd.DataFrame,
    defensive_tickers: Optional[list[str]] = None,
    date_range: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """Compute daily equal-weight defensive ETF basket returns.

    Parameters
    ----------
    price_df : pd.DataFrame
        Adjusted close prices (wide format, index = dates).
    defensive_tickers : list[str], optional
        List of defensive ticker symbols.  Defaults to ``config.DEFENSIVE_TICKERS``.
    date_range : pd.DatetimeIndex, optional
        Date range to restrict the computation.

    Returns
    -------
    pd.Series
        Daily equal-weight returns of the defensive basket.
        Name = ``"defensive_basket_return"``.

    Notes
    -----
    The default defensive basket (XLU, XLP, SPLV, VYM) targets:

    * **XLU** (Utilities): Low beta, high dividend yield.
    * **XLP** (Consumer Staples): Defensive sector, non-cyclical revenues.
    * **SPLV** (S&P 500 Low Volatility): Explicit low-vol factor overlay.
    * **VYM** (Vanguard High Dividend): Dividend yield as quality proxy.

    For periods before these ETFs listed (XLP: 1998, XLU: 1998, SPLV: 2011,
    VYM: 2006), available tickers are used and the rest are excluded.
    """
    if defensive_tickers is None:
        defensive_tickers = config.DEFENSIVE_TICKERS

    available = [t for t in defensive_tickers if t in price_df.columns]
    if not available:
        logger.warning("No defensive tickers found in price_df. Using market returns as proxy.")
        returns = price_df.pct_change().mean(axis=1)
        returns.name = "defensive_basket_return"
        return returns

    prices = price_df[available]
    if date_range is not None:
        prices = prices.loc[prices.index.isin(date_range)]

    returns = prices.pct_change()
    basket_return = returns.mean(axis=1)
    basket_return.name = "defensive_basket_return"
    return basket_return


# ---------------------------------------------------------------------------
# 2. Regime-Based Exposure Scaling
# ---------------------------------------------------------------------------

def apply_regime_filter(
    factor_returns: pd.Series,
    crowding_df: pd.DataFrame,
    price_df: pd.DataFrame,
    defensive_tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Apply regime-conditional exposure scaling to factor portfolio returns.

    Parameters
    ----------
    factor_returns : pd.Series
        Raw factor portfolio returns (daily or monthly).
    crowding_df : pd.DataFrame
        Output of ``crowding_detector.detect_crowding_regime``.
        Must have column ``regime_label``.
    price_df : pd.DataFrame
        Adjusted close prices for computing the defensive basket return.
    defensive_tickers : list[str], optional
        Defensive ETF tickers.  Defaults to ``config.DEFENSIVE_TICKERS``.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``raw_factor_return``, ``filtered_return``,
        ``regime_label``, ``factor_weight``, ``defensive_weight``.

    Notes
    -----
    **CRITICAL: Look-ahead bias prevention**

    The regime classification used on date T is the regime from date T-1
    (lagged by 1 day using ``lag_signal``).  This means:

    * We observe the closing prices on day T-1.
    * Crowding index and regime are computed from those prices.
    * The portfolio position on day T uses the previous day's regime.

    This is the standard "signal from yesterday, trade today" convention.
    It is explicitly enforced here via ``lag_signal``.  Do NOT bypass this
    lag without understanding the look-ahead bias implications.

    **Exposure scaling**

    The factor weight and defensive weight sum to 1.0 in all regimes:
    * WARNING: 50% factor, 50% defensive
    * ELEVATED: 75% factor, 25% defensive
    * NORMAL: 100% factor, 0% defensive
    * DEPRESSED: 110% factor, -10% defensive (notional short of defensive basket,
      experimental — flag this as a margin-requiring strategy)

    **Transaction costs**

    This function does NOT apply transaction costs on the regime transitions.
    These should be applied in ``backtest.py`` after computing turnover from
    the weight series.
    """
    if defensive_tickers is None:
        defensive_tickers = config.DEFENSIVE_TICKERS

    logger.info("Applying regime filter …")

    # Step 1: Lag regime by 1 day (look-ahead bias prevention)
    regime_raw = crowding_df["regime_label"].copy()
    regime_lagged = lag_signal(regime_raw, periods=1)
    regime_lagged.fillna(config.REGIME_NORMAL, inplace=True)  # Assume NORMAL before signal

    # Step 2: Compute defensive basket returns
    def_returns = compute_defensive_basket_return(price_df, defensive_tickers)

    # Step 3: Align all series to the same date range
    common_dates = factor_returns.index.intersection(regime_lagged.index)
    if common_dates.empty:
        logger.error("No overlapping dates between factor_returns and crowding_df.")
        return pd.DataFrame()

    factor_ret_aligned = factor_returns.reindex(common_dates)
    def_ret_aligned = def_returns.reindex(common_dates, fill_value=0.0)
    regime_aligned = regime_lagged.reindex(common_dates, fill_value=config.REGIME_NORMAL)

    # Step 4: Compute factor and defensive weights per day
    factor_weights = regime_aligned.map(config.REGIME_EXPOSURE_MAP).fillna(config.NORMAL_EXPOSURE)
    defensive_weights = 1.0 - factor_weights

    # Step 5: Compute filtered returns
    filtered_returns = (
        factor_weights * factor_ret_aligned
        + defensive_weights * def_ret_aligned.fillna(0.0)
    )

    result = pd.DataFrame(
        {
            "raw_factor_return": factor_ret_aligned,
            "filtered_return": filtered_returns,
            "regime_label": regime_aligned,
            "factor_weight": factor_weights,
            "defensive_weight": defensive_weights,
        },
        index=common_dates,
    )
    result.index.name = "date"

    # Log regime statistics
    regime_days = regime_aligned.value_counts()
    logger.info("Regime days during backtest:\n%s", regime_days.to_string())
    logger.info(
        "Mean factor weight: %.2f | Mean defensive weight: %.2f",
        factor_weights.mean(),
        defensive_weights.mean(),
    )
    return result


# ---------------------------------------------------------------------------
# 3. Regime Transition Matrix
# ---------------------------------------------------------------------------

def compute_regime_transition_matrix(crowding_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the empirical Markov transition probability matrix.

    Parameters
    ----------
    crowding_df : pd.DataFrame
        Output of ``detect_crowding_regime``.  Must have ``regime_label``.

    Returns
    -------
    pd.DataFrame
        4 × 4 transition probability matrix.
        ``transition_matrix.loc[from_regime, to_regime]`` is the empirical
        probability of transitioning from ``from_regime`` to ``to_regime``
        on the next trading day.

    Notes
    -----
    **Regime persistence and trading implications**

    The diagonal of this matrix gives the probability of the regime
    *remaining the same* on the next day.  This is the regime persistence:

    * If ``P(WARNING → WARNING) > 0.70``: the WARNING regime is sticky.
      A single-day trigger is reliable — act on the first WARNING day.
    * If ``P(WARNING → WARNING) < 0.50``: the regime is transient.
      Add a confirmation rule: require 3 consecutive WARNING days before
      reducing exposure.  This reduces false positives at the cost of
      some lag.

    **Computing turnover from transitions**

    The off-diagonal elements ``P(NORMAL → WARNING)`` and
    ``P(WARNING → NORMAL)`` determine how often the portfolio shifts
    between full and reduced exposure.  High transition frequencies
    imply high turnover and higher transaction costs.

    This matrix can also be used to compute the **steady-state regime
    distribution** (the left eigenvector of the transition matrix), which
    tells you what fraction of trading days the portfolio will be in each
    regime in the long run.
    """
    logger.info("Computing regime transition matrix …")

    ordered_regimes = [
        config.REGIME_WARNING,
        config.REGIME_ELEVATED,
        config.REGIME_NORMAL,
        config.REGIME_DEPRESSED,
    ]

    labels = crowding_df["regime_label"]
    transitions = pd.crosstab(
        labels.iloc[:-1].values,
        labels.iloc[1:].values,
        normalize="index",
    )

    # Ensure all regimes are present (fill missing with 0)
    transitions = transitions.reindex(index=ordered_regimes, columns=ordered_regimes, fill_value=0.0)

    logger.info("Regime transition matrix:\n%s", transitions.to_string())

    # Persistence summary
    for regime in ordered_regimes:
        if regime in transitions.index:
            persistence = transitions.loc[regime, regime]
            logger.info("  %s persistence: %.1f%%", regime, 100 * persistence)

    return transitions
