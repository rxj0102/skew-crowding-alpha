"""
config.py
=========
Central configuration for the skew-crowding-alpha research repository.

All hardcoded parameters, file paths, and strategy constants live here.
No other module should define paths or tunable constants — import from config.

Notes
-----
Modify parameters here to run sensitivity analyses (e.g., change
CROWDING_THRESHOLD from 1.5 to 2.0, or SKEW_ROLLING_WINDOW from 20 to 10).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root (all paths are relative to this)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_REFERENCE_DIR = DATA_DIR / "reference"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# Ensure output directories exist at import time
for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_REFERENCE_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
SP500_TICKERS_URL: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UNIVERSE_SIZE: int = 100  # Top 100 S&P 500 by market cap for tractability

# ---------------------------------------------------------------------------
# Factor parameters
# ---------------------------------------------------------------------------
MOMENTUM_LOOKBACK: int = 252     # 12-month lookback in trading days
MOMENTUM_SKIP: int = 21          # Skip most recent month (12-1 momentum convention)
FACTOR_QUINTILE: float = 0.80    # Top quintile threshold (top 20%)
LOW_VOL_WINDOW: int = 63         # Rolling window for realized vol (63 ≈ 1 quarter)
QUALITY_WINDOW: int = 252        # Trailing window for quality drawdown proxy

FACTOR_NAMES: list[str] = ["momentum", "low_vol", "quality"]

# ---------------------------------------------------------------------------
# Skew parameters
# ---------------------------------------------------------------------------
SKEW_ROLLING_WINDOW: int = 20        # Days for rolling skew z-score calculation
SKEW_ZSCORE_HISTORY: int = 60        # Baseline window (days) for crowding index mean/std
CROWDING_THRESHOLD: float = 1.5      # Z-score threshold for WARNING regime
ELEVATED_THRESHOLD: float = 0.75     # Z-score threshold for ELEVATED regime
DEPRESSED_THRESHOLD: float = -0.75   # Z-score threshold for DEPRESSED regime
PUT_DELTA_TARGET: float = 0.25       # 25-delta put target
ATM_DELTA_TARGET: float = 0.50       # ATM option target
SKEW_WINSOR_CLIP: float = 4.0        # Winsorize z-scores at ±4

# ---------------------------------------------------------------------------
# Regime parameters
# ---------------------------------------------------------------------------
WARNING_EXPOSURE: float = 0.50        # Factor exposure in WARNING regime (50%)
ELEVATED_EXPOSURE: float = 0.75       # Factor exposure in ELEVATED regime (75%)
NORMAL_EXPOSURE: float = 1.00         # Factor exposure in NORMAL regime (100%)
DEPRESSED_EXPOSURE: float = 1.10      # Factor exposure in DEPRESSED regime (110%, experimental)

DEFENSIVE_TICKERS: list[str] = ["XLU", "XLP", "SPLV", "VYM"]

# Regime labels
REGIME_WARNING: str = "WARNING"
REGIME_ELEVATED: str = "ELEVATED"
REGIME_NORMAL: str = "NORMAL"
REGIME_DEPRESSED: str = "DEPRESSED"

REGIME_EXPOSURE_MAP: dict[str, float] = {
    REGIME_WARNING: WARNING_EXPOSURE,
    REGIME_ELEVATED: ELEVATED_EXPOSURE,
    REGIME_NORMAL: NORMAL_EXPOSURE,
    REGIME_DEPRESSED: DEPRESSED_EXPOSURE,
}

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------
BACKTEST_START: str = "2010-01-01"
BACKTEST_END: str = "2023-12-31"
REBALANCE_FREQUENCY: str = "monthly"
TRANSACTION_COST_BPS: int = 10  # One-way transaction cost (basis points)
RISK_FREE_RATE: float = 0.02    # Annual risk-free rate for Sharpe/Sortino

# ---------------------------------------------------------------------------
# Factor ETF proxies (for reality-check benchmarking)
# ---------------------------------------------------------------------------
FACTOR_ETF_PROXIES: dict[str, str] = {
    "momentum": "MTUM",
    "low_vol": "USMV",
    "quality": "QUAL",
    "value": "VLUE",
}

# ---------------------------------------------------------------------------
# CBOE data URLs (free, no authentication required)
# ---------------------------------------------------------------------------
CBOE_VIX_URL: str = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
)
CBOE_VIX3M_URL: str = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"
)

# ---------------------------------------------------------------------------
# Known factor crash dates for event study
# ---------------------------------------------------------------------------
FACTOR_CRASH_DATES: dict[str, str] = {
    "Quant Quake Aug 2007": "2007-08-06",
    "GFC Deleveraging Sep 2008": "2008-09-15",
    "Value Trap Feb 2009": "2009-02-09",
    "Vol Spike Feb 2018": "2018-02-05",
    "COVID Crash Mar 2020": "2020-03-16",
    "Momentum Crash Sep 2020": "2020-09-04",
    "Rate Shock Nov 2021": "2021-11-26",
    "Risk-Off May 2012": "2012-05-01",
}

# ---------------------------------------------------------------------------
# File paths (all data artifacts)
# ---------------------------------------------------------------------------
SP500_TICKERS_PATH = DATA_PROCESSED_DIR / "sp500_tickers.csv"
PRICE_DATA_PATH = DATA_PROCESSED_DIR / "price_data.parquet"
MARKET_CAP_PATH = DATA_PROCESSED_DIR / "market_cap.csv"
VIX_TERM_STRUCTURE_PATH = DATA_PROCESSED_DIR / "vix_term_structure.csv"
FACTOR_ETF_PATH = DATA_PROCESSED_DIR / "factor_etf_proxies.parquet"
SKEW_PANEL_PATH = DATA_PROCESSED_DIR / "skew_panel.parquet"
SKEW_ZSCORE_PANEL_PATH = DATA_PROCESSED_DIR / "skew_zscore_panel.parquet"

FACTOR_CRASH_DATES_CSV = DATA_REFERENCE_DIR / "factor_crash_dates.csv"
SECTOR_ETF_MAP_CSV = DATA_REFERENCE_DIR / "sector_etf_map.csv"

def factor_portfolio_path(factor_name: str) -> Path:
    """Return path for a factor portfolio CSV."""
    return DATA_PROCESSED_DIR / f"{factor_name}_portfolio.csv"

def factor_crowding_index_path(factor_name: str) -> Path:
    """Return path for a factor crowding index CSV."""
    return DATA_PROCESSED_DIR / f"{factor_name}_crowding_index.csv"

def raw_skew_path(ticker: str, date: str) -> Path:
    """Return path for a single ticker/date raw skew JSON cache."""
    return DATA_RAW_DIR / f"{ticker}_{date}_skew.json"

# ---------------------------------------------------------------------------
# Visualization settings
# ---------------------------------------------------------------------------
FIGURE_DPI: int = 300
FIGURE_STYLE: str = "seaborn-v0_8-whitegrid"
FIGURE_SIZE_WIDE: tuple[int, int] = (16, 8)
FIGURE_SIZE_SQUARE: tuple[int, int] = (10, 10)
FIGURE_SIZE_MEDIUM: tuple[int, int] = (12, 6)

REGIME_COLORS: dict[str, str] = {
    REGIME_WARNING: "#FFCCCC",    # Light red
    REGIME_ELEVATED: "#FFE5CC",   # Light orange
    REGIME_NORMAL: "white",
    REGIME_DEPRESSED: "#CCE5FF",  # Light blue
}

REGIME_LINE_COLORS: dict[str, str] = {
    REGIME_WARNING: "#CC0000",
    REGIME_ELEVATED: "#FF6600",
    REGIME_NORMAL: "#666666",
    REGIME_DEPRESSED: "#0066CC",
}
