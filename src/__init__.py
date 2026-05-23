"""
skew-crowding-alpha
===================
Options Implied Skew as a Regime Classifier for Equity Factor Crowding.

This package implements a research framework that uses the options market's
forward-looking hedging demand (put skew) as a leading indicator of equity
factor crowding risk. When institutional investors hedge concentrated factor
positions, the resulting demand for downside protection manifests as elevated
put implied volatility relative to ATM volatility — the put skew.

Modules
-------
data_loader      : Data ingestion (prices, options skew, VIX term structure)
factor_engine    : Factor score computation (momentum, low-vol, quality)
skew_engine      : Put skew computation, rolling z-score, VIX overlay
crowding_detector: Crowding index construction and regime classification
regime_filter    : Regime-conditioned exposure scaling
backtest         : Performance evaluation and event study
visualize        : All figures
"""

__version__ = "1.0.0"
__author__ = "skew-crowding-alpha research"
