# Options Implied Skew as a Regime Classifier for Equity Factor Crowding

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Abstract

Equity factor strategies — momentum, low volatility, quality — have earned durable risk premia over decades, but are periodically subject to sudden, severe drawdowns driven by crowding: the simultaneous exit of overlapping institutional positions. Traditional crowding metrics (13F overlap analysis, return correlation, bid-ask spread widening) are *lagging* by construction: they confirm that a crowd existed after it has already begun to unwind.

This repository implements a framework that uses the **options market's forward-looking hedging demand** as a *leading* indicator of factor crowding risk. When institutional investors accumulate large factor positions, they simultaneously buy downside protection — demand that manifests as elevated out-of-the-money put implied volatility relative to at-the-money IV (the *put skew*). By aggregating single-name put skew signals across a factor's constituents into a **Factor Crowding Index**, and classifying the current market regime, this framework aims to provide actionable advance warning of factor unwind risk.

The central contribution is a regime-conditioned portfolio construction rule: in WARNING regimes (crowding index z-score > 1.5), factor exposure is reduced to 50% with the balance allocated to a defensive ETF basket, providing capital protection during factor crashes while maintaining market participation.

---

## Core Hypothesis

**H₀ (Null):** The cross-sectional median put-skew z-score of a factor portfolio's constituents has no predictive power for future factor returns. The observed correlations between put-skew elevation and subsequent factor crashes are attributable to chance.

**H₁ (Alternative):** Elevated cross-sectional median put-skew z-scores among a factor's top-quintile constituents significantly predict negative factor returns over the following 5–20 trading days, reflecting institutional hedging of concentrated factor positions as a leading indicator of crowding risk.

---

## Key Distinction: Why Options Skew?

Existing crowding measures are **lagging by construction**:

| Metric | What it Measures | Lag |
|--------|-----------------|-----|
| 13F overlap (quarterly) | Historical institutional positions | 45–90 days |
| Return correlation spike | Co-movement during unwind | Concurrent |
| Bid-ask spread widening | Liquidity deterioration during selling | Concurrent |
| Factor volatility elevation | Realised vol during unwind | Concurrent |

**This framework uses the options market's forward-looking hedging demand as a leading indicator of factor unwind risk.** When large institutional investors hedge concentrated factor positions, they buy OTM puts *before* the unwind begins. This demand drives up put implied volatility relative to ATM — creating an observable signal in the options market *before* the equity market registers the stress.

The key insight: **the options tail wags the equity dog.** Options markets, dominated by sophisticated institutional hedgers, often price in risk before it becomes visible in equity price action.

---

## Methodology Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                        │
│  Price Data (yfinance) → Adjusted Closes (100 tickers, 2010-2023)  │
│  Options Data (Barchart/yfinance) → IV Surface per ticker per date  │
│  CBOE VIX Term Structure → Macro conditioning variable              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
         ┌─────────────────┐    ┌─────────────────────────┐
         │  FACTOR ENGINE  │    │    SKEW ENGINE           │
         │                 │    │                          │
         │ Momentum (12-1) │    │ put_skew = IV(25Δ) -     │
         │ Low Volatility  │    │           IV(ATM)        │
         │ Quality (proxy) │    │                          │
         │                 │    │ Rolling z-score          │
         │ Top Quintile    │    │ (20-day window, ±4 clip) │
         │ Portfolio       │    │                          │
         └────────┬────────┘    └───────────┬─────────────┘
                  │                         │
                  └──────────┬──────────────┘
                             ▼
            ┌────────────────────────────────────┐
            │         CROWDING DETECTOR          │
            │                                    │
            │  Factor Crowding Index =            │
            │    median{ z_{i,t} | i ∈ TopQ_t }  │
            │                                    │
            │  Regime Z-Score =                  │
            │    (CI_t - μ_{60d}) / σ_{60d}      │
            │                                    │
            │  ┌────────────────────────────┐   │
            │  │ WARNING   z > +1.5         │   │
            │  │ ELEVATED  +0.75 < z ≤ +1.5 │   │
            │  │ NORMAL    |z| ≤ 0.75       │   │
            │  │ DEPRESSED z ≤ -0.75        │   │
            │  └────────────────────────────┘   │
            └───────────────┬────────────────────┘
                            │
            ┌───────────────┴────────────────────┐
            │         REGIME FILTER               │
            │  (Regime lagged 1 day — no look-    │
            │   ahead bias)                       │
            │                                     │
            │  WARNING   → 50% factor + 50% def  │
            │  ELEVATED  → 75% factor + 25% def  │
            │  NORMAL    → 100% factor            │
            │  DEPRESSED → 110% factor [exp.]    │
            └───────────────┬────────────────────┘
                            │
            ┌───────────────┴────────────────────┐
            │         FILTERED PORTFOLIO          │
            │   Risk-adjusted factor exposure     │
            │   + Defensive ETF basket overlay    │
            └────────────────────────────────────┘
```

---

## Mathematical Formulation

### Factor Crowding Index

$$\text{CI}_{f,t} = \text{median}\left\{ z_{i,t}^{\text{skew}} \;\middle|\; i \in \mathcal{P}_{f,t}^{\text{top}} \right\}$$

where:
- $f$ = factor (momentum, low-vol, quality)
- $\mathcal{P}_{f,t}^{\text{top}}$ = set of tickers in the top quintile of factor $f$ on date $t$
- $z_{i,t}^{\text{skew}}$ = rolling 20-day z-score of ticker $i$'s put skew on date $t$:

$$z_{i,t}^{\text{skew}} = \frac{\text{PutSkew}_{i,t} - \overline{\text{PutSkew}}_{i,[t-20,t]}}{\hat{\sigma}_{i,[t-20,t]}}$$

$$\text{PutSkew}_{i,t} = IV_{i,t}^{25\Delta\text{-put}} - IV_{i,t}^{\text{ATM-put}}$$

### Regime Z-Score

$$\text{RegimeZ}_{f,t} = \frac{\text{CI}_{f,t} - \overline{\text{CI}}_{f,[t-60,t]}}{\hat{\sigma}_{f,[t-60,t]}}$$

### Portfolio Exposure Rule

$$w_{f,t} = \begin{cases} 1.10 & \text{if } \text{RegimeZ}_{f,t-1} \leq -0.75 \quad [\text{DEPRESSED, experimental}]\\ 1.00 & \text{if } |\text{RegimeZ}_{f,t-1}| \leq 0.75 \quad [\text{NORMAL}]\\ 0.75 & \text{if } 0.75 < \text{RegimeZ}_{f,t-1} \leq 1.5 \quad [\text{ELEVATED}]\\ 0.50 & \text{if } \text{RegimeZ}_{f,t-1} > 1.5 \quad [\text{WARNING}] \end{cases}$$

**Note:** Regime is lagged by $t-1$ to prevent look-ahead bias.

---

## Factor Crash Validation

Preliminary validation table (populate after running the full pipeline):

| Event | Start Date | Warning Triggered | Days Lead Time | Peak Crowding |
|-------|-----------|------------------|---------------|---------------|
| Quant Quake Aug 2007 | 2007-08-06 | — | — | — |
| GFC Deleveraging Sep 2008 | 2008-09-15 | — | — | — |
| Vol Spike Feb 2018 | 2018-02-05 | — | — | — |
| COVID Crash Mar 2020 | 2020-03-16 | — | — | — |
| Momentum Crash Sep 2020 | 2020-09-04 | — | — | — |
| Rate Shock Nov 2021 | 2021-11-26 | — | — | — |

*Run `python -m src.crowding_detector --factor momentum` to populate this table.*

---

## Repository Structure

```
skew-crowding-alpha/
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── .gitignore
├── config.py                          ← All parameters and paths (modify here)
│
├── data/
│   ├── raw/                           ← Per-ticker options JSON cache
│   ├── processed/                     ← Parquet panels, factor portfolios
│   └── reference/
│       ├── factor_crash_dates.csv     ← 8 historical factor crashes with metadata
│       └── sector_etf_map.csv         ← 11 GICS sectors → SPDR ETF mapping
│
├── outputs/                           ← All figures (300 DPI PNG)
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb  ← 10-section walkthrough notebook
│
└── src/
    ├── __init__.py
    ├── utils.py                       ← lag_signal, Sharpe, drawdown, etc.
    ├── data_loader.py                 ← Price, options, VIX data ingestion
    ├── factor_engine.py               ← Momentum, low-vol, quality factors
    ├── skew_engine.py                 ← Put skew panel, z-score, VIX overlay
    ├── crowding_detector.py           ← Crowding index, regime classification
    ├── regime_filter.py               ← Exposure scaling, transition matrix
    ├── backtest.py                    ← Performance metrics, event study
    └── visualize.py                   ← 7 publication-quality figures
```

---

## How to Run

### 1. Installation

```bash
git clone https://github.com/rxj0102/skew-crowding-alpha.git
cd skew-crowding-alpha
pip install -r requirements.txt
```

### 2. Smoke Test (No Data Required)

Verify the core logic with synthetic data:

```bash
python -m src.crowding_detector --factor momentum
```

This runs the full crowding detection pipeline with randomly generated inputs, prints regime statistics, and confirms no look-ahead bias violations.

### 3. Full Pipeline

```python
# Run in Python or a Jupyter notebook
import config
from src import data_loader, factor_engine, skew_engine, crowding_detector, regime_filter, backtest, visualize

# Step 1: Build universe
tickers = data_loader.fetch_sp500_tickers()
mcap = data_loader.fetch_market_cap_data(tickers)
universe = mcap.sort_values('market_cap', ascending=False).head(config.UNIVERSE_SIZE)['ticker'].tolist()

# Step 2: Download price data
prices = data_loader.fetch_price_data(universe, config.BACKTEST_START, config.BACKTEST_END)

# Step 3: Compute factors and build portfolio
mom_scores = factor_engine.compute_momentum_factor(prices)
mom_portfolio = factor_engine.build_factor_portfolio(mom_scores, 'momentum')
mom_returns = factor_engine.compute_factor_returns(mom_portfolio, prices)

# Step 4: Build skew panel (WARNING: slow on first run ~hours — cached thereafter)
date_range = prices.index
# For full history: skew_panel = skew_engine.build_universe_skew_panel(universe, date_range)
# For demo: use pre-built synthetic panel or factor ETF proxy

# Step 5: Compute crowding index and regime
zscore_panel = skew_engine.compute_rolling_skew_zscore(skew_panel)
crowding_idx = crowding_detector.compute_factor_crowding_index(zscore_panel, mom_portfolio, 'momentum')
crowding_df = crowding_detector.detect_crowding_regime(crowding_idx)

# Step 6: Regime-filtered backtest
filtered = regime_filter.apply_regime_filter(mom_returns, crowding_df, prices)
baseline_bt = backtest.run_baseline_factor_backtest(mom_returns)
filtered_bt = backtest.run_regime_filtered_backtest(filtered)

# Step 7: Evaluate and visualize
backtest.compute_performance_metrics(baseline_bt, 'Baseline')
backtest.compute_performance_metrics(filtered_bt, 'Regime-Filtered')
visualize.plot_cumulative_returns_comparison(baseline_bt, filtered_bt)
```

### 4. Exploratory Notebook

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

The notebook contains 10 sections with fully reproducible synthetic data — no external API calls required for the demonstration path.

---

## Key Results

*Placeholder — run the full pipeline to populate.*

```
Strategy             Ann. Return    Sharpe    Max Drawdown
─────────────────────────────────────────────────────────
Baseline (Momentum)  TBD%           TBD       TBD%
Regime-Filtered      TBD%           TBD       TBD%
Improvement          TBD%           TBD×      TBD%
```

---

## Data Sources

| Data | Source | Cost | Notes |
|------|--------|------|-------|
| S&P 500 constituents | Wikipedia | Free | Survivorship bias; use Compustat for production |
| Adjusted close prices | yfinance | Free | May have data quality issues for older history |
| Options skew | Barchart.com | Free (delayed) | 20-min delay; requires careful session handling |
| Options skew (fallback) | yfinance options | Free | Approximate delta; less accurate than Barchart |
| VIX term structure | CBOE public CSVs | Free | High quality, daily history since 1990 |
| Factor ETF proxies | yfinance | Free | MTUM since 2013; pre-launch gap |

---

## Limitations

1. **Survivorship bias**: Using the current S&P 500 list excludes historical constituents that were delisted or acquired. Point-in-time index membership data (Compustat/WRDS) required for rigorous out-of-sample testing.

2. **Options data quality**: Single-name options skew is noisy for smaller names with wide bid-ask spreads. The 25-delta identification is approximate (moneyness heuristic). Production use requires a paid options data feed with true delta surfaces.

3. **In-sample parameter selection**: The warning threshold (1.5 z-score), rolling windows (20-day skew, 60-day baseline), and regime thresholds were chosen based on economic intuition but not formally optimised out-of-sample.

4. **Factor proxy quality**: The price-based quality proxy (inverse max drawdown) is a weak substitute for fundamental quality (ROE, earnings stability, balance sheet strength). Accounting data via Compustat would improve signal quality.

5. **Transaction costs**: The 10 bps one-way assumption may be conservative for large-cap equity ETFs but optimistic for small-cap single names with wide spreads. Regime transitions that generate high turnover will be more impacted than the simple model suggests.

---

## Academic References

1. **Gârleanu, N. & Pedersen, L.H. (2011)**. "Margin-Based Asset Pricing and Deviations from the Law of One Price." *Review of Financial Studies*, 24(6), 1980–2022. — Foundational framework linking leverage constraints to crowding and asset pricing deviations.

2. **Stein, J.C. (2009)**. "Sophisticated Investors and Market Stability." *Brookings Papers on Economic Activity*, 2009(1), 235–287. — Documents how sophisticated investors (hedge funds, quant strategies) can destabilise markets by trading against each other during crowded unwinds.

3. **Frazzini, A. & Pedersen, L.H. (2014)**. "Betting Against Beta." *Journal of Financial Economics*, 111(1), 1–23. — The low-volatility anomaly and its connection to leverage constraints; highly relevant to understanding low-vol factor crowding.

4. **Han, B. (2008)**. "Investor Sentiment and Option Prices." *Review of Financial Studies*, 21(1), 387–414. — Empirical evidence that options prices, particularly put skew, reflect investor sentiment and predict future returns. Direct antecedent for the skew-as-signal approach.

5. **Asness, C., Moskowitz, T. & Pedersen, L.H. (2013)**. "Value and Momentum Everywhere." *Journal of Finance*, 68(3), 929–985. — Comprehensive documentation of momentum and value premia across asset classes; provides the factor framework this research builds on.

6. **Koijen, R., Moskovitz, T., Pedersen, L.H. & Vrugt, E. (2018)**. "Carry." *Journal of Financial Economics*, 127(2), 197–225. — Relevant methodology for cross-asset factor construction and crowding analysis.

7. **Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006)**. "The Cross-Section of Volatility and Expected Returns." *Journal of Finance*, 61(1), 259–299. — The low-volatility anomaly, foundational to the low-vol factor in this framework.

---

## Disclaimer

This repository is provided for **academic research and educational purposes only**. Nothing in this repository constitutes investment advice, a solicitation to invest, or a recommendation to buy or sell any security. Past performance of simulated backtests is not indicative of future results. All backtests use synthetic or delayed data and are subject to survivorship bias, look-ahead bias, and other methodological limitations documented above.

Use at your own risk. The authors make no representations about the accuracy or completeness of the information contained herein.

---

*Built as a quant finance research portfolio project demonstrating options microstructure, factor investing, and systematic risk management.*
