"""
visualize.py
============
All charts for the skew-crowding-alpha research framework.

All figures are saved to ``config.OUTPUTS_DIR`` at 300 DPI in PNG format.
Style: seaborn-whitegrid for clean, publication-quality appearance.

Figure inventory
----------------
1. ``plot_crowding_index_with_regimes``   — Dual-axis: crowding + factor returns
2. ``plot_lead_lag_correlation``          — Bar chart: crowding → future return lag
3. ``plot_skew_zscore_heatmap``           — Heatmap: ticker × date, skew z-score
4. ``plot_cumulative_returns_comparison`` — Cumulative return overlay, both strategies
5. ``plot_crash_event_study``             — Grouped bar: drawdown per crash event
6. ``plot_regime_transition_matrix``      — 4×4 transition probability heatmap
7. ``plot_skew_vs_factor_return_scatter`` — Scatter: crowding level vs next-30d return
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/notebook environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import config

logger = logging.getLogger(__name__)

# Apply global style
try:
    plt.style.use(config.FIGURE_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8-whitegrid")

CRASH_DATES = config.FACTOR_CRASH_DATES


def _save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save figure to outputs directory and close."""
    out_path = config.OUTPUTS_DIR / filename
    fig.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 1. Crowding Index with Regime Shading
# ---------------------------------------------------------------------------

def plot_crowding_index_with_regimes(
    crowding_df: pd.DataFrame,
    factor_returns: pd.Series,
    factor_name: str = "momentum",
    filename: Optional[str] = None,
) -> Path:
    """Plot crowding index with regime shading and factor return overlay.

    Parameters
    ----------
    crowding_df : pd.DataFrame
        Output of ``detect_crowding_regime``.  Must have:
        ``crowding_index``, ``regime_label``.
    factor_returns : pd.Series
        Factor portfolio returns for cumulative return overlay.
    factor_name : str
        Used in chart title.
    filename : str, optional
        Output filename.  Defaults to ``crowding_index_{factor_name}.png``.

    Returns
    -------
    Path
        Path to the saved figure.

    Notes
    -----
    **This is Figure 1 of the paper** — the visual that makes the thesis
    immediately compelling to a PM audience.

    The chart shows three things simultaneously:
    1. The crowding index (left axis): rising = increasing crowding risk.
    2. Regime background shading: red = WARNING, orange = ELEVATED, blue = DEPRESSED.
    3. Cumulative factor return (right axis): the strategy you're trying to protect.

    The key visual insight: WARNING regimes (red) tend to precede factor
    drawdowns (cumulative return declining).
    """
    if filename is None:
        filename = f"crowding_index_{factor_name}.png"

    fig, ax1 = plt.subplots(figsize=config.FIGURE_SIZE_WIDE)
    ax2 = ax1.twinx()

    # --- Regime shading ---
    regime_colors = {
        config.REGIME_WARNING: config.REGIME_COLORS[config.REGIME_WARNING],
        config.REGIME_ELEVATED: config.REGIME_COLORS[config.REGIME_ELEVATED],
        config.REGIME_DEPRESSED: config.REGIME_COLORS[config.REGIME_DEPRESSED],
    }

    for regime, color in regime_colors.items():
        mask = crowding_df["regime_label"] == regime
        if not mask.any():
            continue
        # Find contiguous blocks
        changes = mask.astype(int).diff().fillna(0)
        starts = crowding_df.index[changes == 1].tolist()
        ends = crowding_df.index[changes == -1].tolist()

        if mask.iloc[0]:
            starts = [crowding_df.index[0]] + starts
        if mask.iloc[-1]:
            ends = ends + [crowding_df.index[-1]]

        for s, e in zip(starts, ends):
            ax1.axvspan(s, e, alpha=0.3, color=color, linewidth=0)

    # --- Crowding index line ---
    ax1.plot(
        crowding_df.index,
        crowding_df["crowding_index"],
        color="#1a1a2e",
        linewidth=1.5,
        label="Crowding Index",
        zorder=5,
    )
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--", alpha=0.5)

    # --- Cumulative factor return (right axis) ---
    cum_ret = (1 + factor_returns.reindex(crowding_df.index, method="ffill").fillna(0)).cumprod()
    ax2.plot(
        cum_ret.index,
        cum_ret.values,
        color="#0066cc",
        linewidth=2,
        linestyle="-",
        label="Cumulative Factor Return",
        alpha=0.8,
        zorder=4,
    )

    # --- Crash date vertical lines ---
    for event_name, date_str in CRASH_DATES.items():
        crash_date = pd.Timestamp(date_str)
        if crowding_df.index.min() <= crash_date <= crowding_df.index.max():
            ax1.axvline(crash_date, color="#cc0000", linewidth=1.2, linestyle="--", alpha=0.7, zorder=6)
            ax1.annotate(
                event_name.split(" ")[-2],  # Short label
                xy=(crash_date, crowding_df["crowding_index"].max() * 0.9),
                rotation=90, fontsize=7, color="#cc0000", alpha=0.8,
            )

    # --- Legend and labels ---
    legend_patches = [
        mpatches.Patch(color=config.REGIME_COLORS[config.REGIME_WARNING], alpha=0.5, label="WARNING Regime"),
        mpatches.Patch(color=config.REGIME_COLORS[config.REGIME_ELEVATED], alpha=0.5, label="ELEVATED Regime"),
        mpatches.Patch(color=config.REGIME_COLORS[config.REGIME_DEPRESSED], alpha=0.5, label="DEPRESSED Regime"),
    ]
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles=handles1 + handles2 + legend_patches,
        labels=labels1 + labels2 + [p.get_label() for p in legend_patches],
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
    )

    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Crowding Index (z-score)", fontsize=11)
    ax2.set_ylabel("Cumulative Factor Return", fontsize=11, color="#0066cc")
    ax2.tick_params(axis="y", labelcolor="#0066cc")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    fig.suptitle(
        f"Options Implied Skew Crowding Index — {factor_name.title()} Factor\n"
        "Cross-sectional median put-skew z-score of top-quintile holdings",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save_figure(fig, filename)


# ---------------------------------------------------------------------------
# 2. Lead-Lag Correlation Bar Chart
# ---------------------------------------------------------------------------

def plot_lead_lag_correlation(
    lead_lag_df: pd.DataFrame,
    factor_name: str = "momentum",
    filename: Optional[str] = None,
) -> Path:
    """Bar chart of crowding index lead-lag correlation with future factor returns.

    Parameters
    ----------
    lead_lag_df : pd.DataFrame
        Output of ``compute_crowding_lead_lag``.
        Columns: ``lag_days``, ``correlation``, ``p_value``.
    factor_name : str
        Used in chart title.
    filename : str, optional
        Output filename.

    Returns
    -------
    Path
        Path to saved figure.

    Notes
    -----
    Red bars = negative correlation (crowding predicts lower future returns).
    Blue bars = positive correlation.
    A vertical dashed line at the peak negative bar is annotated.
    """
    if filename is None:
        filename = f"lead_lag_correlation_{factor_name}.png"

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_MEDIUM)

    corrs = lead_lag_df["correlation"].values
    lags = lead_lag_df["lag_days"].values
    colors = ["#cc0000" if c < 0 else "#0066cc" for c in corrs]

    bars = ax.bar(lags, corrs, color=colors, alpha=0.8, width=0.7, edgecolor="white")

    # Annotate peak negative bar
    min_idx = np.nanargmin(corrs)
    ax.bar(lags[min_idx], corrs[min_idx], color="#8b0000", alpha=1.0, width=0.7, zorder=5)
    ax.annotate(
        f"Peak: lag={lags[min_idx]}d\nr={corrs[min_idx]:.3f}",
        xy=(lags[min_idx], corrs[min_idx]),
        xytext=(lags[min_idx] + 1.5, corrs[min_idx] - 0.02),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
        color="#8b0000",
    )

    # Significance threshold (approximate p=0.05 for n≈200)
    sig_threshold = 2.0 / np.sqrt(200)
    ax.axhline(sig_threshold, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="p=0.05 approx")
    ax.axhline(-sig_threshold, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Lag (days ahead)", fontsize=11)
    ax.set_ylabel("Pearson Correlation", fontsize=11)
    ax.set_title(
        f"Lead-Lag: Crowding Index vs Future {factor_name.title()} Factor Returns\n"
        "Negative correlation at positive lags = crowding predicts underperformance",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.set_xticks(lags)
    fig.tight_layout()
    return _save_figure(fig, filename)


# ---------------------------------------------------------------------------
# 3. Skew Z-Score Heatmap
# ---------------------------------------------------------------------------

def plot_skew_zscore_heatmap(
    skew_zscore_panel: pd.DataFrame,
    factor_portfolio_df: pd.DataFrame,
    factor_name: str = "momentum",
    filename: Optional[str] = None,
    max_tickers: int = 25,
) -> Path:
    """Heatmap of skew z-scores for top-quintile factor holdings.

    Parameters
    ----------
    skew_zscore_panel : pd.DataFrame
        Wide skew z-score panel (index = dates, columns = tickers).
    factor_portfolio_df : pd.DataFrame
        Factor portfolio holdings.
    factor_name : str
        Used in chart title.
    filename : str, optional
        Output filename.
    max_tickers : int, optional
        Maximum number of tickers to show.  Tickers are sorted by average
        skew z-score and the top ``max_tickers`` are shown.

    Returns
    -------
    Path
        Path to saved figure.
    """
    if filename is None:
        filename = f"skew_zscore_heatmap_{factor_name}.png"

    # Get top-quintile tickers from portfolio
    portfolio_tickers = factor_portfolio_df["ticker"].unique().tolist()
    available = [t for t in portfolio_tickers if t in skew_zscore_panel.columns]

    if not available:
        logger.warning("No portfolio tickers found in skew panel for heatmap.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return _save_figure(fig, filename)

    # Select monthly snapshots for readability
    panel = skew_zscore_panel[available]
    monthly_panel = panel.resample("ME").mean()

    # Sort tickers by average skew z-score (highest crowding first)
    avg_skew = monthly_panel.mean(axis=0).sort_values(ascending=False)
    top_tickers = avg_skew.head(max_tickers).index.tolist()
    plot_data = monthly_panel[top_tickers].T  # tickers as rows, dates as columns

    fig, ax = plt.subplots(figsize=(16, max(8, len(top_tickers) * 0.35)))

    sns.heatmap(
        plot_data,
        ax=ax,
        cmap="RdYlBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        linewidths=0.1,
        linecolor="grey",
        cbar_kws={"label": "Put Skew Z-Score", "shrink": 0.5},
        xticklabels=6,  # Show every 6th month
    )

    ax.set_title(
        f"Put Skew Z-Score Heatmap — {factor_name.title()} Factor Top Quintile\n"
        "Red = elevated hedging demand (crowded), Blue = low skew (under-owned)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Ticker", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)

    fig.tight_layout()
    return _save_figure(fig, filename)


# ---------------------------------------------------------------------------
# 4. Cumulative Returns Comparison
# ---------------------------------------------------------------------------

def plot_cumulative_returns_comparison(
    baseline_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    filename: Optional[str] = None,
) -> Path:
    """Overlay cumulative returns of baseline vs regime-filtered strategies.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Output of ``run_baseline_factor_backtest``.
    filtered_df : pd.DataFrame
        Output of ``run_regime_filtered_backtest``.
    filename : str, optional
        Output filename.

    Returns
    -------
    Path
        Path to saved figure.
    """
    if filename is None:
        filename = "cumulative_returns_comparison.png"

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_WIDE)

    # --- WARNING regime shading ---
    if "regime_label" in filtered_df.columns:
        mask = filtered_df["regime_label"] == config.REGIME_WARNING
        if mask.any():
            changes = mask.astype(int).diff().fillna(0)
            starts = filtered_df.index[changes == 1].tolist()
            ends = filtered_df.index[changes == -1].tolist()
            if mask.iloc[0]:
                starts = [filtered_df.index[0]] + starts
            if mask.iloc[-1]:
                ends = ends + [filtered_df.index[-1]]
            for s, e in zip(starts, ends):
                ax.axvspan(s, e, alpha=0.15, color="#cc0000", linewidth=0)

    # --- Cumulative returns ---
    ax.plot(
        baseline_df.index,
        baseline_df["cumulative_return"],
        color="#888888",
        linewidth=2,
        linestyle="--",
        label="Baseline (Unfiltered)",
        alpha=0.9,
    )
    ax.plot(
        filtered_df.index,
        filtered_df["cumulative_return"],
        color="#0066cc",
        linewidth=2.5,
        linestyle="-",
        label="Regime-Filtered",
    )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)

    # --- Annotation box with key stats ---
    baseline_total = float(baseline_df["cumulative_return"].iloc[-1]) - 1.0
    filtered_total = float(filtered_df["cumulative_return"].iloc[-1]) - 1.0
    info_text = (
        f"Baseline Total Return: {baseline_total * 100:.1f}%\n"
        f"Filtered Total Return: {filtered_total * 100:.1f}%"
    )
    ax.text(
        0.02, 0.97, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    # --- Crash date markers ---
    for event_name, date_str in CRASH_DATES.items():
        crash_date = pd.Timestamp(date_str)
        if baseline_df.index.min() <= crash_date <= baseline_df.index.max():
            ax.axvline(crash_date, color="#cc0000", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return (1 = $1 invested)", fontsize=11)
    ax.set_title(
        "Baseline vs Regime-Filtered Factor Strategy — Cumulative Returns\n"
        "Red shading = WARNING regime (reduced factor exposure)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    fig.tight_layout()
    return _save_figure(fig, filename)


# ---------------------------------------------------------------------------
# 5. Crash Event Study Bar Chart
# ---------------------------------------------------------------------------

def plot_crash_event_study(
    event_study_df: pd.DataFrame,
    filename: Optional[str] = None,
) -> Path:
    """Grouped bar chart: drawdowns during crash events, baseline vs filtered.

    Parameters
    ----------
    event_study_df : pd.DataFrame
        Output of ``run_crash_date_event_study``.
    filename : str, optional
        Output filename.

    Returns
    -------
    Path
        Path to saved figure.

    Notes
    -----
    **This is the most impactful single chart for a PM audience.**

    It shows, for each historical crash, how much the regime filter reduced
    the drawdown.  Bars going less negative for the filtered strategy =
    capital protection.  Pairs where bars are equally bad indicate events
    where the filter did not fire (no crowding signal pre-crash).
    """
    if filename is None:
        filename = "crash_event_study.png"

    # Filter to events with data
    df = event_study_df.dropna(subset=["baseline_crash_return", "filtered_crash_return"])
    if df.empty:
        logger.warning("No crash events with data to plot.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No crash event data in backtest range", ha="center", va="center")
        return _save_figure(fig, filename)

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_WIDE)

    bars1 = ax.bar(
        x - width / 2,
        df["baseline_crash_return"],
        width,
        label="Baseline (Unfiltered)",
        color="#cc3300",
        alpha=0.8,
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        df["filtered_crash_return"],
        width,
        label="Regime-Filtered",
        color="#ff8c00",
        alpha=0.8,
        edgecolor="white",
    )

    # Annotate protection
    for i, (_, row) in enumerate(df.iterrows()):
        protection = row.get("protection_pct", np.nan)
        if not np.isnan(protection) and protection > 0:
            ax.text(
                i, max(row["baseline_crash_return"], row["filtered_crash_return"]) + 0.5,
                f"+{protection:.1f}%",
                ha="center",
                fontsize=9,
                color="#006600",
                fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Crash Event", fontsize=11)
    ax.set_ylabel("Cumulative Return During Crash (%)", fontsize=11)
    ax.set_title(
        "Capital Protection: Regime-Filtered vs Baseline During Factor Crashes\n"
        "Green annotation = protection provided by regime filter",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["event_name"], rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    fig.tight_layout()
    return _save_figure(fig, filename)


# ---------------------------------------------------------------------------
# 6. Regime Transition Matrix Heatmap
# ---------------------------------------------------------------------------

def plot_regime_transition_matrix(
    transition_matrix_df: pd.DataFrame,
    filename: Optional[str] = None,
) -> Path:
    """Heatmap of the 4×4 regime transition probability matrix.

    Parameters
    ----------
    transition_matrix_df : pd.DataFrame
        Output of ``compute_regime_transition_matrix``.
    filename : str, optional
        Output filename.

    Returns
    -------
    Path
        Path to saved figure.
    """
    if filename is None:
        filename = "regime_transition_matrix.png"

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_SQUARE)

    sns.heatmap(
        transition_matrix_df,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Transition Probability"},
        annot_kws={"size": 13, "weight": "bold"},
    )

    ax.set_title(
        "Regime Transition Probability Matrix\n"
        "P(tomorrow's regime | today's regime) — diagonal = persistence",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("To Regime (Tomorrow)", fontsize=11)
    ax.set_ylabel("From Regime (Today)", fontsize=11)
    fig.tight_layout()
    return _save_figure(fig, filename)


# ---------------------------------------------------------------------------
# 7. Skew vs Factor Return Scatter
# ---------------------------------------------------------------------------

def plot_skew_vs_factor_return_scatter(
    crowding_df: pd.DataFrame,
    factor_returns: pd.Series,
    forward_days: int = 30,
    filename: Optional[str] = None,
) -> Path:
    """Scatter plot: crowding index level vs next N-day factor return.

    Parameters
    ----------
    crowding_df : pd.DataFrame
        Output of ``detect_crowding_regime``.
    factor_returns : pd.Series
        Factor portfolio returns (daily).
    forward_days : int, optional
        Forward-looking return window in days.  Default 30.
    filename : str, optional
        Output filename.

    Returns
    -------
    Path
        Path to saved figure.

    Notes
    -----
    **The OLS trendline slope is the visual expression of the core hypothesis.**

    A negative slope (crowding index ↑ → future return ↓) is the signal.
    Points are coloured by regime label.  If the WARNING (red) points cluster
    in the bottom-right (high crowding, negative future return), the signal
    is visually compelling.
    """
    if filename is None:
        filename = f"skew_vs_return_scatter_{forward_days}d.png"

    # Compute forward returns
    forward_ret = factor_returns.rolling(window=forward_days).sum().shift(-forward_days)
    forward_ret.name = f"forward_{forward_days}d_return"

    plot_df = pd.concat(
        [crowding_df["crowding_index"], crowding_df["regime_label"], forward_ret],
        axis=1,
        join="inner",
    ).dropna()

    if plot_df.empty:
        logger.warning("No overlapping data for scatter plot.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return _save_figure(fig, filename)

    regime_colors = {
        config.REGIME_WARNING: "#cc0000",
        config.REGIME_ELEVATED: "#ff8c00",
        config.REGIME_NORMAL: "#888888",
        config.REGIME_DEPRESSED: "#0066cc",
    }

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_MEDIUM)

    for regime, color in regime_colors.items():
        mask = plot_df["regime_label"] == regime
        if mask.sum() < 2:
            continue
        ax.scatter(
            plot_df.loc[mask, "crowding_index"],
            plot_df.loc[mask, forward_ret.name] * 100,
            c=color,
            alpha=0.5,
            s=20,
            label=regime,
        )

    # OLS trendline
    x = plot_df["crowding_index"].values
    y = plot_df[forward_ret.name].values * 100
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="black",
        linewidth=2,
        linestyle="--",
        label=f"OLS: y={slope:.3f}x+{intercept:.2f} (R²={r_val**2:.3f}, p={p_val:.3f})",
    )

    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Crowding Index Level", fontsize=11)
    ax.set_ylabel(f"Next {forward_days}-Day Factor Return (%)", fontsize=11)
    ax.set_title(
        f"Crowding Index vs Forward {forward_days}-Day Factor Returns\n"
        "Negative slope = crowding predicts underperformance",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    return _save_figure(fig, filename)
