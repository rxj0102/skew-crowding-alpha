"""Offline tests: factor maths, monthly rebalancing, regime classification and the 1-day regime lag."""

import numpy as np
import pandas as pd
import pytest

import config
from src import factor_engine as fe
from src.crowding_detector import detect_crowding_regime
from src.regime_filter import apply_regime_filter
from src.skew_engine import compute_put_skew_single
from src.utils import rolling_zscore


def _prices(n=400, k=25, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n, k)), axis=0)),
        index=idx,
        columns=[f"T{i}" for i in range(k)],
    )


@pytest.fixture(autouse=True)
def _tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "factor_portfolio_path", lambda name: tmp_path / f"{name}_portfolio.csv")


# ---------------------------------------------------------------- skew leg
def test_put_skew_difference_and_sanity_filters():
    assert compute_put_skew_single("A", "2023-01-03", {"put_iv_25d": 0.35, "atm_iv": 0.30}) == pytest.approx(0.05)
    assert np.isnan(compute_put_skew_single("A", "d", {"put_iv_25d": np.nan, "atm_iv": 0.3}))
    assert np.isnan(compute_put_skew_single("A", "d", {"put_iv_25d": 9.0, "atm_iv": 0.3}))  # implausible IV
    assert np.isnan(compute_put_skew_single("A", "d", {}))


# ------------------------------------------------------------ rolling stats
def test_rolling_zscore_point_in_time_and_winsorised():
    s = pd.Series(np.random.default_rng(1).normal(size=200))
    s.iloc[150] = 50.0
    z = rolling_zscore(s, window=20, winsorise_clip=3.0)
    z_trunc = rolling_zscore(s.iloc[:120], window=20, winsorise_clip=3.0)
    pd.testing.assert_series_equal(z.iloc[:120], z_trunc)
    assert z.abs().max() <= 3.0 + 1e-12


# ------------------------------------------------------------------ factors
def test_momentum_is_12_minus_1_and_point_in_time():
    px = _prices()
    m = fe.compute_momentum_factor(px)
    t = 300
    expected = px.iloc[t - config.MOMENTUM_SKIP] / px.iloc[t - config.MOMENTUM_LOOKBACK] - 1
    pd.testing.assert_series_equal(m.iloc[t], expected, check_names=False)
    trunc = fe.compute_momentum_factor(px.iloc[:350])
    pd.testing.assert_frame_equal(trunc, m.iloc[:350])


def test_low_vol_prefers_calm_names_and_is_point_in_time():
    px = _prices()
    px["CALM"] = 100 * (1 + 1e-4 * np.arange(len(px)))
    lv = fe.compute_low_vol_factor(px)
    assert lv.iloc[-1].idxmax() == "CALM"
    pd.testing.assert_frame_equal(fe.compute_low_vol_factor(px.iloc[:300]), lv.iloc[:300])


def test_rank_universe_percentiles_and_fallback_to_prior_date():
    f = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [2.0, 2.0], "C": [3.0, 3.0], "D": [4.0, 4.0], "E": [5.0, np.nan]},
        index=pd.to_datetime(["2023-01-02", "2023-01-04"]),
    )
    r = fe.rank_factor_universe(f, "2023-01-02")
    assert r.loc["E", "factor_rank"] == 1.0 and bool(r.loc["E", "top_quintile"])
    assert not r.loc["A", "top_quintile"]
    r_gap = fe.rank_factor_universe(f, "2023-01-03")  # non-trading day -> previous available date
    assert len(r_gap) == 5
    assert fe.rank_factor_universe(f, "2022-12-01").empty


def test_factor_portfolio_rebalances_monthly_not_weekly():
    """Regression: an operator-precedence bug made this rebalance every week."""
    px = _prices(n=260)
    factor = fe.compute_momentum_factor(px).dropna(how="all")
    factor = pd.DataFrame(np.random.default_rng(0).normal(size=px.shape), index=px.index, columns=px.columns)
    port = fe.build_factor_portfolio(factor, "testfactor")
    dates = pd.DatetimeIndex(sorted(port["date"].unique()))
    months = factor.index.to_period("M").nunique()
    assert len(dates) == months
    assert dates.to_period("M").is_unique
    # each rebalance date is the first trading day of its month
    first_days = factor.groupby(factor.index.to_period("M")).head(1).index
    assert list(dates) == list(first_days)


def test_factor_portfolio_equal_weights_top_quintile():
    px = _prices(n=60, k=25)
    factor = pd.DataFrame(np.random.default_rng(2).normal(size=px.shape), index=px.index, columns=px.columns)
    port = fe.build_factor_portfolio(factor, "w")
    for _, g in port.groupby("date"):
        np.testing.assert_allclose(g["portfolio_weight"].sum(), 1.0)
        assert (g["factor_rank"] > config.FACTOR_QUINTILE).all()


# -------------------------------------------------------------- regime logic
def test_regime_labels_follow_thresholds():
    rng = np.random.default_rng(3)
    base = pd.Series(rng.normal(0, 1, 200), index=pd.bdate_range("2022-01-03", periods=200))
    base.iloc[-1] = 25.0  # large positive shock at the end -> WARNING
    out = detect_crowding_regime(base, baseline_window=60)
    assert out["regime_label"].iloc[-1] == config.REGIME_WARNING
    assert out["regime_label"].iloc[:10].eq(config.REGIME_NORMAL).all()  # NaN warm-up -> NORMAL
    assert set(out["regime_label"]).issubset(set(config.REGIME_EXPOSURE_MAP))


def test_regime_filter_uses_previous_day_regime(monkeypatch):
    idx = pd.bdate_range("2023-01-02", periods=6)
    px = pd.DataFrame({t: np.linspace(100, 101, 6) for t in config.DEFENSIVE_TICKERS}, index=idx)
    factor_ret = pd.Series(0.01, index=idx)
    labels = [config.REGIME_NORMAL] * 6
    labels[3] = config.REGIME_WARNING  # crowding detected on day 3 (close)
    crowd = pd.DataFrame({"regime_label": labels}, index=idx)
    out = apply_regime_filter(factor_ret, crowd, px)
    # exposure cannot be cut on the day the warning is observed, only the next day
    assert out["factor_weight"].iloc[3] == config.NORMAL_EXPOSURE
    assert out["factor_weight"].iloc[4] == config.WARNING_EXPOSURE
    np.testing.assert_allclose(out["factor_weight"] + out["defensive_weight"], 1.0)
