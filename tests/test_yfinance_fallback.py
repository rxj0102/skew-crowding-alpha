"""The yfinance fallback must not return today's option chain for a historical date."""
import pandas as pd

from src import data_loader


def test_fallback_refuses_historical_dates(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("yfinance must not be called for a historical date")

    monkeypatch.setattr(data_loader.yf, "Ticker", boom)
    out = data_loader._compute_skew_from_yfinance("AAPL", "2015-06-30")
    assert out["data_source"] == "yfinance_no_history"
    assert pd.isna(out["put_skew"])
