"""
data_loader.py
==============
Data ingestion layer for the skew-crowding-alpha research framework.

Responsibilities
----------------
* Fetch and cache S&P 500 constituent tickers from Wikipedia.
* Download adjusted-close price history via yfinance (with parquet cache).
* Pull market-cap data to rank the universe by size.
* Scrape single-name options put-skew data from Barchart (with yfinance fallback).
* Download free CBOE VIX term-structure CSVs.
* Download factor ETF proxy price histories (MTUM, USMV, QUAL, VLUE).

All persistent artefacts are written to paths defined in ``config.py``.
No path is hardcoded in this module.

Notes
-----
Barchart free tier shows delayed (20-minute) data and requires a valid browser
session for some endpoints.  The scraper uses a realistic ``User-Agent`` header
and a short sleep between requests.  If the scraper fails (common in CI or
restricted environments), the yfinance fallback is automatically used and the
``data_source`` key in the returned dict will reflect this.

For production use, replace the Barchart scraper with a paid options data feed
(e.g., CBOE LiveVol, OptionMetrics, or Interactive Brokers API).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. S&P 500 Tickers
# ---------------------------------------------------------------------------

def fetch_sp500_tickers(force_refresh: bool = False) -> list[str]:
    """Scrape the S&P 500 constituent list from Wikipedia.

    Parameters
    ----------
    force_refresh : bool, optional
        If True, bypass the local cache and re-scrape.  Default False.

    Returns
    -------
    list[str]
        List of ticker symbols, cleaned for yfinance compatibility
        (e.g. ``BRK.B`` → ``BRK-B``).

    Notes
    -----
    Wikipedia's S&P 500 list is updated on a rolling basis and may lag real
    index changes by a day or two.  For historical backtesting purposes this
    survivorship bias is acceptable; for live trading use a paid index-membership
    database (e.g. Compustat or Bloomberg).

    The result is cached to ``config.SP500_TICKERS_PATH`` with the fetch date
    stamped in a companion column.
    """
    cache_path = config.SP500_TICKERS_PATH

    if cache_path.exists() and not force_refresh:
        logger.info("Loading S&P 500 tickers from cache: %s", cache_path)
        df = pd.read_csv(cache_path)
        return df["ticker"].tolist()

    logger.info("Fetching S&P 500 tickers from Wikipedia …")
    try:
        tables = pd.read_html(config.SP500_TICKERS_URL, attrs={"id": "constituents"})
        tickers_raw: list[str] = tables[0]["Symbol"].tolist()
    except Exception as exc:
        logger.error("Wikipedia scrape failed: %s", exc)
        # Fallback: try second table approach
        tables = pd.read_html(config.SP500_TICKERS_URL)
        tickers_raw = tables[0]["Symbol"].tolist()

    # Clean for yfinance: replace dots with hyphens
    tickers = [t.replace(".", "-") for t in tickers_raw]

    df = pd.DataFrame({"ticker": tickers, "fetch_date": pd.Timestamp.today().date()})
    df.to_csv(cache_path, index=False)
    logger.info("Cached %d tickers to %s", len(tickers), cache_path)
    return tickers


# ---------------------------------------------------------------------------
# 2. Price Data
# ---------------------------------------------------------------------------

def fetch_price_data(
    tickers: list[str],
    start: str,
    end: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted-close price history for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols compatible with yfinance.
    start : str
        Start date in ``YYYY-MM-DD`` format.
    end : str
        End date in ``YYYY-MM-DD`` format.
    force_refresh : bool, optional
        Bypass cache and re-download.  Default False.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame: index = trading dates (DatetimeIndex),
        columns = ticker symbols, values = adjusted close prices.
        Missing tickers are silently dropped and logged.

    Notes
    -----
    Data is cached in Parquet format (``config.PRICE_DATA_PATH``) for fast
    re-loading.  Parquet is ~10× faster than CSV for panel data of this size.
    Cache is re-used if it already covers the requested date range.
    """
    cache_path = config.PRICE_DATA_PATH

    if cache_path.exists() and not force_refresh:
        logger.info("Loading price data from cache: %s", cache_path)
        df = pd.read_parquet(cache_path)
        # Validate coverage
        cache_start = df.index.min()
        cache_end = df.index.max()
        if (
            pd.Timestamp(start) >= cache_start
            and pd.Timestamp(end) <= cache_end
        ):
            logger.info(
                "Cache covers requested range [%s, %s]. Returning cached data.", start, end
            )
            return df.loc[start:end]
        logger.info("Cache does not fully cover requested range. Re-downloading …")

    logger.info("Downloading price data for %d tickers via yfinance …", len(tickers))

    chunk_size = 50  # yfinance performs better with smaller batches
    all_chunks: list[pd.DataFrame] = []

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        logger.debug("Downloading chunk %d/%d …", i // chunk_size + 1, (len(tickers) + chunk_size - 1) // chunk_size)
        try:
            raw = yf.download(
                chunk,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            # Handle multi-level columns returned by yfinance
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]] if "Close" in raw.columns else raw
            all_chunks.append(close)
        except Exception as exc:
            logger.warning("Download failed for chunk starting %s: %s", chunk[0], exc)

    if not all_chunks:
        raise RuntimeError("Price download returned no data.")

    price_df = pd.concat(all_chunks, axis=1)
    price_df = price_df.loc[:, ~price_df.columns.duplicated()]

    # Report missing tickers
    fetched = set(price_df.columns.tolist())
    requested = set(tickers)
    missing = requested - fetched
    if missing:
        logger.warning("Missing tickers (not available in yfinance): %s", sorted(missing))

    price_df.index = pd.to_datetime(price_df.index)
    price_df.sort_index(inplace=True)

    # Save to parquet
    price_df.to_parquet(cache_path)
    logger.info(
        "Cached price data [%s → %s] for %d tickers to %s",
        price_df.index.min().date(),
        price_df.index.max().date(),
        price_df.shape[1],
        cache_path,
    )
    return price_df


# ---------------------------------------------------------------------------
# 3. Market-Cap Data
# ---------------------------------------------------------------------------

def fetch_market_cap_data(
    tickers: list[str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Pull market capitalisation, sector, and industry for each ticker.

    Parameters
    ----------
    tickers : list[str]
        yfinance-compatible ticker symbols.
    force_refresh : bool, optional
        Bypass cache.  Default False.

    Returns
    -------
    pd.DataFrame
        Columns: ``ticker``, ``market_cap``, ``sector``, ``industry``.
        Rows sorted descending by ``market_cap``.
        Tickers for which yfinance returns no data are excluded and logged.

    Notes
    -----
    yfinance's ``Ticker.info`` endpoint is a scrape-backed API with rate limits.
    The function adds a 0.2-second sleep between requests.  For large universes
    run this once and rely on the cache.
    """
    cache_path = config.MARKET_CAP_PATH

    if cache_path.exists() and not force_refresh:
        logger.info("Loading market cap data from cache: %s", cache_path)
        return pd.read_csv(cache_path)

    logger.info("Fetching market cap for %d tickers …", len(tickers))
    records: list[dict] = []

    for i, ticker in enumerate(tickers):
        if i % 20 == 0:
            logger.info("  … %d / %d", i, len(tickers))
        try:
            info = yf.Ticker(ticker).info
            records.append(
                {
                    "ticker": ticker,
                    "market_cap": info.get("marketCap", np.nan),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                }
            )
        except Exception as exc:
            logger.debug("yfinance info failed for %s: %s", ticker, exc)
        time.sleep(0.2)

    df = pd.DataFrame(records)
    df.sort_values("market_cap", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(cache_path, index=False)
    logger.info("Market cap data cached to %s", cache_path)
    return df


# ---------------------------------------------------------------------------
# 4. Options Skew — Barchart Scraper with yfinance fallback
# ---------------------------------------------------------------------------

_BARCHART_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.barchart.com/",
}


YFINANCE_MAX_AGE_DAYS = 3


def _compute_skew_from_yfinance(ticker: str, date: str) -> dict:
    """Fallback skew calculation using yfinance options chain.

    Parameters
    ----------
    ticker : str
        Equity ticker.
    date : str
        Reference date string (used for cache key only; yfinance returns
        current data regardless of the ``date`` parameter).

    Returns
    -------
    dict
        Same schema as ``fetch_options_skew_barchart`` but ``data_source``
        is set to ``"yfinance_fallback"``.

    Notes
    -----
    yfinance does not provide greeks (delta) directly.  We approximate delta
    using moneyness:
        * ATM ≈ closest strike to current spot price
        * 25-delta put ≈ strike at 90–95% of spot (rough approximation)
    This is less accurate than true 25-delta computed from a vol surface.
    """
    # yfinance only serves the current option chain. Using it for a past date would stamp
    # today's skew onto that date, so historical requests get an empty result instead.
    if (pd.Timestamp.today().normalize() - pd.Timestamp(date).normalize()).days > YFINANCE_MAX_AGE_DAYS:
        return _empty_skew_dict(ticker, date, "yfinance_no_history")
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return _empty_skew_dict(ticker, date, "yfinance_no_options")

        # Find front-month expiry with > 7 DTE
        today = pd.Timestamp(date)
        valid_expiries = [
            e for e in expirations
            if (pd.Timestamp(e) - today).days > 7
        ]
        if not valid_expiries:
            valid_expiries = expirations[:1]
        expiry = valid_expiries[0]

        chain = t.option_chain(expiry)
        puts = chain.puts.copy()
        if puts.empty:
            return _empty_skew_dict(ticker, date, "yfinance_no_puts")

        spot = t.fast_info.get("lastPrice") or t.fast_info.get("regularMarketPrice")
        if spot is None or spot == 0:
            hist = t.history(period="5d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan

        if np.isnan(spot):
            return _empty_skew_dict(ticker, date, "yfinance_no_spot")

        # ATM: closest strike to spot
        puts["strike_dist"] = abs(puts["strike"] - spot)
        atm_row = puts.loc[puts["strike_dist"].idxmin()]
        atm_iv = float(atm_row["impliedVolatility"])

        # 25-delta approx: strike at ~93% of spot for a put
        target_25d_strike = spot * 0.93
        puts["dist_25d"] = abs(puts["strike"] - target_25d_strike)
        put_25d_row = puts.loc[puts["dist_25d"].idxmin()]
        put_25d_iv = float(put_25d_row["impliedVolatility"])

        put_skew = put_25d_iv - atm_iv
        return {
            "ticker": ticker,
            "date": date,
            "put_iv_25d": round(put_25d_iv, 4),
            "atm_iv": round(atm_iv, 4),
            "put_skew": round(put_skew, 4),
            "expiry": expiry,
            "data_source": "yfinance_fallback",
        }

    except Exception as exc:
        logger.debug("yfinance options fallback failed for %s: %s", ticker, exc)
        return _empty_skew_dict(ticker, date, f"error:{exc}")


def _empty_skew_dict(ticker: str, date: str, reason: str) -> dict:
    """Return a uniform NaN-filled skew dict."""
    return {
        "ticker": ticker,
        "date": date,
        "put_iv_25d": np.nan,
        "atm_iv": np.nan,
        "put_skew": np.nan,
        "expiry": None,
        "data_source": reason,
    }


def fetch_options_skew_barchart(
    ticker: str,
    date: str,
    use_cache: bool = True,
    sleep_sec: float = 1.5,
) -> dict:
    """Fetch 25-delta put skew for a single ticker from Barchart, with fallback.

    Parameters
    ----------
    ticker : str
        Equity ticker (e.g. ``"AAPL"``).
    date : str
        Reference date string ``YYYY-MM-DD``.  Used as cache key.
    use_cache : bool, optional
        If True (default), check local JSON cache before fetching.
    sleep_sec : float, optional
        Seconds to sleep after a successful Barchart fetch (rate-limit courtesy).

    Returns
    -------
    dict
        Keys: ``ticker``, ``date``, ``put_iv_25d``, ``atm_iv``, ``put_skew``,
        ``expiry``, ``data_source``.

    Notes
    -----
    **Data source priority**

    1. Local JSON cache (``data/raw/{ticker}_{date}_skew.json``).
    2. Barchart free-tier HTML scrape.  Note: Barchart shows delayed (20-minute)
       data on the free tier and may require browser cookies for full access.
       In a production context, replace with a paid options API.
    3. yfinance options chain (least accurate delta approximation).

    The ``data_source`` key in the returned dict records which source was used.

    **Delta calculation**

    Barchart does not expose delta directly on the free tier.  We identify the
    25-delta put as the OTM put strike whose absolute delta is closest to 0.25
    (after parsing the options table).  If delta data is unavailable, we fall
    back to the 90–95% moneyness heuristic as in the yfinance fallback.
    """
    cache_path = config.raw_skew_path(ticker, date)

    # Check local cache
    if use_cache and cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            logger.debug("Loaded skew cache for %s @ %s", ticker, date)
            return data
        except json.JSONDecodeError:
            logger.warning("Corrupt skew cache for %s @ %s — re-fetching", ticker, date)

    # --- Barchart scrape ---
    url = f"https://www.barchart.com/stocks/quotes/{ticker}/options"
    try:
        session = requests.Session()
        resp = session.get(url, headers=_BARCHART_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Barchart renders options data in a structured table; find it
        table = soup.find("table", class_=lambda c: c and "options" in c.lower())
        if table is None:
            # Try a generic table search
            tables = soup.find_all("table")
            table = tables[0] if tables else None

        if table is None:
            raise ValueError("No options table found on Barchart page")

        rows = table.find_all("tr")
        records: list[dict] = []
        for row in rows[1:]:  # skip header
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) >= 6:
                records.append({"raw": cells})

        if not records:
            raise ValueError("Options table parsed but no rows found")

        # Parse implied vol from Barchart table structure
        # Barchart put table columns typically: Strike, Last, Change, Bid, Ask, Vol, OI, IV, Delta
        # We locate IV and Delta columns by header
        headers = [c.get_text(strip=True).lower() for c in rows[0].find_all(["td", "th"])]

        def safe_col(name_fragment: str) -> int:
            for i, h in enumerate(headers):
                if name_fragment in h:
                    return i
            return -1

        iv_col = safe_col("iv") if safe_col("iv") >= 0 else safe_col("impl")
        delta_col = safe_col("delta")
        strike_col = safe_col("strike")

        if iv_col < 0 or strike_col < 0:
            raise ValueError(f"Cannot locate IV/Strike columns. Headers: {headers}")

        parsed: list[dict] = []
        for row in rows[1:]:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) <= max(iv_col, strike_col):
                continue
            try:
                strike = float(cells[strike_col].replace(",", ""))
                iv_str = cells[iv_col].rstrip("%")
                iv_val = float(iv_str) / 100.0
                delta_val = np.nan
                if delta_col >= 0 and delta_col < len(cells):
                    try:
                        delta_val = float(cells[delta_col])
                    except ValueError:
                        pass
                parsed.append({"strike": strike, "iv": iv_val, "delta": delta_val})
            except (ValueError, IndexError):
                continue

        if not parsed:
            raise ValueError("No parseable options data rows")

        pdf = pd.DataFrame(parsed)
        # Find ATM by delta ≈ -0.50 or closest to 0 moneyness
        if pdf["delta"].notna().any():
            pdf["delta_dist_atm"] = abs(abs(pdf["delta"]) - config.ATM_DELTA_TARGET)
            pdf["delta_dist_25d"] = abs(abs(pdf["delta"]) - config.PUT_DELTA_TARGET)
            atm_row = pdf.loc[pdf["delta_dist_atm"].idxmin()]
            put25_row = pdf.loc[pdf["delta_dist_25d"].idxmin()]
        else:
            # Fallback: use index-based selection (first/third decile of OTM puts)
            pdf.sort_values("strike", inplace=True)
            mid = len(pdf) // 2
            atm_row = pdf.iloc[mid]
            put25_row = pdf.iloc[max(0, mid - len(pdf) // 4)]

        atm_iv = float(atm_row["iv"])
        put_25d_iv = float(put25_row["iv"])
        put_skew = put_25d_iv - atm_iv

        # Determine front-month expiry from page title or URL params
        expiry_tag = soup.find("option", {"selected": True})
        expiry = expiry_tag.get_text(strip=True) if expiry_tag else "front-month"

        result = {
            "ticker": ticker,
            "date": date,
            "put_iv_25d": round(put_25d_iv, 4),
            "atm_iv": round(atm_iv, 4),
            "put_skew": round(put_skew, 4),
            "expiry": expiry,
            "data_source": "barchart",
        }
        time.sleep(sleep_sec)

    except Exception as exc:
        logger.warning(
            "Barchart scrape failed for %s @ %s (%s) — using yfinance fallback",
            ticker, date, exc,
        )
        result = _compute_skew_from_yfinance(ticker, date)

    # Write to cache
    try:
        with open(cache_path, "w") as f:
            json.dump(result, f, default=str)
    except Exception as exc:
        logger.warning("Could not cache skew result for %s: %s", ticker, exc)

    return result


# ---------------------------------------------------------------------------
# 5. CBOE VIX Term Structure
# ---------------------------------------------------------------------------

def fetch_cboe_vix_term_structure(force_refresh: bool = False) -> pd.DataFrame:
    """Download CBOE VIX and VIX3M daily history and compute term spread.

    Parameters
    ----------
    force_refresh : bool, optional
        Bypass cache and re-download.  Default False.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``vix_spot``, ``vix3m``, ``vix_term_spread``.
        ``vix_term_spread = vix3m - vix_spot``.
        Positive spread indicates a normal (contango) VIX term structure.
        Negative spread (backwardation) signals stress / elevated near-term fear.

    Notes
    -----
    CBOE publishes these CSVs publicly at no charge:
    * VIX_History.csv  — CBOE VIX (30-day expected vol)
    * VIX3M_History.csv — CBOE VIX 3-Month

    The VIX term spread is a macro conditioning variable that can be combined
    with the single-name put skew crowding index.  During stress events, both
    tend to spike simultaneously, providing confirmation.
    """
    cache_path = config.VIX_TERM_STRUCTURE_PATH

    if cache_path.exists() and not force_refresh:
        logger.info("Loading VIX term structure from cache.")
        df = pd.read_csv(cache_path, parse_dates=["date"])
        df.set_index("date", inplace=True)
        return df

    logger.info("Downloading CBOE VIX and VIX3M histories …")
    try:
        vix = pd.read_csv(config.CBOE_VIX_URL)
        vix3m = pd.read_csv(config.CBOE_VIX3M_URL)
    except Exception as exc:
        logger.error("CBOE download failed: %s", exc)
        raise

    # Normalize column names — CBOE uses DATE, OPEN, HIGH, LOW, CLOSE
    def _norm(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        df.columns = df.columns.str.strip().str.upper()
        df = df.rename(columns={"DATE": "date", "CLOSE": value_col})
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df[[value_col]]

    vix_df = _norm(vix, "vix_spot")
    vix3m_df = _norm(vix3m, "vix3m")

    merged = vix_df.join(vix3m_df, how="inner")
    merged["vix_term_spread"] = merged["vix3m"] - merged["vix_spot"]
    merged.sort_index(inplace=True)

    merged.reset_index().to_csv(cache_path, index=False)
    logger.info("VIX term structure cached to %s (%d rows)", cache_path, len(merged))
    return merged


# ---------------------------------------------------------------------------
# 6. Factor ETF Proxies
# ---------------------------------------------------------------------------

def fetch_factor_etf_proxies(
    start: str = config.BACKTEST_START,
    end: str = config.BACKTEST_END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted-close history for factor ETF benchmarks.

    Parameters
    ----------
    start : str
        Start date ``YYYY-MM-DD``.
    end : str
        End date ``YYYY-MM-DD``.
    force_refresh : bool, optional
        Bypass cache.  Default False.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame: index = dates, columns = ETF tickers
        (``MTUM``, ``USMV``, ``QUAL``, ``VLUE``).

    Notes
    -----
    These ETFs serve two purposes in this research:

    1. **Reality check**: our price-derived factor portfolios should broadly
       correlate with the corresponding ETF returns.  Low correlation indicates
       a methodological issue.
    2. **Crash date validation**: ETF drawdowns at known crash dates provide
       objective benchmarks for evaluating the crowding signal's effectiveness.

    Note that MTUM launched in April 2013 and QUAL/USMV in 2011–2012, so
    pre-launch data is not available.  Backfill with factor index returns
    (e.g. MSCI USA Momentum Index) for pre-2013 analysis if needed.
    """
    cache_path = config.FACTOR_ETF_PATH

    if cache_path.exists() and not force_refresh:
        logger.info("Loading factor ETF proxies from cache.")
        return pd.read_parquet(cache_path)

    tickers = list(config.FACTOR_ETF_PROXIES.values())
    logger.info("Downloading factor ETF proxies: %s", tickers)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"]
    else:
        df = raw

    df.index = pd.to_datetime(df.index)
    df.to_parquet(cache_path)
    logger.info("Factor ETF proxies cached to %s", cache_path)
    return df
