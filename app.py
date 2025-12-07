import os
import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# -------------------------------------------------------------------
# App-level constants
# -------------------------------------------------------------------

APP_TITLE = "WAVES Institutional Console"
BENCH_TICKER = "SPY"
VIX_TICKER = "^VIX"

# If we have to fall back because wave_weights.csv is bad/missing,
# we’ll use this in-code default mapping.
DEFAULT_WAVE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "AI_Wave": {
        "NVDA": 0.33,
        "MSFT": 0.33,
        "META": 0.34,
    },
    "Growth_Wave": {
        "AAPL": 0.34,
        "AMZN": 0.33,
        "TSLA": 0.33,
    },
    "FuturePower_Wave": {
        "NEE": 0.34,
        "ENPH": 0.33,
        "PLUG": 0.33,
    },
    "CleanTransitInfra_Wave": {
        "TSLA": 0.34,
        "NIO": 0.33,
        "BLDR": 0.33,
    },
    "SmallCapGrowth_Wave": {
        "IWM": 0.50,
        "IJT": 0.50,
    },
    "SmallMidGrowth_Wave": {
        "VO": 0.50,
        "VOE": 0.50,
    },
    "Quantum_Wave": {
        "NVDA": 0.34,
        "AMD": 0.33,
        "AVGO": 0.33,
    },
    "CryptoIncome_Wave": {
        "BTC-USD": 0.50,
        "ETH-USD": 0.50,
    },
    "SP500_Wave": {
        BENCH_TICKER: 1.0,
    },
}


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(w, 0.0) for w in weights.values())
    if total <= 0:
        # Equal weight if all zeros/negative
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: max(w, 0.0) / total for k, w in weights.items()}


def _default_weights_df() -> pd.DataFrame:
    rows = []
    for wave, mapping in DEFAULT_WAVE_WEIGHTS.items():
        norm = _normalize_weights(mapping)
        for ticker, w in norm.items():
            rows.append({"wave": wave, "ticker": ticker, "weight": w})
    return pd.DataFrame(rows, columns=["wave", "ticker", "weight"])


# -------------------------------------------------------------------
# Wave weights loader with robust fallback
# -------------------------------------------------------------------

@st.cache_data
def load_wave_weights(path: str = "wave_weights.csv") -> pd.DataFrame:
    """
    Load wave weights from CSV, clean them, and if anything goes wrong
    fall back to the code-managed DEFAULT_WAVE_WEIGHTS.
    """
    default_df = _default_weights_df()

    # Try reading CSV in the most forgiving way we can.
    try:
        # Newer pandas supports on_bad_lines
        try:
            raw = pd.read_csv(path, on_bad_lines="skip")
        except TypeError:
            raw = pd.read_csv(path, error_bad_lines=False)

    except FileNotFoundError:
        st.warning("wave_weights.csv not found. Using code-managed default weights.")
        return default_df
    except Exception as e:
        st.warning(f"Could not parse wave_weights.csv ({e}). Using code-managed default weights.")
        return default_df

    if raw is None or raw.empty:
        st.warning("wave_weights.csv is empty. Using code-managed default weights.")
        return default_df

    # Canonicalize column names (case / whitespace)
    col_map = {c.strip().lower(): c for c in raw.columns}
    required = ["wave", "ticker", "weight"]
    missing = [r for r in required if r not in col_map]
    if missing:
        st.warning(
            "wave_weights.csv is missing required columns "
            f"{missing}. Using code-managed default weights."
        )
        return default_df

    wave_col = col_map["wave"]
    ticker_col = col_map["ticker"]
    weight_col = col_map["weight"]

    df = raw[[wave_col, ticker_col, weight_col]].copy()
    df.columns = ["wave", "ticker", "weight"]

    # Clean types and whitespace
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    df = df.dropna(subset=["wave", "ticker", "weight"])
    if df.empty:
        st.warning(
            "wave_weights.csv has no usable rows after cleaning. "
            "Using code-managed default weights."
        )
        return default_df

    # Normalize weights within each wave
    df["weight"] = df["weight"].clip(lower=0.0)
    total_by_wave = df.groupby("wave")["weight"].transform("sum")
    # Avoid divide-by-zero: where total is 0, fall back to equal-weight
    df.loc[total_by_wave > 0, "weight"] = (
        df.loc[total_by_wave > 0, "weight"] / total_by_wave[total_by_wave > 0]
    )

    zero_waves = df.loc[total_by_wave == 0, "wave"].unique()
    for wname in zero_waves:
        mask = df["wave"] == wname
        n = mask.sum()
        if n > 0:
            df.loc[mask, "weight"] = 1.0 / n

    # If anything still looks off, fall back
    if df["weight"].isna().all():
        st.warning(
            "wave_weights.csv produced NaN weights after normalization. "
            "Using code-managed default weights."
        )
        return default_df

    return df


# -------------------------------------------------------------------
# Price history loader
# -------------------------------------------------------------------

@st.cache_data
def fetch_price_history(
    tickers: List[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for tickers between start and end dates.
    Returns a DataFrame indexed by date with one column per ticker.

    If yfinance fails or returns nothing, we return an empty DataFrame.
    """
    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end + dt.timedelta(days=1),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        st.warning(f"Price fetch failed: {e}")
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    else:
        prices = data.copy()

    # Make sure we only keep our tickers and drop empty columns
    prices = prices[tickers].dropna(how="all")
    return prices


# -------------------------------------------------------------------
# Portfolio & alpha calculations
# -------------------------------------------------------------------

def compute_wave_series(
    wave_name: str,
    weights_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compute equity curve for a given wave and its benchmark (SPY).

    Returns:
        wave_equity, bench_equity (both indexed by date)
    """
    if price_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    if BENCH_TICKER not in price_df.columns:
        st.warning(f"Benchmark {BENCH_TICKER} not in price data.")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Select weights for this wave
    w = weights_df[weights_df["wave"] == wave_name].copy()
    if w.empty:
        st.warning(f"No weights found for {wave_name}.")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    w = w.groupby("ticker")["weight"].sum()
    # Filter to tickers we actually have prices for
    w = w[w.index.isin(price_df.columns)]
    if w.empty:
        st.warning(f"No overlapping tickers between weights and prices for {wave_name}.")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Normalize weights again just to be safe
    w = w.clip(lower=0.0)
    if w.sum() <= 0:
        w[:] = 1.0 / len(w)
    else:
        w = w / w.sum()

    # Compute daily returns
    prices = price_df[w.index.tolist() + [BENCH_TICKER]].dropna(how="all")
    rets = prices.pct_change().dropna()

    wave_ret = (rets[w.index] * w.values).sum(axis=1)
    bench_ret = rets[BENCH_TICKER]

    wave_equity = (1.0 + wave_ret).cumprod()
    bench_equity = (1.0 + bench_ret).cumprod()

    return wave_equity, bench_equity


def compute_today_stats(
    wave_equity: pd.Series,
    bench_equity: pd.Series,
    exposure: float = 0.75,
) -> Dict[str, float]:
    if len(wave_equity) < 2 or len(bench_equity) < 2:
        return {
            "wave_today": np.nan,
            "bench_today": np.nan,
            "alpha_today": np.nan,
            "beta_realized": np.nan,
        }

    # Daily returns
    wave_ret = wave_equity.pct_change().dropna()
    bench_ret = bench_equity.pct_change().dropna()

    today_wave = wave_ret.iloc[-1]
    today_bench = bench_ret.iloc[-1]

    # Simple alpha today = wave - exposure * bench
    alpha_today = today_wave - exposure * today_bench

    # Realized beta via covariance / variance over full lookback
    if bench_ret.var() > 0 and len(wave_ret) > 10:
        beta_real = np.cov(wave_ret, bench_ret)[0, 1] / bench_ret.var()
    else:
        beta_real = np.nan

    return {
        "wave_today": today_wave,
        "bench_today": today_bench,
        "alpha_today": alpha_today,
        "beta_realized": beta_real,
    }


def _window_alpha(
    wave_equity: pd.Series,
    bench_equity: pd.Series,
    days: int,
) -> float:
    """
    Compute cumulative alpha over the last `days` trading days:
    Alpha_window = (1 + R_wave) / (1 + R_bench) - 1
    """
    if len(wave_equity) < days + 1 or len(bench_equity) < days + 1:
        return np.nan

    wave_slice = wave_equity.iloc[-(days + 1):]
    bench_slice = bench_equity.iloc[-(days + 1):]

    wave_ret = wave_slice.pct_change().dropna()
    bench_ret = bench_slice.pct_change().dropna()

    if wave_ret.empty or bench_ret.empty:
        return np.nan

    cw = (1.0 + wave_ret).prod() - 1.0
    cb = (1.0 + bench_ret).prod() - 1.0

    return (1.0 + cw) / (1.0 + cb) - 1.0


def compute_alpha_windows(
    wave_equity: pd.Series,
    bench_equity: pd.Series,
) -> Dict[str, float]:
    windows = {
        "30D": 30,
        "60D": 60,
        "6M": 126,   # ~21 trading days * 6
        "1Y": 252,   # ~252 trading days per year
    }
    out = {}
    for label, n in windows.items():
        out[label] = _window_alpha(wave_equity, bench_equity, n)
    return out


# -------------------------------------------------------------------
# Streamlit layout
# -------------------------------------------------------------------

def format_pct(x: float) -> str:
    if pd.isna(x):
        return "–"
    return f"{x * 100:0.2f}%"


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Adaptive Portfolio Waves • AIWs/APWs • SmartSafe™ • VIX-gated risk • "
        "Alpha-Minus-Beta & Private Logic™"
    )

    # Sidebar controls ------------------------------------------------
    weights_df = load_wave_weights()

    waves = sorted(weights_df["wave"].unique())
    default_wave = "SP500_Wave" if "SP500_Wave" in waves else waves[0]

    selected_wave = st.sidebar.selectbox("Select Wave", options=waves, index=waves.index(default_wave))
    mode = st.sidebar.selectbox("Mode", options=["Standard", "Alpha-Minus-Beta", "Private Logic™"])
    lookback_days = st.sidebar.slider("Lookback (trading days)", min_value=60, max_value=365, value=365, step=5)
    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    # Simple exposure rule for now (can be wired to VIX later)
    exposure = 0.75
    smartsafe = 1.0 - exposure

    # Date range for prices ------------------------------------------
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=int(lookback_days * 1.5))

    tickers = sorted(weights_df["ticker"].unique().tolist())
    if BENCH_TICKER not in tickers:
        tickers.append(BENCH_TICKER)
    if VIX_TICKER not in tickers:
        tickers.append(VIX_TICKER)

    prices = fetch_price_history(tickers, start_date, end_date)

    # Split out SPY and VIX
    vix_series = None
    if not prices.empty and VIX_TICKER in prices.columns:
        vix_series = prices[VIX_TICKER].dropna()
        prices = prices.drop(columns=[VIX_TICKER])

    # Compute series for the selected wave ---------------------------
    wave_eq, bench_eq = compute_wave_series(selected_wave, weights_df, prices)

    # Main metric row -------------------------------------------------
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    if wave_eq.empty or bench_eq.empty:
        with col1:
            st.metric("Wave Today", "–")
        with col2:
            st.metric("Benchmark Today (SPY)", "–")
        with col3:
            st.metric("Today Alpha Captured", "–")
        with col4:
            st.metric("Realized Beta vs SPY", "–")
        with col5:
            st.metric("Current Exposure", format_pct(exposure))
        with col6:
            st.metric("SmartSafe™ Allocation Now", format_pct(smartsafe))
        st.info("No price data available for this wave / benchmark.")
    else:
        stats = compute_today_stats(wave_eq, bench_eq, exposure=exposure)
        with col1:
            st.metric("Wave Today", format_pct(stats["wave_today"]))
        with col2:
            st.metric("Benchmark Today (SPY)", format_pct(stats["bench_today"]))
        with col3:
            st.metric("Today Alpha Captured", format_pct(stats["alpha_today"]))
        with col4:
            st.metric(
                "Realized Beta vs SPY",
                "–" if pd.isna(stats["beta_realized"]) else f"{stats['beta_realized']:.2f}",
                help="Rolling beta estimated from covariance vs SPY over the current lookback window.",
            )
        with col5:
            st.metric("Current Exposure", format_pct(exposure), help="Equity allocation for this wave.")
        with col6:
            st.metric("SmartSafe™ Allocation Now", format_pct(smartsafe))

    # Alpha captured windows -----------------------------------------
    st.subheader("Alpha Captured Windows (This Wave Only)")
    col30, col60, col6m, col1y = st.columns(4)

    if wave_eq.empty or bench_eq.empty:
        for c in (col30, col60, col6m, col1y):
            with c:
                st.metric("", "–")
    else:
        alpha_w = compute_alpha_windows(wave_eq, bench_eq)
        with col30:
            st.metric("30D", format_pct(alpha_w["30D"]))
        with col60:
            st.metric("60D", format_pct(alpha_w["60D"]))
        with col6m:
            st.metric("6M", format_pct(alpha_w["6M"]))
        with col1y:
            st.metric("1Y", format_pct(alpha_w["1Y"]))

    # Equity curves & holdings ---------------------------------------
    left, right = st.columns([2, 1])

    with left:
        st.subheader(f"{selected_wave} vs Benchmark (Equity Curves)")
        if wave_eq.empty or bench_eq.empty:
            st.info("No price data available for this wave / benchmark.")
        else:
            eq_df = pd.DataFrame(
                {
                    selected_wave: wave_eq,
                    "Benchmark (SPY)": bench_eq,
                }
            ).dropna()
            st.line_chart(eq_df)

        st.subheader(f"{BENCH_TICKER} (Benchmark) – Price")
        if prices.empty or BENCH_TICKER not in prices.columns:
            st.info(f"No {BENCH_TICKER} price data available.")
        else:
            st.line_chart(prices[BENCH_TICKER].dropna())

        st.subheader("VIX – Level")
        if vix_series is None or vix_series.empty:
            st.info("No VIX data available.")
        else:
            st.line_chart(vix_series)

    with right:
        st.subheader("Top Holdings (Live)")
        w = weights_df[weights_df["wave"] == selected_wave].copy()
        if w.empty:
            st.info("No holdings for this wave.")
        else:
            w = (
                w.groupby("ticker")["weight"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            # Build Google quote links
            w["Google Quote"] = w["ticker"].apply(
                lambda t: f"https://www.google.com/finance/quote/{t}:NYSE"
            )
            w["Weight %"] = (w["weight"] * 100).round(2)
            display = w[["ticker", "Weight %", "Google Quote"]]
            st.dataframe(display, use_container_width=True)

    # Engine status ---------------------------------------------------
    st.markdown(
        f"""
        <div style="text-align:right; font-size:0.8rem; color:#00FF00;">
        Engine Status: SANDBOX • Last refresh: {dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Debug panel -----------------------------------------------------
    if show_debug:
        st.subheader("Debug Info")
        st.write("Wave weights (head):", weights_df.head())
        st.write("Prices (head):", prices.head())
        st.write("Wave equity (tail):", wave_eq.tail())
        st.write("Benchmark equity (tail):", bench_eq.tail())


if __name__ == "__main__":
    main()
