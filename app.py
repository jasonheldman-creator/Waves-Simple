import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# 1. Core configuration
# ============================================================

BENCHMARK_TICKER = "SPY"
VIX_TICKER = "^VIX"

# 9 locked-in waves (names must match wave_weights.csv if used)
WAVES = [
    "AI_Wave",
    "Growth_Wave",
    "Quantum_Wave",
    "FuturePower_Wave",
    "CleanTransitInfra_Wave",
    "Crypto_Wave",
    "SmallCap_Wave",
    "SmallMidGrowth_Wave",
    "SP500_Wave",
]

MODES = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

# Default lookback in trading days
DEFAULT_LOOKBACK_DAYS = 365

# ============================================================
# 2. Default weights (safe fallback if CSV is missing/broken)
#    You can edit these tickers/weights later.
# ============================================================

# Dict[wave_name, List[(ticker, weight)]]
DEFAULT_WAVE_WEIGHTS: Dict[str, List[Tuple[str, float]]] = {
    "AI_Wave": [
        ("NVDA", 0.34),
        ("MSFT", 0.33),
        ("META", 0.33),
    ],
    "Growth_Wave": [
        ("AAPL", 0.34),
        ("AMZN", 0.33),
        ("TSLA", 0.33),
    ],
    "Quantum_Wave": [
        ("NVDA", 0.4),
        ("AMD", 0.3),
        ("TSM", 0.3),
    ],
    "FuturePower_Wave": [
        ("NEE", 0.34),
        ("ENPH", 0.33),
        ("FSLR", 0.33),
    ],
    "CleanTransitInfra_Wave": [
        ("TSLA", 0.34),
        ("NIO", 0.33),
        ("BLDR", 0.33),
    ],
    "Crypto_Wave": [
        ("BTC-USD", 0.5),
        ("ETH-USD", 0.5),
    ],
    "SmallCap_Wave": [
        ("IWM", 1.0),
    ],
    "SmallMidGrowth_Wave": [
        ("IJT", 0.5),
        ("IJK", 0.5),
    ],
    "SP500_Wave": [
        ("SPY", 1.0),
    ],
}


def _defaults_as_df() -> pd.DataFrame:
    rows = []
    for wave, pairs in DEFAULT_WAVE_WEIGHTS.items():
        for t, w in pairs:
            rows.append({"wave": wave, "ticker": t, "weight": float(w)})
    df = pd.DataFrame(rows)
    # Ensure each wave’s weights sum to 1
    df["weight"] = (
        df["weight"] / df.groupby("wave")["weight"].transform("sum")
    )
    return df


# ============================================================
# 3. Data loading helpers
# ============================================================

@st.cache_data
def load_wave_weights(path: str = "wave_weights.csv") -> pd.DataFrame:
    """
    Load wave_weights.csv if possible.
    Requirements:
        - Columns: wave, ticker, weight (case-insensitive)
        - Comma-delimited
    On any error, falls back to DEFAULT_WAVE_WEIGHTS.
    """
    default_df = _defaults_as_df()

    try:
        raw = pd.read_csv(path)
    except Exception as e:
        st.warning(
            f"Could not read {path}: {e}. Using internal default weights."
        )
        return default_df

    # Normalize column names
    col_map = {c.strip().lower(): c for c in raw.columns}
    required = ["wave", "ticker", "weight"]
    if not all(r in col_map for r in required):
        st.warning(
            f"{path} is missing required columns {required}. "
            "Using internal default weights."
        )
        return default_df

    wave_col = col_map["wave"]
    ticker_col = col_map["ticker"]
    weight_col = col_map["weight"]

    df = raw[[wave_col, ticker_col, weight_col]].copy()
    df.rename(
        columns={
            wave_col: "wave",
            ticker_col: "ticker",
            weight_col: "weight",
        },
        inplace=True,
    )

    # Clean up
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    # Drop bad rows
    df = df.dropna(subset=["wave", "ticker", "weight"])
    if df.empty:
        st.warning(
            f"{path} had no valid rows after cleaning. "
            "Using internal default weights."
        )
        return default_df

    # Keep only known waves to avoid typos killing the app
    df = df[df["wave"].isin(WAVES)]
    if df.empty:
        st.warning(
            f"{path} contains no rows for the configured waves {WAVES}. "
            "Using internal default weights."
        )
        return default_df

    # Normalize weights within each wave
    df["weight"] = (
        df["weight"]
        / df.groupby("wave")["weight"].transform("sum")
    )

    return df


@st.cache_data
def fetch_price_history(
    tickers: List[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """
    Fetch Adj Close for a list of tickers.
    Returns a DataFrame indexed by date with one column per ticker.
    """
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end + dt.timedelta(days=1),
        progress=False,
    )

    # yfinance returns different shapes for 1 vs many tickers
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    else:
        prices = data.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.dropna(how="all")
    return prices


@st.cache_data
def fetch_single_price(
    ticker: str,
    start: dt.date,
    end: dt.date,
) -> pd.Series:
    df = fetch_price_history([ticker], start, end)
    if ticker in df.columns:
        return df[ticker].dropna()
    elif df.shape[1] == 1:
        return df.iloc[:, 0].dropna()
    return pd.Series(dtype=float)


# ============================================================
# 4. Portfolio / alpha calculations
# ============================================================

def compute_wave_series(
    wave_name: str,
    weights_df: pd.DataFrame,
    all_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily returns, equity curve, and alpha vs SPY
    for a single wave.
    """
    # Subset weights for this wave
    w = weights_df[weights_df["wave"] == wave_name].copy()
    if w.empty:
        return pd.DataFrame()

    tickers = list(w["ticker"].unique())
    w = w.set_index("ticker")["weight"]

    # Align prices
    prices = all_prices.reindex(columns=tickers).dropna(how="all")
    prices = prices.dropna(how="any", axis=0)
    if prices.empty:
        return pd.DataFrame()

    # Daily returns
    rets = prices.pct_change().dropna(how="all")
    # Wave return: weighted sum each day
    aligned_weights = w.reindex(rets.columns).fillna(0.0)
    wave_ret = (rets * aligned_weights).sum(axis=1)

    # Benchmark
    if BENCHMARK_TICKER in all_prices.columns:
        bench_price = all_prices[BENCHMARK_TICKER].loc[rets.index]
    else:
        bench_price = fetch_single_price(
            BENCHMARK_TICKER,
            rets.index.min().date(),
            rets.index.max().date(),
        )
        bench_price = bench_price.reindex(rets.index).ffill().bfill()

    bench_ret = bench_price.pct_change().reindex(wave_ret.index).fillna(0.0)

    # Equity curves (normalized to 1.0)
    wave_equity = (1 + wave_ret).cumprod()
    bench_equity = (1 + bench_ret).cumprod()

    # Simple alpha = wave_ret - bench_ret
    alpha_daily = wave_ret - bench_ret
    alpha_cum = alpha_daily.cumsum()

    df = pd.DataFrame(
        {
            "wave_ret": wave_ret,
            "bench_ret": bench_ret,
            "wave_equity": wave_equity,
            "bench_equity": bench_equity,
            "alpha_daily": alpha_daily,
            "alpha_cum": alpha_cum,
        }
    )
    return df


def compute_alpha_windows(
    series_df: pd.DataFrame,
    windows: List[int],
) -> Dict[int, float]:
    """
    Compute cumulative alpha for the last N days
    for each window in `windows`.
    """
    results: Dict[int, float] = {}
    if series_df.empty:
        for w in windows:
            results[w] = np.nan
        return results

    alpha = series_df["alpha_daily"].copy()
    n = len(alpha)
    for w in windows:
        if n < 2:
            results[w] = np.nan
            continue
        look = min(w, n)
        window_alpha = alpha.iloc[-look:].sum()
        results[w] = window_alpha
    return results


def estimate_realized_beta(series_df: pd.DataFrame) -> float:
    if series_df.empty:
        return np.nan
    x = series_df["bench_ret"].values
    y = series_df["wave_ret"].values
    if len(x) < 10:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return np.nan
    beta = float(np.dot(x, y) / denom)
    return beta


# ============================================================
# 5. Streamlit UI
# ============================================================

def main():
    st.set_page_config(
        page_title="WAVES Institutional Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ----- Sidebar: wave & mode -----
    st.sidebar.header("Wave & Mode")

    wave_name = st.sidebar.selectbox("Select Wave", WAVES, index=0)
    mode = st.sidebar.selectbox("Mode", MODES, index=0)
    lookback_days = st.sidebar.slider(
        "Lookback (trading days)",
        min_value=60,
        max_value=730,
        value=DEFAULT_LOOKBACK_DAYS,
        step=5,
    )

    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    # ----- Dates -----
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=int(lookback_days * 1.6))

    # ----- Load weights safely -----
    weights_df = load_wave_weights("wave_weights.csv")

    # Build list of all tickers we need prices for
    all_tickers = sorted(weights_df["ticker"].unique().tolist())
    if BENCHMARK_TICKER not in all_tickers:
        all_tickers.append(BENCHMARK_TICKER)
    if VIX_TICKER not in all_tickers:
        all_tickers.append(VIX_TICKER)

    prices = fetch_price_history(all_tickers, start_date, end_date)

    # ----- Compute series for selected wave -----
    series_df = compute_wave_series(wave_name, weights_df, prices)

    st.markdown("## WAVES Institutional Console")
    st.markdown(
        "Adaptive Portfolio Waves • AIWs/APWs • SmartSafe™ • "
        "VIX-gated risk • Alpha-Minus-Beta & Private Logic™"
    )

    # Engine status
    st.markdown(
        f"<div style='text-align:right;color:#00FF00;'>"
        f"Engine Status: SANDBOX &nbsp;&nbsp;&nbsp; "
        f"Last refresh: {dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ----- Top metrics row -----
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    if series_df.empty:
        wave_today = np.nan
        bench_today = np.nan
        today_alpha = np.nan
        realized_beta = np.nan
    else:
        wave_today = series_df["wave_ret"].iloc[-1]
        bench_today = series_df["bench_ret"].iloc[-1]
        today_alpha = series_df["alpha_daily"].iloc[-1]
        realized_beta = estimate_realized_beta(series_df)

    # Simple placeholder exposure logic (can refine later)
    if mode == "Standard":
        exposure = 0.75
    elif mode == "Alpha-Minus-Beta":
        exposure = 0.60
    else:  # Private Logic
        exposure = 0.90

    smartsafe_alloc = 1.0 - exposure

    # VIX latest
    vix_series = fetch_single_price(VIX_TICKER, start_date, end_date)
    vix_latest = float(vix_series.iloc[-1]) if not vix_series.empty else np.nan

    col1.metric("Wave Today", f"{wave_today * 100:0.2f}%" if pd.notna(wave_today) else "–")
    col2.metric(
        "Benchmark Today (SPY)",
        f"{bench_today * 100:0.2f}%" if pd.notna(bench_today) else "–",
    )
    col3.metric(
        "Today Alpha Captured",
        f"{today_alpha * 100:0.2f}%" if pd.notna(today_alpha) else "–",
    )
    col4.metric(
        "Realized Beta vs SPY",
        f"{realized_beta:0.2f}" if pd.notna(realized_beta) else "–",
    )
    col5.metric("Current Exposure", f"{exposure * 100:0.1f}%")
    col6.metric(
        "SmartSafe™ Allocation Now", f"{smartsafe_alloc * 100:0.1f}%"
    )

    st.markdown("---")

    # ----- Alpha captured windows -----
    st.markdown("### Alpha Captured Windows (This Wave Only)")

    win_cols = st.columns(4)
    windows = [30, 60, 126, 252]  # 30D, 60D, ~6M, ~1Y
    alpha_windows = compute_alpha_windows(series_df, windows)

    labels = ["30D", "60D", "6M", "1Y"]
    for c, lbl, w in zip(win_cols, labels, windows):
        val = alpha_windows.get(w, np.nan)
        c.metric(
            lbl,
            f"{val * 100:0.2f}%" if pd.notna(val) else "–",
        )

    st.markdown("---")

    # ----- Equity curve + Top holdings -----
    left, right = st.columns([2, 1])

    with left:
        st.markdown(f"### {wave_name} vs Benchmark (Equity Curves)")
        if series_df.empty:
            st.info("No price data available for this wave / benchmark.")
        else:
            eq = series_df[["wave_equity", "bench_equity"]].copy()
            eq.rename(
                columns={
                    "wave_equity": wave_name,
                    "bench_equity": "SPY",
                },
                inplace=True,
            )
            st.line_chart(eq)

    with right:
        st.markdown("### Top Holdings (Live)")

        ww = weights_df[weights_df["wave"] == wave_name].copy()
        if ww.empty:
            st.info("No holdings defined for this wave.")
        else:
            # Make sure we don't hit the previous pandas error:
            # use a plain DataFrame, unique column names, simple sort_values
            ww = ww[["ticker", "weight"]].copy()
            ww["weight_pct"] = ww["weight"] * 100.0
            ww = ww.sort_values(by="weight", ascending=False)

            display_df = ww[["ticker", "weight_pct"]].copy()
            display_df.rename(columns={"weight_pct": "Weight %"}, inplace=True)
            display_df["Google Quote"] = display_df["ticker"].apply(
                lambda t: f"https://www.google.com/finance/quote/{t}:NASDAQ"
            )
            st.dataframe(display_df, use_container_width=True)

    st.markdown("---")

    # ----- SPY & VIX charts -----
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### SPY (Benchmark) – Price")
        spy_series = fetch_single_price(BENCHMARK_TICKER, start_date, end_date)
        if spy_series.empty:
            st.info("No SPY price data available.")
        else:
            st.line_chart(spy_series.to_frame("SPY"))

    with c2:
        st.markdown("### VIX – Level")
        if vix_series.empty:
            st.info("No VIX data available.")
        else:
            st.line_chart(vix_series.to_frame("VIX"))

    # ----- Debug section -----
    if show_debug:
        st.markdown("### Debug Info")
        st.write("Wave weights (cleaned):", weights_df.head())
        st.write("Prices (sample):", prices.iloc[:5, :5])
        st.write("Series DF (tail):", series_df.tail())


if __name__ == "__main__":
    main()