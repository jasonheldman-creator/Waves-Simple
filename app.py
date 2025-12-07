# app.py — WAVES Institutional Console (Full Engine + UI Polish)

import datetime as dt
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

WAVE_WEIGHTS_PATH = "wave_weights.csv"
UNIVERSE_PATH = "universe.csv"        # optional; used for Name/Sector lookup

BENCHMARK_BY_WAVE = {
    # You can extend this mapping as needed
    "AL_Wave": "SPY",
    "Growth_Wave": "SPYG",
    "Infinity_Wave": "QQQ",
    "Quantum_Wave": "QTUM",
    "Future_Power_Wave": "XLE",
    "Small_Cap_Growth_Wave": "IWO",
    "Small_Mid_Cap_Growth_Wave": "IJT",
}

DEFAULT_BENCHMARK = "SPY"
VIX_TICKER = "^VIX"

TARGET_BETA = 0.90
DEFAULT_LOOKBACK_DAYS = 365

# -------------------------------------------------------------------
# UTILS
# -------------------------------------------------------------------


def _normalize_column_names(columns: List[str]) -> List[str]:
    return [c.strip().lower() for c in columns]


def _map_columns(cols: List[str], wanted: dict) -> dict:
    """
    Map fuzzy column names in CSV to canonical names.

    wanted: {"wave": ["wave"], "ticker": ["ticker", "symbol"], ...}
    """
    lower = _normalize_column_names(cols)
    mapping = {}
    for canon, candidates in wanted.items():
        for c in candidates:
            if c.lower() in lower:
                idx = lower.index(c.lower())
                mapping[cols[idx]] = canon
                break
    return mapping


# -------------------------------------------------------------------
# DATA LOADERS
# -------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_weights(path: str) -> pd.DataFrame:
    """Load wave weights CSV robustly and normalize weights within each wave."""
    df = pd.read_csv(path)
    orig_cols = list(df.columns)

    col_map = _map_columns(
        orig_cols,
        {
            "wave": ["wave"],
            "ticker": ["ticker", "symbol"],
            "weight": ["weight", "w", "pct"],
        },
    )
    df = df.rename(columns=col_map)

    required = ["wave", "ticker", "weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"wave_weights.csv missing columns: {missing}. "
            f"Found columns: {orig_cols}"
        )

    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    # Drop zero / negative weights
    df = df[df["weight"] > 0]

    if df.empty:
        raise ValueError("wave_weights.csv contains no positive weights.")

    # Normalize within each wave to sum to 1.0
    def _norm(group: pd.Series) -> pd.Series:
        s = group.sum()
        if s <= 0:
            return group
        return group / s

    df["weight"] = df.groupby("wave")["weight"].transform(_norm)

    return df[["wave", "ticker", "weight"]]


@st.cache_data(show_spinner=False)
def load_universe(path: str) -> Optional[pd.DataFrame]:
    """Load optional universe file with Ticker / Name / Sector info."""
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    orig_cols = list(df.columns)
    col_map = _map_columns(
        orig_cols,
        {
            "ticker": ["ticker", "symbol"],
            "name": ["name", "company", "company name"],
            "sector": ["sector", "industry"],
        },
    )
    df = df.rename(columns=col_map)

    if "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype(str).str.strip()

    keep_cols = [c for c in ["ticker", "name", "sector"] if c in df.columns]
    return df[keep_cols]


@st.cache_data(show_spinner=False)
def fetch_price_history(
    tickers: List[str],
    start: dt.date,
    end: dt.date,
) -> Optional[pd.DataFrame]:
    """Fetch daily close prices for a list of tickers."""
    if not tickers:
        return None
    if yf is None:
        return None

    tickers = sorted(set(tickers))
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end + dt.timedelta(days=1),
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception:
        return None

    if data is None or data.empty:
        return None

    # Handle single vs multi-ticker structure
    if isinstance(data.columns, pd.MultiIndex):
        # Prefer "Adj Close" then "Close"
        lvl0 = set(data.columns.get_level_values(0))
        if "Adj Close" in lvl0:
            prices = data["Adj Close"]
        elif "Close" in lvl0:
            prices = data["Close"]
        else:
            # fallback to first level
            first = list(lvl0)[0]
            prices = data[first]
    else:
        # Single ticker case
        if "Adj Close" in data.columns:
            prices = data["Adj Close"].to_frame()
        elif "Close" in data.columns:
            prices = data["Close"].to_frame()
        else:
            prices = data.iloc[:, [0]]

        prices.columns = tickers

    prices = prices.dropna(how="all")
    return prices


@st.cache_data(show_spinner=False)
def fetch_single_series(
    ticker: str, start: dt.date, end: dt.date
) -> Optional[pd.Series]:
    """Fetch a single price series (e.g., SPY, VIX)."""
    df = fetch_price_history([ticker], start, end)
    if df is None or df.empty:
        return None
    if ticker in df.columns:
        return df[ticker]
    # Fallback to first column
    return df.iloc[:, 0]


# -------------------------------------------------------------------
# PORTFOLIO / ALPHA ENGINE
# -------------------------------------------------------------------


def compute_wave_returns(
    wave: str,
    weights_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> Optional[pd.Series]:
    """Compute daily returns for a given wave from prices + weights."""
    subset = weights_df[weights_df["wave"] == wave]
    if subset.empty:
        return None

    available = [t for t in subset["ticker"].unique() if t in prices.columns]
    if not available:
        return None

    w = (
        subset.set_index("ticker")["weight"]
        .reindex(available)
        .fillna(0.0)
    )

    sub_prices = prices[available].dropna(how="all")
    if sub_prices.shape[0] < 3:
        return None

    rets = sub_prices.pct_change().dropna()

    # align weights to columns
    w = w.reindex(rets.columns).fillna(0.0)
    wave_ret = (rets * w).sum(axis=1)
    return wave_ret


def compute_cumulative_returns(ret: pd.Series) -> pd.Series:
    return (1.0 + ret).cumprod() - 1.0


def compute_beta(
    wave_ret: pd.Series,
    bench_ret: pd.Series,
) -> Optional[float]:
    common = wave_ret.index.intersection(bench_ret.index)
    if len(common) < 20:
        return None
    x = bench_ret.loc[common].values
    y = wave_ret.loc[common].values
    if np.allclose(x, 0) or np.allclose(y, 0):
        return None
    try:
        beta, _ = np.polyfit(x, y, 1)
    except Exception:
        return None
    return float(beta)


def vix_to_exposure(vix_level: float) -> float:
    """Simple VIX ladder → target equity exposure."""
    if np.isnan(vix_level):
        return 0.6
    if vix_level <= 15:
        return 1.0
    if vix_level <= 20:
        return 0.85
    if vix_level <= 25:
        return 0.70
    if vix_level <= 30:
        return 0.55
    if vix_level <= 40:
        return 0.40
    return 0.25


# -------------------------------------------------------------------
# UI HELPERS
# -------------------------------------------------------------------


def google_finance_link(ticker: str) -> str:
    # Use a generic Google Finance quote URL (falls back for most tickers)
    return f"https://www.google.com/finance/quote/{ticker}:NASDAQ"


def format_percent(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def get_today_return_from_prices(prices: pd.DataFrame) -> pd.Series:
    if prices is None or prices.shape[0] < 2:
        return pd.Series(dtype=float)
    last_two = prices.tail(2)
    return (last_two.iloc[-1] / last_two.iloc[-2] - 1.0)


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="WAVES Institutional Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---------------- HEADER ----------------

    col_title, col_status = st.columns([3, 1])

    with col_title:
        st.markdown(
            "<h1 style='color:white;'>WAVES Institutional Console</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#cccccc;'>"
            "Adaptive Portfolio Waves • AIWs/APWs • SmartSafe™ • "
            "VIX-gated risk • Alpha-Minus-Beta & Private Logic™"
            "</p>",
            unsafe_allow_html=True,
        )

    now_utc = dt.datetime.utcnow().replace(microsecond=0)

    with col_status:
        st.markdown(
            "<div style='text-align:right; color:#00ff9a; font-size:14px;'>"
            "<strong>Engine Status:</strong> LIVE / SANDBOX<br>"
            f"<span style='color:#aaaaaa;'>Last refresh: {now_utc} UTC</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---------------- LOAD DATA ----------------

    try:
        weights_df = load_weights(WAVE_WEIGHTS_PATH)
    except Exception as e:
        st.error(
            "Error loading wave_weights.csv. "
            f"Please verify the file and columns. Details: {e}"
        )
        st.stop()

    universe_df = load_universe(UNIVERSE_PATH)

    available_waves = sorted(weights_df["wave"].unique())

    # ---------------- SIDEBAR ----------------

    st.sidebar.title("Wave & Mode")

    selected_wave = st.sidebar.selectbox(
        "Select Wave", available_waves, index=0
    )

    mode = st.sidebar.radio(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    lookback_days = st.sidebar.slider(
        "Lookback (trading days)",
        min_value=90,
        max_value=730,
        value=DEFAULT_LOOKBACK_DAYS,
        step=30,
    )

    st.sidebar.markdown("---")
    show_alpha_curve = st.sidebar.checkbox("Show alpha curve", value=True)
    show_drawdown = st.sidebar.checkbox("Show drawdown", value=False)
    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    # ---------------- DATE RANGE ----------------

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=int(lookback_days * 1.4))

    benchmark_ticker = BENCHMARK_BY_WAVE.get(
        selected_wave, DEFAULT_BENCHMARK
    )

    tickers_for_wave = sorted(
        weights_df.loc[weights_df["wave"] == selected_wave, "ticker"].unique()
    )

    all_price_tickers = list(set(tickers_for_wave + [benchmark_ticker]))

    prices = fetch_price_history(all_price_tickers, start_date, end_date)
    bench_series = fetch_single_series(benchmark_ticker, start_date, end_date)
    vix_series = fetch_single_series(VIX_TICKER, start_date, end_date)

    if prices is None or bench_series is None or prices.empty:
        st.error(
            "No price data available. Check network, yfinance, or ticker "
            "symbols. The engine handled the error without crashing."
        )
        if show_debug:
            st.write("DEBUG prices:", prices)
            st.write("DEBUG benchmark:", bench_series)
        st.stop()

    wave_ret = compute_wave_returns(selected_wave, weights_df, prices)
    bench_ret = bench_series.pct_change().dropna()

    if wave_ret is None or wave_ret.empty:
        st.error(
            f"No returns could be computed for {selected_wave}. "
            "Please verify tickers and weights."
        )
        if show_debug:
            st.write("DEBUG wave_ret:", wave_ret)
        st.stop()

    # Align time series
    common_idx = wave_ret.index.intersection(bench_ret.index)
    wave_ret = wave_ret.loc[common_idx]
    bench_ret = bench_ret.loc[common_idx]

    wave_cum = compute_cumulative_returns(wave_ret)
    bench_cum = compute_cumulative_returns(bench_ret)

    # Daily metrics
    today_wave = wave_ret.iloc[-1] if len(wave_ret) else np.nan
    today_bench = bench_ret.iloc[-1] if len(bench_ret) else np.nan
    today_alpha = today_wave - today_bench

    beta_est = compute_beta(wave_ret, bench_ret)

    latest_vix = float(vix_series.dropna().iloc[-1]) if vix_series is not None and not vix_series.empty else np.nan
    exposure = vix_to_exposure(latest_vix)
    smartsafe_alloc = 1.0 - exposure

    # ---------------- METRICS ROW ----------------

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric("Wave Today", format_percent(today_wave))

    with m2:
        st.metric(f"Benchmark Today ({benchmark_ticker})", format_percent(today_bench))

    with m3:
        beta_str = f"{beta_est:.2f}" if beta_est is not None else "N/A"
        st.metric("Estimated Beta vs SPY", beta_str, help="Based on daily returns")

    with m4:
        st.metric("Exposure", format_percent(exposure))

    with m5:
        vix_str = f"{latest_vix:.1f}" if not np.isnan(latest_vix) else "N/A"
        st.metric("SmartSafe™ Allocation", format_percent(smartsafe_alloc))
        st.caption(f"VIX (latest): {vix_str}")

    st.markdown("---")

    # ---------------- ALPHA & PERFORMANCE CHARTS ----------------

    top_row = st.columns([3, 2])

    # Wave vs Benchmark
    with top_row[0]:
        perf_df = pd.DataFrame(
            {
                "Wave": wave_cum,
                "Benchmark": bench_cum,
            }
        )
        st.subheader(f"{selected_wave} vs Benchmark")
        st.line_chart(perf_df)

        if show_alpha_curve:
            alpha_series = wave_cum - bench_cum
            st.subheader("Cumulative Alpha (Wave − Benchmark)")
            st.line_chart(alpha_series)

        if show_drawdown:
            peak = (1 + wave_cum).cummax()
            dd = (1 + wave_cum) / peak - 1.0
            st.subheader("Wave Drawdown")
            st.line_chart(dd)

    # Top Holdings — live
    with top_row[1]:
        st.subheader("Top Holdings (Live)")

        weights_wave = (
            weights_df.loc[weights_df["wave"] == selected_wave]
            .copy()
            .sort_values("weight", ascending=False)
        )

        # Today's % change for each ticker
        today_ret_by_ticker = get_today_return_from_prices(prices)

        top_n = weights_wave.head(10).copy()
        top_n["Weight%"] = top_n["weight"] * 100.0
        top_n["Today%"] = top_n["ticker"].map(today_ret_by_ticker).fillna(0.0) * 100.0

        # Name / sector from universe
        if universe_df is not None:
            top_n = top_n.merge(
                universe_df,
                how="left",
                left_on="ticker",
                right_on="ticker",
            )

        # Build clickable ticker column
        links = []
        for t in top_n["ticker"]:
            url = google_finance_link(t)
            links.append(f"[{t}]({url})")
        top_n_display = pd.DataFrame(
            {
                "Ticker": links,
                "Weight": top_n["Weight%"].map(lambda x: f"{x:.2f}%"),
                "Today%": top_n["Today%"],
            }
        )
        if "name" in top_n.columns:
            top_n_display["Name"] = top_n["name"]
        if "sector" in top_n.columns:
            top_n_display["Sector"] = top_n["sector"]

        def _color_today(val):
            try:
                v = float(val)
            except Exception:
                return ""
            color = "#00ff7f" if v > 0 else "#ff4b4b" if v < 0 else "#ffffff"
            return f"color: {color};"

        st.dataframe(
            top_n_display.style.format(
                {
                    "Today%": "{:.2f}%",
                }
            ).applymap(_color_today, subset=["Today%"]),
            use_container_width=True,
        )

    # ---------------- SECOND ROW: ALPHA, SPY, VIX ----------------

    bottom_row = st.columns([2, 2, 2])

    # Cumulative alpha small chart
    with bottom_row[0]:
        alpha_small = wave_cum - bench_cum
        st.subheader("Cumulative Alpha (Compact)")
        st.line_chart(alpha_small)

    # SPY / Benchmark price
    with bottom_row[1]:
        st.subheader(f"{benchmark_ticker} (Benchmark) – Price")
        if bench_series is not None and not bench_series.empty:
            st.line_chart(
                pd.DataFrame({benchmark_ticker: bench_series})
            )
        else:
            st.caption("No benchmark data available.")

    # VIX level
    with bottom_row[2]:
        st.subheader("VIX – Level")
        if vix_series is not None and not vix_series.empty:
            st.line_chart(pd.DataFrame({"VIX": vix_series}))
        else:
            st.caption("No VIX data available.")

    # ---------------- DEBUG PANEL ----------------

    if show_debug:
        st.markdown("---")
        st.subheader("DEBUG PANEL")

        st.write("Selected Wave:", selected_wave)
        st.write("Mode:", mode)
        st.write("Benchmark:", benchmark_ticker)
        st.write("Lookback days:", lookback_days)

        st.write("Wave weights (head):")
        st.write(weights_df.head())

        st.write("Prices shape:", prices.shape)
        st.write("Wave returns (tail):")
        st.write(wave_ret.tail())
        st.write("Benchmark returns (tail):")
        st.write(bench_ret.tail())

        st.write("Beta estimate:", beta_est)
        st.write("Latest VIX:", latest_vix)
        st.write("Exposure:", exposure)
        st.write("SmartSafe allocation:", smartsafe_alloc)


if __name__ == "__main__":
    main()
