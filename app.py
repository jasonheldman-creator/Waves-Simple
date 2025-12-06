# app.py
# WAVES Institutional Console — Live Engine · Multi-Wave
#
# Pure Streamlit version (no raw HTML) so nothing shows as HTML text.
# Uses wave_weights.csv as the master positions file and optional
# performance logs in logs/performance/.

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# -------------------------------------------------------------------
# PATHS & CONSTANTS
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_PERF_DIR = os.path.join(BASE_DIR, "logs", "performance")
WAVE_WEIGHTS_FILE = os.path.join(BASE_DIR, "wave_weights.csv")

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

BETA_TARGET = 0.90
DEFAULT_EXPOSURE = 90

# -------------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Institutional Console",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

@st.cache_data(ttl=60)
def fetch_index_snapshot(ticker: str):
    """Fetch last price and % change for an index."""
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if data.empty:
            data = yf.Ticker(ticker).history(period="5d", interval="1d")
        if data.empty:
            return None
        last = float(data["Close"].iloc[-1])
        info = yf.Ticker(ticker).info or {}
        if "previousClose" in info:
            prev = float(info["previousClose"])
        elif len(data) >= 2:
            prev = float(data["Close"].iloc[-2])
        else:
            prev = last
        change = last - prev
        pct = (change / prev) * 100 if prev else 0.0
        return {"last": last, "change": change, "pct": pct}
    except Exception:
        return None


def pct_str(x):
    try:
        return f"{x:+.2f}%"
    except Exception:
        return "—"


def bps_str(x):
    try:
        return f"{x:+.1f} bps"
    except Exception:
        return "—"


@st.cache_data(ttl=60)
def load_wave_weights():
    """Load wave_weights.csv and normalize internal columns."""
    if not os.path.exists(WAVE_WEIGHTS_FILE):
        st.error("wave_weights.csv not found in repo root.")
        st.stop()

    df = pd.read_csv(WAVE_WEIGHTS_FILE)

    # Map flexible column names
    cols = {c.lower(): c for c in df.columns}
    wave_col = cols.get("wave") or cols.get("wavename") or cols.get("wave_name")
    ticker_col = cols.get("ticker") or cols.get("symbol")
    weight_col = cols.get("weight") or cols.get("target_weight")
    basket_col = cols.get("basket")  # optional

    if not wave_col or not ticker_col:
        st.error("wave_weights.csv must have at least 'Wave' and 'Ticker' columns.")
        st.write(df.head())
        st.stop()

    df["__wave"] = df[wave_col].astype(str).str.strip()
    df["__ticker"] = df[ticker_col].astype(str).str.strip().str.upper()

    if weight_col:
        df["__weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    else:
        df["__weight"] = 1.0  # equal weights fallback

    if basket_col:
        df["__basket"] = df[basket_col].astype(str).str.strip().str.title()
    else:
        df["__basket"] = "Primary"

    return df


def discover_waves(weights_df: pd.DataFrame):
    return sorted(weights_df["__wave"].unique().tolist())


def get_basket(weights_df: pd.DataFrame, wave: str, basket: str):
    sub = weights_df[
        (weights_df["__wave"] == wave) &
        (weights_df["__basket"] == basket)
    ].copy()

    if sub.empty:
        return sub

    total = sub["__weight"].sum()
    if total > 0:
        sub["Weight %"] = sub["__weight"] / total * 100.0
    else:
        sub["Weight %"] = np.nan

    sub = sub.sort_values("Weight %", ascending=False)
    return sub


def google_link(ticker: str) -> str:
    t = str(ticker).strip().upper()
    # You can change NASDAQ to NYSE if needed later
    return f"[{t}](https://www.google.com/finance/quote/{t}:NASDAQ)"


def get_latest_perf_file(wave: str):
    pattern = os.path.join(LOGS_PERF_DIR, f"{wave}_performance_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort()
    return files[-1]


def load_performance_df(wave: str):
    latest = get_latest_perf_file(wave)
    if not latest:
        return None, None
    try:
        df = pd.read_csv(latest)
    except Exception as e:
        st.error(f"Could not read performance file for {wave}: {e}")
        return None, latest

    dt_col = None
    for c in ["timestamp", "Timestamp", "datetime", "Datetime", "date", "Date"]:
        if c in df.columns:
            dt_col = c
            break
    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col]).sort_values(dt_col).set_index(dt_col)

    return df, latest


def compute_metrics(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    nav_col = None
    for c in ["nav", "NAV", "equity_curve", "portfolio_value", "value", "Value"]:
        if c in df.columns:
            nav_col = c
            break

    ret_col = None
    for c in ["return", "Return", "daily_return", "strategy_return"]:
        if c in df.columns:
            ret_col = c
            break

    bench_ret_col = None
    for c in ["benchmark_return", "bench_return", "BenchmarkReturn"]:
        if c in df.columns:
            bench_ret_col = c
            break

    # Build curve
    if nav_col is not None:
        curve = df[nav_col].astype(float)
    elif ret_col is not None:
        r = df[ret_col].astype(float).fillna(0.0)
        curve = (1 + r).cumprod()
    else:
        return None

    if len(curve) < 2:
        return None

    total_return = curve.iloc[-1] / curve.iloc[0] - 1.0

    today_ret = None
    if ret_col is not None:
        today_ret = float(df[ret_col].iloc[-1])

    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    max_dd = float(drawdown.min())

    alpha_bps = None
    if bench_ret_col is not None and ret_col is not None:
        r = df[ret_col].astype(float).fillna(0.0)
        rb = df[bench_ret_col].astype(float).fillna(0.0)
        my_curve = (1 + r).cumprod()
        bench_curve = (1 + rb).cumprod()
        alpha = my_curve.iloc[-1] - bench_curve.iloc[-1]
        alpha_bps = float((alpha - 1.0) * 10_000)

    return {
        "curve": curve,
        "total_return": float(total_return),
        "today": today_ret,
        "max_dd": max_dd,
        "alpha_bps": alpha_bps,
    }

# -------------------------------------------------------------------
# RENDER BLOCKS
# -------------------------------------------------------------------

def render_header(selected_wave: str, mode: str):
    st.title("WAVES Institutional Console")
    st.caption("Live engine console for WAVES Intelligence™ — Adaptive Index Waves™")

    col1, col2, col3 = st.columns([2, 1, 1])

    # Index tiles
    with col1:
        st.subheader("Market Snapshot")
        c_spx, c_vix = st.columns(2)

        spx_snap = fetch_index_snapshot(SPX_TICKER)
        vix_snap = fetch_index_snapshot(VIX_TICKER)

        with c_spx:
            st.markdown("**SPX**")
            if spx_snap:
                st.metric(
                    label="",
                    value=f"{spx_snap['last']:.2f}",
                    delta=pct_str(spx_snap["pct"]),
                )
            else:
                st.write("no data")

        with c_vix:
            st.markdown("**VIX**")
            if vix_snap:
                st.metric(
                    label="",
                    value=f"{vix_snap['last']:.2f}",
                    delta=pct_str(vix_snap["pct"]),
                )
            else:
                st.write("no data")

    with col2:
        st.subheader("Wave")
        st.write(f"**Selected Wave:** {selected_wave}")
        st.write(f"**Mode:** {mode}")

    with col3:
        st.subheader("Console")
        st.write(f"**Console Time (UTC):**")
        st.write(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

    return spx_snap, vix_snap


def render_ticker_strip(spx_snap, vix_snap, wave_name, metrics):
    """Simple text ticker strip."""
    parts = []

    if spx_snap:
        parts.append(
            f"SPX {spx_snap['last']:.2f} ({pct_str(spx_snap['pct'])})"
        )
    if vix_snap:
        parts.append(
            f"VIX {vix_snap['last']:.2f} ({pct_str(vix_snap['pct'])})"
        )

    if metrics and metrics.get("today") is not None:
        today = metrics["today"] * 100
        parts.append(f"{wave_name} {pct_str(today)} today")

    if parts:
        st.info(" ·  ".join(parts))


def render_metric_strip(metrics):
    c1, c2, c3, c4 = st.columns(4)

    if metrics is None:
        cards = [
            ("Total Return (live)", "—"),
            ("Today", "—"),
            ("Max Drawdown", "—"),
            ("Alpha Captured vs Benchmark", "—"),
        ]
    else:
        total = metrics.get("total_return")
        today = metrics.get("today")
        max_dd = metrics.get("max_dd")
        alpha_bps = metrics.get("alpha_bps")
        cards = [
            ("Total Return (live)", pct_str(total * 100) if total is not None else "—"),
            ("Today", pct_str(today * 100) if today is not None else "—"),
            ("Max Drawdown", pct_str(max_dd * 100) if max_dd is not None else "—"),
            ("Alpha vs Benchmark", bps_str(alpha_bps) if alpha_bps is not None else "—"),
        ]

    for col, (label, value) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"**{label}**")
            st.write(value)


def render_perf_curve(metrics, df_perf):
    st.subheader("Performance Curve")
    if metrics is None or df_perf is None:
        st.info(
            "No performance log found yet for this Wave. "
            "Once the live engine writes CSVs into logs/performance/, "
            "the performance curve will appear here."
        )
        return
    curve = metrics["curve"]
    st.line_chart(pd.DataFrame({"Performance": curve}))


def render_exposure_cards(exposure_pct: int, mode: str):
    st.subheader("Exposure & Risk")
    cash_pct = 100 - exposure_pct

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Equity Exposure**")
        st.write(f"{exposure_pct}%")
        st.caption(f"Target β ≈ {BETA_TARGET:.2f} in Standard mode")
    with c2:
        st.markdown("**Cash Buffer**")
        st.write(f"{cash_pct}%")
        st.caption("Dynamic SmartSafe™ allocation")
    with c3:
        st.markdown("**Risk Mode**")
        st.write(mode)
        st.caption("Mode-aware engine parameters")


def render_basket(title: str, df_basket: pd.DataFrame):
    st.markdown(f"**{title}**")
    if df_basket.empty:
        st.info("No holdings in this basket yet.")
        return
    df = df_basket.copy()
    df["Ticker"] = df["__ticker"].apply(google_link)
    df["Weight %"] = df["Weight %"].round(2)
    cols = ["Ticker", "Weight %"]
    extra = [c for c in df.columns if c not in cols and not c.startswith("__")]
    cols.extend(extra)
    st.dataframe(df[cols], use_container_width=True)


def render_top10(df_primary: pd.DataFrame, df_secondary: pd.DataFrame):
    st.subheader("Top 10 Positions — Google Finance Links")
    combined = pd.concat([df_primary, df_secondary], ignore_index=True)
    if combined.empty:
        st.info("No holdings defined for this Wave in wave_weights.csv.")
        return
    combined = combined.sort_values("Weight %", ascending=False).head(10).copy()
    combined["Ticker"] = combined["__ticker"].apply(google_link)
    combined["Weight %"] = combined["Weight %"].round(2)
    st.table(combined[["Ticker", "Weight %"]])


def render_alpha_dashboard(df_perf):
    st.subheader("Alpha Dashboard")
    if df_perf is None or df_perf.empty:
        st.info(
            "No performance log with benchmark returns yet. "
            "When the engine logs `return` and `benchmark_return`, "
            "a rolling alpha chart will appear here."
        )
        return

    ret_col = None
    for c in ["return", "Return", "daily_return", "strategy_return"]:
        if c in df_perf.columns:
            ret_col = c
            break
    bench_col = None
    for c in ["benchmark_return", "bench_return", "BenchmarkReturn"]:
        if c in df_perf.columns:
            bench_col = c
            break

    if not ret_col or not bench_col:
        st.info(
            "This performance file does not include both strategy and benchmark return columns."
        )
        st.dataframe(df_perf.head())
        return

    r = df_perf[ret_col].astype(float).fillna(0.0)
    rb = df_perf[bench_col].astype(float).fillna(0.0)
    alpha_daily = r - rb
    alpha_30 = alpha_daily.rolling(30).sum()

    st.line_chart(pd.DataFrame({"30-day Rolling Alpha": alpha_30 * 100}))
    st.caption("Rolling 30-day alpha in percentage points vs benchmark.")


def render_engine_logs_tab(wave_list):
    st.subheader("Engine Logs & Discovery")

    st.markdown("**Discovered Waves (from wave_weights.csv)**")
    if not wave_list:
        st.warning("No Waves discovered. Check the Wave column in wave_weights.csv.")
    else:
        st.write(", ".join(wave_list))

    st.markdown("**Performance Logs (logs/performance/)**")
    if os.path.isdir(LOGS_PERF_DIR):
        files = sorted(glob.glob(os.path.join(LOGS_PERF_DIR, "*.csv")))
        if files:
            st.write("\n".join(os.path.basename(f) for f in files))
        else:
            st.write("(none yet)")
    else:
        st.write("logs/performance/ folder not found")

    st.markdown("**Base Paths**")
    st.code(
        f"BASE_DIR      = {BASE_DIR}\n"
        f"LOGS_PERF_DIR = {LOGS_PERF_DIR}\n"
        f"WAVE_WEIGHTS  = {WAVE_WEIGHTS_FILE}"
    )

# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------

weights_df = load_wave_weights()
wave_list = discover_waves(weights_df)

# Sidebar
with st.sidebar:
    st.header("Engine Controls")

    if wave_list:
        st.success("Using Wave universe from wave_weights.csv.")
    else:
        st.error("No Waves discovered in wave_weights.csv.")
        st.stop()

    selected_wave = st.selectbox("Select Wave", options=wave_list, index=0)

    st.subheader("Mode")
    mode = st.radio(
        "Risk Mode",
        options=["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    exposure = st.slider(
        "Equity Exposure",
        min_value=0,
        max_value=100,
        value=DEFAULT_EXPOSURE,
        step=5,
        help="Target equity allocation; remaining % held in SmartSafe™ / cash.",
    )

    st.caption(f"Target β ≈ {BETA_TARGET:.2f} · Cash buffer: {100 - exposure}%")

# Header & ticker strip
spx_snap, vix_snap = render_header(selected_wave, mode)

perf_df, perf_file = load_performance_df(selected_wave)
metrics = compute_metrics(perf_df)

render_ticker_strip(spx_snap, vix_snap, selected_wave, metrics)

st.markdown("---")
st.subheader("WAVES Engine Dashboard")
st.caption("Live / demo console for WAVES Intelligence™ — Adaptive Index Waves™")

render_metric_strip(metrics)

st.markdown("---")

tab_overview, tab_alpha, tab_logs = st.tabs(["Overview", "Alpha Dashboard", "Engine Logs"])

with tab_overview:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_perf_curve(metrics, perf_df)
    with col_right:
        render_exposure_cards(exposure, mode)

    st.markdown("---")

    c_primary, c_secondary = st.columns(2)
    primary_df = get_basket(weights_df, selected_wave, "Primary")
    secondary_df = get_basket(weights_df, selected_wave, "Secondary")

    with c_primary:
        render_basket("Primary Basket", primary_df)
    with c_secondary:
        render_basket("Secondary Basket", secondary_df)

    st.markdown("---")
    render_top10(primary_df, secondary_df)

with tab_alpha:
    render_alpha_dashboard(perf_df)

with tab_logs:
    render_engine_logs_tab(wave_list)
