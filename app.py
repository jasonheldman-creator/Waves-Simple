# app.py
# WAVES Institutional Console — Live Engine · Multi-Wave
#
# Features:
# - Uses wave_weights.csv as the master Wave universe (Primary / Secondary baskets).
# - Discovers Waves automatically from wave_weights.csv.
# - Uses logs/positions/<Wave>_positions_YYYYMMDD.csv (if present)
#   for live Top-10 holdings with Google Finance links, falling back
#   to weights if no positions log exists yet.
# - Uses logs/performance/<Wave>_performance_*.csv (if present)
#   for performance curve & metrics.
# - Bloomberg-style header with SPX & VIX tiles and Wave / Mode badges.

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
LOGS_POS_DIR = os.path.join(BASE_DIR, "logs", "positions")
WAVE_WEIGHTS_FILE = os.path.join(BASE_DIR, "wave_weights.csv")

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

BETA_TARGET = 0.90
DEFAULT_EXPOSURE = 90

# -------------------------------------------------------------------
# STREAMLIT CONFIG & LIGHT CSS
# -------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Institutional Console",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Just enough CSS to make it feel like a mini terminal
st.markdown(
    """
    <style>
    .waves-header-box {
        border-radius: 10px;
        padding: 0.75rem 1.0rem;
        background: radial-gradient(circle at top left, #06263b 0, #02020a 50%, #000000 100%);
        border: 1px solid rgba(0, 255, 120, 0.35);
        box-shadow: 0 0 16px rgba(0, 255, 120, 0.22);
        margin-bottom: 0.75rem;
    }
    .waves-pill {
        display: inline-block;
        padding: 0.12rem 0.5rem;
        margin-right: 0.35rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border: 1px solid rgba(0, 255, 120, 0.55);
        color: #9fffbf;
        background: rgba(0, 255, 120, 0.08);
    }
    .waves-header-title {
        font-size: 1.0rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #f5f7ff;
        margin-bottom: 0.25rem;
    }
    .waves-header-sub {
        font-size: 0.8rem;
        color: #c2e0ff;
    }
    .ticker-strip {
        font-size: 0.8rem;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        border: 1px solid rgba(0,255,120,0.25);
        background: linear-gradient(90deg, #050810 0, #071624 50%, #050810 100%);
        margin-bottom: 0.8rem;
    }
    .ticker-label {
        color: #c7e9ff;
        font-weight: 600;
        margin-right: 0.18rem;
    }
    .ticker-value-up { color: #3dff96; font-weight: 700; }
    .ticker-value-down { color: #ff4d6a; font-weight: 700; }
    .ticker-value-flat { color: #e0e0e0; font-weight: 700; }
    .ticker-sep {
        color: #6f8195;
        margin: 0 0.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------------------------

@st.cache_data(ttl=60)
def fetch_index_snapshot(ticker: str):
    """Fetch last price and % change for an index using yfinance."""
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
        return {"last": last, "pct": pct}
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
    """Load wave_weights.csv and normalize internal column names."""
    if not os.path.exists(WAVE_WEIGHTS_FILE):
        st.error("wave_weights.csv not found in repo root.")
        st.stop()

    df = pd.read_csv(WAVE_WEIGHTS_FILE)

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
    """Slice weights dataframe for a given Wave and basket name."""
    sub = weights_df[
        (weights_df["__wave"] == wave) & (weights_df["__basket"] == basket)
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
    # Default to NASDAQ on Google Finance; still works for many tickers.
    return f"[{t}](https://www.google.com/finance/quote/{t}:NASDAQ)"


def sanitize_wave_for_filename(wave: str) -> str:
    """Turn 'Growth Wave' into 'Growth_Wave' etc. for file matching."""
    return wave.replace(" ", "_")


def get_latest_perf_file(wave: str):
    pattern = os.path.join(LOGS_PERF_DIR, f"{sanitize_wave_for_filename(wave)}_performance_*.csv")
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

    # Build equity curve
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


# ---------- Positions / Top-10 helpers ----------

def get_latest_positions_file(wave: str):
    pattern = os.path.join(
        LOGS_POS_DIR, f"{sanitize_wave_for_filename(wave)}_positions_*.csv"
    )
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort()
    return files[-1]


def load_positions_top10(wave: str):
    """
    Load top-10 holdings from logs/positions/<Wave>_positions_*.csv if present.
    Assumes columns like: ticker, weight, market_value.
    Returns (df_top10, source_string) or (None, None) if not found.
    """
    latest = get_latest_positions_file(wave)
    if not latest:
        return None, None

    try:
        df = pd.read_csv(latest)
    except Exception as e:
        st.warning(f"Could not read positions file for {wave}: {e}")
        return None, None

    cols_l = {c.lower(): c for c in df.columns}
    ticker_col = cols_l.get("ticker") or cols_l.get("symbol")
    weight_col = cols_l.get("weight")
    mv_col = cols_l.get("market_value") or cols_l.get("marketvalue") or cols_l.get("value")

    if not ticker_col:
        st.warning(f"Positions file {os.path.basename(latest)} has no 'ticker' column.")
        return None, None

    df["__ticker"] = df[ticker_col].astype(str).str.strip().str.upper()

    if weight_col:
        df["Weight %"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0) * 100.0
    elif mv_col:
        mv = pd.to_numeric(df[mv_col], errors="coerce").fillna(0.0)
        total = mv.sum()
        df["Weight %"] = (mv / total * 100.0) if total > 0 else 0.0
    else:
        # equal weights
        n = len(df)
        df["Weight %"] = 100.0 / n if n > 0 else 0.0

    df = df.sort_values("Weight %", ascending=False).head(10).copy()
    return df, f"logs/positions/{os.path.basename(latest)}"


def build_top10_from_weights(primary_df: pd.DataFrame, secondary_df: pd.DataFrame):
    """Fallback top-10 using wave_weights if we don't have positions logs."""
    combined = pd.concat([primary_df, secondary_df], ignore_index=True)
    if combined.empty:
        return None
    combined = combined.sort_values("Weight %", ascending=False).head(10).copy()
    combined["__ticker"] = combined["__ticker"].astype(str).str.upper()
    return combined


# -------------------------------------------------------------------
# RENDER HELPERS
# -------------------------------------------------------------------

def render_header(selected_wave: str, mode: str):
    """Bloomberg-style header using Streamlit columns + light HTML."""
    spx = fetch_index_snapshot(SPX_TICKER)
    vix = fetch_index_snapshot(VIX_TICKER)

    with st.container():
        st.markdown('<div class="waves-header-box">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown(
                """
                <div class="waves-header-title">WAVES INSTITUTIONAL CONSOLE</div>
                <div class="waves-header-sub">
                    <span class="waves-pill">LIVE ENGINE</span>
                    <span class="waves-pill">MULTI-WAVE</span>
                    <span class="waves-pill">ADAPTIVE INDEX WAVES™</span>
                    <span style="margin-left:0.3rem; opacity:0.85;">
                        Mini Bloomberg-style terminal for WAVES Intelligence™
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.write(
                f"**Selected Wave:** {selected_wave}  |  **Mode:** {mode}  "
                f"|  **UTC:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
            )

        st.markdown("</div>", unsafe_allow_html=True)

    return spx, vix


def render_ticker_strip(spx_snap, vix_snap, wave_name, metrics):
    parts = []

    def add(label, snap):
        if not snap:
            return
        pct = snap["pct"]
        cls = "ticker-value-flat"
        if pct > 0.05:
            cls = "ticker-value-up"
        elif pct < -0.05:
            cls = "ticker-value-down"
        parts.append(
            f'<span class="ticker-label">{label}</span>'
            f'<span class="{cls}">{snap["last"]:.2f} ({pct_str(pct)})</span>'
        )

    add("SPX", spx_snap)
    add("VIX", vix_snap)

    if metrics and metrics.get("today") is not None:
        today = metrics["today"] * 100
        cls = "ticker-value-flat"
        if today > 0.05:
            cls = "ticker-value-up"
        elif today < -0.05:
            cls = "ticker-value-down"
        parts.append(
            f'<span class="ticker-label">{wave_name}</span>'
            f'<span class="{cls}">{pct_str(today)} today</span>'
        )

    if not parts:
        return

    html = '<div class="ticker-strip">' + '<span class="ticker-sep">|</span>'.join(parts) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_metric_strip(metrics):
    c1, c2, c3, c4 = st.columns(4)

    if metrics is None:
        cards = [
            ("Total Return (live)", "—"),
            ("Today", "—"),
            ("Max Drawdown", "—"),
            ("Alpha vs Benchmark", "—"),
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
    st.table(df[cols])


def render_top10(top10_df: pd.DataFrame, source: str):
    st.subheader("Top 10 Positions — Google Finance Links")
    if top10_df is None or top10_df.empty:
        st.info("No holdings available yet for this Wave.")
        return

    df = top10_df.copy()
    if "__ticker" in df.columns:
        df["Ticker"] = df["__ticker"].apply(google_link)
    else:
        df["Ticker"] = df.iloc[:, 0].astype(str).apply(google_link)

    if "Weight %".lower() in [c.lower() for c in df.columns]:
        # normalize "Weight %" name
        name_map = {c: "Weight %" for c in df.columns if c.lower() == "weight %"}
        df.rename(columns=name_map, inplace=True)

    if "Weight %" in df.columns:
        df["Weight %"] = df["Weight %"].round(2)

    cols = ["Ticker"]
    if "Weight %" in df.columns:
        cols.append("Weight %")

    st.table(df[cols])
    st.caption(f"Source: {source}")


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

    st.markdown("**Positions Logs (logs/positions/)**")
    if os.path.isdir(LOGS_POS_DIR):
        files = sorted(glob.glob(os.path.join(LOGS_POS_DIR, "*.csv")))
        if files:
            st.write("\n".join(os.path.basename(f) for f in files))
        else:
            st.write("(none yet)")
    else:
        st.write("logs/positions/ folder not found")

    st.markdown("**Base Paths**")
    st.code(
        f"BASE_DIR      = {BASE_DIR}\n"
        f"LOGS_PERF_DIR = {LOGS_PERF_DIR}\n"
        f"LOGS_POS_DIR  = {LOGS_POS_DIR}\n"
        f"WAVE_WEIGHTS  = {WAVE_WEIGHTS_FILE}"
    )


# -------------------------------------------------------------------
# MAIN APP FLOW
# -------------------------------------------------------------------

weights_df = load_wave_weights()
wave_list = discover_waves(weights_df)

# Sidebar controls
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

    # Baskets from wave_weights
    c_primary, c_secondary = st.columns(2)
    primary_df = get_basket(weights_df, selected_wave, "Primary")
    secondary_df = get_basket(weights_df, selected_wave, "Secondary")

    with c_primary:
        render_basket("Primary Basket", primary_df)

    with c_secondary:
        render_basket("Secondary Basket", secondary_df)

    st.markdown("---")

    # Top 10: prefer positions logs, else fall back to weights
    top10_df, source = load_positions_top10(selected_wave)
    if top10_df is None:
        weights_top10 = build_top10_from_weights(primary_df, secondary_df)
        if weights_top10 is not None:
            render_top10(weights_top10, "wave_weights.csv (no positions logs yet)")
        else:
            render_top10(None, "")
    else:
        render_top10(top10_df, source)

with tab_alpha:
    render_alpha_dashboard(perf_df)

with tab_logs:
    render_engine_logs_tab(wave_list)
