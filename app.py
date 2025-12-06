# app.py
# WAVES INSTITUTIONAL CONSOLE — LIVE ENGINE · MULTI-WAVE
#
# Single source of truth: wave_weights.csv
#   - Wave list
#   - Primary / Secondary baskets
#   - Full holdings + Top 10 with Google Finance links
#
# Optional: performance logs in logs/performance/<Wave>_performance_*.csv
#   - Returns, performance curve, alpha dashboard

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
# STREAMLIT PAGE CONFIG & STYLES
# -------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Institutional Console",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS and layout styling
st.markdown(
    """
    <style>
    body {
        background-color: #050510;
        color: #f5f7ff;
    }
    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
    }

    /* Header */
    .waves-header {
        border: 1px solid #0b7736;
        box-shadow: 0 0 18px rgba(0, 255, 120, 0.25);
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        background: radial-gradient(circle at top left, #06263b 0, #02020a 45%, #000000 100%);
        margin-bottom: 0.75rem;
    }
    .waves-header-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #f5f7ff;
        margin-bottom: 0.25rem;
    }
    .waves-header-subtitle {
        font-size: 0.85rem;
        color: #cde8ff;
    }
    .waves-header-pill {
        display: inline-block;
        font-size: 0.7rem;
        padding: 0.15rem 0.45rem;
        margin-right: 0.35rem;
        border-radius: 999px;
        background: rgba(0, 255, 120, 0.1);
        color: #9fffbf;
        border: 1px solid rgba(0, 255, 120, 0.35);
    }

    .index-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .index-tile {
        min-width: 120px;
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        border: 1px solid rgba(153, 255, 195, 0.25);
        background: rgba(1, 12, 20, 0.85);
        font-size: 0.75rem;
    }
    .index-tile-label {
        font-weight: 600;
        color: #c7e9ff;
        margin-bottom: 0.1rem;
    }
    .index-tile-value {
        font-size: 0.9rem;
        font-weight: 700;
    }
    .index-up { color: #4bff9a; }
    .index-down { color: #ff5d73; }
    .index-flat { color: #e0e0e0; }
    .index-tile-change {
        font-size: 0.7rem;
        opacity: 0.9;
    }

    .waves-meta {
        text-align: right;
        font-size: 0.75rem;
        color: #c0d8ff;
    }
    .waves-meta strong { color: #ffffff; }

    /* Ticker tape */
    .ticker-bar {
        margin-top: 0.35rem;
        margin-bottom: 0.7rem;
        border-top: 1px solid rgba(0, 255, 120, 0.25);
        border-bottom: 1px solid rgba(0, 255, 120, 0.25);
        padding: 0.25rem 0;
        overflow: hidden;
        background: linear-gradient(90deg, #050810 0, #041018 50%, #050810 100%);
    }
    .ticker-track {
        display: inline-block;
        white-space: nowrap;
        animation: tickerMove 22s linear infinite;
    }
    .ticker-item {
        display: inline-block;
        margin-right: 2.5rem;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
    }
    .ticker-label {
        color: #c7e9ff;
        font-weight: 600;
        margin-right: 0.25rem;
    }
    .ticker-value {
        font-weight: 700;
        margin-right: 0.25rem;
    }
    .ticker-up { color: #3dff96; }
    .ticker-down { color: #ff4d6a; }
    .ticker-flat { color: #e0e0e0; }
    @keyframes tickerMove {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-50%, 0, 0); }
    }

    /* Metric strip */
    .metric-card {
        border-radius: 8px;
        border: 1px solid rgba(120, 255, 190, 0.2);
        padding: 0.5rem 0.75rem;
        background: radial-gradient(circle at top left, #04121f 0, #020410 65%, #000000 100%);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #c9ddff;
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 1rem;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.7rem;
        color: #9faad0;
    }

    .mini-card {
        border-radius: 8px;
        border: 1px solid rgba(120, 255, 190, 0.2);
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.45rem;
        background: radial-gradient(circle at top left, #071424 0, #02030b 70%);
        font-size: 0.8rem;
    }
    .mini-label {
        font-weight: 600;
        color: #c9ddff;
        margin-bottom: 0.1rem;
    }
    .mini-value {
        font-size: 0.9rem;
        font-weight: 700;
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
    """Fetch latest price + % change for an index."""
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if data.empty:
            data = yf.Ticker(ticker).history(period="5d", interval="1d")
        if data.empty:
            return None
        last = data.iloc[-1]
        close = float(last["Close"])
        info = yf.Ticker(ticker).info or {}
        if "previousClose" in info:
            prev = float(info["previousClose"])
        else:
            if len(data) >= 2:
                prev = float(data["Close"].iloc[-2])
            else:
                prev = close
        change = close - prev
        pct = (change / prev) * 100 if prev else 0.0
        return {"last": close, "change": change, "pct": pct}
    except Exception:
        return None


def format_pct(x):
    try:
        return f"{x:+.2f}%"
    except Exception:
        return "—"


def format_bps(x):
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
    sub = weights_df[(weights_df["__wave"] == wave) &
                     (weights_df["__basket"] == basket)].copy()
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
# RENDER FUNCTIONS
# -------------------------------------------------------------------

def render_header(selected_wave: str, mode: str):
    spx = fetch_index_snapshot(SPX_TICKER)
    vix = fetch_index_snapshot(VIX_TICKER)

    def tile(label, snap):
        if snap is None:
            return f"""
            <div class="index-tile">
              <div class="index-tile-label">{label}</div>
              <div class="index-tile-value index-flat">—</div>
              <div class="index-tile-change">no data</div>
            </div>
            """
        cls = "index-flat"
        if snap["pct"] > 0.05:
            cls = "index-up"
        elif snap["pct"] < -0.05:
            cls = "index-down"
        return f"""
        <div class="index-tile">
          <div class="index-tile-label">{label}</div>
          <div class="index-tile-value {cls}">{snap['last']:.2f}</div>
          <div class="index-tile-change {cls}">{format_pct(snap['pct'])}</div>
        </div>
        """

    header_html = f"""
    <div class="waves-header">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:0.5rem;">
        <div style="flex:2;">
          <div class="waves-header-title">WAVES INSTITUTIONAL CONSOLE</div>
          <div class="waves-header-subtitle">
            <span class="waves-header-pill">LIVE ENGINE</span>
            <span class="waves-header-pill">MULTI-WAVE</span>
            <span class="waves-header-pill">ADAPTIVE INDEX WAVES™</span>
            <span style="margin-left:0.3rem; opacity:0.9;">Mini Bloomberg-style terminal for WAVES Intelligence™</span>
          </div>
          <div class="index-row">
            {tile("SPX", spx)}
            {tile("VIX", vix)}
          </div>
        </div>
        <div style="flex:1;" class="waves-meta">
          <div><strong>Selected Wave:</strong> {selected_wave}</div>
          <div><strong>Mode:</strong> {mode}</div>
          <div><strong>Console Time:</strong> {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</div>
        </div>
      </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    return spx, vix


def render_ticker(spx_snap, vix_snap, wave_name, metrics):
    items = []

    def add(label, snap):
        if snap is None:
            return
        pct = snap["pct"]
        cls = "ticker-flat"
        if pct > 0.05:
            cls = "ticker-up"
        elif pct < -0.05:
            cls = "ticker-down"
        items.append(
            f'<span class="ticker-item">'
            f'<span class="ticker-label">{label}</span>'
            f'<span class="ticker-value {cls}">{snap["last"]:.2f}</span>'
            f'<span class="{cls}">{format_pct(pct)}</span>'
            f'</span>'
        )

    add("SPX", spx_snap)
    add("VIX", vix_snap)

    if metrics and metrics.get("today") is not None:
        today = metrics["today"] * 100
        cls = "ticker-flat"
        if today > 0.05:
            cls = "ticker-up"
        elif today < -0.05:
            cls = "ticker-down"
        items.append(
            f'<span class="ticker-item">'
            f'<span class="ticker-label">{wave_name}</span>'
            f'<span class="ticker-value {cls}">{format_pct(today)}</span>'
            f'<span class="{cls}">today</span>'
            f'</span>'
        )

    bar_html = '<div class="ticker-bar"><div class="ticker-track">' + " ".join(items) + "</div></div>"
    st.markdown(bar_html, unsafe_allow_html=True)


def render_metric_strip(metrics):
    c1, c2, c3, c4 = st.columns(4)
    if metrics is None:
        labels = [
            "Total Return (live)",
            "Today",
            "Max Drawdown",
            "Alpha Captured vs Benchmark",
        ]
        for col, label in zip([c1, c2, c3, c4], labels):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="metric-label">{label}</div>
                      <div class="metric-value">—</div>
                      <div class="metric-sub">waiting for first engine logs</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        return

    total = metrics.get("total_return")
    today = metrics.get("today")
    max_dd = metrics.get("max_dd")
    alpha_bps = metrics.get("alpha_bps")

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Total Return (live)</div>
              <div class="metric-value">{format_pct(total * 100) if total is not None else "—"}</div>
              <div class="metric-sub">from first log to latest</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Today</div>
              <div class="metric-value">{format_pct(today * 100) if today is not None else "—"}</div>
              <div class="metric-sub">last logged period</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Max Drawdown</div>
              <div class="metric-value">{format_pct(max_dd * 100) if max_dd is not None else "—"}</div>
              <div class="metric-sub">peak-to-trough from logs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Alpha Captured vs Benchmark</div>
              <div class="metric-value">{format_bps(alpha_bps) if alpha_bps is not None else "—"}</div>
              <div class="metric-sub">simple cumulative return differential</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_perf_curve(metrics, df_perf):
    st.markdown("### Performance Curve")
    if metrics is None or df_perf is None:
        st.info(
            "No performance log found yet for this Wave. Once the live engine writes CSVs "
            "into logs/performance/, the performance curve will plot here."
        )
        return
    curve = metrics["curve"]
    st.line_chart(pd.DataFrame({"Performance": curve}))


def render_exposure_cards(exposure_pct: int, mode: str):
    st.markdown("### Exposure & Risk")
    cash_pct = 100 - exposure_pct

    st.markdown(
        f"""
        <div class="mini-card">
          <div class="mini-label">Equity Exposure</div>
          <div class="mini-value">{exposure_pct} %</div>
          <div class="metric-sub">Target β ≈ {BETA_TARGET:.2f} in Standard mode</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="mini-card">
          <div class="mini-label">Cash Buffer</div>
          <div class="mini-value">{cash_pct} %</div>
          <div class="metric-sub">dynamic SmartSafe™ allocation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="mini-card">
          <div class="mini-label">Risk Mode</div>
          <div class="mini-value">{mode}</div>
          <div class="metric-sub">mode-aware engine parameters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_basket_table(title: str, df_basket: pd.DataFrame):
    st.markdown(f"#### {title}")
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
    st.markdown("### Top 10 Positions — Google Finance Links")
    combined = pd.concat([df_primary, df_secondary], ignore_index=True)
    if combined.empty:
        st.info("No holdings defined for this Wave in wave_weights.csv.")
        return
    combined = combined.sort_values("Weight %", ascending=False).head(10).copy()
    combined["Ticker"] = combined["__ticker"].apply(google_link)
    combined["Weight %"] = combined["Weight %"].round(2)
    st.table(combined[["Ticker", "Weight %"]])


def render_alpha_dashboard(df_perf):
    st.markdown("### Alpha Dashboard")
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
    st.markdown("### Engine Logs & Discovery")
    st.markdown("#### Discovered Waves (from wave_weights.csv)")
    if not wave_list:
        st.warning("No Waves discovered. Check the Wave column in wave_weights.csv.")
    else:
        st.write(", ".join(wave_list))

    st.markdown("#### Performance Logs (logs/performance/)")
    if os.path.isdir(LOGS_PERF_DIR):
        files = sorted(glob.glob(os.path.join(LOGS_PERF_DIR, "*.csv")))
        st.write("\n".join(os.path.basename(f) for f in files) if files else "_(none yet)_")
    else:
        st.write("_folder not found_")

    st.markdown("#### Base Paths")
    st.code(
        f"BASE_DIR      = {BASE_DIR}\n"
        f"LOGS_PERF_DIR = {LOGS_PERF_DIR}\n"
        f"WAVE_WEIGHTS  = {WAVE_WEIGHTS_FILE}"
    )

# -------------------------------------------------------------------
# MAIN APP FLOW
# -------------------------------------------------------------------

weights_df = load_wave_weights()
wave_list = discover_waves(weights_df)

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Engine Controls")
    if wave_list:
        st.success("Using Wave universe from wave_weights.csv.")
    else:
        st.error("No Waves discovered in wave_weights.csv.")
        st.stop()

    selected_wave = st.selectbox("Select Wave", options=wave_list, index=0)

    st.markdown("#### Mode")
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

    st.markdown(
        f"**Target β ≈ {BETA_TARGET:.2f}** · Cash buffer: **{100 - exposure}%**",
    )

# Header + ticker strip
spx_snap, vix_snap = render_header(selected_wave, mode)

perf_df, perf_file = load_performance_df(selected_wave)
metrics = compute_metrics(perf_df)

render_ticker(spx_snap, vix_snap, selected_wave, metrics)

st.markdown("#### WAVES Engine Dashboard")
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

    col_p, col_s = st.columns(2)
    primary_df = get_basket(weights_df, selected_wave, "Primary")
    secondary_df = get_basket(weights_df, selected_wave, "Secondary")

    with col_p:
        render_basket_table("Primary Basket", primary_df)
    with col_s:
        render_basket_table("Secondary Basket", secondary_df)

    st.markdown("---")
    render_top10(primary_df, secondary_df)

with tab_alpha:
    render_alpha_dashboard(perf_df)

with tab_logs:
    render_engine_logs_tab(wave_list)
