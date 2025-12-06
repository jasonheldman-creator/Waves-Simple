# app.py
# WAVES INSTITUTIONAL CONSOLE — LIVE ENGINE · MULTI-WAVE

import os
import glob
from datetime import datetime, timedelta

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
LIST_FILE = os.path.join(BASE_DIR, "list.csv")

SPX_TICKER = "^GSPC"   # S&P 500 index
VIX_TICKER = "^VIX"    # VIX index

BETA_TARGET = 0.90      # Target beta for Standard mode
DEFAULT_EXPOSURE = 90   # Default equity exposure %


# -------------------------------------------------------------------
# STREAMLIT PAGE CONFIG & GLOBAL STYLE
# -------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Institutional Console",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global CSS for dark institutional look, header, tiles & ticker tape
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

    /* Header shell */
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

    .index-up {
        color: #4bff9a;
    }

    .index-down {
        color: #ff5d73;
    }

    .index-tile-change {
        font-size: 0.7rem;
        opacity: 0.9;
    }

    .waves-meta {
        text-align: right;
        font-size: 0.75rem;
        color: #c0d8ff;
    }

    .waves-meta strong {
        color: #ffffff;
    }

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

    .ticker-up {
        color: #3dff96;
    }

    .ticker-down {
        color: #ff4d6a;
    }

    .ticker-flat {
        color: #e0e0e0;
    }

    @keyframes tickerMove {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-50%, 0, 0); }
    }

    /* Metric strip card */
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

    /* Exposure/Risk cards */
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

    /* Streamlit tweaks */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.25rem 0.6rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------

@st.cache_data(ttl=60)
def fetch_index_snapshot(ticker: str):
    try:
        data = yf.Ticker(ticker).history(period="1d", interval="1m")
        if data.empty:
            data = yf.Ticker(ticker).history(period="5d", interval="1d")
        if data.empty:
            return None

        last = data.iloc[-1]
        close = float(last["Close"])

        # Use previous close as reference if available
        if "Previous Close" in yf.Ticker(ticker).info:
            prev = float(yf.Ticker(ticker).info["Previous Close"])
        else:
            if len(data) >= 2:
                prev = float(data["Close"].iloc[-2])
            else:
                prev = close

        change = close - prev
        pct = (change / prev) * 100 if prev != 0 else 0.0

        return {
            "last": close,
            "change": change,
            "pct": pct,
            "series": data["Close"],
        }
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


def discover_waves_from_logs():
    waves = set()
    if not os.path.isdir(LOGS_PERF_DIR):
        return []

    pattern = os.path.join(LOGS_PERF_DIR, "*_performance_*.csv")
    files = glob.glob(pattern)
    for f in files:
        base = os.path.basename(f)
        if "_performance_" in base:
            wave = base.split("_performance_")[0]
            if wave:
                waves.add(wave)
    return sorted(list(waves))


def discover_waves_from_weights():
    if not os.path.exists(WAVE_WEIGHTS_FILE):
        return []
    try:
        df = pd.read_csv(WAVE_WEIGHTS_FILE)
    except Exception:
        return []

    for candidate in ["Wave", "wave", "WaveName", "wave_name"]:
        if candidate in df.columns:
            waves = sorted(df[candidate].dropna().unique().tolist())
            return waves
    return []


def get_wave_universe_source():
    """Return (waves, source_label)."""
    waves_logs = discover_waves_from_logs()
    if waves_logs:
        return waves_logs, "performance logs"
    waves_weights = discover_waves_from_weights()
    if waves_weights:
        return waves_weights, "wave_weights.csv"
    return [], "none"


def get_latest_performance_file(wave_name: str):
    pattern = os.path.join(LOGS_PERF_DIR, f"{wave_name}_performance_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort()
    return files[-1]


def get_latest_positions_file(wave_name: str):
    pattern = os.path.join(LOGS_POS_DIR, f"{wave_name}_positions_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort()
    return files[-1]


def load_performance_df(wave_name: str):
    latest = get_latest_performance_file(wave_name)
    if latest is None:
        return None, None
    try:
        df = pd.read_csv(latest)
    except Exception as e:
        st.error(f"Could not read performance file for {wave_name}: {e}")
        return None, latest

    # Detect datetime column
    date_col = None
    for c in ["timestamp", "Timestamp", "datetime", "Datetime", "date", "Date"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(by=date_col)
        df = df.set_index(date_col)

    return df, latest


def compute_wave_metrics(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    # Try to infer key columns
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

    # “Today” = last period return if we have it
    today_ret = None
    if ret_col is not None:
        today_ret = float(df[ret_col].iloc[-1])

    # Max drawdown
    running_max = curve.cummax()
    drawdown = curve / running_max - 1.0
    max_dd = float(drawdown.min())

    # Alpha vs benchmark (simple: diff of cumulative return)
    alpha_bps = None
    if bench_ret_col is not None and ret_col is not None:
        r = df[ret_col].astype(float).fillna(0.0)
        rb = df[bench_ret_col].astype(float).fillna(0.0)
        my_curve = (1 + r).cumprod()
        bench_curve = (1 + rb).cumprod()
        alpha = my_curve.iloc[-1] - bench_curve.iloc[-1]
        alpha_bps = float((alpha - 1.0) * 10_000)  # difference in bps vs bench

    return {
        "curve": curve,
        "total_return": float(total_return),
        "today": today_ret,
        "max_dd": max_dd,
        "alpha_bps": alpha_bps,
    }


# -------------------------------------------------------------------
# RENDER HELPERS
# -------------------------------------------------------------------

def render_header(selected_wave: str, selected_mode: str):
    spx = fetch_index_snapshot(SPX_TICKER)
    vix = fetch_index_snapshot(VIX_TICKER)

    def index_tile(label, snap):
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
                <div class="waves-header-title">
                    WAVES INSTITUTIONAL CONSOLE
                </div>
                <div class="waves-header-subtitle">
                    <span class="waves-header-pill">LIVE ENGINE</span>
                    <span class="waves-header-pill">MULTI-WAVE</span>
                    <span class="waves-header-pill">ADAPTIVE INDEX WAVES™</span>
                    <span style="margin-left:0.3rem; opacity:0.9;">Mini Bloomberg-style terminal for WAVES Intelligence™</span>
                </div>
                <div class="index-row">
                    {index_tile("SPX", spx)}
                    {index_tile("VIX", vix)}
                </div>
            </div>
            <div style="flex:1;" class="waves-meta">
                <div><strong>Selected Wave:</strong> {selected_wave if selected_wave else "—"}</div>
                <div><strong>Mode:</strong> {selected_mode}</div>
                <div><strong>Console Time:</strong> {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</div>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    return spx, vix


def render_ticker(spx_snap, vix_snap, wave_name, wave_metrics):
    items = []

    def add_item(label, snap):
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
            f"</span>"
        )

    add_item("SPX", spx_snap)
    add_item("VIX", vix_snap)

    if wave_name and wave_metrics is not None and wave_metrics.get("today") is not None:
        today = wave_metrics["today"] * 100
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
            f"</span>"
        )

    ticker_html = """
    <div class="ticker-bar">
        <div class="ticker-track">
            """ + " ".join(items) + """
        </div>
    </div>
    """
    st.markdown(ticker_html, unsafe_allow_html=True)


def render_metric_strip(wave_metrics):
    col1, col2, col3, col4 = st.columns(4)

    if wave_metrics is None:
        for col, label in zip(
            [col1, col2, col3, col4],
            ["Total Return (live)", "Today", "Max Drawdown", "Alpha Captured vs Benchmark"],
        ):
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

    total = wave_metrics.get("total_return")
    today = wave_metrics.get("today")
    max_dd = wave_metrics.get("max_dd")
    alpha_bps = wave_metrics.get("alpha_bps")

    with col1:
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
    with col2:
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
    with col3:
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
    with col4:
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


def render_performance_curve(wave_name, wave_metrics, df_perf):
    st.markdown("### Performance Curve")

    if wave_metrics is None or df_perf is None:
        st.info(
            "No performance log found yet for this Wave. "
            "Once the live engine writes CSVs into `logs/performance/`, "
            "the performance curve will plot here."
        )
        return

    curve = wave_metrics["curve"]
    perf_df = pd.DataFrame({"Performance": curve})
    st.line_chart(perf_df)


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


def render_top_positions(selected_wave: str, top_n: int = 10):
    latest_file = get_latest_positions_file(selected_wave)
    if latest_file is None:
        st.markdown("### Top 10 Positions — Google Finance Links")
        st.info(
            f"No positions log found yet for **{selected_wave}** in `logs/positions/`.\n\n"
            "Once the engine writes positions CSVs, this panel will show live top holdings."
        )
        return

    try:
        df = pd.read_csv(latest_file)
    except Exception as e:
        st.markdown("### Top 10 Positions — Google Finance Links")
        st.error(f"Could not read positions file for {selected_wave}: {e}")
        return

    # Detect columns
    ticker_col = None
    for c in ["ticker", "Ticker", "symbol", "Symbol"]:
        if c in df.columns:
            ticker_col = c
            break

    if ticker_col is None:
        st.markdown("### Top 10 Positions — Google Finance Links")
        st.warning("No ticker/symbol column found in positions file.")
        st.dataframe(df.head(20))
        return

    weight_col = None
    for c in ["weight", "Weight", "target_weight", "TargetWeight", "position_weight", "PositionWeight"]:
        if c in df.columns:
            weight_col = c
            break

    value_col = None
    for c in ["value", "Value", "market_value", "MarketValue"]:
        if c in df.columns:
            value_col = c
            break

    # Sort by value or weight if available
    df_sorted = df.copy()
    if value_col:
        df_sorted = df_sorted.sort_values(by=value_col, ascending=False)
    elif weight_col:
        df_sorted = df_sorted.sort_values(by=weight_col, ascending=False)
    else:
        df_sorted = df_sorted.sort_values(by=ticker_col)

    top = df_sorted.head(top_n).copy()

    def google_link(t):
        t = str(t).strip().upper()
        return f"[{t}](https://www.google.com/finance/quote/{t}:NASDAQ)"

    top["Ticker"] = top[ticker_col].apply(google_link)

    display_cols = ["Ticker"]

    if weight_col:
        top["Weight %"] = (top[weight_col].astype(float) * 100).round(2)
        display_cols.append("Weight %")

    if value_col:
        top["Value"] = top[value_col]
        display_cols.append("Value")

    st.markdown("### Top 10 Positions — Google Finance Links")
    st.caption(f"From latest engine positions log: `{os.path.basename(latest_file)}`")
    st.table(top[display_cols])


def render_alpha_dashboard(df_perf):
    st.markdown("### Alpha Dashboard")

    if df_perf is None or df_perf.empty:
        st.info(
            "No performance log with benchmark columns detected yet. "
            "When the engine logs `return` and `benchmark_return`, "
            "a rolling alpha chart will appear here."
        )
        return

    # Detect columns
    ret_col = None
    for c in ["return", "Return", "daily_return", "strategy_return"]:
        if c in df_perf.columns:
            ret_col = c
            break

    bench_ret_col = None
    for c in ["benchmark_return", "bench_return", "BenchmarkReturn"]:
        if c in df_perf.columns:
            bench_ret_col = c
            break

    if ret_col is None or bench_ret_col is None:
        st.info(
            "This performance file does not contain both strategy and benchmark return columns. "
            "Rolling alpha requires `return` and `benchmark_return` (or close variants)."
        )
        st.dataframe(df_perf.head(20))
        return

    r = df_perf[ret_col].astype(float).fillna(0.0)
    rb = df_perf[bench_ret_col].astype(float).fillna(0.0)
    alpha_daily = r - rb
    alpha_30d = alpha_daily.rolling(window=30).sum()

    alpha_df = pd.DataFrame({"30-day Rolling Alpha": alpha_30d * 100})
    st.line_chart(alpha_df)
    st.caption("Rolling 30-day alpha, in percentage points vs benchmark.")


def render_engine_logs_tab(waves, source_label):
    st.markdown("### Engine Logs & Discovery")

    st.markdown("#### Discovered Waves")
    if not waves:
        st.warning(
            "No Waves discovered yet. "
            "Create performance logs in `logs/performance/` or provide `wave_weights.csv`."
        )
    else:
        st.write(", ".join(waves))

    st.markdown("#### Source of Wave List")
    st.info(f"Wave universe discovered from **{source_label}**.")

    st.markdown("#### Performance Logs (logs/performance/)")
    if os.path.isdir(LOGS_PERF_DIR):
        files = sorted(glob.glob(os.path.join(LOGS_PERF_DIR, "*.csv")))
        if files:
            st.write("\n".join(os.path.basename(f) for f in files))
        else:
            st.write("_(none yet)_")
    else:
        st.write("_folder not found_")

    st.markdown("#### Positions Logs (logs/positions/)")
    if os.path.isdir(LOGS_POS_DIR):
        files = sorted(glob.glob(os.path.join(LOGS_POS_DIR, "*.csv")))
        if files:
            st.write("\n".join(os.path.basename(f) for f in files))
        else:
            st.write("_(none yet)_")
    else:
        st.write("_folder not found_")

    st.markdown("#### Base Paths")
    st.code(
        f"BASE_DIR        = {BASE_DIR}\n"
        f"LOGS_PERF_DIR   = {LOGS_PERF_DIR}\n"
        f"LOGS_POS_DIR    = {LOGS_POS_DIR}\n"
        f"WAVE_WEIGHTS    = {WAVE_WEIGHTS_FILE}\n"
        f"LIST_FILE       = {LIST_FILE}"
    )


# -------------------------------------------------------------------
# SIDEBAR / ENGINE CONTROLS
# -------------------------------------------------------------------

waves, source_label = get_wave_universe_source()

with st.sidebar:
    st.markdown("### ⚙️ Engine Controls")

    if source_label == "performance logs":
        st.success("Using Wave list discovered from `logs/performance/`.")
    elif source_label == "wave_weights.csv":
        st.warning("Using Wave list from `wave_weights.csv` (no performance logs found yet).")
    else:
        st.error("No Wave universe detected. Please add logs or `wave_weights.csv`.")
        st.stop()

    selected_wave = st.selectbox("Select Wave", options=waves, index=0 if waves else None)

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
        help="Target equity allocation; remaining % held in SmartSafe™ / cash."
    )

    st.markdown(
        f"**Target β ≈ {BETA_TARGET:.2f}** · Cash buffer: **{100 - exposure}%**",
    )


# -------------------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------------------

spx_snap, vix_snap = render_header(selected_wave, mode)

# Load performance df + metrics for selected wave
perf_df, perf_file = load_performance_df(selected_wave)
wave_metrics = compute_wave_metrics(perf_df)

render_ticker(spx_snap, vix_snap, selected_wave, wave_metrics)

st.markdown("#### WAVES Engine Dashboard")
st.caption("Live / demo console for WAVES Intelligence™ — Adaptive Index Waves™")

render_metric_strip(wave_metrics)

st.markdown("---")

tab_overview, tab_alpha, tab_logs = st.tabs(["Overview", "Alpha Dashboard", "Engine Logs"])

with tab_overview:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        render_performance_curve(selected_wave, wave_metrics, perf_df)

    with col_right:
        render_exposure_cards(exposure, mode)

    st.markdown("---")
    render_top_positions(selected_wave)

with tab_alpha:
    render_alpha_dashboard(perf_df)

with tab_logs:
    render_engine_logs_tab(waves, source_label)
