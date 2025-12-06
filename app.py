# app.py
# WAVES Intelligence™ — Institutional Console
# Mini Bloomberg-style terminal for Adaptive Index Waves.

from pathlib import Path
from datetime import datetime, timedelta
import glob
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None  # Cloud/local env might not have it; we fail gracefully.

# ----------------------------------------------------
# PATHS / CONSTANTS
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
PERF_DIR = LOGS_DIR / "performance"
POS_DIR = LOGS_DIR / "positions"
WEIGHTS_PATH = BASE_DIR / "wave_weights.csv"

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

WAVE_DISPLAY_NAMES: Dict[str, str] = {
    "SP500_Wave": "S&P 500 Wave",
    "Growth_Wave": "Growth Wave",
    "Income_Wave": "Income Wave",
    "AL_Wave": "AI Leaders Wave",
}

# ----------------------------------------------------
# HELPERS: DISCOVER WAVES
# ----------------------------------------------------
def discover_waves_from_logs() -> List[str]:
    """Find Waves from files like <Wave>_performance_YYYYMMDD.csv."""
    if not PERF_DIR.exists():
        return []

    waves = set()
    for path in PERF_DIR.glob("*_performance_*.csv"):
        name = path.name
        if "_performance_" in name:
            wave_name = name.split("_performance_")[0]
            if wave_name:
                waves.add(wave_name)
    return sorted(waves)


def discover_waves_from_weights() -> List[str]:
    """Fallback: read unique Wave names from wave_weights.csv."""
    if not WEIGHTS_PATH.exists():
        return []

    try:
        df = pd.read_csv(WEIGHTS_PATH)
    except Exception:
        return []

    # Try common wave name columns
    for col in ["Wave", "Wave_Name", "wave", "wave_name"]:
        if col in df.columns:
            return sorted(df[col].dropna().unique().tolist())

    # Otherwise just use first column
    if len(df.columns) > 0:
        return sorted(df.iloc[:, 0].dropna().unique().tolist())

    return []


def discover_waves() -> (List[str], str):
    """Return (waves, source) where source is 'logs', 'weights', or 'none'."""
    from_logs = discover_waves_from_logs()
    if from_logs:
        return from_logs, "logs"

    from_weights = discover_waves_from_weights()
    if from_weights:
        return from_weights, "weights"

    return [], "none"


# ----------------------------------------------------
# HELPERS: LOAD DATA
# ----------------------------------------------------
def load_performance(wave_name: str) -> Optional[pd.DataFrame]:
    """Load latest performance CSV for a Wave."""
    if not PERF_DIR.exists():
        return None

    pattern = str(PERF_DIR / f"{wave_name}_performance_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None

    latest_path = max(matches, key=lambda p: Path(p).stat().st_mtime)
    try:
        df = pd.read_csv(latest_path)
    except Exception:
        return None

    # Normalize date
    for col in ["date", "Date", "timestamp", "Timestamp"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col)
            except Exception:
                pass
            break

    return df


def load_positions(wave_name: str) -> Optional[pd.DataFrame]:
    """Load latest positions CSV for a Wave."""
    if not POS_DIR.exists():
        return None

    pattern = str(POS_DIR / f"{wave_name}_positions_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None

    latest_path = max(matches, key=lambda p: Path(p).stat().st_mtime)
    try:
        df = pd.read_csv(latest_path)
    except Exception:
        return None

    return df


# ----------------------------------------------------
# HELPERS: MARKET DATA (SPX / VIX)
# ----------------------------------------------------
def fetch_index_timeseries(ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
    """Download recent daily close series for ticker."""
    if yf is None:
        return None
    try:
        data = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    except Exception:
        return None
    if data is None or data.empty:
        return None
    out = data[["Close"]].rename(columns={"Close": "close"})
    out.index.name = "date"
    return out


def fetch_index_snapshot(ticker: str) -> Optional[Dict[str, float]]:
    """Latest price + daily change %."""
    if yf is None:
        return None
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
    except Exception:
        return None
    if data is None or data.empty:
        return None

    latest = float(data.iloc[-1]["Close"])
    if len(data) > 1:
        prev = float(data.iloc[-2]["Close"])
    else:
        prev = latest
    change = latest - prev
    pct = change / prev if prev != 0 else 0.0
    return {"last": latest, "change": change, "pct": pct}


# ----------------------------------------------------
# HELPERS: METRICS
# ----------------------------------------------------
def compute_summary_metrics(perf_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    metrics = {
        "total_return": None,
        "today_return": None,
        "max_drawdown": None,
        "alpha_captured": None,
    }
    if perf_df is None or perf_df.empty:
        return metrics

    # Try to find return and benchmark columns
    ret_cols = [c for c in perf_df.columns if "return" in c.lower() and "benchmark" not in c.lower()]
    bench_cols = [c for c in perf_df.columns if "benchmark" in c.lower()]
    price_cols = [c for c in perf_df.columns if "nav" in c.lower() or "value" in c.lower()]

    if ret_cols:
        r = perf_df[ret_cols[0]].astype(float)
    elif price_cols:
        p = perf_df[price_cols[0]].astype(float)
        r = p.pct_change().fillna(0.0)
    else:
        return metrics

    # total
    metrics["total_return"] = (1 + r).prod() - 1
    # today
    metrics["today_return"] = float(r.iloc[-1])

    # max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    metrics["max_drawdown"] = float(dd.min())

    # alpha captured vs benchmark if available
    if bench_cols:
        br = perf_df[bench_cols[0]].astype(float)
        if br.abs().max() > 5:  # assume in %
            br = br / 100.0
        alpha = r - br
        metrics["alpha_captured"] = float(alpha.mean())

    return metrics


def fmt_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


def sign_class(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "neutral"
    return "pos" if x > 0 else ("neg" if x < 0 else "neutral")


def wave_display_name(w: str) -> str:
    return WAVE_DISPLAY_NAMES.get(w, w)


# ----------------------------------------------------
# STREAMLIT CONFIG & GLOBAL CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="WAVES Institutional Console",
    page_icon="💹",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left, #0b1120 0%, #020617 45%, #000000 100%);
        color: #f9fafb;
    }
    header[data-testid="stHeader"] {display: none;}

    /* Top bar */
    .waves-topbar {
        background: linear-gradient(90deg, #020617 0%, #020617 40%, #022c22 100%);
        border-bottom: 1px solid #111827;
        padding: 0.6rem 1.5rem 0.4rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .waves-topbar-left {
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
    }
    .waves-title {
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #e5e7eb;
    }
    .waves-badge {
        font-size: 0.70rem;
        padding: 0.12rem 0.45rem;
        border-radius: 0.25rem;
        background: #020617;
        border: 1px solid #1f2937;
        color: #9ca3af;
    }
    .waves-subtitle {
        font-size: 0.75rem;
        color: #9ca3af;
    }
    .waves-topbar-right {
        display: flex;
        gap: 0.75rem;
        align-items: center;
    }
    .index-tile {
        padding: 0.35rem 0.6rem;
        border-radius: 0.35rem;
        background: #020617;
        border: 1px solid #1f2937;
        font-size: 0.75rem;
        min-width: 135px;
    }
    .index-label { color: #9ca3af; font-size: 0.70rem; }
    .index-value { font-weight: 600; }
    .index-pct.pos { color: #4ade80; }
    .index-pct.neg { color: #f97373; }
    .index-pct.neutral { color: #e5e7eb; }

    /* Ticker tape */
    .ticker-tape {
        background: #020617;
        border-bottom: 1px solid #111827;
        padding: 0.25rem 1.5rem;
        font-size: 0.75rem;
        white-space: nowrap;
        overflow: hidden;
    }
    .ticker-inner {
        display: inline-block;
        animation: ticker-move 25s linear infinite;
    }
    .ticker-item {
        display: inline-block;
        margin-right: 1.75rem;
    }
    .ticker-symbol {
        font-weight: 600;
        margin-right: 0.35rem;
    }
    .ticker-pct.pos {
        color: #22c55e;
        animation: blink-pos 1.5s infinite;
    }
    .ticker-pct.neg {
        color: #f97373;
        animation: blink-neg 1.5s infinite;
    }
    .ticker-pct.neutral {
        color: #e5e7eb;
    }
    @keyframes ticker-move {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    @keyframes blink-pos {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    @keyframes blink-neg {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Metric strip */
    .metric-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.9rem 0 0.25rem 0;
    }
    .metric-card {
        background: #020617;
        border-radius: 0.6rem;
        padding: 0.65rem 0.8rem;
        border: 1px solid #111827;
        font-size: 0.8rem;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.72rem;
    }
    .metric-value {
        margin-top: 0.15rem;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .metric-value.pos { color: #4ade80; }
    .metric-value.neg { color: #f97373; }
    .metric-value.neutral { color: #e5e7eb; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label {
        color: #e5e7eb !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# TOP BAR: TITLE + INDEX SNAPSHOTS
# ----------------------------------------------------
spx_snap = fetch_index_snapshot(SPX_TICKER)
vix_snap = fetch_index_snapshot(VIX_TICKER)

def index_tile(label: str, snap: Optional[Dict[str, float]], flip_vix: bool = False) -> str:
    if not snap:
        return f"""
        <div class="index-tile">
          <div class="index-label">{label}</div>
          <div class="index-value">—</div>
        </div>
        """
    pct = snap["pct"]
    if flip_vix and pct is not None and not pd.isna(pct):
        pct = -pct  # lower VIX is "green"
    cls = sign_class(pct)
    return f"""
    <div class="index-tile">
      <div class="index-label">{label}</div>
      <div class="index-value">{snap['last']:.2f}</div>
      <div class="index-pct {cls}">{fmt_pct(pct)}</div>
    </div>
    """

topbar_html = f"""
<div class="waves-topbar">
  <div class="waves-topbar-left">
    <div class="waves-title">WAVES INSTITUTIONAL CONSOLE</div>
    <div class="waves-badge">LIVE ENGINE · MULTI-WAVE</div>
    <div class="waves-subtitle">Adaptive Index Waves · Mini Bloomberg-Style Terminal</div>
  </div>
  <div class="waves-topbar-right">
    {index_tile("S&P 500", spx_snap)}
    {index_tile("VIX (Risk Pulse)", vix_snap, flip_vix=True)}
    <div class="waves-subtitle">{datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC</div>
  </div>
</div>
"""
st.markdown(topbar_html, unsafe_allow_html=True)

# ----------------------------------------------------
# DISCOVER WAVES & SIDEBAR CONTROLS
# ----------------------------------------------------
st.sidebar.header("⚙️ Engine Controls")

waves, source = discover_waves()
if not waves:
    st.sidebar.error(
        "No Waves discovered in `logs/performance` or `wave_weights.csv`.\n\n"
        "• To see live data, run `waves_engine.py` to generate logs, or\n"
        "• Ensure `wave_weights.csv` is present in this folder."
    )
    st.stop()

# default wave preference
default_wave = None
for candidate in ["Growth_Wave", "SP500_Wave"]:
    if candidate in waves:
        default_wave = candidate
        break
if default_wave is None:
    default_wave = waves[0]

if source == "logs":
    st.sidebar.success("Waves discovered from engine logs ✅")
elif source == "weights":
    st.sidebar.warning("Using Wave list from wave_weights.csv (no performance logs found yet).")

selected_wave = st.sidebar.selectbox(
    "Select Wave",
    waves,
    index=waves.index(default_wave),
    format_func=wave_display_name,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Mode")
st.sidebar.radio("Risk Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic™"], index=0)
equity_exposure = st.sidebar.slider("Equity Exposure", 0, 100, 90, step=5)
st.sidebar.caption(f"Target β ≈ 0.90 · Cash buffer: {100 - equity_exposure}%")

# ----------------------------------------------------
# LOAD DATA FOR SELECTED WAVE
# ----------------------------------------------------
perf_df = load_performance(selected_wave)
pos_df = load_positions(selected_wave)
metrics = compute_summary_metrics(perf_df)

# ----------------------------------------------------
# TICKER TAPE
# ----------------------------------------------------
tape_items = []

if spx_snap:
    tape_items.append(
        f'<span class="ticker-item">'
        f'<span class="ticker-symbol">SPX</span>'
        f'<span class="ticker-pct {sign_class(spx_snap["pct"])}">{fmt_pct(spx_snap["pct"])}</span>'
        f'</span>'
    )

if vix_snap:
    vix_pct = vix_snap["pct"]
    if vix_pct is not None and not pd.isna(vix_pct):
        vix_pct = -vix_pct  # inverse risk
    tape_items.append(
        f'<span class="ticker-item">'
        f'<span class="ticker-symbol">VIX</span>'
        f'<span class="ticker-pct {sign_class(vix_pct)}">{fmt_pct(vix_pct)}</span>'
        f'</span>'
    )

tape_items.append(
    f'<span class="ticker-item">'
    f'<span class="ticker-symbol">{wave_display_name(selected_wave)}</span>'
    f'<span class="ticker-pct {sign_class(metrics["today_return"])}">{fmt_pct(metrics["today_return"])}</span>'
    f'</span>'
)

tape_html = f"""
<div class="ticker-tape">
  <div class="ticker-inner">
    {' '.join(tape_items)} {' '.join(tape_items)}
  </div>
</div>
"""
st.markdown(tape_html, unsafe_allow_html=True)

# ----------------------------------------------------
# MAIN HEADER + METRIC STRIP
# ----------------------------------------------------
st.markdown("### WAVES Engine Dashboard")
st.caption("Live / demo console for WAVES Intelligence™ — Adaptive Index Waves")

metric_html = f"""
<div class="metric-strip">
  <div class="metric-card">
    <div class="metric-label">Total Return (since inception)</div>
    <div class="metric-value {sign_class(metrics['total_return'])}">{fmt_pct(metrics['total_return'])}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Today</div>
    <div class="metric-value {sign_class(metrics['today_return'])}">{fmt_pct(metrics['today_return'])}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Max Drawdown</div>
    <div class="metric-value neg">{fmt_pct(metrics['max_drawdown'])}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Alpha Captured vs Benchmark</div>
    <div class="metric-value {sign_class(metrics['alpha_captured'])}">{fmt_pct(metrics['alpha_captured'])}</div>
  </div>
</div>
"""
st.markdown(metric_html, unsafe_allow_html=True)

if perf_df is None:
    st.info(
        f"No performance log found yet for `{selected_wave}` in `logs/performance`.\n\n"
        "The dashboard is running in **structure/demo mode**. "
        "Run `waves_engine.py` locally and upload the resulting CSVs to see full live charts."
    )

# ----------------------------------------------------
# TABS
# ----------------------------------------------------
tab_overview, tab_alpha, tab_logs = st.tabs(["Overview", "Alpha Dashboard", "Engine Logs"])

# ----------------------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------------------
with tab_overview:
    left_col, right_col = st.columns([2.2, 1.3])

    # Performance curve
    with left_col:
        st.subheader("Performance Curve")
        if perf_df is not None and not perf_df.empty:
            date_cols = [c for c in perf_df.columns if "date" in c.lower() or "time" in c.lower()]
            nav_cols = [c for c in perf_df.columns if "nav" in c.lower() or "value" in c.lower()]
            if date_cols and nav_cols:
                df_plot = perf_df[[date_cols[0], nav_cols[0]]].copy()
                df_plot = df_plot.rename(columns={date_cols[0]: "date", nav_cols[0]: "NAV"})
                df_plot = df_plot.set_index("date")
                st.line_chart(df_plot)
            else:
                st.info("Could not identify NAV/date columns — showing raw performance table.")
                st.dataframe(perf_df.tail(50), use_container_width=True)
        else:
            st.caption("No performance data yet — waiting for first engine logs.")

        st.markdown("#### Top 10 Positions — Google Finance Links")
        if pos_df is not None and not pos_df.empty:
            top10 = pos_df.head(10).copy()
            ticker_col = None
            for c in ["Ticker", "ticker", "Symbol", "symbol"]:
                if c in top10.columns:
                    ticker_col = c
                    break
            if ticker_col:
                lines = []
                for _, row in top10.iterrows():
                    t = str(row[ticker_col]).strip()
                    if not t:
                        continue
                    url = f"https://www.google.com/finance/quote/{t}:NASDAQ"
                    weight_text = ""
                    for w_col in ["Weight", "weight", "PortfolioWeight", "portfolio_weight"]:
                        if w_col in top10.columns and not pd.isna(row[w_col]):
                            try:
                                w = float(row[w_col])
                                if w > 1.5:
                                    w = w / 100.0
                                weight_text = f" · {w * 100:0.2f}%"
                            except Exception:
                                pass
                            break
                    lines.append(f"- [{t}]({url}){weight_text}")
                st.markdown("\n".join(lines))
            else:
                st.caption("Ticker/Symbol column not found — cannot build Google links.")
        else:
            st.caption("No positions file yet for this Wave in `logs/positions`.")

    # Right column: risk + SPX/VIX charts
    with right_col:
        st.subheader("Exposure & Risk")
        st.metric("Equity Exposure", f"{equity_exposure} %")
        st.metric("Cash Buffer", f"{100 - equity_exposure} %")
        st.metric("Target β", "0.90")

        st.markdown("#### S&P 500 & VIX")
        if yf is None:
            st.caption("`yfinance` not installed — market charts unavailable in this environment.")
        else:
            spx_hist = fetch_index_timeseries(SPX_TICKER, days=120)
            vix_hist = fetch_index_timeseries(VIX_TICKER, days=120)
            if spx_hist is not None:
                st.line_chart(spx_hist["close"], height=150)
            if vix_hist is not None:
                st.line_chart(vix_hist["close"], height=120)

# ----------------------------------------------------
# TAB 2: ALPHA DASHBOARD
# ----------------------------------------------------
with tab_alpha:
    st.subheader("Alpha Dashboard")
    if perf_df is None or perf_df.empty:
        st.info("Alpha metrics will appear once performance logs are available for this Wave.")
    else:
        bench_cols = [c for c in perf_df.columns if "benchmark" in c.lower()]
        ret_cols = [c for c in perf_df.columns if "return" in c.lower() and "benchmark" not in c.lower()]
        if bench_cols and ret_cols:
            r = perf_df[ret_cols[0]].astype(float)
            br = perf_df[bench_cols[0]].astype(float)
            if br.abs().max() > 5:
                br = br / 100.0
            alpha = r - br
            alpha_30 = alpha.rolling(30).mean()
            st.line_chart(alpha_30, height=260)
            st.caption("Rolling 30-day average alpha vs benchmark.")
        else:
            st.caption("Benchmark / return columns not found — alpha view is currently in placeholder mode.")

# ----------------------------------------------------
# TAB 3: ENGINE LOGS / DEBUG
# ----------------------------------------------------
with tab_logs:
    st.subheader("Engine Logs / Discovery Debug")
    st.write("Base directory:", str(BASE_DIR))
    st.write("Logs directory:", str(LOGS_DIR))

    st.markdown("**Discovered Waves**")
    st.json({"waves": waves, "source": source})

    st.markdown("**Performance files**")
    if PERF_DIR.exists():
        perf_files = [p.name for p in PERF_DIR.glob('*.csv')]
        st.write(perf_files if perf_files else "No CSV files in `logs/performance`.")
    else:
        st.write("`logs/performance` directory does not exist.")

    st.markdown("**Position files**")
    if POS_DIR.exists():
        pos_files = [p.name for p in POS_DIR.glob('*.csv')]
        st.write(pos_files if pos_files else "No CSV files in `logs/positions`.")
    else:
        st.write("`logs/positions` directory does not exist.")

    st.caption(
        "All live data is sourced from WAVES Engine logs when present. "
        "If you see no files above, run `waves_engine.py` to generate them."
    )
