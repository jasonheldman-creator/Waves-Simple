# app.py
# WAVES Intelligence™ – Institutional Console
# Works both with real engine logs (logs/performance, logs/positions)
# and in demo/structure mode using wave_weights.csv only.

from pathlib import Path
from datetime import datetime, timedelta
import glob

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None  # In case yfinance is not available in some environments

# ----------------------------------------------------
# PATHS / FOLDERS
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
PERF_DIR = LOGS_DIR / "performance"
POS_DIR = LOGS_DIR / "positions"
WEIGHTS_PATH = BASE_DIR / "wave_weights.csv"

# ----------------------------------------------------
# DISPLAY NAMES & INDEX TICKERS
# ----------------------------------------------------
WAVE_DISPLAY_NAMES = {
    "SP500_Wave": "S&P 500 Wave",
    "Growth_Wave": "Growth Wave",
    "Income_Wave": "Income Wave",
    "AL_Wave": "AI Leaders Wave",
}

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

# ----------------------------------------------------
# DISCOVERY HELPERS
# ----------------------------------------------------
def discover_waves_from_logs() -> list[str]:
    """Look for performance CSVs named like:
       <WaveName>_performance_YYYYMMDD_daily.csv and extract WaveName.
    """
    if not PERF_DIR.exists():
        return []

    perf_files = list(PERF_DIR.glob("*_performance_*.csv"))
    waves = set()

    for f in perf_files:
        name = f.name
        if "_performance_" in name:
            wave_name = name.split("_performance_")[0]
            if wave_name:
                waves.add(wave_name)

    return sorted(waves)


def discover_waves_from_weights() -> list[str]:
    """Fallback: read unique Wave names from wave_weights.csv."""
    if not WEIGHTS_PATH.exists():
        return []

    try:
        df = pd.read_csv(WEIGHTS_PATH)
    except Exception:
        return []

    # Try several likely column names
    for col in ["Wave", "Wave_Name", "wave", "wave_name"]:
        if col in df.columns:
            waves = sorted(df[col].dropna().unique().tolist())
            return waves

    # If no obvious column, try to infer from first column
    if len(df.columns) >= 1:
        waves = sorted(df.iloc[:, 0].dropna().unique().tolist())
        return waves

    return []


def discover_waves() -> tuple[list[str], str]:
    """Main discovery function: returns (waves, source)."""
    waves_from_logs = discover_waves_from_logs()
    if waves_from_logs:
        return waves_from_logs, "logs"

    waves_from_weights = discover_waves_from_weights()
    if waves_from_weights:
        return waves_from_weights, "weights"

    return [], "none"

# ----------------------------------------------------
# DATA LOADERS
# ----------------------------------------------------
def load_performance(wave_name: str) -> pd.DataFrame | None:
    """Load performance CSV for a given Wave."""
    if not PERF_DIR.exists():
        return None

    pattern = str(PERF_DIR / f"{wave_name}_performance_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None

    latest_file = max(matches, key=lambda p: Path(p).stat().st_mtime)

    try:
        df = pd.read_csv(latest_file)
    except Exception:
        return None

    # Try to parse date column if present
    for col in ["date", "Date", "timestamp", "Timestamp"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.sort_values(col)
            except Exception:
                pass
            break

    return df


def load_positions(wave_name: str) -> pd.DataFrame | None:
    """Load positions CSV for a given Wave."""
    if not POS_DIR.exists():
        return None

    pattern = str(POS_DIR / f"{wave_name}_positions_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None

    latest_file = max(matches, key=lambda p: Path(p).stat().st_mtime)

    try:
        df = pd.read_csv(latest_file)
    except Exception:
        return None

    return df


# ----------------------------------------------------
# MARKET INDEX DATA (SPX & VIX)
# ----------------------------------------------------
def fetch_index_timeseries(ticker: str, days: int = 90) -> pd.DataFrame | None:
    """Fetch recent history for an index using yfinance."""
    if yf is None:
        return None
    try:
        data = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
        if data.empty:
            return None
        data = data[["Close"]].rename(columns={"Close": "close"})
        data.index.name = "date"
        return data
    except Exception:
        return None


def fetch_index_snapshot(ticker: str) -> dict | None:
    """Fetch the latest price & daily change for an index."""
    if yf is None:
        return None
    try:
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        if data.empty:
            return None
        latest = data.iloc[-1]["Close"]
        prev = data.iloc[-2]["Close"] if len(data) > 1 else np.nan
        change = latest - prev
        pct = change / prev if prev and not np.isnan(prev) else np.nan
        return {"last": latest, "change": change, "pct": pct}
    except Exception:
        return None


# ----------------------------------------------------
# METRIC HELPERS
# ----------------------------------------------------
def compute_summary_metrics(perf_df: pd.DataFrame | None) -> dict:
    """Compute total return, today return, max drawdown, alpha captured."""
    metrics = {
        "total_return": None,
        "today_return": None,
        "max_drawdown": None,
        "alpha_captured": None,
    }

    if perf_df is None or perf_df.empty:
        return metrics

    # Generic column guesses
    price_cols = [c for c in perf_df.columns if "nav" in c.lower() or "value" in c.lower()]
    ret_cols = [c for c in perf_df.columns if "return" in c.lower() and "benchmark" not in c.lower()]
    bench_cols = [c for c in perf_df.columns if "benchmark" in c.lower()]

    # Daily return series
    if ret_cols:
        r = perf_df[ret_cols[0]].astype(float)
    elif price_cols:
        p = perf_df[price_cols[0]].astype(float)
        r = p.pct_change().fillna(0.0)
    else:
        return metrics

    # Total return
    metrics["total_return"] = (1 + r).prod() - 1

    # Today
    metrics["today_return"] = r.iloc[-1]

    # Max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    metrics["max_drawdown"] = dd.min()

    # Alpha captured (vs benchmark) if available
    if bench_cols:
        br = perf_df[bench_cols[0]].astype(float)
        if br.abs().max() > 5:
            br = br / 100.0
        alpha = r - br
        metrics["alpha_captured"] = alpha.mean()

    return metrics


def fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


def sign_class(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "neutral"
    return "pos" if x > 0 else ("neg" if x < 0 else "neutral")


# ----------------------------------------------------
# STREAMLIT PAGE CONFIG & CSS
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
        background: radial-gradient(circle at top left, #0b1220 0%, #050810 45%, #01040b 100%);
        color: #f9fafb;
    }
    /* Hide default Streamlit header */
    header[data-testid="stHeader"] {display: none;}
    /* Bloomberg-style top bar */
    .waves-topbar {
        background: linear-gradient(90deg, #020617 0%, #020617 40%, #022c22 100%);
        border-bottom: 1px solid #111827;
        padding: 0.6rem 1.5rem 0.4rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: system-ui, sans-serif;
    }
    .waves-topbar-left {
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
    }
    .waves-title {
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #e5e7eb;
    }
    .waves-badge {
        font-size: 0.70rem;
        padding: 0.10rem 0.4rem;
        border-radius: 0.25rem;
        background: #0f172a;
        border: 1px solid #1d293b;
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
        border-radius: 0.25rem;
        background: #020617;
        border: 1px solid #111827;
        font-size: 0.75rem;
        min-width: 130px;
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
        margin-right: 1.5rem;
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

    /* Metric strip cards */
    .metric-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0 0.25rem 0;
    }
    .metric-card {
        background: #020617;
        border-radius: 0.5rem;
        padding: 0.6rem 0.75rem;
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

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {
        background: #020617;
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

def format_index_tile(label: str, snap: dict | None, pct_flip_sign: bool = False) -> str:
    if not snap:
        return f"""
        <div class="index-tile">
          <div class="index-label">{label}</div>
          <div class="index-value">—</div>
        </div>
        """
    pct = snap["pct"]
    if pct_flip_sign and pct is not None and not pd.isna(pct):
        pct = -pct  # VIX moves inverse to risk-on
    cls = sign_class(pct)
    pct_txt = fmt_pct(pct)
    return f"""
    <div class="index-tile">
      <div class="index-label">{label}</div>
      <div class="index-value">{snap['last']:.2f}</div>
      <div class="index-pct {cls}">{pct_txt}</div>
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
    {format_index_tile("S&P 500", spx_snap)}
    {format_index_tile("VIX (Risk Pulse)", vix_snap, pct_flip_sign=True)}
    <div class="waves-subtitle">
      {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC
    </div>
  </div>
</div>
"""
st.markdown(topbar_html, unsafe_allow_html=True)

# ----------------------------------------------------
# DISCOVER WAVES
# ----------------------------------------------------
st.sidebar.header("⚙️ Engine Controls")

waves, source = discover_waves()

if not waves:
    st.sidebar.error(
        "No Waves discovered in `logs/performance` or `wave_weights.csv` yet.\n\n"
        "• To see live data, run `waves_engine.py` to generate logs, or\n"
        "• Ensure `wave_weights.csv` is present in the repo."
    )
    st.stop()

# pick default wave
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

def display_name(w: str) -> str:
    return WAVE_DISPLAY_NAMES.get(w, w)

selected_wave = st.sidebar.selectbox(
    "Select Wave",
    waves,
    index=waves.index(default_wave),
    format_func=display_name,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Mode")
st.sidebar.radio("Risk Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic™"], index=0)
equity_exposure = st.sidebar.slider("Equity Exposure", 0, 100, 90, step=5)
st.sidebar.caption(f"Target β ~0.90 · Cash buffer: {100 - equity_exposure}%")

# ----------------------------------------------------
# LOAD DATA FOR SELECTED WAVE
# ----------------------------------------------------
perf_df = load_performance(selected_wave)
pos_df = load_positions(selected_wave)
metrics = compute_summary_metrics(perf_df)

# ----------------------------------------------------
# BLOOMBERG-STYLE TICKER TAPE
# ----------------------------------------------------
tape_items = []

# Indices first
if spx_snap:
    tape_items.append(
        f'<span class="ticker-item">'
        f'<span class="ticker-symbol">SPX</span>'
        f'<span class="ticker-pct {sign_class(spx_snap["pct"])}">{fmt_pct(spx_snap["pct"])}</span>'
        f"</span>"
    )
if vix_snap:
    vix_pct = None
    if vix_snap["pct"] is not None and not pd.isna(vix_snap["pct"]):
        vix_pct = -vix_snap["pct"]  # treat lower VIX as positive
    tape_items.append(
        f'<span class="ticker-item">'
        f'<span class="ticker-symbol">VIX</span>'
        f'<span class="ticker-pct {sign_class(vix_pct)}">{fmt_pct(vix_pct)}</span>'
        f"</span>"
    )

# Add a couple of Wave "pseudo-tickers"
tape_items.append(
    f'<span class="ticker-item">'
    f'<span class="ticker-symbol">{display_name(selected_wave)}</span>'
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
# MAIN TITLE & METRIC STRIP
# ----------------------------------------------------
st.markdown("### WAVES Engine Dashboard")
st.caption("Live / demo console for WAVES Intelligence™ – Adaptive Index Waves")

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
# OVERVIEW TAB
# ----------------------------------------------------
with tab_overview:
    left_col, right_col = st.columns([2, 1])

    # Performance curve
    with left_col:
        st.subheader("Performance Curve")
        if perf_df is not None and not perf_df.empty:
            nav_cols = [c for c in perf_df.columns if "nav" in c.lower() or "value" in c.lower()]
            date_cols = [c for c in perf_df.columns if "date" in c.lower() or "time" in c.lower()]
            if date_cols and nav_cols:
                df_plot = perf_df[[date_cols[0], nav_cols[0]]].copy()
                df_plot = df_plot.rename(columns={date_cols[0]: "date", nav_cols[0]: "NAV"})
                df_plot = df_plot.set_index("date")
                st.line_chart(df_plot)
            else:
                st.info("Performance columns not in expected format; showing raw table instead.")
                st.dataframe(perf_df.tail(30), use_container_width=True)
        else:
            st.caption("No performance data yet – waiting for first engine logs.")

    # Exposure & dual market charts
    with right_col:
        st.subheader("Exposure & Risk")
        st.metric("Equity Exposure", f"{equity_exposure} %")
        st.metric("Cash Buffer", f"{100 - equity_exposure} %")
        st.metric("Target β", "0.90")

        st.markdown("#### SPX & VIX")
        if yf is None:
            st.caption("yfinance not installed – market charts unavailable in this environment.")
        else:
            spx_hist = fetch_index_timeseries(SPX_TICKER, days=120)
            vix_hist = fetch_index_timeseries(VIX_TICKER, days=120)
            if spx_hist is not None:
                st.line_chart(spx_hist["close"], height=140)
            if vix_hist is not None:
                st.line_chart(vix_hist["close"], height=100)

    st.markdown("---")

    # Top 10 positions with Google links
    st.subheader("Top 10 Positions – Google Finance Links")
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
                            # assume 0–1 or 0–100
                            if w > 1.5:
                                w = w / 100.0
                            weight_text = f" · {w*100:0.2f}%"
                        except Exception:
                            pass
                        break
                lines.append(f"- [{t}]({url}){weight_text}")
            st.markdown("\n".join(lines))
        else:
            st.caption("Could not locate a Ticker/Symbol column to build links.")

        st.markdown("##### Positions Snapshot (raw)")
        st.dataframe(pos_df, use_container_width=True)
    else:
        st.caption("No positions file found yet for this Wave in `logs/positions`.")

# ----------------------------------------------------
# ALPHA DASHBOARD TAB
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
            st.caption("Benchmark / return columns not found – alpha view is in placeholder mode.")

# ----------------------------------------------------
# ENGINE LOGS TAB
# ----------------------------------------------------
with tab_logs:
    st.subheader("Engine Logs (Discovery Debug View)")
    st.write("Base directory:", str(BASE_DIR))
    st.write("Logs directory:", str(LOGS_DIR))

    st.markdown("**Discovered Waves**")
    st.json({"waves": waves, "source": source})

    st.markdown("**Performance files**")
    if PERF_DIR.exists():
        perf_files = [p.name for p in PERF_DIR.glob("*.csv")]
        st.write(perf_files if perf_files else "No CSV files in logs/performance.")
    else:
        st.write("logs/performance directory does not exist.")

    st.markdown("**Position files**")
    if POS_DIR.exists():
        pos_files = [p.name for p in POS_DIR.glob("*.csv")]
        st.write(pos_files if pos_files else "No CSV files in logs/positions.")
    else:
        st.write("logs/positions directory does not exist.")

    st.caption(
        "All data sourced from WAVES Engine logs when available. "
        "If you see no files listed above, run `waves_engine.py` to generate them."
    )
