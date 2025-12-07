# app.py
# WAVES Intelligence™ — Institutional Console
# Mini Bloomberg-style terminal with LIVE-ONLY internal alpha engine (no index benchmark alpha).

from pathlib import Path
from datetime import datetime
import glob
from typing import List, Optional, Dict, Tuple

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

# Long-run market drift assumption (for internal alpha)
MU_ANNUAL = 0.07     # 7% per year
TRADING_DAYS = 252
MU_DAILY = MU_ANNUAL / TRADING_DAYS  # ≈ 0.000278/day

WAVE_DISPLAY_NAMES: Dict[str, str] = {
    "SP500_Wave": "S&P 500 Wave",
    "Growth_Wave": "Growth Wave",
    "Income_Wave": "Income Wave",
    "AL_Wave": "AI Leaders Wave",
}

# Windows for alpha: trading days
ALPHA_WINDOWS = {
    "30d": 30,
    "60d": 60,
    "6m": 126,
    "1y": 252,
    "since_inception": None,
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
    for col in ["Wave", "Wave_Name", "wave", "wave_name"]:
        if col in df.columns:
            return sorted(df[col].dropna().unique().tolist())
    if len(df.columns) > 0:
        return sorted(df.iloc[:, 0].dropna().unique().tolist())
    return []


def discover_waves() -> Tuple[List[str], str]:
    """Return (waves, source) where source is 'logs', 'weights', or 'none'."""
    from_logs = discover_waves_from_logs()
    if from_logs:
        return from_logs, "logs"
    from_weights = discover_waves_from_weights()
    if from_weights:
        return from_weights, "weights"
    return [], "none"


def wave_display_name(w: str) -> str:
    return WAVE_DISPLAY_NAMES.get(w, w)

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
    date_col = None
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc or "timestamp" in lc:
            date_col = c
            break
    if date_col is not None:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        except Exception:
            pass
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
# MARKET DATA (SPX / VIX) — VISUAL ONLY
# ----------------------------------------------------
def fetch_index_timeseries(ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
    """Download recent daily close series for ticker (for charts only)."""
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
    """Latest price + daily change % (visual only)."""
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
# RETURNS & INTERNAL ALPHA (LIVE-ONLY)
# ----------------------------------------------------
def _normalize_return_series(raw: pd.Series) -> pd.Series:
    """Ensure returns are in decimal form (0.01 = 1%)."""
    s = pd.to_numeric(raw, errors="coerce").dropna()
    if s.empty:
        return s
    if s.abs().median() > 0.5:
        s = s / 100.0
    return s


def get_wave_return_series(perf_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """
    Extract daily Wave return series from performance DF.
    Prefers daily_return; falls back to NAV/value pct_change.
    """
    if perf_df is None or perf_df.empty:
        return None
    df = perf_df.copy()
    ret_col = None
    nav_col = None

    # Try return column
    for c in df.columns:
        lc = c.lower()
        if "daily_return" in lc or ("return" in lc and "benchmark" not in lc):
            ret_col = c
            break

    # Try NAV/value
    if ret_col is None:
        for c in df.columns:
            lc = c.lower()
            if "nav" in lc or "value" in lc:
                nav_col = c
                break

    if ret_col:
        r = _normalize_return_series(df[ret_col])
    elif nav_col:
        prices = pd.to_numeric(df[nav_col], errors="coerce").dropna()
        r = prices.pct_change().fillna(0.0)
    else:
        return None

    # Align index to date if present
    date_col = None
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc or "timestamp" in lc:
            date_col = c
            break
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            r.index = df[date_col]
        except Exception:
            pass

    return r.sort_index()


def get_beta_target(perf_df: Optional[pd.DataFrame], default_beta: float = 0.90) -> float:
    if perf_df is None or perf_df.empty:
        return default_beta
    if "beta_target" in perf_df.columns:
        try:
            beta_series = pd.to_numeric(perf_df["beta_target"], errors="coerce").dropna()
            if not beta_series.empty:
                return float(beta_series.iloc[-1])
        except Exception:
            pass
    return default_beta


def compute_return_metrics(perf_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    """Total return, today's return, max drawdown from live returns."""
    metrics = {"total_return": None, "today_return": None, "max_drawdown": None}
    r = get_wave_return_series(perf_df)
    if r is None or r.empty:
        return metrics

    metrics["total_return"] = (1 + r).prod() - 1
    metrics["today_return"] = float(r.iloc[-1])
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    metrics["max_drawdown"] = float(dd.min())
    return metrics


def compute_internal_alpha_metrics(
    perf_df: Optional[pd.DataFrame],
    beta_default: float = 0.90,
    mu_daily: float = MU_DAILY,
    min_days_threshold: int = 15,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Internal (benchmark-free) alpha vs expected drift:

        expected_daily = beta * mu_daily
        alpha_daily    = wave_r - expected_daily

    For each window:
        cum_wave     = ∏(1 + wave_r)
        cum_expected = (1 + expected_daily) ^ N
        alpha_window = cum_wave - cum_expected
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for key in ALPHA_WINDOWS:
        out[key] = {"alpha": None, "wave_return": None, "expected_return": None, "days": 0}
    out["_one_day_alpha"] = None

    r = get_wave_return_series(perf_df)
    if r is None or r.empty:
        return out

    beta = get_beta_target(perf_df, default_beta=beta_default)
    expected_daily = beta * mu_daily

    # Intraday / 1-Day Alpha = last day's alpha vs expected
    one_day_alpha = float(r.iloc[-1] - expected_daily)
    out["_one_day_alpha"] = one_day_alpha

    # Window alphas
    for label, window in ALPHA_WINDOWS.items():
        if window is None:
            window_r = r.copy()
        else:
            window_r = r.tail(window)

        n_days = len(window_r)
        out[label]["days"] = n_days

        if window is not None and n_days < min_days_threshold:
            continue  # not enough history, leave None

        cum_wave = (1.0 + window_r).prod()
        cum_expected = (1.0 + expected_daily) ** n_days

        alpha_window = cum_wave - cum_expected
        wave_total = cum_wave - 1.0
        expected_total = cum_expected - 1.0

        out[label] = {
            "alpha": float(alpha_window),
            "wave_return": float(wave_total),
            "expected_return": float(expected_total),
            "days": int(n_days),
        }

    return out


def compute_daily_alpha_series(
    perf_df: Optional[pd.DataFrame],
    beta_default: float = 0.90,
    mu_daily: float = MU_DAILY,
) -> Optional[pd.Series]:
    """Daily alpha series: alpha_daily = wave_r - beta * mu_daily."""
    r = get_wave_return_series(perf_df)
    if r is None or r.empty:
        return None
    beta = get_beta_target(perf_df, default_beta=beta_default)
    expected_daily = beta * mu_daily
    alpha = r - expected_daily
    alpha.name = "alpha_daily"
    return alpha

# ----------------------------------------------------
# SMALL FORMAT HELPERS
# ----------------------------------------------------
def fmt_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


def sign_class(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "neutral"
    return "pos" if x > 0 else ("neg" if x < 0 else "neutral")

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
# TOP BAR: TITLE + INDEX SNAPSHOTS (VISUAL ONLY)
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
    <div class="waves-subtitle">Adaptive Index Waves · Internal Alpha (β-Adjusted)</div>
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
ret_metrics = compute_return_metrics(perf_df)
alpha_metrics = compute_internal_alpha_metrics(perf_df, beta_default=0.90, mu_daily=MU_DAILY)
alpha_daily_series = compute_daily_alpha_series(perf_df, beta_default=0.90, mu_daily=MU_DAILY)

one_day_alpha = alpha_metrics.get("_one_day_alpha")
alpha_30d = alpha_metrics["30d"]["alpha"]
alpha_60d = alpha_metrics["60d"]["alpha"]
alpha_1y = alpha_metrics["1y"]["alpha"]

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
    f'<span class="ticker-symbol">{wave_display_name(selected_wave)} 1D α</span>'
    f'<span class="ticker-pct {sign_class(one_day_alpha)}">{fmt_pct(one_day_alpha)}</span>'
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
st.caption(
    "Live console for WAVES Intelligence™ — Adaptive Index Waves. "
    "**All alpha metrics are internal, live-only, and β-adjusted vs long-run market drift (no index benchmark alpha).**"
)

# Row 1: performance metrics
metric_html_row1 = f"""
<div class="metric-strip">
  <div class="metric-card">
    <div class="metric-label">Total Return (Since Inception)</div>
    <div class="metric-value {sign_class(ret_metrics['total_return'])}">{fmt_pct(ret_metrics['total_return'])}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Intraday Return (Today)</div>
    <div class="metric-value {sign_class(ret_metrics['today_return'])}">{fmt_pct(ret_metrics['today_return'])}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Max Drawdown</div>
    <div class="metric-value neg">{fmt_pct(ret_metrics['max_drawdown'])}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Intraday Internal Alpha (1-Day)</div>
    <div class="metric-value {sign_class(one_day_alpha)}">{fmt_pct(one_day_alpha)}</div>
  </div>
</div>
"""
st.markdown(metric_html_row1, unsafe_allow_html=True)

# Row 2: 30D / 60D / 1Y Internal Alpha
metric_html_row2 = f"""
<div class="metric-strip">
  <div class="metric-card">
    <div class="metric-label">30-Day Internal Alpha</div>
    <div class="metric-value {sign_class(alpha_30d)}">{fmt_pct(alpha_30d)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">60-Day Internal Alpha</div>
    <div class="metric-value {sign_class(alpha_60d)}">{fmt_pct(alpha_60d)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">1-Year Internal Alpha</div>
    <div class="metric-value {sign_class(alpha_1y)}">{fmt_pct(alpha_1y)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">β-Adjusted Expected Drift (Daily)</div>
    <div class="metric-value neutral">{fmt_pct(get_beta_target(perf_df) * MU_DAILY)}</div>
  </div>
</div>
"""
st.markdown(metric_html_row2, unsafe_allow_html=True)

if perf_df is None:
    st.info(
        f"No performance log found yet for `{selected_wave}` in `logs/performance`.\n\n"
        "The dashboard is running in **structure/demo mode**. "
        "Run `waves_engine.py` locally and upload the resulting CSVs to see full live charts and alpha."
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
        st.subheader("Performance Curve (Live Engine)")
        if perf_df is not None and not perf_df.empty:
            date_col = None
            nav_col = None
            for c in perf_df.columns:
                lc = c.lower()
                if date_col is None and ("date" in lc or "time" in lc or "timestamp" in lc):
                    date_col = c
                if nav_col is None and ("nav" in lc or "value" in lc):
                    nav_col = c
            if date_col and nav_col:
                df_plot = perf_df[[date_col, nav_col]].copy()
                df_plot = df_plot.rename(columns={date_col: "date", nav_col: "NAV"})
                df_plot = df_plot.set_index("date")
                st.line_chart(df_plot)
            else:
                st.info("Could not identify NAV/value and date columns — showing raw performance table.")
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

        st.markdown("#### S&P 500 & VIX (Visual Only)")
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
# TAB 2: ALPHA DASHBOARD (LIVE-ONLY INTERNAL)
# ----------------------------------------------------
with tab_alpha:
    st.subheader("Live Internal Alpha Dashboard (β-Adjusted · No Benchmarks)")

    if perf_df is None or perf_df.empty:
        st.info("Alpha metrics will appear once performance logs are available for this Wave.")
    else:
        # Summary table for windows
        rows = []
        for label, window in ALPHA_WINDOWS.items():
            m = alpha_metrics[label]
            label_pretty = {
                "30d": "30-Day",
                "60d": "60-Day",
                "6m": "6-Month",
                "1y": "1-Year",
                "since_inception": "Since Inception",
            }.get(label, label)
            rows.append({
                "Window": label_pretty,
                "Internal Alpha vs Drift": fmt_pct(m["alpha"]),
                "Wave Return": fmt_pct(m["wave_return"]),
                "Expected Return (β·μ)": fmt_pct(m["expected_return"]),
                "Days in Window": m["days"],
            })
        alpha_table = pd.DataFrame(rows)
        st.dataframe(alpha_table, use_container_width=True)

        st.markdown("#### Daily Internal Alpha Series")
        if alpha_daily_series is not None and not alpha_daily_series.empty:
            alpha_df = alpha_daily_series.to_frame()
            st.line_chart(alpha_df, height=260)
            st.caption("alpha_daily = Wave_return – β_target × μ_daily  (μ_daily ≈ 7% / 252)")
        else:
            st.caption("No daily alpha series available yet for this Wave.")

        st.caption(
            "All alpha metrics are computed **only** from live engine logs, using internal expected drift, "
            "not external index benchmarks. This is β-adjusted excess return vs long-run market drift."
        )

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
        perf_files = [p.name for p in PERF_DIR.glob("*.csv")]
        st.write(perf_files if perf_files else "No CSV files in `logs/performance`.")
    else:
        st.write("`logs/performance` directory does not exist.")

    st.markdown("**Position files**")
    if POS_DIR.exists():
        pos_files = [p.name for p in POS_DIR.glob("*.csv")]
        st.write(pos_files if pos_files else "No CSV files in `logs/positions`.")
    else:
        st.write("`logs/positions` directory does not exist.")

    st.caption(
        "All live performance and internal alpha are sourced exclusively from WAVES Engine logs. "
        "If you see no files above, run `waves_engine.py` to generate them."
    )
