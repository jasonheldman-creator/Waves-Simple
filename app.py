# app.py
# WAVES Intelligence™ Live Engine Console
# Works both:
#   - with real logs (logs/performance, logs/positions)
#   - and without logs (falls back to wave_weights.csv for Wave list)

from pathlib import Path
from datetime import datetime
import glob

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------
# PATHS / FOLDERS
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
PERF_DIR = LOGS_DIR / "performance"
POS_DIR = LOGS_DIR / "positions"
WEIGHTS_PATH = BASE_DIR / "wave_weights.csv"

# ----------------------------------------------------
# DISCOVERY HELPERS
# ----------------------------------------------------
def discover_waves_from_logs() -> list[str]:
    """
    Look for performance CSVs named like:
        <WaveName>_performance_YYYYMMDD_daily.csv
    and extract WaveName.
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
    """
    Fallback: read unique Wave names from wave_weights.csv
    Expecting a column named 'Wave' (e.g. 'Growth_Wave', 'SP500_Wave').
    """
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
    """
    Main discovery function.
    Returns (waves, source) where source is 'logs' or 'weights' or 'none'.
    """
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
    """
    Load performance CSV for a given Wave.
    If none exists, return None.
    """
    if not PERF_DIR.exists():
        return None

    pattern = str(PERF_DIR / f"{wave_name}_performance_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None

    # Use the latest file by modified time
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
    """
    Load positions CSV for a given Wave.
    """
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
# METRIC HELPERS
# ----------------------------------------------------
def compute_summary_metrics(perf_df: pd.DataFrame | None) -> dict:
    """
    Compute total return, today return, max drawdown, alpha captured.
    All metrics are optional if perf_df is None.
    """
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
        # assume benchmark is daily return or percentage
        if br.abs().max() > 5:  # probably percentage
            br = br / 100.0
        alpha = r - br
        metrics["alpha_captured"] = alpha.mean()

    return metrics


def fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


# ----------------------------------------------------
# STREAMLIT LAYOUT
# ----------------------------------------------------
st.set_page_config(
    page_title="WAVES Engine Dashboard",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left, #101725 0%, #050812 55%, #020308 100%);
        color: #f5f7ff;
    }
    .stMetric {
        background-color: #111827 !important;
        border-radius: 10px;
        padding: 12px;
    }
    .css-18ni7ap, .css-1d391kg {
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌊 WAVES Engine Dashboard")
st.caption("Live / demo console for WAVES Intelligence™ – Adaptive Index Waves")

# ----------------------------------------------------
# SIDEBAR – ENGINE CONTROLS
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

if source == "logs":
    st.sidebar.success("Waves discovered from engine logs ✅")
elif source == "weights":
    st.sidebar.warning("Using Wave list from wave_weights.csv (no performance logs found yet).")

selected_wave = st.sidebar.selectbox("Select Wave", waves, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Mode")
st.sidebar.radio("Risk Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic™"], index=0)
exposure = st.sidebar.slider("Equity Exposure", 0, 100, 90, step=5)
st.sidebar.caption(f"Target β ~0.90 · Cash buffer: {100 - exposure}%")

# ----------------------------------------------------
# LOAD DATA FOR SELECTED WAVE
# ----------------------------------------------------
perf_df = load_performance(selected_wave)
pos_df = load_positions(selected_wave)
metrics = compute_summary_metrics(perf_df)

# ----------------------------------------------------
# TOP METRICS STRIP
# ----------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Return (since inception)", fmt_pct(metrics["total_return"]))
c2.metric("Today", fmt_pct(metrics["today_return"]))
c3.metric("Max Drawdown", fmt_pct(metrics["max_drawdown"]))
c4.metric("Alpha Captured vs Benchmark", fmt_pct(metrics["alpha_captured"]))

if perf_df is None:
    st.info(
        f"No performance log found yet for `{selected_wave}` in `logs/performance`.\n\n"
        "The dashboard is running in **structure/demo mode**. "
        "Run `waves_engine.py` locally and upload the resulting CSVs to see full live charts."
    )

# ----------------------------------------------------
# MAIN LAYOUT – OVERVIEW TAB
# ----------------------------------------------------
tab_overview, tab_alpha, tab_logs = st.tabs(["Overview", "Alpha Dashboard", "Engine Logs"])

with tab_overview:
    left, right = st.columns([2, 1])

    # Performance curve
    with left:
        st.subheader("Performance Curve")
        if perf_df is not None and not perf_df.empty:
            # Try to find a cumulative value or NAV column
            nav_cols = [c for c in perf_df.columns if "nav" in c.lower() or "value" in c.lower()]
            date_cols = [c for c in perf_df.columns if "date" in c.lower() or "time" in c.lower()]

            if date_cols and nav_cols:
                df_plot = perf_df[[date_cols[0], nav_cols[0]]].copy()
                df_plot = df_plot.rename(columns={date_cols[0]: "date", nav_cols[0]: "NAV"})
                df_plot = df_plot.set_index("date")
                st.line_chart(df_plot)
            else:
                st.info("Performance columns not in expected format; showing raw table instead.")
                st.dataframe(perf_df.tail(30))
        else:
            st.caption("No performance data yet – waiting for first engine logs.")

    # Exposure & risk
    with right:
        st.subheader("Exposure & Risk")
        st.metric("Equity Exposure", f"{exposure}%")
        st.metric("Cash Buffer", f"{100 - exposure}%")
        st.metric("Target β", "0.90")

    st.markdown("---")

    # Positions snapshot
    st.subheader("Positions Snapshot")
    if pos_df is not None and not pos_df.empty:
        st.dataframe(pos_df.head(50), use_container_width=True)
    else:
        st.caption("No positions file found yet for this Wave in `logs/positions`.")

with tab_alpha:
    st.subheader("Alpha Dashboard")
    if perf_df is None or perf_df.empty:
        st.info("Alpha metrics will appear once performance logs are available.")
    else:
        # Simple rolling 30-day alpha illustration if benchmark column exists
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
            st.caption("Rolling 30-day average alpha.")
        else:
            st.caption("Benchmark / return columns not found – alpha view is in placeholder mode.")

with tab_logs:
    st.subheader("Engine Logs (File Discovery)")
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
