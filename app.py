import os
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ============================================================
# PAGE CONFIG & GLOBALS
# ============================================================

st.set_page_config(
    page_title="WAVES Institutional Console",
    page_icon="🌊",
    layout="wide",
)

ROOT_DIR = Path(__file__).parent.resolve()
LOGS_DIR = ROOT_DIR / "logs"
PERF_DIR = LOGS_DIR / "performance"
POS_DIR = LOGS_DIR / "positions"

# Ensure directories exist (safe on cloud + local)
for d in [LOGS_DIR, PERF_DIR, POS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# SMALL STYLING TOUCHES
# ============================================================

CUSTOM_CSS = """
<style>
/* Make app a bit more "terminal / console" style */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stMetricValue"] {
    font-weight: 600;
    font-size: 1.4rem;
}

.wave-pill {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    background: rgba(0, 255, 135, 0.08);
    border: 1px solid rgba(0, 255, 135, 0.35);
    color: #00ff87;
}

.mode-pill {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    background: rgba(0, 168, 255, 0.08);
    border: 1px solid rgba(0, 168, 255, 0.35);
    color: #00a8ff;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV safely; return None on failure."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_universe_file(root: Path) -> Optional[Path]:
    """Find a plausible wave universe / weights file in the repo."""
    candidates = [
        root / "wave_weights.csv",
        root / "Wave_Weights.csv",
        root / "weights.csv",
        root / "universe.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: first CSV in root (if any)
    for c in root.glob("*.csv"):
        return c
    return None


def load_universe_df() -> Optional[pd.DataFrame]:
    """Load the Wave universe / weights file and normalize columns."""
    universe_path = find_universe_file(ROOT_DIR)
    if universe_path is None:
        return None

    df = safe_read_csv(universe_path)
    if df is None or df.empty:
        return None

    # Normalize columns
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Map to canonical names
    col_map = {}

    # Wave name
    for candidate in ["wave", "wave_name", "portfolio", "strategy"]:
        if candidate in df.columns:
            col_map["wave"] = candidate
            break

    # Ticker
    for candidate in ["ticker", "symbol"]:
        if candidate in df.columns:
            col_map["ticker"] = candidate
            break

    # Weight
    for candidate in ["weight", "target_weight", "wgt"]:
        if candidate in df.columns:
            col_map["weight"] = candidate
            break

    # Name
    for candidate in ["name", "company", "security_name"]:
        if candidate in df.columns:
            col_map["name"] = candidate
            break

    # Sector
    for candidate in ["sector", "industry"]:
        if candidate in df.columns:
            col_map["sector"] = candidate
            break

    # Build normalized view
    out = pd.DataFrame()
    if "wave" in col_map:
        out["wave"] = df[col_map["wave"]].astype(str)
    if "ticker" in col_map:
        out["ticker"] = df[col_map["ticker"]].astype(str)
    if "weight" in col_map:
        out["weight"] = pd.to_numeric(df[col_map["weight"]], errors="coerce")
    if "name" in col_map:
        out["name"] = df[col_map["name"]].astype(str)
    if "sector" in col_map:
        out["sector"] = df[col_map["sector"]].astype(str)

    if out.empty or "wave" not in out.columns or "ticker" not in out.columns:
        # Universe file exists but not in expected schema
        return None

    # Clean weights if present
    if "weight" in out.columns:
        # Ensure weights sum to 1 per wave where possible
        out["weight"] = out["weight"].fillna(0.0)

    return out


def get_available_waves(universe_df: Optional[pd.DataFrame]) -> List[str]:
    """Return list of Wave names from the universe or a default list."""
    if universe_df is not None and "wave" in universe_df.columns:
        waves = sorted(universe_df["wave"].dropna().unique().tolist())
        if waves:
            return waves

    # Fallback: known / typical Waves
    return [
        "Growth_Wave",
        "SP500_Wave",
        "Income_Wave",
        "Future_Power_Wave",
        "Crypto_Income_Wave",
        "Quantum_Wave",
        "Clean_Transit_Wave",
        "SMID_Growth_Wave",
    ]


def find_latest_perf_file_for_wave(wave_name: str) -> Optional[Path]:
    """
    Find the most recent performance CSV for a Wave.
    Supports:
        WaveName_performance_YYYYMMDD_daily.csv
        WaveName_performance_daily.csv
    """
    if not PERF_DIR.exists():
        return None

    pattern_dated = PERF_DIR.glob(f"{wave_name}_performance_*_daily.csv")
    pattern_simple = PERF_DIR.glob(f"{wave_name}_performance_daily.csv")

    candidates = list(pattern_dated) + list(pattern_simple)
    if not candidates:
        return None

    # Choose by latest modified time
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def load_perf_df_for_wave(wave_name: str) -> Optional[pd.DataFrame]:
    """Load the latest performance CSV for a Wave and normalize."""
    perf_path = find_latest_perf_file_for_wave(wave_name)
    if perf_path is None:
        return None

    df = safe_read_csv(perf_path)
    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Try to parse a timestamp if present
    for candidate in ["timestamp", "date", "asof"]:
        if candidate in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df[candidate])
            except Exception:
                df["timestamp"] = df[candidate].astype(str)
            break

    return df


def compute_perf_metrics(perf_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Compute summary metrics from the performance DataFrame.
    We try to be flexible with column names.
    Returns values in decimal units (e.g. 0.042 = 4.2%).
    """
    metrics = {
        "total_return": None,
        "today_return": None,
        "max_drawdown": None,
        "alpha": None,
    }

    if perf_df is None or perf_df.empty:
        return metrics

    cols = perf_df.columns

    # Total return
    for candidate in ["cumulative_return", "cum_return", "total_return"]:
        if candidate in cols:
            metrics["total_return"] = perf_df[candidate].iloc[-1]
            break

    # Today return
    for candidate in ["daily_return", "return", "day_return"]:
        if candidate in cols:
            metrics["today_return"] = perf_df[candidate].iloc[-1]
            break

    # Max drawdown
    for candidate in ["max_drawdown", "drawdown"]:
        if candidate in cols:
            # If it's a series of drawdown values, take min
            dd_series = perf_df[candidate]
            if dd_series.min() < 0:
                metrics["max_drawdown"] = dd_series.min()
            else:
                metrics["max_drawdown"] = dd_series.iloc[-1]
            break

    # Alpha vs benchmark
    for candidate in ["alpha", "excess_return", "alpha_vs_benchmark"]:
        if candidate in cols:
            metrics["alpha"] = perf_df[candidate].iloc[-1]
            break

    return metrics


def load_positions_for_wave(wave_name: str) -> Optional[pd.DataFrame]:
    """
    Load latest positions CSV for a Wave (if available).
    Not required for the app to function; used for richer tables if present.
    """
    if not POS_DIR.exists():
        return None

    # Look for pattern: WaveName_positions_*.csv
    candidates = list(POS_DIR.glob(f"{wave_name}_positions_*.csv"))
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    df = safe_read_csv(latest)
    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def fmt_pct(x: Optional[float]) -> str:
    """Format decimal -> 'x.xx%' or em dash if None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:,.2f}%"


def fmt_bp(x: Optional[float]) -> str:
    """Format decimal -> basis points string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 10000:,.0f} bp"


# ============================================================
# SIDEBAR – GLOBAL CONTROLS
# ============================================================

universe_df = load_universe_df()
available_waves = get_available_waves(universe_df)

with st.sidebar:
    st.markdown("## 🌊 WAVES Intelligence™")
    st.markdown(
        "<span class='wave-pill'>Cloud Snapshot</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "This view is optimized for **mobile and cloud**: weights, holdings, "
        "and risk. Full live engine runs on your **desktop terminal**."
    )

    st.markdown("---")

    selected_wave = st.selectbox("Select Wave", available_waves, index=0)

    mode = st.radio(
        "Mode (label only in cloud view)",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
        help="In the cloud view this is a label; live mode logic runs on the desktop engine.",
    )

    st.markdown(
        f"<span class='mode-pill'>{mode}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        "Tip: In the **desktop engine**, this same Wave will show full live performance, "
        "drawdowns, and alpha vs benchmark."
    )

# ============================================================
# HEADER
# ============================================================

col_title, col_badge = st.columns([0.8, 0.2])

with col_title:
    st.title("WAVES Institutional Console")
    st.caption(
        "📱 **Cloud / Mobile View** — Live **weights, holdings, and risk snapshot**. "
        "Full performance curves and alpha metrics run on the **desktop WAVES engine**."
    )

with col_badge:
    st.markdown(
        "<div style='text-align:right;margin-top:0.6rem;'>"
        "<span class='wave-pill'>WAVES Intelligence™</span>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ============================================================
# LOAD DATA FOR CURRENT WAVE
# ============================================================

perf_df = load_perf_df_for_wave(selected_wave)
perf_metrics = compute_perf_metrics(perf_df)

positions_df = load_positions_for_wave(selected_wave)

if universe_df is not None:
    wave_universe = universe_df[universe_df["wave"] == selected_wave].copy()
else:
    wave_universe = None

# ============================================================
# SUMMARY METRICS ROW
# ============================================================

summary_cols = st.columns(4)
total_ret_str = fmt_pct(perf_metrics["total_return"])
today_ret_str = fmt_pct(perf_metrics["today_return"])
max_dd_str = fmt_pct(perf_metrics["max_drawdown"])
alpha_str = fmt_bp(perf_metrics["alpha"])  # alpha in bp feels nice

with summary_cols[0]:
    st.metric("Total Return (since inception / logs)", total_ret_str)

with summary_cols[1]:
    st.metric("Today", today_ret_str)

with summary_cols[2]:
    st.metric("Max Drawdown", max_dd_str)

with summary_cols[3]:
    st.metric("Alpha vs Benchmark", alpha_str)

st.caption(
    "If these show **0.00%** or **—**, the cloud app either has no performance logs yet "
    "or only a very short history. The **desktop WAVES engine** runs the full live analytics."
)

st.markdown("---")

# ============================================================
# LAYOUT: LEFT = PERFORMANCE / SNAPSHOT, RIGHT = HOLDINGS
# ============================================================

left, right = st.columns([0.55, 0.45])

# -------------------- LEFT: PERFORMANCE PANEL --------------------
with left:
    st.subheader("Performance View")

    if perf_df is None or perf_df.empty:
        st.info(
            """
            **Mobile Engine View — What you're seeing**

            • Live **weights & holdings** for this Wave  
            • Current **equity exposure** and **cash buffer** (when logs are present)  
            • Wave **risk mode** and SmartSafe™ framing  

            The full **performance curve, historical drawdowns, and alpha vs benchmark** 
            are calculated by the **WAVES desktop engine** and shown on your desktop console.

            This cloud view is optimized for checking **positioning and risk on the go**.
            """
        )
    else:
        # Try to use timestamp on x-axis and any reasonable numeric column on y
        df = perf_df.copy()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
            df = df.set_index("timestamp")

        # Prefer plotting cumulative_return-style series if present
        y_cols_preferred = [
            c for c in numeric_cols if "cum" in c or "cumulative" in c or "nav" in c
        ]
        if y_cols_preferred:
            plot_df = df[y_cols_preferred]
        elif numeric_cols:
            plot_df = df[numeric_cols]
        else:
            plot_df = None

        if plot_df is not None and not plot_df.empty:
            st.line_chart(plot_df)
            st.caption(
                "Chart based on the latest performance log found in `logs/performance/` "
                f"for **{selected_wave}**."
            )
        else:
            st.info(
                "Performance data is present, but no numeric columns were detected to plot."
            )

# -------------------- RIGHT: HOLDINGS PANEL --------------------
with right:
    st.subheader("Wave Holdings & Weights")

    if wave_universe is None or wave_universe.empty:
        st.warning(
            "No universe/weights file could be loaded for this Wave. "
            "Ensure a `wave_weights.csv` (or similar) is present in the repo."
        )
    else:
        df = wave_universe.copy()

        # Normalize weights to sum to 100% (if present)
        if "weight" in df.columns and df["weight"].sum() > 0:
            df["weight_pct"] = df["weight"] / df["weight"].sum()
        else:
            df["weight_pct"] = np.nan

        # Sort by weight descending
        df = df.sort_values("weight_pct", ascending=False)

        # Build a display frame
        display_cols = []
        col_order = []

        if "ticker" in df.columns:
            col_order.append("ticker")
        if "name" in df.columns:
            col_order.append("name")
        if "sector" in df.columns:
            col_order.append("sector")
        col_order.append("weight_pct")

        display_df = df[col_order].copy()
        display_df.rename(
            columns={
                "ticker": "Ticker",
                "name": "Name",
                "sector": "Sector",
                "weight_pct": "Weight",
            },
            inplace=True,
        )
        if "Weight" in display_df.columns:
            display_df["Weight"] = display_df["Weight"].apply(
                lambda x: f"{x * 100:,.2f}%" if pd.notnull(x) else "—"
            )

        st.caption("Top 10 Holdings")
        st.dataframe(display_df.head(10), use_container_width=True)

        with st.expander("View full Wave holdings"):
            st.dataframe(display_df, use_container_width=True)

# ============================================================
# ENGINE STATUS / LOGS PANEL
# ============================================================

st.markdown("---")
st.subheader("Engine & Logs Snapshot")

status_cols = st.columns(2)

with status_cols[0]:
    st.markdown("**Local Engine Status**")
    st.caption(
        "On your **desktop**, the WAVES engine writes CSV logs into `logs/positions/` "
        "and `logs/performance/` for each Wave."
    )

    if PERF_DIR.exists():
        perf_files = sorted(PERF_DIR.glob(f"{selected_wave}_performance*.csv"))
    else:
        perf_files = []

    if perf_files:
        latest = max(perf_files, key=lambda p: p.stat().st_mtime)
        ts = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        st.success(
            f"Latest performance log for **{selected_wave}**: `{latest.name}`\n\n"
            f"_Last modified (server time): **{ts}**_"
        )
    else:
        st.info(
            f"No performance logs were found for **{selected_wave}** in `logs/performance/` "
            "on this deployment. The desktop engine will populate these files during live runs."
        )

with status_cols[1]:
    st.markdown("**How to Read This Console**")
    st.markdown(
        """
        - Use the **Wave selector** (left) to switch between portfolios  
        - Review **top holdings & weights** on the right  
        - Check **Total Return / Today / Max DD / Alpha** at the top  
        - Use the **desktop WAVES engine** for full-depth analytics, order routing, and intraday views  
        """
    )

st.caption(
    "WAVES Intelligence™ — AI-managed Waves, desktop engine for live trading & performance, "
    "cloud view for fast mobile oversight."
)