import os
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG & PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
POSITIONS_DIR = LOGS_DIR / "positions"
PERF_DIR = LOGS_DIR / "performance"

st.set_page_config(
    page_title="WAVES Intelligence™ Live Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# BRANDING & DARK THEME CSS
# ============================================================

DARK_BG = "#050816"      # deep navy
CARD_BG = "#0b1020"      # slightly lighter
ACCENT = "#40e0d0"       # teal / WAVES accent
ACCENT_SOFT = "#00b894"  # soft green
TEXT_PRIMARY = "#f5f7ff"
TEXT_MUTED = "#9aa4c6"

CUSTOM_CSS = f"""
<style>
/* Global background */
.stApp {{
    background: radial-gradient(circle at top, #111933 0, {DARK_BG} 45%, #02040a 100%);
    color: {TEXT_PRIMARY};
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    max-width: 1400px;
}}

.stSidebar {{
    background: linear-gradient(180deg, #050816 0%, #02030a 100%);
    border-right: 1px solid #151a2c;
}}

.stSidebar [data-testid="stMetricValue"],
.stSidebar [data-testid="stMetricLabel"] {{
    color: {TEXT_PRIMARY};
}}

h1, h2, h3, h4, h5, h6 {{
    color: {TEXT_PRIMARY};
    letter-spacing: 0.02em;
}}

.small-muted {{
    font-size: 0.75rem;
    color: {TEXT_MUTED};
}}

/* Top header */
.wave-header {{
    padding: 0.75rem 1rem 0.75rem 1.25rem;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(64,224,208,0.06), rgba(0,0,0,0.3));
    border: 1px solid rgba(64,224,208,0.25);
    box-shadow: 0 18px 40px rgba(0,0,0,0.45);
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.wave-header-left {{
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
}}

.wave-title {{
    font-size: 1.3rem;
    font-weight: 650;
}}

.wave-subtitle {{
    font-size: 0.8rem;
    color: {TEXT_MUTED};
}}

/* Badge styles */
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    border: 1px solid rgba(255,255,255,0.2);
}}

.badge-live {{
    background: rgba(0, 200, 120, 0.16);
    border-color: rgba(0, 200, 120, 0.6);
}}

.badge-mode {{
    background: rgba(111, 66, 193, 0.12);
    border-color: rgba(132, 94, 247, 0.7);
}}

.badge-beta {{
    background: rgba(64, 224, 208, 0.12);
    border-color: rgba(64, 224, 208, 0.7);
}}

/* KPI cards */
.kpi-card {{
    background: radial-gradient(circle at top left, rgba(64,224,208,0.08), {CARD_BG});
    border-radius: 14px;
    padding: 0.8rem 0.9rem;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 16px 35px rgba(0,0,0,0.5);
}}

.kpi-label {{
    font-size: 0.75rem;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.12em;
}}

.kpi-value {{
    font-size: 1.3rem;
    font-weight: 650;
    margin-top: 0.35rem;
}}

.kpi-delta-pos {{
    font-size: 0.75rem;
    color: {ACCENT_SOFT};
}}

.kpi-delta-neg {{
    font-size: 0.75rem;
    color: #ff7675;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    border-bottom: 1px solid #192038;
}}

.stTabs [data-baseweb="tab"] {{
    padding-top: 0.4rem;
    padding-bottom: 0.5rem;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT_SOFT});
}}

/* Tables */
.dataframe {{
    background: {CARD_BG} !important;
    color: {TEXT_PRIMARY} !important;
}}

.dataframe td, .dataframe th {{
    border-color: #1b2238 !important;
}}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================

def ensure_directories():
    missing = []
    if not LOGS_DIR.exists():
        missing.append(str(LOGS_DIR))
    if not POSITIONS_DIR.exists():
        missing.append(str(POSITIONS_DIR))
    if not PERF_DIR.exists():
        missing.append(str(PERF_DIR))
    return missing


@st.cache_data
def get_available_waves():
    if not PERF_DIR.exists():
        return []
    files = PERF_DIR.glob("*_performance_daily.csv")
    waves = sorted(f.name.replace("_performance_daily.csv", "") for f in files)
    return waves


@st.cache_data
def load_performance(wave_name: str) -> pd.DataFrame | None:
    path = PERF_DIR / f"{wave_name}_performance_daily.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Date column detection
    for cand in ["date", "Date", "timestamp", "Timestamp"]:
        if cand in df.columns:
            df["date"] = pd.to_datetime(df[cand])
            break
    else:
        df["date"] = pd.to_datetime(df.index)

    df = df.sort_values("date").reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]

    # Cumulative return
    if "cumulative_return" not in df.columns:
        if "daily_return" in df.columns:
            df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
        elif "nav" in df.columns:
            df["cumulative_return"] = df["nav"] / df["nav"].iloc[0] - 1

    # Benchmark cumulative if available
    if "benchmark_return" in df.columns and "benchmark_cumulative" not in df.columns:
        df["benchmark_cumulative"] = (1 + df["benchmark_return"]).cumprod() - 1

    return df


@st.cache_data
def load_latest_positions(wave_name: str):
    pattern = str(POSITIONS_DIR / f"{wave_name}_positions_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None

    latest_path = Path(files[-1])
    df = pd.read_csv(latest_path)
    df.columns = [c.strip() for c in df.columns]

    cols_lower = {c.lower(): c for c in df.columns}

    # Try to compute Value / Weight if missing
    if "value" not in cols_lower:
        price_col = cols_lower.get("price")
        qty_col = cols_lower.get("quantity") or cols_lower.get("shares")
        if price_col and qty_col:
            df["Value"] = df[price_col] * df[qty_col]

    if "weight" not in cols_lower and "Value" in df.columns:
        total_val = df["Value"].sum()
        if total_val > 0:
            df["Weight"] = df["Value"] / total_val

    return df, latest_path.name


def format_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x*100:,.2f}%"


def summarize_performance(df: pd.DataFrame):
    if df is None or df.empty:
        return {}

    last_row = df.iloc[-1]

    # Latest cumulative
    cum = last_row.get("cumulative_return", np.nan)

    # One-day change
    daily = last_row.get("daily_return", np.nan)

    # Simple drawdown calc
    if "cumulative_return" in df.columns:
        peak = df["cumulative_return"].cummax()
        dd = df["cumulative_return"] - peak
        max_dd = dd.min()
    else:
        max_dd = np.nan

    # Alpha if we have benchmark
    alpha = np.nan
    if "benchmark_cumulative" in df.columns:
        alpha = last_row["cumulative_return"] - last_row["benchmark_cumulative"]

    return {
        "cum": cum,
        "daily": daily,
        "max_dd": max_dd,
        "alpha": alpha,
    }


# ============================================================
# SIDEBAR — ENGINE CONTROLS
# ============================================================

missing_dirs = ensure_directories()
if missing_dirs:
    st.error(
        "⚠️ `logs/` folder not found.\n\n"
        "Make sure `logs/performance` and `logs/positions` are in the **same folder as `app.py` and `waves_engine.py`.**"
    )
    st.stop()

waves = get_available_waves()
st.sidebar.markdown("### ⚙️ Engine Controls")

if not waves:
    st.sidebar.error("No Waves discovered in `logs/performance` yet.")
    st.stop()

selected_wave = st.sidebar.selectbox("Wave", waves, index=0)

# Exposure slider (for demo — doesn’t change logs, just UI)
exposure = st.sidebar.slider("Equity Exposure", min_value=0, max_value=100, value=90, step=5)

mode = st.sidebar.radio(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=1,
)

beta_target = 0.90 if mode == "Alpha-Minus-Beta" else 1.00
st.sidebar.markdown(
    f"<span class='small-muted'>Target β: <b>{beta_target:.2f}</b> · Cash buffer implied from exposure: <b>{100-exposure}%</b></span>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span class='small-muted'>WAVES Intelligence™ · Live Engine Console</span>",
    unsafe_allow_html=True,
)

# ============================================================
# MAIN LAYOUT
# ============================================================

perf_df = load_performance(selected_wave)
pos_df, pos_filename = load_latest_positions(selected_wave)

perf_summary = summarize_performance(perf_df)

# ---------- HEADER ----------
col_header, col_header_right = st.columns([3, 2])

with col_header:
    st.markdown(
        f"""
        <div class="wave-header">
            <div class="wave-header-left">
                <div class="wave-title">🌊 {selected_wave}</div>
                <div class="wave-subtitle">
                    WAVES Intelligence™ • Adaptive Index Wave · Real-time console
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

with col_header_right:
    live_badge = """
        <div style="display:flex; gap:0.4rem; justify-content:flex-end; align-items:center;">
            <span class="badge badge-live">● Live Engine</span>
            <span class="badge badge-mode">{mode}</span>
            <span class="badge badge-beta">β Target {beta:.2f}</span>
        </div>
    """.format(mode=mode, beta=beta_target)
    st.markdown(live_badge, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # close header div

st.markdown("")

# ---------- KPI STRIP ----------
k1, k2, k3, k4 = st.columns(4)

cum = perf_summary.get("cum", np.nan)
daily = perf_summary.get("daily", np.nan)
max_dd = perf_summary.get("max_dd", np.nan)
alpha = perf_summary.get("alpha", np.nan)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Return (since inception)</div>
            <div class="kpi-value">{format_pct(cum)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    delta_class = "kpi-delta-pos" if (not pd.isna(daily) and daily >= 0) else "kpi-delta-neg"
    daily_str = format_pct(daily)
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Today</div>
            <div class="kpi-value">{daily_str}</div>
            <div class="{delta_class}">vs prior close</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Max Drawdown</div>
            <div class="kpi-value">{format_pct(max_dd)}</div>
            <div class="small-muted">Rolling since inception</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    alpha_str = format_pct(alpha)
    alpha_class = "kpi-delta-pos" if (not pd.isna(alpha) and alpha >= 0) else "kpi-delta-neg"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Alpha Captured</div>
            <div class="kpi-value">{alpha_str}</div>
            <div class="{alpha_class}">vs benchmark</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# ---------- TABS ----------
tab_overview, tab_alpha, tab_logs = st.tabs(["OVERVIEW", "ALPHA DASHBOARD", "ENGINE LOGS"])

# ------------------------------------------------------------
# TAB 1 — OVERVIEW
# ------------------------------------------------------------
with tab_overview:
    left, right = st.columns([2, 1.4])

    with left:
        st.markdown("#### Performance Curve")

        if perf_df is not None and not perf_df.empty:
            chart_df = perf_df.set_index("date")[["cumulative_return"]].copy()
            chart_df.rename(columns={"cumulative_return": "Wave"}, inplace=True)
            st.line_chart(chart_df)
        else:
            st.info("No performance history yet for this Wave.")

        st.markdown("#### Positions Snapshot")

        if pos_df is not None and not pos_df.empty:
            show_cols = []
            for col in ["Ticker", "Name", "Weight", "Value", "Sector"]:
                if col in pos_df.columns:
                    show_cols.append(col)
            if not show_cols:
                show_cols = pos_df.columns.tolist()

            df_view = pos_df.copy()
            if "Weight" in df_view.columns:
                df_view["Weight"] = df_view["Weight"].apply(lambda x: f"{x*100:,.2f}%" if pd.notna(x) else "—")

            st.dataframe(
                df_view[show_cols].sort_values(
                    by="Weight" if "Weight" in show_cols else show_cols[0],
                    ascending=False
                ),
                use_container_width=True,
                height=420,
            )
        else:
            st.info("No positions log found yet for this Wave.")

    with right:
        st.markdown("#### Exposure & Risk")

        st.metric("Equity Exposure", f"{exposure}%")
        st.metric("Cash Buffer", f"{100 - exposure}%")
        st.metric("β Target", f"{beta_target:.2f}")

        st.markdown("---")
        st.markdown("#### Benchmark Trace")

        if perf_df is not None and "benchmark_cumulative" in perf_df.columns:
            bench_df = perf_df.set_index("date")[["cumulative_return", "benchmark_cumulative"]].copy()
            bench_df.rename(
                columns={
                    "cumulative_return": selected_wave,
                    "benchmark_cumulative": "Benchmark",
                },
                inplace=True,
            )
            st.line_chart(bench_df)
        else:
            st.info("Benchmark series not available yet for this Wave.")

# ------------------------------------------------------------
# TAB 2 — ALPHA DASHBOARD
# ------------------------------------------------------------
with tab_alpha:
    st.markdown("#### Alpha Capture Timeline")

    if perf_df is not None and "benchmark_return" in perf_df.columns:
        alpha_daily = perf_df.copy()
        alpha_daily["alpha_daily"] = alpha_daily["daily_return"] - alpha_daily["benchmark_return"]
        alpha_daily.set_index("date", inplace=True)

        st.bar_chart(alpha_daily[["alpha_daily"]])

        st.markdown("#### Rolling Alpha (30-day)")

        alpha_daily["alpha_30d"] = alpha_daily["alpha_daily"].rolling(30).sum()
        st.line_chart(alpha_daily[["alpha_30d"]])

    else:
        st.info("Alpha-vs-benchmark series not yet available for this Wave.")

# ------------------------------------------------------------
# TAB 3 — ENGINE LOGS
# ------------------------------------------------------------
with tab_logs:
    st.markdown("#### Raw Engine Feeds")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Performance Feed (tail)**")
        if perf_df is not None and not perf_df.empty:
            st.dataframe(perf_df.tail(50), use_container_width=True, height=420)
        else:
            st.info("No performance feed found.")

    with c2:
        st.markdown("**Positions Feed (tail)**")
        if pos_df is not None and not pos_df.empty:
            st.dataframe(pos_df.tail(50), use_container_width=True, height=420)
        else:
            st.info("No positions feed found.")

    st.markdown("---")
    st.caption(
        "All data sourced from WAVES Engine logs · LIVE/SANDBOX status inferred from underlying data regime."
    )
