# app.py — WAVES Intelligence™ Institutional Console
#
# Features:
#   • Portfolio-Level Overview (All Waves)
#   • Multi-Window Alpha Capture (All Waves)
#   • Risk & WaveScore Ingredients (All Waves, 1Y)
#   • Recent Correction & Recovery (Last 60 Days, All Waves)  <-- NEW
#   • Benchmark ETF Mix table
#   • Wave Detail view:
#       - NAV chart (Wave vs benchmark)
#       - Performance vs benchmark (30D / 365D)
#       - Mode comparison (Standard / AMB / Private Logic)
#       - Top-10 holdings with Google Finance links
#
# Mobile-friendly: everything is internal, no terminal required.

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252


def _safe_pct(x: float) -> float:
    if x is None or np.isnan(x):
        return 0.0
    return float(x)


def compute_window_stats(
    nav_df: pd.DataFrame, days: int
) -> Tuple[float, float]:
    """
    Compute total returns over given window (Wave, Benchmark).
    nav_df: columns ['wave_nav', 'bm_nav'], index Date ascending.
    """
    if nav_df.empty or len(nav_df) < 2:
        return 0.0, 0.0

    nav = nav_df.iloc[-days:] if len(nav_df) > days else nav_df
    w_start, w_end = nav["wave_nav"].iloc[0], nav["wave_nav"].iloc[-1]
    b_start, b_end = nav["bm_nav"].iloc[0], nav["bm_nav"].iloc[-1]
    if w_start <= 0 or b_start <= 0:
        return 0.0, 0.0

    wave_ret = w_end / w_start - 1.0
    bm_ret = b_end / b_start - 1.0
    return float(wave_ret), float(bm_ret)


def compute_risk_stats(nav_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute 1Y risk stats for Risk & WaveScore table.
    Returns dict with:
        'wave_vol', 'bm_vol', 'wave_maxdd', 'bm_maxdd',
        'tracking_error', 'information_ratio'
    """
    if nav_df.empty:
        return {k: np.nan for k in [
            "wave_vol", "bm_vol", "wave_maxdd", "bm_maxdd",
            "tracking_error", "information_ratio"
        ]}

    wave_ret = nav_df["wave_ret"]
    bm_ret = nav_df["bm_ret"]

    # Vol (annualized)
    wave_vol = wave_ret.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    bm_vol = bm_ret.std() * math.sqrt(TRADING_DAYS_PER_YEAR)

    # Max drawdown
    wave_nav = nav_df["wave_nav"]
    bm_nav = nav_df["bm_nav"]

    wave_peak = wave_nav.cummax()
    wave_dd = wave_nav / wave_peak - 1.0
    wave_maxdd = wave_dd.min()

    bm_peak = bm_nav.cummax()
    bm_dd = bm_nav / bm_peak - 1.0
    bm_maxdd = bm_dd.min()

    # Tracking error & IR
    diff = wave_ret - bm_ret
    te = diff.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
    excess = (1.0 + wave_ret).prod() - (1.0 + bm_ret).prod()
    # Approx 1Y excess: convert to simple % over period
    if te > 0:
        ir = excess / te
    else:
        ir = np.nan

    return {
        "wave_vol": float(wave_vol),
        "bm_vol": float(bm_vol),
        "wave_maxdd": float(wave_maxdd),
        "bm_maxdd": float(bm_maxdd),
        "tracking_error": float(te),
        "information_ratio": float(ir),
    }


def compute_correction_cycle(nav_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute correction + recovery metrics for last 60 days:
        Peak→Trough (Wave / Benchmark)
        Trough→Now (Wave / Benchmark)
        Net Cycle Alpha (Trough→Now Wave - Benchmark)
    nav_df assumed to be last 60 days (or fewer), Date ascending.
    """
    if nav_df.empty or len(nav_df) < 5:
        return {k: np.nan for k in [
            "wave_p2t", "bm_p2t", "wave_t2n", "bm_t2n", "cycle_alpha"
        ]}

    wave_nav = nav_df["wave_nav"]
    bm_nav = nav_df["bm_nav"]

    # Wave peak->trough
    wave_peak_cum = wave_nav.cummax()
    wave_dd = wave_nav / wave_peak_cum - 1.0
    trough_idx = wave_dd.idxmin()
    # Peak is highest nav up to trough
    wave_peak_val = wave_nav.loc[:trough_idx].max()
    bm_peak_val = bm_nav.loc[:trough_idx].max()

    wave_trough_val = wave_nav.loc[trough_idx]
    bm_trough_val = bm_nav.loc[trough_idx]

    wave_last = wave_nav.iloc[-1]
    bm_last = bm_nav.iloc[-1]

    if wave_peak_val <= 0 or bm_peak_val <= 0:
        return {k: np.nan for k in [
            "wave_p2t", "bm_p2t", "wave_t2n", "bm_t2n", "cycle_alpha"
        ]}

    wave_p2t = wave_trough_val / wave_peak_val - 1.0
    bm_p2t = bm_trough_val / bm_peak_val - 1.0

    # Recovery from trough to now
    if wave_trough_val > 0:
        wave_t2n = wave_last / wave_trough_val - 1.0
    else:
        wave_t2n = np.nan

    if bm_trough_val > 0:
        bm_t2n = bm_last / bm_trough_val - 1.0
    else:
        bm_t2n = np.nan

    cycle_alpha = wave_t2n - bm_t2n if not (np.isnan(wave_t2n) or np.isnan(bm_t2n)) else np.nan

    return {
        "wave_p2t": float(wave_p2t),
        "bm_p2t": float(bm_p2t),
        "wave_t2n": float(wave_t2n),
        "bm_t2n": float(bm_t2n),
        "cycle_alpha": float(cycle_alpha),
    }


def format_pct(x: float) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{x*100:,.2f}%"


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Console")
st.caption("Vector OS · Dynamic APW/AIW Engine · Live Alpha & Risk")

# Sidebar controls
modes = we.get_modes()
default_mode_index = 0 if "Standard" not in modes else modes.index("Standard")
selected_mode = st.sidebar.selectbox(
    "Mode",
    options=modes,
    index=default_mode_index,
    help="View performance in Standard / Alpha-Minus-Beta / Private Logic modes.",
)

lookback_days_main = 365
lookback_days_short = 60

waves = we.get_all_waves()

# ------------------------------------------------------------
# Pre-compute NAV histories for all waves (365D + 60D)
# ------------------------------------------------------------

nav_cache_365: Dict[str, pd.DataFrame] = {}
nav_cache_60: Dict[str, pd.DataFrame] = {}

for w in waves:
    try:
        df_365 = we.compute_history_nav(w, mode=selected_mode, days=lookback_days_main)
    except Exception:
        df_365 = pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
    nav_cache_365[w] = df_365

    try:
        df_60 = we.compute_history_nav(w, mode=selected_mode, days=lookback_days_short)
    except Exception:
        df_60 = pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
    nav_cache_60[w] = df_60

# ------------------------------------------------------------
# Portfolio-Level Overview (All Waves)
# ------------------------------------------------------------

st.subheader("Portfolio-Level Overview (All Waves)")

overview_rows: List[Dict[str, object]] = []

for w in waves:
    df = nav_cache_365[w]
    if df.empty:
        overview_rows.append({
            "Wave": w,
            "30D Return": np.nan,
            "30D Alpha": np.nan,
            "365D Return": np.nan,
            "365D Alpha": np.nan,
        })
        continue

    wave_30, bm_30 = compute_window_stats(df, 30)
    wave_365, bm_365 = compute_window_stats(df, 365)

    overview_rows.append({
        "Wave": w,
        "30D Return": wave_30,
        "30D Alpha": wave_30 - bm_30,
        "365D Return": wave_365,
        "365D Alpha": wave_365 - bm_365,
    })

overview_df = pd.DataFrame(overview_rows)
if not overview_df.empty:
    display_df = overview_df.copy()
    for col in ["30D Return", "30D Alpha", "365D Return", "365D Alpha"]:
        display_df[col] = display_df[col].apply(format_pct)
    st.dataframe(display_df, use_container_width=True)

# ------------------------------------------------------------
# Multi-Window Alpha Capture (All Waves)
# ------------------------------------------------------------

st.subheader("Multi-Window Alpha Capture (All Waves)")

alpha_rows: List[Dict[str, object]] = []

for w in waves:
    df = nav_cache_365[w]
    if df.empty:
        alpha_rows.append({
            "Wave": w,
            "1D Return": np.nan,
            "1D Alpha": np.nan,
            "30D Return": np.nan,
            "30D Alpha": np.nan,
            "60D Return": np.nan,
            "60D Alpha": np.nan,
            "365D Return": np.nan,
            "365D Alpha": np.nan,
        })
        continue

    # 1D return vs benchmark (last day)
    wave_1d = df["wave_ret"].iloc[-1]
    bm_1d = df["bm_ret"].iloc[-1]

    wave_30, bm_30 = compute_window_stats(df, 30)
    wave_60, bm_60 = compute_window_stats(df, 60)
    wave_365, bm_365 = compute_window_stats(df, 365)

    alpha_rows.append({
        "Wave": w,
        "1D Return": wave_1d,
        "1D Alpha": wave_1d - bm_1d,
        "30D Return": wave_30,
        "30D Alpha": wave_30 - bm_30,
        "60D Return": wave_60,
        "60D Alpha": wave_60 - bm_60,
        "365D Return": wave_365,
        "365D Alpha": wave_365 - bm_365,
    })

alpha_df = pd.DataFrame(alpha_rows)
if not alpha_df.empty:
    display_alpha = alpha_df.copy()
    for col in [
        "1D Return", "1D Alpha",
        "30D Return", "30D Alpha",
        "60D Return", "60D Alpha",
        "365D Return", "365D Alpha",
    ]:
        display_alpha[col] = display_alpha[col].apply(format_pct)
    st.dataframe(display_alpha, use_container_width=True)

# ------------------------------------------------------------
# Risk & WaveScore Ingredients (All Waves, 1Y)
# ------------------------------------------------------------

st.subheader("Risk & WaveScore Ingredients (All Waves, 1Y (365D))")

risk_rows: List[Dict[str, object]] = []

for w in waves:
    df = nav_cache_365[w]
    if df.empty:
        risk_rows.append({
            "Wave": w,
            "Wave Vol (Ann.)": np.nan,
            "Benchmark Vol (Ann.)": np.nan,
            "Max Drawdown (Wave)": np.nan,
            "Max Drawdown (Benchmark)": np.nan,
            "Tracking Error": np.nan,
            "Information Ratio": np.nan,
        })
        continue

    stats = compute_risk_stats(df)
    risk_rows.append({
        "Wave": w,
        "Wave Vol (Ann.)": stats["wave_vol"],
        "Benchmark Vol (Ann.)": stats["bm_vol"],
        "Max Drawdown (Wave)": stats["wave_maxdd"],
        "Max Drawdown (Benchmark)": stats["bm_maxdd"],
        "Tracking Error": stats["tracking_error"],
        "Information Ratio": stats["information_ratio"],
    })

risk_df = pd.DataFrame(risk_rows)
if not risk_df.empty:
    display_risk = risk_df.copy()
    for col in [
        "Wave Vol (Ann.)",
        "Benchmark Vol (Ann.)",
        "Max Drawdown (Wave)",
        "Max Drawdown (Benchmark)",
        "Tracking Error",
    ]:
        display_risk[col] = display_risk[col].apply(format_pct)
    # IR is not percentage
    display_risk["Information Ratio"] = display_risk["Information Ratio"].apply(
        lambda x: "—" if np.isnan(x) else f"{x:,.2f}"
    )
    st.dataframe(display_risk, use_container_width=True)

# ------------------------------------------------------------
# Recent Correction & Recovery (Last 60 Days, All Waves)  <-- NEW
# ------------------------------------------------------------

st.subheader("Recent Correction & Recovery (Last 60 Days, All Waves)")

cycle_rows: List[Dict[str, object]] = []

for w in waves:
    df = nav_cache_60[w]
    if df.empty:
        cycle_rows.append({
            "Wave": w,
            "Peak→Trough (Wave)": np.nan,
            "Peak→Trough (Benchmark)": np.nan,
            "Trough→Now (Wave)": np.nan,
            "Trough→Now (Benchmark)": np.nan,
            "Net Cycle Alpha (T→Now)": np.nan,
        })
        continue

    stats = compute_correction_cycle(df)
    cycle_rows.append({
        "Wave": w,
        "Peak→Trough (Wave)": stats["wave_p2t"],
        "Peak→Trough (Benchmark)": stats["bm_p2t"],
        "Trough→Now (Wave)": stats["wave_t2n"],
        "Trough→Now (Benchmark)": stats["bm_t2n"],
        "Net Cycle Alpha (T→Now)": stats["cycle_alpha"],
    })

cycle_df = pd.DataFrame(cycle_rows)
if not cycle_df.empty:
    display_cycle = cycle_df.copy()
    for col in [
        "Peak→Trough (Wave)",
        "Peak→Trough (Benchmark)",
        "Trough→Now (Wave)",
        "Trough→Now (Benchmark)",
        "Net Cycle Alpha (T→Now)",
    ]:
        display_cycle[col] = display_cycle[col].apply(format_pct)
    st.dataframe(display_cycle, use_container_width=True)

# ------------------------------------------------------------
# Benchmark ETF Mix table
# ------------------------------------------------------------

st.subheader("Benchmark ETF Mix (Auto-Constructed)")

try:
    bm_mix = we.get_benchmark_mix_table()
except Exception:
    bm_mix = pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])

if not bm_mix.empty:
    bm_display = bm_mix.copy()
    bm_display["Weight"] = bm_display["Weight"].apply(format_pct)
    st.dataframe(bm_display, use_container_width=True)

# ------------------------------------------------------------
# Wave Detail View
# ------------------------------------------------------------

st.subheader("Wave Detail View")

col_left, col_right = st.columns([2, 1])

with col_left:
    selected_wave = st.selectbox("Select Wave", options=waves)

with col_left:
    df_wave = nav_cache_365.get(selected_wave, pd.DataFrame())
    if df_wave is None or df_wave.empty:
        st.info("No data available for this Wave.")
    else:
        chart_df = df_wave[["wave_nav", "bm_nav"]].copy()
        chart_df.columns = ["Wave NAV", "Benchmark NAV"]
        st.line_chart(chart_df, use_container_width=True)

with col_right:
    if df_wave is not None and not df_wave.empty:
        w_30, b_30 = compute_window_stats(df_wave, 30)
        w_365, b_365 = compute_window_stats(df_wave, 365)

        st.markdown("**Performance vs Benchmark**")
        perf_tbl = pd.DataFrame(
            {
                "Window": ["30D", "365D"],
                "Wave Return": [format_pct(w_30), format_pct(w_365)],
                "Benchmark Return": [format_pct(b_30), format_pct(b_365)],
                "Alpha": [format_pct(w_30 - b_30), format_pct(w_365 - b_365)],
            }
        )
        st.table(perf_tbl)

        # Mode comparison
        st.markdown("**Mode Comparison (365D)**")
        mode_rows = []
        for m in we.get_modes():
            try:
                df_m = we.compute_history_nav(selected_wave, mode=m, days=365)
            except Exception:
                df_m = pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
            if df_m.empty:
                mode_rows.append({
                    "Mode": m,
                    "365D Return": np.nan,
                    "365D Alpha": np.nan,
                })
                continue
            mw, mb = compute_window_stats(df_m, 365)
            mode_rows.append({
                "Mode": m,
                "365D Return": mw,
                "365D Alpha": mw - mb,
            })
        mode_df = pd.DataFrame(mode_rows)
        if not mode_df.empty:
            mode_display = mode_df.copy()
            mode_display["365D Return"] = mode_display["365D Return"].apply(format_pct)
            mode_display["365D Alpha"] = mode_display["365D Alpha"].apply(format_pct)
            st.table(mode_display)

with col_left:
    # Top-10 holdings with Google Finance links
    st.markdown("**Top-10 Holdings**")
    holdings_df = we.get_wave_holdings(selected_wave)
    if holdings_df is None or holdings_df.empty:
        st.info("No holdings defined for this Wave.")
    else:
        top10 = holdings_df.head(10).copy()
        link_col = []
        for _, row in top10.iterrows():
            ticker = row["Ticker"]
            # Simple Google Finance link
            url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
            link_col.append(f"[{ticker}]({url})")
        top10_display = pd.DataFrame(
            {
                "Ticker": link_col,
                "Name": top10["Name"],
                "Weight": top10["Weight"].apply(format_pct),
            }
        )
        st.table(top10_display)