"""
app.py — WAVES Intelligence™ Institutional Console (Hybrid Engine)

Front-end on top of waves_engine hybrid backend.

Features
--------
• Overview tab
    - Discovers Waves from wave_weights.csv
    - Shows NAV (last), 365D return, benchmark 365D, alpha 365D, status

• Wave Detail tab
    - Wave selector
    - Mode selector (placeholder, engine-ready)
    - History window: 30 / 90 / 365 / 2y / 5y
    - Positions snapshot (weights, prices, dollar_weight)
    - Google Finance quote links
    - Performance metrics table:
        * Total return
        * Benchmark return
        * Alpha
        * Ann. volatility
        * Sharpe
        * Max drawdown
    - Wave NAV vs Benchmark NAV chart
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from waves_engine import (
    get_available_waves,
    get_wave_snapshot,
    get_wave_history,
    compute_risk_stats,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Console")
st.caption("Hybrid engine: Full_Wave_History.csv + yfinance live prices")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{x * 100.0:0.2f}%"
    except Exception:
        return "—"


def _fmt_pct_direct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{x:0.2f}%"
    except Exception:
        return "—"


def _fmt_nav(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{x:0.3f}"
    except Exception:
        return "—"


def _google_link(ticker: str) -> str:
    if not isinstance(ticker, str) or not ticker:
        return ""
    url = f"https://www.google.com/finance/quote/{ticker}"
    return f"[{ticker}]({url})"


# ---------------------------------------------------------------------------
# Layout: tabs
# ---------------------------------------------------------------------------

tab_overview, tab_detail = st.tabs(["📊 Overview", "📈 Wave Detail"])


# ---------------------------------------------------------------------------
# 1) Portfolio Overview
# ---------------------------------------------------------------------------

with tab_overview:
    st.subheader("Portfolio-Level Overview")

    try:
        waves = get_available_waves()
    except Exception as e:
        st.error(f"Error loading Waves from wave_weights.csv: {e}")
        st.stop()

    rows = []
    lookback_days = 365

    for w in waves:
        nav_last = np.nan
        total_ret = np.nan
        bench_ret = np.nan
        alpha_ret = np.nan
        status = "ok"

        try:
            hist = get_wave_history(wave_name=w, lookback_days=lookback_days)
            stats = compute_risk_stats(hist)

            if not hist.empty:
                nav_last = float(hist["wave_nav"].iloc[-1])
            total_ret = stats["total_return"]
            bench_ret = stats["bench_return"]
            alpha_ret = stats["alpha_total"]
        except Exception as e:
            status = f"error: {type(e).__name__}"

        rows.append(
            {
                "Wave": w,
                "NAV (last)": nav_last,
                "365D Return %": total_ret * 100.0 if not pd.isna(total_ret) else np.nan,
                "Benchmark 365D %": bench_ret * 100.0 if not pd.isna(bench_ret) else np.nan,
                "Alpha 365D %": alpha_ret * 100.0 if not pd.isna(alpha_ret) else np.nan,
                "Status": status,
            }
        )

    if rows:
        df = pd.DataFrame(rows)

        display_df = df.copy()
        display_df["NAV (last)"] = display_df["NAV (last)"].map(_fmt_nav)
        display_df["365D Return %"] = display_df["365D Return %"].map(_fmt_pct_direct)
        display_df["Benchmark 365D %"] = display_df["Benchmark 365D %"].map(
            _fmt_pct_direct
        )
        display_df["Alpha 365D %"] = display_df["Alpha 365D %"].map(_fmt_pct_direct)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.caption(
            "NAV is normalized to 1.0 at the start of the 365D window. "
            "Returns are cumulative over the selected window. "
            "History comes from Full_Wave_History.csv when available, "
            "otherwise from live yfinance prices."
        )
    else:
        st.info("No Waves found in wave_weights.csv.")


# ---------------------------------------------------------------------------
# 2) Wave Detail
# ---------------------------------------------------------------------------

with tab_detail:
    st.subheader("Wave Detail")

    waves = get_available_waves()
    if not waves:
        st.warning("No Waves available.")
        st.stop()

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        wave_name = st.selectbox("Select Wave", options=waves, index=0)

    with col2:
        mode = st.selectbox(
            "Mode (engine-ready; currently cosmetic)",
            options=["standard", "alpha_minus_beta", "private_logic"],
            index=0,
        )

    with col3:
        window_label = st.radio(
            "History window",
            options=["30 days", "90 days", "365 days", "2 years", "5 years"],
            index=2,
            horizontal=True,
        )
        window_days_map = {
            "30 days": 30,
            "90 days": 90,
            "365 days": 365,
            "2 years": 730,
            "5 years": 365 * 5,
        }
        lookback_days = window_days_map[window_label]

    # ---------------- Positions Snapshot ----------------
    st.markdown("### Positions Snapshot")

    try:
        snap = get_wave_snapshot(wave_name, mode=mode)
        pos = snap["positions"].copy()
    except Exception as e:
        st.error(f"Error loading snapshot for {wave_name}: {e}")
        pos = pd.DataFrame()

    if not pos.empty:
        pos_display = pos.copy()
        pos_display["weight"] = pos_display["weight"].map(
            lambda x: f"{float(x):0.3f}" if pd.notna(x) else "—"
        )
        if "price" in pos_display.columns:
            pos_display["price"] = pos_display["price"].map(
                lambda x: f"${float(x):0.2f}" if pd.notna(x) else "—"
            )
        if "dollar_weight" in pos_display.columns:
            pos_display["dollar_weight"] = pos_display["dollar_weight"].map(
                lambda x: f"{float(x):0.3f}" if pd.notna(x) else "—"
            )

        # Add quote links
        pos_display.insert(
            0,
            "Quote",
            pos_display["ticker"].map(_google_link),
        )

        st.dataframe(pos_display, use_container_width=True, hide_index=True)
        st.caption(
            f"As of {snap.get('as_of')} · Prices via yfinance; missing quotes appear as '—'."
        )
    else:
        st.info("No positions available for this Wave.")

    # ---------------- Performance History ----------------
    st.markdown("### Performance History")

    try:
        hist = get_wave_history(
            wave_name=wave_name,
            mode=mode,
            lookback_days=lookback_days,
        )
    except Exception as e:
        st.error(f"Error loading history for {wave_name}: {e}")
        hist = pd.DataFrame()

    if hist is not None and not hist.empty:
        # Ensure date is datetime index
        h = hist.copy()
        h["date"] = pd.to_datetime(h["date"])
        h = h.sort_values("date")

        # Chart: Wave NAV vs Benchmark NAV
        chart_df = pd.DataFrame(
            {
                "Wave NAV": h["wave_nav"].values,
                "Benchmark NAV": h["bench_nav"].values,
            },
            index=h["date"],
        )
        st.line_chart(chart_df, use_container_width=True)

        # Metrics
        stats = compute_risk_stats(h)

        metrics_df = pd.DataFrame(
            {
                "Metric": [
                    "Window",
                    "Total Return",
                    "Benchmark Return",
                    "Alpha (Total)",
                    "Ann. Volatility",
                    "Sharpe vs Benchmark",
                    "Max Drawdown",
                ],
                "Value": [
                    window_label,
                    _fmt_pct(stats["total_return"]),
                    _fmt_pct(stats["bench_return"]),
                    _fmt_pct(stats["alpha_total"]),
                    _fmt_pct(stats["ann_vol"]),
                    f"{stats['sharpe']:0.2f}"
                    if not pd.isna(stats["sharpe"])
                    else "—",
                    _fmt_pct(stats["max_drawdown"]),
                ],
            }
        )

        st.table(metrics_df)

        st.caption(
            "NAV series are normalized to 1.0 at the start of the selected window. "
            "Alpha is Wave minus benchmark."
        )
    else:
        st.info("No history data available for this Wave / window.")