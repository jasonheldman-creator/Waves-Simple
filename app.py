"""
app.py — WAVES Intelligence™ Console (Option B)

Front-end for the waves_engine v2.

Tabs:
  • Portfolio-Level Overview
  • Wave Detail

Features:
  • Auto-discovers Waves from wave_weights.csv
  • Overview table: NAV (last) and 365D return vs benchmark, per Wave
  • Wave Detail:
        - Mode selector (standard / alpha_minus_beta / private_logic)
        - History window (30 / 90 / 365 days)
        - Positions snapshot (weights, prices, dollar weights)
        - Performance history chart with Wave NAV vs Benchmark NAV
"""

import math
from typing import Dict, Any

import pandas as pd
import streamlit as st

from waves_engine import (
    get_available_waves,
    get_wave_snapshot,
    get_wave_history,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)

st.markdown(
    """
    <style>
    .small-text { font-size: 0.8rem; color: #888888; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:0.1f}%"


def _fmt_nav(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:0.3f}"


def _google_finance_link(ticker: str) -> str:
    if not isinstance(ticker, str) or not ticker:
        return ""
    url = f"https://www.google.com/finance/quote/{ticker}"
    return f"[{ticker}]({url})"


# ---------------------------------------------------------------------------
# Layout: Title & Tabs
# ---------------------------------------------------------------------------

st.title("WAVES Intelligence™ Console")
st.caption("Live engine: wave_weights.csv + yfinance-driven history")

tab_overview, tab_detail = st.tabs(["📊 Overview", "📈 Wave Detail"])


# ---------------------------------------------------------------------------
# Tab 1: Portfolio-Level Overview
# ---------------------------------------------------------------------------

with tab_overview:
    st.subheader("Portfolio-Level Overview")

    waves = get_available_waves()

    rows = []
    for wave in waves:
        try:
            hist = get_wave_history(
                wave_name=wave,
                mode="standard",
                lookback_days=365,
            )
            if hist.empty:
                nav_last = None
                ret_365 = None
                status = "no data"
            else:
                nav_last = float(hist["wave_nav"].iloc[-1])
                ret_365 = float(hist["cum_wave_return"].iloc[-1])
                status = ""
        except Exception as exc:
            nav_last = None
            ret_365 = None
            status = f"error: {type(exc).__name__}"

        rows.append(
            {
                "Wave": wave,
                "NAV (last)": nav_last,
                "365D Return %": ret_365,
                "Status": status,
            }
        )

    df_overview = pd.DataFrame(rows)

    # Pretty format for display
    df_display = df_overview.copy()
    df_display["NAV (last)"] = df_display["NAV (last)"].map(_fmt_nav)
    df_display["365D Return %"] = df_display["365D Return %"].map(_fmt_pct)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <p class="small-text">
        NAV is normalized to 1.0 at the start of the 365D window.
        Returns are cumulative over the selected window.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 2: Wave Detail
# ---------------------------------------------------------------------------

with tab_detail:
    st.subheader("Wave Detail")

    waves = get_available_waves()
    if not waves:
        st.error("No Waves found in wave_weights.csv")
        st.stop()

    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        selected_wave = st.selectbox("Select Wave", options=waves)

    with col2:
        mode = st.selectbox(
            "Mode",
            options=["standard", "alpha_minus_beta", "private_logic"],
            index=0,
            help="Engine currently implements modes as different exposure levels.",
        )

    with col3:
        window_label = st.radio(
            "History window",
            options=["30 days", "90 days", "365 days"],
            index=2,
            horizontal=True,
        )
        if window_label.startswith("30"):
            lookback_days = 30
        elif window_label.startswith("90"):
            lookback_days = 90
        else:
            lookback_days = 365

    # -------------------- Positions snapshot --------------------
    st.markdown("### Positions Snapshot")

    try:
        snap = get_wave_snapshot(selected_wave, mode=mode)
        pos_df = snap["positions"].copy()

        # Add Google Finance link
        pos_df.insert(
            0,
            "Quote",
            pos_df["ticker"].map(_google_finance_link),
        )

        # Formatting
        display_pos = pos_df.copy()
        display_pos["weight"] = display_pos["weight"].map(lambda x: f"{x:0.3f}")
        display_pos["eff_weight"] = display_pos["eff_weight"].map(
            lambda x: f"{x:0.3f}"
        )
        display_pos["price"] = display_pos["price"].map(
            lambda x: "—" if pd.isna(x) else f"${x:0.2f}"
        )
        display_pos["dollar_weight"] = display_pos["dollar_weight"].map(
            lambda x: f"${x:0.2f}"
        )

        st.dataframe(
            display_pos,
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Eff_weight reflects mode exposure (e.g., 0.80x for alpha_minus_beta)."
        )

    except Exception as exc:
        st.error(f"Error loading positions for '{selected_wave}': {exc}")

    # -------------------- Performance history --------------------
    st.markdown("### Performance History")

    try:
        hist = get_wave_history(
            wave_name=selected_wave,
            mode=mode,
            lookback_days=lookback_days,
        )
    except Exception as exc:
        st.error(
            f"Error loading history for '{selected_wave}' "
            f"({window_label}): {exc}"
        )
        hist = pd.DataFrame()

    if hist is not None and not hist.empty:
        # Chart NAV vs Benchmark NAV
        chart_df = pd.DataFrame(
            {
                "Wave NAV": hist["wave_nav"].values,
                "Benchmark NAV": hist["bench_nav"].values,
            },
            index=pd.to_datetime(hist["date"]),
        )
        st.line_chart(chart_df, use_container_width=True)

        # Summary metrics
        wave_cum = float(hist["cum_wave_return"].iloc[-1])
        bench_cum = float(hist["cum_bench_return"].iloc[-1])
        alpha_cum = float(hist["cum_alpha"].iloc[-1])

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Wave Cumulative Return", _fmt_pct(wave_cum))
        col_b.metric("Benchmark Cumulative Return", _fmt_pct(bench_cum))
        col_c.metric("Cumulative Alpha", _fmt_pct(alpha_cum))

        st.markdown(
            """
            <p class="small-text">
            Returns are cumulative over the selected window. NAV is normalized
            to 1.0 at the start of the window. Alpha is Wave minus benchmark.
            </p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No history data available for this Wave / window.")