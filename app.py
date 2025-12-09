"""
WAVES Intelligence™ Institutional Console – Streamlit UI
Uses the rebuilt waves_engine.run_full_engine()
"""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import streamlit as st

from waves_engine import ENGINE_VERSION, run_full_engine

# ---------------------------------------------------------------------
# Page config & cache clearing
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence – Institutional Console",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clear Streamlit caches at startup so ONLY latest code/data are used
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    # Older Streamlit versions may not have these; ignore
    pass

# ---------------------------------------------------------------------
# Load engine results
# ---------------------------------------------------------------------

with st.spinner("Starting WAVES Engine and computing live metrics…"):
    try:
        engine_result = run_full_engine()
    except Exception as e:
        st.error("Engine failed to start. See details below.")
        st.exception(e)
        st.stop()

# Convenience handles
wave_list = engine_result.wave_list
alpha_capture = engine_result.alpha_capture  # dict of mode -> df
top_holdings = engine_result.top_holdings
system_status = engine_result.system_status

# ---------------------------------------------------------------------
# Layout: header
# ---------------------------------------------------------------------

st.markdown(
    f"""
    <h1 style="margin-bottom:0.2rem;">WAVES Intelligence™ – Institutional Console</h1>
    <p style="color:#AAAAAA;margin-top:0;">
        Engine version <b>{ENGINE_VERSION}</b> • As of <b>{engine_result.as_of:%Y-%m-%d %H:%M} UTC</b>
    </p>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Wave Details", "Alpha Capture", "WaveScore", "System Status"])

# ---------------------------------------------------------------------
# Tab 1 – Wave Details
# ---------------------------------------------------------------------

with tabs[0]:
    st.subheader("Wave Lineup")

    st.dataframe(
        wave_list.rename(
            columns={
                "wave": "Wave",
                "category": "Category",
                "benchmark": "Benchmark",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Top 10 Holdings per Wave")

    selected_wave = st.selectbox(
        "Select Wave to view holdings",
        options=["All Waves"] + sorted(wave_list["wave"].tolist()),
        index=0,
    )

    holdings_df = top_holdings.copy()
    # Convert weights to % for nicer display
    holdings_df["Weight_%"] = (holdings_df["Weight"] * 100.0).round(2)

    if selected_wave != "All Waves":
        holdings_df = holdings_df[holdings_df["Wave"] == selected_wave]

    st.dataframe(
        holdings_df[["Wave", "Ticker", "Weight_%", "Google_Finance_URL"]],
        use_container_width=True,
    )

# ---------------------------------------------------------------------
# Tab 2 – Alpha Capture
# ---------------------------------------------------------------------

def _format_alpha_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "Alpha_1d",
        "Alpha_30d",
        "Alpha_60d",
        "Alpha_1y",
        "Return_1d",
        "Return_30d",
        "Return_60d",
        "Return_1y",
    ]:
        if col in df.columns:
            df[col] = (df[col] * 100.0).round(2)
    return df


with tabs[1]:
    st.subheader("Alpha Capture Matrix")

    mode = st.radio(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic"],
        horizontal=True,
    )

    df_mode = alpha_capture.get(mode, pd.DataFrame())
    if df_mode.empty:
        st.warning("No alpha data available for this mode.")
    else:
        df_display = _format_alpha_df(df_mode)
        # Reorder columns for readability
        col_order = [
            "Wave",
            "Category",
            "Benchmark",
            "Alpha_1d",
            "Alpha_30d",
            "Alpha_60d",
            "Alpha_1y",
            "Return_1d",
            "Return_30d",
            "Return_60d",
            "Return_1y",
        ]
        df_display = df_display[[c for c in col_order if c in df_display.columns]]

        st.dataframe(df_display, use_container_width=True)

# ---------------------------------------------------------------------
# Tab 3 – WaveScore (simple placeholder based on alpha)
# ---------------------------------------------------------------------

with tabs[2]:
    st.subheader("WaveScore™ – Prototype")

    # Use Standard mode as base scoring input
    base = alpha_capture.get("Standard", pd.DataFrame()).copy()
    if base.empty:
        st.info("WaveScore prototype requires Standard alpha data.")
    else:
        df = base.copy()

        # Simple prototype: focus on 1y alpha & 60d alpha
        for col in ["Alpha_60d", "Alpha_1y"]:
            if col not in df.columns:
                df[col] = np.nan

        # Raw numeric scores (0–100 scaled by alpha)
        df["WaveScore"] = (
            (df["Alpha_1y"].fillna(0) * 100.0)
            + (df["Alpha_60d"].fillna(0) * 50.0)
        )

        # Normalize to 0–100 band
        if df["WaveScore"].abs().max() > 0:
            max_abs = df["WaveScore"].abs().max()
            df["WaveScore"] = 50 + 50 * (df["WaveScore"] / max_abs)

        df["WaveScore"] = df["WaveScore"].clip(0, 100).round(1)

        df_display = df[["Wave", "Category", "Benchmark", "Alpha_60d", "Alpha_1y", "WaveScore"]].copy()
        df_display["Alpha_60d"] = (df_display["Alpha_60d"] * 100.0).round(2)
        df_display["Alpha_1y"] = (df_display["Alpha_1y"] * 100.0).round(2)

        st.dataframe(
            df_display.sort_values("WaveScore", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

        st.caption(
            "Prototype WaveScore™: scaled off 60-day and 1-year alpha. "
            "Final spec is governed by locked WAVESCORE™ v1.0."
        )

# ---------------------------------------------------------------------
# Tab 4 – System Status
# ---------------------------------------------------------------------

with tabs[3]:
    st.subheader("System Status — Engine & Data Health")

    st.success("waves_engine module loaded — engine AVAILABLE.")

    st.markdown("### Engine")
    st.json(system_status)

    st.markdown("### Controls")

    if st.button("Force hard refresh (clear cache & rerun)"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception:
            pass
        st.experimental_rerun()

    st.markdown("---")
    st.caption(
        "This console is a live simulation using market data via yfinance. "
        "Results are for research & demonstration only and do not represent "
        "actual trading or investment performance."
    )