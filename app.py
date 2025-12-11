"""
app.py — WAVES Intelligence™ Institutional Console (History-Aware)

Features
--------
• Discovers Waves from wave_weights.csv.
• Uses waves_engine.get_wave_snapshot() for all analytics.
• Uses Full_Wave_History.csv (auto-built by build_full_wave_history.py).
• Shows:
    - Overview table: returns & alpha for all Waves
    - Detail view per Wave with metrics, chart, and positions
    - Mode selector: Standard / Alpha-Minus-Beta / Private Logic™

This file is designed to be a clean, indentation-safe drop-in for Streamlit.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from waves_engine import (
    get_available_waves,
    get_wave_snapshot,
    get_wave_history,
)

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)

st.title("🌊 WAVES Intelligence™ Console")
st.caption("History-aware, benchmark-aware, alpha-oriented analytics.")


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_wave_list() -> list[str]:
    return get_available_waves()


waves = _cached_wave_list()
if not waves:
    st.error("No Waves found in wave_weights.csv")
    st.stop()

with st.sidebar:
    st.header("Wave Controls")

    selected_wave = st.selectbox("Select Wave", waves, index=0)

    mode_label = st.radio(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    mode_map = {
        "Standard": "standard",
        "Alpha-Minus-Beta": "amb",
        "Private Logic™": "pl",
    }
    selected_mode = mode_map[mode_label]

    st.markdown("---")
    st.caption("Tip: metrics use historical daily returns from Full_Wave_History.csv.")


# ---------------------------------------------------------------------
# Helper: cached snapshots
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_snapshot(wave: str, mode: str) -> dict:
    return get_wave_snapshot(wave, mode)


@st.cache_data(show_spinner=False)
def _cached_all_snapshots(mode: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for w in waves:
        out[w] = get_wave_snapshot(w, mode)
    return out


# ---------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------


tab_overview, tab_detail = st.tabs(["📊 Overview", "📈 Wave Detail"])

with tab_overview:
    st.subheader("Portfolio-Level Overview")

    snapshots = _cached_all_snapshots("standard")  # overview always uses standard

    rows = []
    for w, snap in snapshots.items():
        m = snap["metrics"]
        rows.append(
            {
                "Wave": w,
                "30D Return": m["ret_30d"],
                "30D Alpha": m["alpha_30d"],
                "60D Return": m["ret_60d"],
                "60D Alpha": m["alpha_60d"],
                "1Y Return": m["ret_1y"],
                "1Y Alpha": m["alpha_1y"],
                "SI Return": m["ret_si"],
                "SI Alpha": m["alpha_si"],
                "Vol (1Y)": m["vol_1y"],
                "Max Drawdown": m["maxdd"],
                "Beta (1Y)": m["beta_1y"],
                "Info Ratio (1Y)": m["info_ratio_1y"],
                "Hit Rate (1Y)": m["hit_rate_1y"],
            }
        )

    df_overview = pd.DataFrame(rows)
    df_overview = df_overview.set_index("Wave")

    # Sort by 1Y Alpha descending
    df_overview = df_overview.sort_values("1Y Alpha", ascending=False)

    st.dataframe(
        df_overview.style.format(
            {
                "30D Return": "{:.2%}",
                "30D Alpha": "{:.2%}",
                "60D Return": "{:.2%}",
                "60D Alpha": "{:.2%}",
                "1Y Return": "{:.2%}",
                "1Y Alpha": "{:.2%}",
                "SI Return": "{:.2%}",
                "SI Alpha": "{:.2%}",
                "Vol (1Y)": "{:.2%}",
                "Max Drawdown": "{:.2%}",
                "Beta (1Y)": "{:.2f}",
                "Info Ratio (1Y)": "{:.2f}",
                "Hit Rate (1Y)": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

    st.markdown(
        "_Note: Returns and alpha are computed from daily history; "
        "alpha is vs each Wave's custom benchmark as defined in build_full_wave_history.py._"
    )


# ---------------------------------------------------------------------
# Detail tab
# ---------------------------------------------------------------------


with tab_detail:
    st.subheader(f"Wave Detail — {selected_wave} ({mode_label})")

    snap = _cached_snapshot(selected_wave, selected_mode)
    m = snap["metrics"]
    positions_df = snap["positions"]
    history_df = snap["history"]

    # --- Top metrics row ---
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("30D Return", f"{m['ret_30d']:.2%}", f"{m['alpha_30d']:.2%} alpha")
    col2.metric("60D Return", f"{m['ret_60d']:.2%}", f"{m['alpha_60d']:.2%} alpha")
    col3.metric("1Y Return", f"{m['ret_1y']:.2%}", f"{m['alpha_1y']:.2%} alpha")
    col4.metric("Since Inception", f"{m['ret_si']:.2%}", f"{m['alpha_si']:.2%} alpha")

    st.markdown("### Risk & Quality")

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    col_r1.metric("Volatility (1Y)", f"{m['vol_1y']:.2%}")
    col_r2.metric("Max Drawdown", f"{m['maxdd']:.2%}")
    col_r3.metric("Beta (1Y)", f"{m['beta_1y']:.2f}")
    col_r4.metric("Info Ratio (1Y)", f"{m['info_ratio_1y']:.2f}")

    col_r5, col_r6 = st.columns(2)
    col_r5.metric("Hit Rate (1Y)", f"{m['hit_rate_1y']:.1%}")
    col_r6.metric(
        "SmartSafe",
        f"{m['smartsafe_state']} ({m['smartsafe_sweep']:.0%} sweep)",
        f"VIX {m['vix_level']:.1f}",
    )

    # --- Chart: NAV vs Benchmark NAV ---
    st.markdown("### Performance vs Benchmark")

    chart_df = history_df.copy()
    chart_df = chart_df.sort_values("Date").set_index("Date")

    # Rebuild benchmark NAV from returns
    chart_df["WaveNAV"] = chart_df["NAV"]
    chart_df["BenchNAV"] = (1.0 + chart_df["BenchReturn"]).cumprod()

    chart_to_plot = chart_df[["WaveNAV", "BenchNAV"]].rename(
        columns={"WaveNAV": "Wave NAV", "BenchNAV": "Benchmark NAV"}
    )

    st.line_chart(chart_to_plot, use_container_width=True)

    # --- Positions table ---
    st.markdown("### Current Positions (Top-Down)")

    if not positions_df.empty:
        # Format weights as %
        pos_df = positions_df.copy()
        pos_df["Weight"] = pos_df["Weight"].astype(float)
        pos_df["Price"] = pos_df["Price"].astype(float)
        pos_df["MarketValue"] = pos_df["MarketValue"].astype(float)

        st.dataframe(
            pos_df.style.format(
                {
                    "Weight": "{:.2%}",
                    "Price": "{:.2f}",
                    "MarketValue": "{:.4f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("No positions data available for this Wave.")

    st.markdown(
        "_Note: Prices & market values are taken from the latest date in "
        "Full_Wave_History.csv for this Wave._"
    )