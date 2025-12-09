# app.py
#
# WAVES Intelligence™ Institutional Console — Hard Reset Edition
#
# - On each run, calls run_engine_once_for_all_waves(), which wipes logs and rebuilds them.
# - Shows Wave details (returns + alpha + top 10 holdings).
# - Shows Alpha Capture matrix across waves & modes.
# - Shows Engine version + log tag + universe.

import streamlit as st
import pandas as pd
from datetime import datetime

from waves_engine import (
    ENGINE_VERSION,
    ENGINE_LOG_TAG,
    run_engine_once_for_all_waves,
    load_wave_universe,
    get_latest_perf_row,
    get_latest_positions_frame,
    build_alpha_matrix_for_mode,
)

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)


# ---------------------------------------------------------------------
# Run engine (HARD RESET each time)
# ---------------------------------------------------------------------

with st.spinner("Running WAVES Engine (hard reset + fresh logs)…"):
    run_engine_once_for_all_waves()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

st.sidebar.title("WAVES Intelligence™")
st.sidebar.caption("10-Wave Equity Lineup + SmartSafe™ — Hard Reset Engine")

waves_df, _ = load_wave_universe()
wave_names = waves_df["Wave"].tolist()

selected_wave = st.sidebar.selectbox("Select Wave", wave_names)

mode = st.sidebar.radio(
    "Mode",
    options=["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Engine Version**  \n{ENGINE_VERSION}")
st.sidebar.markdown(f"**Engine Log Tag:** `{ENGINE_LOG_TAG}`")
st.sidebar.markdown(
    f"_Last refreshed: {datetime.utcnow().isoformat(timespec='seconds')} UTC_"
)


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tab_wave, tab_alpha, tab_wavescore, tab_system = st.tabs(
    ["Wave Details", "Alpha Capture", "WaveScore Preview", "System Status"]
)

# ---------------------------------------------------------------------
# Wave Details tab
# ---------------------------------------------------------------------

with tab_wave:
    st.header(f"{selected_wave} — Wave Details")

    col1, col2 = st.columns([2, 1])

    # Performance & alpha
    with col1:
        perf = get_latest_perf_row(selected_wave)
        if perf is None:
            st.warning(
                f"No performance history yet for {selected_wave} "
                f"(no logs found for engine tag `{ENGINE_LOG_TAG}`)."
            )
        else:
            st.subheader("Performance Snapshot (Base Engine Metrics)")

            # Returns table
            ret_df = pd.DataFrame(
                {
                    "Metric": ["Return_1d", "Return_30d", "Return_60d", "Return_1y"],
                    "Value": [
                        perf.get("Return_1d", float("nan")),
                        perf.get("Return_30d", float("nan")),
                        perf.get("Return_60d", float("nan")),
                        perf.get("Return_1y", float("nan")),
                    ],
                }
            ).set_index("Metric")

            st.markdown("**Total Returns (unscaled)**")
            st.dataframe(
                (ret_df * 100).style.format("{:.2f}%"),
                use_container_width=True,
            )

            # Alpha table — scaled by mode
            if mode == "Standard":
                alpha_scale = 1.0
            elif mode == "Alpha-Minus-Beta":
                alpha_scale = 0.8
            else:
                alpha_scale = 1.3

            alpha_df = pd.DataFrame(
                {
                    "Metric": [
                        "Alpha_1d",
                        "Alpha_30d",
                        "Alpha_60d",
                        "Alpha_1y",
                        "Alpha_IR",
                    ],
                    "Value": [
                        perf.get("Alpha_1d", float("nan")) * alpha_scale,
                        perf.get("Alpha_30d", float("nan")) * alpha_scale,
                        perf.get("Alpha_60d", float("nan")) * alpha_scale,
                        perf.get("Alpha_1y", float("nan")) * alpha_scale,
                        perf.get("Alpha_IR", float("nan")),
                    ],
                }
            ).set_index("Metric")

            st.markdown(f"**Alpha (scaled for mode: {mode})**")
            st.dataframe(
                (alpha_df * 100).style.format("{:.2f}%"),
                use_container_width=True,
            )

    # Holdings
    with col2:
        st.subheader("Top 10 Holdings (Latest Snapshot)")
        pos = get_latest_positions_frame(selected_wave)
        if pos is None or pos.empty:
            st.info(
                f"No holdings available for {selected_wave} "
                f"(no positions log found for tag `{ENGINE_LOG_TAG}`)."
            )
        else:
            pos_sorted = pos.sort_values("Weight", ascending=False).head(10).copy()
            pos_sorted["Weight (%)"] = pos_sorted["Weight"] * 100.0
            display_cols = ["Ticker", "Weight (%)", "LastPrice", "GoogleFinanceURL"]
            st.dataframe(
                pos_sorted[display_cols]
                .rename(
                    columns={
                        "LastPrice": "Last Price",
                        "GoogleFinanceURL": "Google Finance",
                    }
                )
                .style.format({"Weight (%)": "{:.2f}%"}),
                use_container_width=True,
            )


# ---------------------------------------------------------------------
# Alpha Capture tab
# ---------------------------------------------------------------------

with tab_alpha:
    st.header(f"Alpha Capture Matrix — {mode}")

    matrix = build_alpha_matrix_for_mode(mode=mode)
    if matrix.empty:
        st.warning(
            "No alpha capture data yet for current engine tag. "
            "Check that the engine is running and logs exist."
        )
    else:
        display_df = matrix.copy()
        for c in [
            "Alpha_1d",
            "Alpha_30d",
            "Alpha_60d",
            "Alpha_1y",
            "Return_30d",
            "Return_60d",
            "Return_1y",
        ]:
            display_df[c] = display_df[c] * 100.0

        st.dataframe(
            display_df.style.format("{:.2f}%"),
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# WaveScore preview tab (simple approximation using 1y alpha + IR)
# ---------------------------------------------------------------------

with tab_wavescore:
    st.header("WaveScore™ Preview (Based on Engine Metrics)")

    matrix_std = build_alpha_matrix_for_mode(mode="Standard")
    if matrix_std.empty:
        st.info(
            "WaveScore preview requires Standard mode alpha metrics. "
            "No data available for current engine tag."
        )
    else:
        rows = []
        for _, row in matrix_std.iterrows():
            wave_name = row["Wave"]
            perf = get_latest_perf_row(wave_name)
            ir = perf.get("Alpha_IR", float("nan")) if perf is not None else float("nan")
            alpha_1y = row["Alpha_1y"]

            # Simple provisional score (0–100)
            score = 50 + alpha_1y * 200 + (0 if pd.isna(ir) else ir * 5)
            score = max(0, min(100, score))

            rows.append(
                {
                    "Wave": wave_name,
                    "Category": row["Category"],
                    "WaveScore_preview": score,
                    "Alpha_1y": alpha_1y,
                    "Alpha_IR": ir,
                }
            )

        score_df = pd.DataFrame(rows).sort_values(
            "WaveScore_preview", ascending=False
        )
        st.dataframe(
            score_df.style.format(
                {
                    "WaveScore_preview": "{:.1f}",
                    "Alpha_1y": "{:.2%}",
                    "Alpha_IR": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        st.caption(
            "WaveScore™ preview only — real WaveScore v1.0 uses the full locked spec "
            "with volatility, drawdown, consistency, and governance factors."
        )


# ---------------------------------------------------------------------
# System Status tab
# ---------------------------------------------------------------------

with tab_system:
    st.header("System Status — Engine & Universe")

    st.subheader("Engine")
    st.success("waves_engine module loaded — engine AVAILABLE.")
    st.info(ENGINE_VERSION)
    st.code(f"ENGINE_LOG_TAG = '{ENGINE_LOG_TAG}'")

    st.subheader("Wave Universe (from list.csv)")
    st.dataframe(
        waves_df[["Wave", "Category", "Benchmark"]],
        use_container_width=True,
    )

    st.markdown(
        "On every run, the engine **deletes all prior logs** and rebuilds fresh "
        "positions and performance for each Wave using the latest prices.\n\n"
        "The console only uses logs whose filenames include the current "
        "`ENGINE_LOG_TAG`, so older engine versions are never read."
    )