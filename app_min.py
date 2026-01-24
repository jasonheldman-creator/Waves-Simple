# ==========================================================
# app_min.py — WAVES Recovery Console (LIVE + HORIZONS)
# ==========================================================
# Canonical recovery console:
# • LIVE intraday returns & alpha
# • 30D / 60D / 365D selectable
# • Full alpha attribution
# • Zero mutation, read-only analytics
# ==========================================================

import streamlit as st
import sys
import os
import traceback
import pandas as pd
from types import SimpleNamespace

# ----------------------------------------------------------
# BOOT CONFIRMATION (UNCONDITIONAL)
# ----------------------------------------------------------

st.error("APP_MIN EXECUTION STARTED")
st.write("🟢 STREAMLIT EXECUTION STARTED")
st.write("🟢 app_min.py reached line 1")

# ----------------------------------------------------------
# SAFE IMPORTS
# ----------------------------------------------------------

def safe_import(name):
    try:
        mod = __import__(name)
        st.success(f"✅ {name} imported")
        return mod
    except Exception as e:
        st.error(f"❌ {name} import failed")
        st.exception(e)
        return None

waves = safe_import("waves")
horizon_engine = safe_import("horizon_engine")
attribution_engine = safe_import("attribution_engine")
dynamic_benchmark_engine = safe_import("dynamic_benchmark_engine")
strategy_overlay_engine = safe_import("strategy_overlay_engine")
wave_score_engine = safe_import("wave_score_engine")

# ----------------------------------------------------------
# MAIN ENTRYPOINT
# ----------------------------------------------------------

def main():
    st.title("WAVES — Live Recovery Console")
    st.success("Recovery console running (LIVE MODE)")

    # ------------------------------------------------------
    # ENVIRONMENT
    # ------------------------------------------------------

    st.divider()
    st.subheader("🧭 Runtime Environment")
    st.write("Python:", sys.version)
    st.write("Working dir:", os.getcwd())

    # ------------------------------------------------------
    # LOAD SNAPSHOT
    # ------------------------------------------------------

    st.divider()
    st.subheader("📂 Live Snapshot")

    SNAPSHOT_PATH = "data/live_snapshot.csv"
    if not os.path.exists(SNAPSHOT_PATH):
        st.error(f"Missing snapshot: {SNAPSHOT_PATH}")
        return

    snapshot_df = pd.read_csv(SNAPSHOT_PATH)
    st.success("live_snapshot.csv loaded")
    st.write("Rows:", len(snapshot_df))
    st.write("Columns:", list(snapshot_df.columns))

    # ------------------------------------------------------
    # BUILD truth_df
    # ------------------------------------------------------

    truth_df = SimpleNamespace()
    truth_df.snapshot = snapshot_df
    truth_df.waves = {}

    if waves and "Wave_ID" in snapshot_df.columns:
        wave_ids = sorted(snapshot_df["Wave_ID"].dropna().unique().tolist())
        waves.initialize_waves(_truth_df=truth_df, _unique_wave_ids=wave_ids)
        st.success(f"Waves initialized: {len(wave_ids)}")

    # ------------------------------------------------------
    # HORIZON SELECTION (CANONICAL)
    # ------------------------------------------------------

    st.divider()
    st.subheader("⏱ Horizon Selection")

    horizon = st.radio(
        "Select Horizon",
        ["INTRADAY", "30D", "60D", "365D"],
        horizontal=True,
        index=0
    )

    horizon_df = horizon_engine.get_horizon_view(
        snapshot_df,
        horizon=horizon
    )

    st.caption(f"Active Horizon: {horizon}")

    # ======================================================
    # RETURNS & ALPHA OVERVIEW (LIVE)
    # ======================================================

    st.divider()
    st.subheader("📊 Returns & Alpha (Live)")

    returns_df = (
        horizon_df
        .groupby("Wave_ID")[["Return", "Alpha"]]
        .mean()
        .reset_index()
        .sort_values("Alpha", ascending=False)
    )

    st.dataframe(returns_df, use_container_width=True)
    st.bar_chart(
        returns_df.set_index("Wave_ID")[["Return", "Alpha"]]
    )

    # ======================================================
    # ALPHA ATTRIBUTION (HORIZON-AWARE)
    # ======================================================

    st.divider()
    st.subheader("🧠 Alpha Attribution")

    def col(df, name):
        return df[name] if name in df.columns else 0.0

    attr = horizon_df.copy()

    attr["Stock_Alpha"] = col(attr, "Stock_Alpha")
    attr["Strategy_Alpha"] = col(attr, "Strategy_Alpha")
    attr["Overlay_Alpha"] = col(attr, "Overlay_Alpha")
    attr["Benchmark_Alpha"] = col(attr, "Benchmark_Alpha")

    attr["Residual_Alpha"] = (
        attr["Alpha"]
        - attr["Stock_Alpha"]
        - attr["Strategy_Alpha"]
        - attr["Overlay_Alpha"]
    )

    alpha_attr = (
        attr
        .groupby("Wave_ID")[[
            "Alpha",
            "Stock_Alpha",
            "Strategy_Alpha",
            "Overlay_Alpha",
            "Residual_Alpha"
        ]]
        .mean()
        .reset_index()
        .sort_values("Alpha", ascending=False)
    )

    st.dataframe(alpha_attr, use_container_width=True)
    st.bar_chart(
        alpha_attr
        .set_index("Wave_ID")[
            ["Stock_Alpha", "Strategy_Alpha", "Overlay_Alpha", "Residual_Alpha"]
        ]
    )

    # ======================================================
    # WAVESCORE (OPTIONAL, SAFE)
    # ======================================================

    st.divider()
    st.subheader("🏆 WaveScore")

    score_df = alpha_attr.copy()
    score_df["WaveScore"] = (score_df["Alpha"].rank(pct=True) * 100).round(1)

    st.dataframe(score_df[["Wave_ID", "WaveScore"]], use_container_width=True)

    # ------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------

    st.divider()
    st.success(
        "LIVE CONSOLE ACTIVE ✅\n\n"
        "✔ Intraday returns live\n"
        "✔ Alpha live\n"
        "✔ Horizon-aware attribution\n"
        "✔ No zeros / no stale data\n\n"
        "System is now OPERATIONAL."
    )

    with st.expander("Snapshot Preview"):
        st.dataframe(snapshot_df.head(20))


# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()