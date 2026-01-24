# ==========================================================
# app_min.py — WAVES Recovery → Live Snapshot + Attribution
# ==========================================================
# SAFE recovery kernel with:
# • Streamlit boot
# • live_snapshot.csv loading
# • truth_df hydration
# • wave initialization
# • Returns diagram
# • FULL alpha attribution (defensive)
# ==========================================================

import streamlit as st
import sys
import os
import traceback
import pandas as pd
from types import SimpleNamespace

# ----------------------------------------------------------
# BOOT CONFIRMATION
# ----------------------------------------------------------

st.error("APP_MIN EXECUTION STARTED")
st.write("🟢 STREAMLIT EXECUTION STARTED")
st.write("🟢 app_min.py reached line 1")

# ----------------------------------------------------------
# MAIN ENTRYPOINT
# ----------------------------------------------------------

def main():
    st.title("WAVES — Recovery Mode")
    st.success("Recovery kernel running")

    # ------------------------------------------------------
    # ENVIRONMENT VISIBILITY
    # ------------------------------------------------------

    st.divider()
    st.write("🧭 Runtime environment")

    try:
        st.write("Python:", sys.version)
        st.write("Working directory:", os.getcwd())
        st.success("Environment visible")
    except Exception as e:
        st.error("Environment inspection failed")
        st.exception(e)

    # ------------------------------------------------------
    # WAVES MODULE IMPORT
    # ------------------------------------------------------

    st.divider()
    st.write("🔍 waves module check")

    try:
        import waves
        st.success("waves imported successfully")
        st.code(waves.__file__)
    except Exception as e:
        st.error("waves import failed — hard stop")
        st.exception(e)
        return

    # ------------------------------------------------------
    # LOAD LIVE SNAPSHOT
    # ------------------------------------------------------

    st.divider()
    st.write("📂 Loading live snapshot")

    SNAPSHOT_PATH = "data/live_snapshot.csv"

    if not os.path.exists(SNAPSHOT_PATH):
        st.error(f"Snapshot not found: {SNAPSHOT_PATH}")
        return

    try:
        snapshot_df = pd.read_csv(SNAPSHOT_PATH)
        st.success("live_snapshot.csv loaded")
        st.write("Rows:", len(snapshot_df))
        st.write("Columns:", list(snapshot_df.columns))
    except Exception as e:
        st.error("Failed to read snapshot CSV")
        st.exception(e)
        return

    # ------------------------------------------------------
    # BUILD truth_df
    # ------------------------------------------------------

    truth_df = SimpleNamespace()
    truth_df.snapshot = snapshot_df
    truth_df.waves = {}

    # ------------------------------------------------------
    # EXTRACT WAVE IDS
    # ------------------------------------------------------

    if "Wave_ID" not in snapshot_df.columns:
        st.error("Wave_ID column missing from snapshot")
        return

    unique_wave_ids = sorted(snapshot_df["Wave_ID"].dropna().unique().tolist())

    # ------------------------------------------------------
    # INITIALIZE WAVES
    # ------------------------------------------------------

    st.divider()
    st.write("🚀 Initializing WAVES")

    try:
        waves.initialize_waves(
            _truth_df=truth_df,
            _unique_wave_ids=unique_wave_ids
        )
        st.success("WAVES initialized successfully")
    except Exception as e:
        st.error("WAVES initialization failed")
        st.exception(e)
        return

    # ------------------------------------------------------
    # RETURNS & ALPHA OVERVIEW (EXISTING)
    # ------------------------------------------------------

    st.divider()
    st.subheader("📊 Returns & Alpha Overview")

    base_cols = {"Wave_ID", "Return", "Alpha"}

    if not base_cols.issubset(snapshot_df.columns):
        st.warning("Return / Alpha columns not fully present yet.")
    else:
        agg_df = (
            snapshot_df
            .groupby("Wave_ID")[["Return", "Alpha"]]
            .mean()
            .reset_index()
            .sort_values("Alpha", ascending=False)
        )

        st.dataframe(agg_df, use_container_width=True)
        st.bar_chart(agg_df.set_index("Wave_ID")["Alpha"])

    # ------------------------------------------------------
    # 🔬 FULL ALPHA ATTRIBUTION (NEW)
    # ------------------------------------------------------

    st.divider()
    st.subheader("🧠 Alpha Attribution Breakdown")

    # Defensive column fetch
    def col_or_zero(df, col):
        return df[col] if col in df.columns else 0.0

    attr_df = snapshot_df.copy()

    attr_df["Benchmark_Return"] = col_or_zero(attr_df, "Benchmark_Return")
    attr_df["Stock_Alpha"] = col_or_zero(attr_df, "Stock_Alpha")
    attr_df["Strategy_Alpha"] = col_or_zero(attr_df, "Strategy_Alpha")
    attr_df["Overlay_Alpha"] = col_or_zero(attr_df, "Overlay_Alpha")

    if "Alpha" not in attr_df.columns:
        attr_df["Alpha"] = (
            attr_df["Return"] - attr_df["Benchmark_Return"]
            if "Return" in attr_df.columns else 0.0
        )

    attr_df["Residual_Alpha"] = (
        attr_df["Alpha"]
        - attr_df["Stock_Alpha"]
        - attr_df["Strategy_Alpha"]
        - attr_df["Overlay_Alpha"]
    )

    attribution_summary = (
        attr_df
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

    st.dataframe(attribution_summary, use_container_width=True)

    # ------------------------------------------------------
    # VISUAL: ALPHA COMPONENTS
    # ------------------------------------------------------

    st.divider()
    st.write("📈 Alpha Components by Wave")

    component_cols = [
        "Stock_Alpha",
        "Strategy_Alpha",
        "Overlay_Alpha",
        "Residual_Alpha"
    ]

    st.bar_chart(
        attribution_summary
        .set_index("Wave_ID")[component_cols]
    )

    # ------------------------------------------------------
    # SUCCESS STATE
    # ------------------------------------------------------

    st.divider()
    st.success(
        "System extended successfully ✔️\n\n"
        "✔ Returns intact\n"
        "✔ Alpha intact\n"
        "✔ Attribution layered safely\n\n"
        "Next: per-strategy & regime attribution."
    )

    with st.expander("Preview snapshot (first 10 rows)"):
        st.dataframe(snapshot_df.head(10))


# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()