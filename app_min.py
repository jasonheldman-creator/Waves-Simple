# ==========================================================
# app_min.py — WAVES Recovery Console (Canonical)
# ==========================================================
# Full recovery console with:
# • Live snapshot ingestion
# • Intraday + 30D + 60D + 365D returns & alpha
# • Full alpha attribution (benchmark / strategy / overlay)
# • Defensive, non-destructive execution
# ==========================================================

import streamlit as st
import os
import sys
import pandas as pd
from types import SimpleNamespace
import traceback

# ----------------------------------------------------------
# BOOT CONFIRMATION
# ----------------------------------------------------------

st.error("APP_MIN EXECUTION STARTED")
st.write("🟢 STREAMLIT EXECUTION STARTED")
st.write("🟢 app_min.py reached line 1")

# ----------------------------------------------------------
# SAFE ENGINE IMPORTS
# ----------------------------------------------------------

def safe_import(name):
    try:
        module = __import__(name)
        st.success(f"✅ {name} imported")
        return module
    except Exception as e:
        st.warning(f"⚠️ {name} not available")
        return None

waves = safe_import("waves")
attribution_engine = safe_import("attribution_engine")
dynamic_benchmark_engine = safe_import("dynamic_benchmark_engine")
strategy_overlay_engine = safe_import("strategy_overlay_engine")
wave_score_engine = safe_import("wave_score_engine")
regime_engine = safe_import("regime_engine")

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    st.title("WAVES — Recovery Console")
    st.success("Recovery console running")

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
        st.error("live_snapshot.csv not found")
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

    # ------------------------------------------------------
    # INITIALIZE WAVES
    # ------------------------------------------------------

    if waves:
        try:
            wave_ids = snapshot_df["Wave_ID"].dropna().unique().tolist()
            waves.initialize_waves(
                _truth_df=truth_df,
                _unique_wave_ids=wave_ids
            )
            st.success(f"Waves initialized: {len(truth_df.waves)}")
        except Exception as e:
            st.error("Wave initialization failed")
            st.exception(e)
            return

    # ======================================================
    # RETURNS & ALPHA (ALL HORIZONS)
    # ======================================================

    st.divider()
    st.header("📊 Returns & Alpha (LIVE → 365D)")

    cols = [
        "Wave_ID",
        "Return_ID", "Alpha_1D",
        "Return_30D", "Alpha_30D",
        "Return_60D", "Alpha_60D",
        "Return_365D", "Alpha_365D"
    ]

    available = [c for c in cols if c in snapshot_df.columns]
    returns_df = snapshot_df[available].copy()

    returns_df = (
        returns_df
        .groupby("Wave_ID")
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("Alpha_365D", ascending=False)
    )

    st.dataframe(returns_df, use_container_width=True)
    st.bar_chart(
        returns_df.set_index("Wave_ID")[["Alpha_365D"]]
    )

    # ======================================================
    # FULL ALPHA ATTRIBUTION (365D CANONICAL)
    # ======================================================

    st.divider()
    st.header("🧠 Alpha Attribution (365D)")

    def col(df, name):
        return df[name] if name in df.columns else 0.0

    attr = snapshot_df.copy()

    attr["Total_Alpha"] = col(attr, "Alpha_365D")
    attr["Benchmark_Selection_Alpha"] = (
        col(attr, "Benchmark_Return_365D") -
        col(attr, "Benchmark_Static_365D")
    )

    attr["Strategy_Alpha"] = col(attr, "Strategy_Alpha_365D")
    attr["Overlay_Alpha"] = col(attr, "Overlay_Alpha_365D")

    attr["Residual_Alpha"] = (
        attr["Total_Alpha"]
        - attr["Benchmark_Selection_Alpha"]
        - attr["Strategy_Alpha"]
        - attr["Overlay_Alpha"]
    )

    attribution = (
        attr
        .groupby("Wave_ID")[[
            "Total_Alpha",
            "Benchmark_Selection_Alpha",
            "Strategy_Alpha",
            "Overlay_Alpha",
            "Residual_Alpha"
        ]]
        .mean()
        .reset_index()
        .sort_values("Total_Alpha", ascending=False)
    )

    st.dataframe(attribution, use_container_width=True)

    st.bar_chart(
        attribution
        .set_index("Wave_ID")[[
            "Benchmark_Selection_Alpha",
            "Strategy_Alpha",
            "Overlay_Alpha",
            "Residual_Alpha"
        ]]
    )

    # ------------------------------------------------------
    # SUCCESS
    # ------------------------------------------------------

    st.divider()
    st.success(
        "Recovery console fully operational ✔️\n\n"
        "• LIVE + 30D + 60D + 365D returns\n"
        "• Full alpha attribution\n"
        "• No destructive changes\n\n"
        "Next: WaveScore + regime dashboards"
    )


# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()