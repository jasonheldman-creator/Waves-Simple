# ==========================================================
# app_min.py — WAVES AGGRESSIVE RECOVERY APP
# ==========================================================
# PURPOSE
# • Force Streamlit execution visibility
# • Load live_snapshot.csv
# • Hydrate truth_df
# • Initialize waves safely
# • Render:
#     - Returns
#     - Alpha
#     - FULL alpha attribution
#     - Diagnostic WaveScore placeholder
# • No dependency on app.py
# • Defensive against missing columns
# ==========================================================

import streamlit as st
import sys
import os
import traceback
import pandas as pd
from types import SimpleNamespace

# ----------------------------------------------------------
# 🚨 HARD BOOT MARKER (CANNOT MISS THIS)
# ----------------------------------------------------------

st.markdown("## 🔥 WAVES RECOVERY APP (app_min.py)")
st.markdown("**Build:** AGGRESSIVE-ALPHA-ATTRIBUTION-v1")
st.markdown("---")

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    st.success("✅ Streamlit execution confirmed")

    # ------------------------------------------------------
    # ENVIRONMENT DIAGNOSTICS
    # ------------------------------------------------------

    with st.expander("🧭 Environment Diagnostics", expanded=False):
        st.write("Python:", sys.version)
        st.write("Working dir:", os.getcwd())
        st.write("Files:", sorted(os.listdir(".")))

    # ------------------------------------------------------
    # IMPORT WAVES (HARD GATE)
    # ------------------------------------------------------

    st.subheader("📦 Module Check")

    try:
        import waves
        st.success("waves module imported")
        st.code(waves.__file__)
    except Exception as e:
        st.error("❌ waves import FAILED — stopping")
        st.exception(e)
        return

    # ------------------------------------------------------
    # LOAD LIVE SNAPSHOT
    # ------------------------------------------------------

    st.subheader("📂 Load Live Snapshot")

    SNAPSHOT_PATH = "data/live_snapshot.csv"

    if not os.path.exists(SNAPSHOT_PATH):
        st.error(f"Missing file: {SNAPSHOT_PATH}")
        return

    try:
        df = pd.read_csv(SNAPSHOT_PATH)
        st.success("live_snapshot.csv loaded")
        st.write("Rows:", len(df))
        st.write("Columns:", list(df.columns))
    except Exception as e:
        st.error("Failed reading snapshot")
        st.exception(e)
        return

    # ------------------------------------------------------
    # BUILD truth_df
    # ------------------------------------------------------

    truth_df = SimpleNamespace()
    truth_df.snapshot = df
    truth_df.waves = {}

    # ------------------------------------------------------
    # EXTRACT WAVE IDS
    # ------------------------------------------------------

    if "Wave_ID" not in df.columns:
        st.error("Wave_ID column missing — cannot proceed")
        return

    wave_ids = sorted(df["Wave_ID"].dropna().unique().tolist())
    st.write(f"Detected {len(wave_ids)} waves")

    # ------------------------------------------------------
    # INITIALIZE WAVES
    # ------------------------------------------------------

    st.subheader("🚀 Initialize WAVES")

    try:
        waves.initialize_waves(
            _truth_df=truth_df,
            _unique_wave_ids=wave_ids
        )
        st.success("WAVES initialized")
    except Exception as e:
        st.error("Wave initialization failed")
        st.exception(e)
        st.code(traceback.format_exc())
        return

    # ------------------------------------------------------
    # RETURNS + ALPHA OVERVIEW
    # ------------------------------------------------------

    st.subheader("📊 Returns & Alpha Overview")

    if not {"Return", "Alpha"}.issubset(df.columns):
        st.warning("Return / Alpha columns missing")
    else:
        overview = (
            df.groupby("Wave_ID")[["Return", "Alpha"]]
            .mean()
            .reset_index()
            .sort_values("Alpha", ascending=False)
        )

        st.dataframe(overview, use_container_width=True)
        st.bar_chart(overview.set_index("Wave_ID")["Alpha"])

    # ------------------------------------------------------
    # FULL ALPHA ATTRIBUTION (AGGRESSIVE + SAFE)
    # ------------------------------------------------------

    st.subheader("🧠 Alpha Attribution Breakdown")

    def col(df, name):
        return df[name] if name in df.columns else 0.0

    attr = df.copy()

    attr["Benchmark_Return"] = col(attr, "Benchmark_Return")
    attr["Stock_Alpha"] = col(attr, "Stock_Alpha")
    attr["Strategy_Alpha"] = col(attr, "Strategy_Alpha")
    attr["Overlay_Alpha"] = col(attr, "Overlay_Alpha")

    if "Alpha" not in attr.columns and "Return" in attr.columns:
        attr["Alpha"] = attr["Return"] - attr["Benchmark_Return"]

    attr["Residual_Alpha"] = (
        attr["Alpha"]
        - attr["Stock_Alpha"]
        - attr["Strategy_Alpha"]
        - attr["Overlay_Alpha"]
    )

    attribution = (
        attr.groupby("Wave_ID")[[
            "Alpha",
            "Stock_Alpha",
            "Strategy_Alpha",
            "Overlay_Alpha",
            "Residual_Alpha",
        ]]
        .mean()
        .reset_index()
        .sort_values("Alpha", ascending=False)
    )

    st.dataframe(attribution, use_container_width=True)

    st.bar_chart(
        attribution
        .set_index("Wave_ID")[
            ["Stock_Alpha", "Strategy_Alpha", "Overlay_Alpha", "Residual_Alpha"]
        ]
    )

    # ------------------------------------------------------
    # WAVESCORE (PLACEHOLDER — SAFE)
    # ------------------------------------------------------

    st.subheader("⭐ WaveScore (Diagnostic)")

    if "Alpha" in attribution.columns:
        attribution["WaveScore"] = (
            attribution["Alpha"].rank(pct=True) * 100
        ).round(1)
        st.dataframe(
            attribution[["Wave_ID", "WaveScore"]]
            .sort_values("WaveScore", ascending=False),
            use_container_width=True
        )
    else:
        st.info("WaveScore pending Alpha availability")

    # ------------------------------------------------------
    # SUCCESS
    # ------------------------------------------------------

    st.success(
        "Recovery App ACTIVE ✅\n\n"
        "• Streamlit executing\n"
        "• Snapshot loaded\n"
        "• Waves initialized\n"
        "• Returns rendered\n"
        "• Alpha attribution live\n\n"
        "This is now a stable foundation to expand."
    )

    with st.expander("🔍 Snapshot Preview"):
        st.dataframe(df.head(20))


# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()