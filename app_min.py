# ==========================================================
# app_min.py — WAVES Recovery → Live Snapshot Rehydration
# ==========================================================
# This file:
# • Boots Streamlit safely
# • Loads data/live_snapshot.csv
# • Builds a minimal truth_df object
# • Initializes waves via waves.initialize_waves
# • Probes attribution engine (alpha / beta)
# • Leaves full app.py untouched
# ==========================================================

import streamlit as st
import sys
import os
import traceback
import pandas as pd
from types import SimpleNamespace

# ----------------------------------------------------------
# SAFE IMPORT PROBE — ATTRIBUTION ENGINE
# ----------------------------------------------------------

try:
    import attribution_engine
    st.success("✅ attribution_engine imported successfully")
    if hasattr(attribution_engine, "_import_check"):
        st.caption(attribution_engine._import_check())
except Exception as e:
    st.warning("⚠️ attribution_engine not active yet (safe to ignore)")
    st.exception(e)

# ----------------------------------------------------------
# BOOT CONFIRMATION (UNCONDITIONAL)
# ----------------------------------------------------------

st.error("APP_MIN EXECUTION STARTED")
st.write("🟢 STREAMLIT EXECUTION STARTED")
st.write("🟢 app_min.py reached line 1")

# ----------------------------------------------------------
# MAIN ENTRYPOINT
# ----------------------------------------------------------

def main():
    st.title("WAVES — Recovery Mode (Live Snapshot)")
    st.success("Recovery kernel running")

    # ------------------------------------------------------
    # ENVIRONMENT VISIBILITY
    # ------------------------------------------------------

    st.divider()
    st.write("🧭 Runtime environment")

    try:
        st.write("Python:", sys.version)
        st.write("Executable:", sys.executable)
        st.write("Working directory:", os.getcwd())
        st.write("Files in root:", sorted(os.listdir(".")))
        st.success("Environment visible")
    except Exception as e:
        st.error("Environment inspection failed")
        st.exception(e)

    # ------------------------------------------------------
    # WAVES MODULE IMPORT (HARD GATE)
    # ------------------------------------------------------

    st.divider()
    st.write("🔍 waves module check")

    try:
        import waves
        st.success("✅ waves imported successfully")
        st.code(waves.__file__)
    except Exception as e:
        st.error("❌ waves import failed — hard stop")
        st.exception(e)
        st.code(traceback.format_exc())
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
        st.success("✅ live_snapshot.csv loaded")
        st.write("Rows:", len(snapshot_df))
        st.write("Columns:", list(snapshot_df.columns))
    except Exception as e:
        st.error("❌ Failed to read snapshot CSV")
        st.exception(e)
        return

    # ------------------------------------------------------
    # BUILD truth_df (MINIMAL, SAFE)
    # ------------------------------------------------------

    st.divider()
    st.write("🧠 Building truth_df")

    try:
        truth_df = SimpleNamespace()
        truth_df.snapshot = snapshot_df
        truth_df.waves = {}
        st.success("truth_df object created")
    except Exception as e:
        st.error("truth_df construction failed")
        st.exception(e)
        return

    # ------------------------------------------------------
    # EXTRACT UNIQUE WAVE IDS
    # ------------------------------------------------------

    st.divider()
    st.write("🧬 Extracting wave IDs")

    if "Wave_ID" not in snapshot_df.columns:
        st.error("❌ Wave_ID column missing from snapshot")
        return

    unique_wave_ids = (
        snapshot_df["Wave_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    unique_wave_ids = sorted(unique_wave_ids)

    st.write("Number of waves:", len(unique_wave_ids))
    st.write("Wave IDs (sample):", unique_wave_ids[:10])

    # ------------------------------------------------------
    # INITIALIZE WAVES (SAFE, GATED)
    # ------------------------------------------------------

    st.divider()
    st.write("🚀 Initializing WAVES")

    try:
        waves.initialize_waves(
            _truth_df=truth_df,
            _unique_wave_ids=unique_wave_ids
        )
        st.success("✅ WAVES initialized successfully")
    except Exception as e:
        st.error("❌ WAVES initialization failed")
        st.exception(e)
        st.code(traceback.format_exc())
        return

    # ------------------------------------------------------
    # VERIFICATION
    # ------------------------------------------------------

    st.divider()
    st.write("🔎 Verification")

    st.write("Initialized wave keys (sample):", list(truth_df.waves.keys())[:10])
    st.write("Total initialized waves:", len(truth_df.waves))

    # ------------------------------------------------------
    # SUCCESS STATE
    # ------------------------------------------------------

    st.divider()
    st.success(
        "Recovery SUCCESSFUL\n\n"
        "✔ live_snapshot loaded\n"
        "✔ truth_df hydrated\n"
        "✔ waves initialized\n"
        "✔ attribution engine reachable\n"
        "✔ system execution restored\n\n"
        "Next step: detailed alpha / beta attribution rendering"
    )

    with st.expander("Preview snapshot (first 10 rows)"):
        st.dataframe(snapshot_df.head(10))


# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()