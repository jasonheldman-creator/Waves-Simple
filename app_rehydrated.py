"""
app_rehydrated.py - WAVES full application entrypoint

This file replaces app.py as the operational Streamlit entrypoint.
It assumes:
• live_snapshot.csv exists
• waves.py is import-safe
• truth_df hydration is valid
"""

import streamlit as st
import pandas as pd
from pathlib import Path

import waves


st.set_page_config(
    page_title="WAVES Intelligence",
    layout="wide",
)

st.success("🚀 WAVES full app (rehydrated) starting")


# --------------------------------------------------
# Load live snapshot
# --------------------------------------------------

SNAPSHOT_PATH = Path("data/live_snapshot.csv")

if not SNAPSHOT_PATH.exists():
    st.error("❌ live_snapshot.csv not found")
    st.stop()

snapshot_df = pd.read_csv(SNAPSHOT_PATH)

st.success(f"✅ Snapshot loaded ({len(snapshot_df)} rows)")


# --------------------------------------------------
# Hydrate truth_df
# --------------------------------------------------

class TruthFrame:
    pass

truth_df = TruthFrame()
truth_df.snapshot = snapshot_df
truth_df.waves = {}

unique_wave_ids = snapshot_df["Wave_ID"].unique().tolist()

waves.initialize_waves(truth_df, unique_wave_ids)

st.success(f"✅ {len(truth_df.waves)} waves initialized")


# --------------------------------------------------
# Basic UI (safe)
# --------------------------------------------------

st.header("📊 WAVES Snapshot Overview")
st.dataframe(snapshot_df.head(25), use_container_width=True)

st.header("🌊 Initialized Waves")
st.write(sorted(truth_df.waves.keys()))

st.success("🎉 WAVES application fully live")