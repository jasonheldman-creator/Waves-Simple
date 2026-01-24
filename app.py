"""
app.py — WAVES Safe Rehydration (Phase B)

Purpose:
- Safely transition from recovery to full app
- Load live snapshot
- Hydrate truth_df
- Initialize waves
- Expose ONLY core UI (no advanced tabs yet)

This file is intentionally conservative.
"""

import streamlit as st
import pandas as pd
import sys
import os
import traceback
from types import SimpleNamespace

# ─────────────────────────────────────────────
# BOOT CONFIRMATION
# ─────────────────────────────────────────────

st.write("🟢 STREAMLIT EXECUTION STARTED")
st.write("🟢 app.py reached line 1")

st.set_page_config(
    page_title="WAVES — Rehydration",
    layout="wide"
)

st.title("WAVES — System Rehydration")
st.success("Core kernel running")

# ─────────────────────────────────────────────
# ENVIRONMENT SNAPSHOT
# ─────────────────────────────────────────────

with st.expander("🧭 Runtime environment", expanded=False):
    st.write("Python:", sys.version)
    st.write("Executable:", sys.executable)
    st.write("Working directory:", os.getcwd())

# ─────────────────────────────────────────────
# LOAD WAVES MODULE (IMPORT SAFE)
# ─────────────────────────────────────────────

st.divider()
st.subheader("🔌 Module load")

try:
    import waves
    st.success("waves module imported successfully")
    st.code(waves.__file__)
except Exception as e:
    st.error("waves import failed — aborting")
    st.exception(e)
    st.stop()

# ─────────────────────────────────────────────
# LOAD LIVE SNAPSHOT
# ─────────────────────────────────────────────

st.divider()
st.subheader("📦 Live snapshot load")

SNAPSHOT_PATH = "data/live_snapshot.csv"

try:
    snapshot_df = pd.read_csv(SNAPSHOT_PATH)
    st.success(f"Loaded snapshot: {SNAPSHOT_PATH}")
    st.write("Rows:", len(snapshot_df))
    st.write("Columns:", list(snapshot_df.columns))
except Exception as e:
    st.error("Failed to load live snapshot")
    st.exception(e)
    st.stop()

with st.expander("Preview snapshot (first 10 rows)", expanded=False):
    st.dataframe(snapshot_df.head(10), use_container_width=True)

# ─────────────────────────────────────────────
# HYDRATE truth_df (CONTROLLED)
# ─────────────────────────────────────────────

st.divider()
st.subheader("🧠 truth_df hydration")

# Create a safe, explicit container
truth_df = SimpleNamespace()
truth_df.snapshot = snapshot_df
truth_df.waves = {}

st.success("truth_df created and hydrated")

# ─────────────────────────────────────────────
# DERIVE WAVE IDS
# ─────────────────────────────────────────────

try:
    wave_ids = sorted(snapshot_df["Wave_ID"].dropna().unique().tolist())
    st.success(f"Derived {len(wave_ids)} wave IDs")
except Exception as e:
    st.error("Failed to derive wave IDs")
    st.exception(e)
    st.stop()

with st.expander("Wave IDs", expanded=False):
    st.write(wave_ids)

# ─────────────────────────────────────────────
# INITIALIZE WAVES (SAFE CALL)
# ─────────────────────────────────────────────

st.divider()
st.subheader("🚀 Initialize WAVES")

try:
    waves_state = waves.initialize_waves(
        truth_df=truth_df,
        unique_wave_ids=wave_ids
    )
    st.success(f"Waves initialized: {len(waves_state)}")
except Exception as e:
    st.error("initialize_waves() failed")
    st.exception(e)
    st.stop()

# ─────────────────────────────────────────────
# CORE VERIFICATION UI (PHASE B ONLY)
# ─────────────────────────────────────────────

st.divider()
st.subheader("✅ System state verification")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Snapshot rows", len(snapshot_df))

with col2:
    st.metric("Wave IDs", len(wave_ids))

with col3:
    st.metric("Initialized waves", len(truth_df.waves))

with st.expander("Initialized wave objects (sample)", expanded=False):
    sample_keys = list(truth_df.waves.keys())[:5]
    for k in sample_keys:
        st.write(k, truth_df.waves[k])

# ─────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────

st.divider()
st.success(
    "Rehydration COMPLETE\n\n"
    "✔ live_snapshot loaded\n"
    "✔ truth_df hydrated\n"
    "✔ waves initialized\n"
    "✔ core execution stable\n\n"
    "Next step: progressively re-enable full UI."
)

# ─────────────────────────────────────────────
# NO AUTOMATIC ADVANCED EXECUTION YET
# ─────────────────────────────────────────────