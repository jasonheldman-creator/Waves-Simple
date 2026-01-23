# app_min.py
# WAVES Recovery Kernel — SAFE HYDRATION VERSION

import streamlit as st
import traceback

st.set_page_config(
    page_title="WAVES — Recovery Mode",
    layout="wide",
)

st.markdown("# 🌊 WAVES — Recovery Mode")
st.success("Recovery kernel running")

# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------
st.markdown("### 🧭 Runtime environment")

try:
    import sys
    st.code(f"Python: {sys.version}")
    st.code(f"Executable: {sys.executable}")
    st.code(f"Working dir: {sys.path[0]}")
    st.success("Environment visible")
except Exception as e:
    st.error("Environment inspection failed")
    st.exception(e)

# -------------------------------------------------------------------
# Import WAVES module
# -------------------------------------------------------------------
st.markdown("### 🔍 waves module check")

try:
    import waves
    st.success("waves imported successfully")
    st.code(waves.__file__)
except Exception as e:
    st.error("Failed to import waves")
    st.exception(e)
    st.stop()

# -------------------------------------------------------------------
# Contract discovery
# -------------------------------------------------------------------
st.markdown("### 🧪 Contract discovery (read-only)")

public_symbols = [
    s for s in dir(waves)
    if not s.startswith("_")
]

st.code(public_symbols)

has_init = hasattr(waves, "initialize_waves")
has_truth_df = hasattr(waves, "truth_df")
has_ids = hasattr(waves, "unique_wave_ids")

st.write("initialize_waves:", has_init)
st.write("truth_df exists:", has_truth_df)
st.write("unique_wave_ids exists:", has_ids)

if not has_init:
    st.error("initialize_waves() missing — cannot proceed")
    st.stop()

# -------------------------------------------------------------------
# Hydrate inputs SAFELY
# -------------------------------------------------------------------
st.markdown("### 🧠 Current state (hydration)")

truth_df = getattr(waves, "truth_df", None)
unique_wave_ids = getattr(waves, "unique_wave_ids", None)

st.write("truth_df type:", type(truth_df))
st.write("unique_wave_ids type:", type(unique_wave_ids))

# Validate truth_df
if truth_df is None:
    st.error("truth_df is None — initialization blocked")
    st.stop()

# Validate wave IDs
if not isinstance(unique_wave_ids, (list, tuple)):
    st.error("unique_wave_ids is not a list/tuple")
    st.stop()

st.write("Number of wave IDs:", len(unique_wave_ids))

if len(unique_wave_ids) == 0:
    st.error("unique_wave_ids is empty — no waves to initialize")
    st.stop()

# -------------------------------------------------------------------
# Controlled execution gate
# -------------------------------------------------------------------
st.warning(
    "Initialization will only run if the contract is satisfied."
)

if st.button("🚀 Initialize WAVES (safe)"):
    st.markdown("### ⏳ Initializing WAVES...")
    try:
        result = waves.initialize_waves(
            truth_df=truth_df,
            unique_wave_ids=unique_wave_ids,
        )
        st.success("initialize_waves() completed successfully")
        st.write(result)
    except Exception as e:
        st.error("initialize_waves() failed")
        st.code(traceback.format_exc())
        st.stop()

# -------------------------------------------------------------------
# Status
# -------------------------------------------------------------------
st.markdown("---")
st.info(
    "Recovery Mode ACTIVE\n\n"
    "✓ Streamlit healthy\n"
    "✓ Environment healthy\n"
    "✓ waves module loadable\n"
    "✓ Contract inspected\n"
    "✓ Execution gated safely\n\n"
    "Next step: full app rehydration."
)