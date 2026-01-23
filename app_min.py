import streamlit as st
import sys
import os
import traceback
from types import ModuleType

# ==========================================================
# WAVES — CONTRACT-AWARE RECOVERY KERNEL
# ==========================================================

st.error("APP_MIN EXECUTION STARTED")
st.write("🟢 STREAMLIT EXECUTION STARTED")
st.write("🟢 app_min.py reached line 1")

def main():
    st.title("WAVES — Recovery Mode")
    st.success("Recovery kernel running")

    # ------------------------------------------------------
    # ENVIRONMENT
    # ------------------------------------------------------

    st.divider()
    st.write("🧭 Runtime environment")

    st.write("Python:", sys.version)
    st.write("Executable:", sys.executable)
    st.write("Working directory:", os.getcwd())
    st.success("Environment visible")

    # ------------------------------------------------------
    # WAVES MODULE LOAD
    # ------------------------------------------------------

    st.divider()
    st.write("🔍 waves module check")

    try:
        import waves
        st.success("✅ waves imported successfully")
        st.code(waves.__file__)
    except Exception as e:
        st.error("❌ waves import failed")
        st.exception(e)
        return

    # ------------------------------------------------------
    # CONTRACT DISCOVERY (READ-ONLY)
    # ------------------------------------------------------

    st.divider()
    st.write("🧪 Contract discovery (read-only)")

    public_symbols = [s for s in dir(waves) if not s.startswith("_")]
    st.write("Public symbols:", public_symbols)

    has_init = hasattr(waves, "initialize_waves")
    has_truth = hasattr(waves, "truth_df")
    has_ids = hasattr(waves, "unique_wave_ids")

    st.write("initialize_waves:", has_init)
    st.write("truth_df exists:", has_truth)
    st.write("unique_wave_ids exists:", has_ids)

    # ------------------------------------------------------
    # SAFE STATE INSPECTION
    # ------------------------------------------------------

    st.divider()
    st.write("🧠 Current state")

    truth_df = getattr(waves, "truth_df", None)
    wave_ids = getattr(waves, "unique_wave_ids", None)

    st.write("truth_df:", type(truth_df))
    st.write("unique_wave_ids:", type(wave_ids))

    # ------------------------------------------------------
    # CONTROLLED EXECUTION GATE
    # ------------------------------------------------------

    st.divider()
    st.warning(
        "⚠️ Controlled execution gate\n\n"
        "Initialization will only run if the contract is satisfied."
    )

    if st.button("🚀 Initialize WAVES (safe)"):
        if truth_df is None:
            st.error("❌ truth_df is None — cannot initialize safely")
            return

        if not isinstance(wave_ids, (list, tuple)):
            st.error("❌ unique_wave_ids invalid — cannot initialize safely")
            return

        try:
            st.write("Initializing with existing contract…")
            result = waves.initialize_waves(truth_df, wave_ids)
            st.success("✅ initialize_waves completed safely")
            st.write(result)
        except Exception as e:
            st.error("❌ initialize_waves failed")
            st.exception(e)
            st.code(traceback.format_exc())

    # ------------------------------------------------------
    # STATUS
    # ------------------------------------------------------

    st.divider()
    st.info(
        "Recovery Mode ACTIVE\n\n"
        "✔ Environment healthy\n"
        "✔ waves module loadable\n"
        "✔ Contract inspected\n"
        "✔ Execution gated safely\n\n"
        "Next step: full app rehydration."
    )

if __name__ == "__main__":
    main()