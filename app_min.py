import streamlit as st
import sys
import os
import traceback
import pandas as pd
from types import SimpleNamespace

# ==========================================================
# WAVES — AGGRESSIVE RECOVERY KERNEL
# Purpose: Force minimal hydration to restore full execution
# ==========================================================

# ----------------------------------------------------------
# BOOT CONFIRMATION (must run unconditionally)
# ----------------------------------------------------------

st.error("APP_MIN EXECUTION STARTED")
st.write("🟢 STREAMLIT EXECUTION STARTED")
st.write("🟢 app_min.py reached line 1")

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

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
    st.write("Working dir:", os.getcwd())

    st.success("Environment visible")

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
        st.error("waves import failed")
        st.exception(e)
        return

    # ------------------------------------------------------
    # CONTRACT DISCOVERY
    # ------------------------------------------------------

    st.divider()
    st.write("🧪 Contract discovery (read-only)")

    public_symbols = [s for s in dir(waves) if not s.startswith("_")]
    st.write(public_symbols)

    has_init = hasattr(waves, "initialize_waves")
    has_truth = hasattr(waves, "truth_df")
    has_ids = hasattr(waves, "unique_wave_ids")

    st.write("initialize_waves:", has_init)
    st.write("truth_df exists:", has_truth)
    st.write("unique_wave_ids exists:", has_ids)

    if not has_init:
        st.error("initialize_waves not found — cannot proceed")
        return

    # ------------------------------------------------------
    # AGGRESSIVE HYDRATION (CONTROLLED)
    # ------------------------------------------------------

    st.divider()
    st.write("🚀 Aggressive hydration (controlled)")

    # Minimal but valid truth_df
    truth_df = SimpleNamespace()
    truth_df.waves = {}

    # Stub wave IDs (safe, deterministic)
    unique_wave_ids = [
        "sp500_wave",
        "ai_cloud_megacap_wave",
        "clean_energy_wave"
    ]

    st.write("Injected wave IDs:", unique_wave_ids)

    # ------------------------------------------------------
    # INITIALIZE WAVES
    # ------------------------------------------------------

    try:
        result = waves.initialize_waves(
            truth_df,
            unique_wave_ids
        )

        st.success("initialize_waves executed successfully")

    except Exception as e:
        st.error("initialize_waves failed")
        st.exception(e)
        st.code(traceback.format_exc())
        return

    # ------------------------------------------------------
    # POST-HYDRATION STATE
    # ------------------------------------------------------

    st.divider()
    st.write("🧠 Post-hydration state")

    st.write("truth_df type:", type(truth_df))
    st.write("Number of waves:", len(truth_df.waves))
    st.write("Wave keys:", list(truth_df.waves.keys()))

    st.json(truth_df.waves)

    # ------------------------------------------------------
    # SUCCESS STATE
    # ------------------------------------------------------

    st.divider()
    st.success(
        "Recovery SUCCESSFUL\n\n"
        "✔ Waves initialized\n"
        "✔ truth_df hydrated\n"
        "✔ System execution restored\n\n"
        "Next step: swap stub data for real snapshot."
    )


# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()