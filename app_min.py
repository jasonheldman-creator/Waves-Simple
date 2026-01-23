import streamlit as st
import sys
import os
import traceback
from types import SimpleNamespace

# ==========================================================
# WAVES — STREAMLIT RECOVERY KERNEL (HYDRATION ENABLED)
# ==========================================================

# ----------------------------------------------------------
# BOOT CONFIRMATION (must execute unconditionally)
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
    # ENVIRONMENT SNAPSHOT
    # ------------------------------------------------------

    st.divider()
    st.subheader("🧭 Runtime environment")

    try:
        st.write("Python:", sys.version)
        st.write("Executable:", sys.executable)
        st.write("Working directory:", os.getcwd())
        st.success("Environment visible")
    except Exception as e:
        st.error("Environment snapshot failed")
        st.exception(e)

    # ------------------------------------------------------
    # WAVES MODULE CHECK
    # ------------------------------------------------------

    st.divider()
    st.subheader("🔎 waves module check")

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
    st.subheader("🧪 Contract discovery (read-only)")

    public_symbols = [s for s in dir(waves) if not s.startswith("_")]
    st.write(public_symbols)

    has_init = hasattr(waves, "initialize_waves")
    has_truth = hasattr(waves, "truth_df")
    has_ids = hasattr(waves, "unique_wave_ids")

    st.write("initialize_waves:", has_init)
    st.write("truth_df exists:", has_truth)
    st.write("unique_wave_ids exists:", has_ids)

    # ------------------------------------------------------
    # HYDRATION (EXPLICIT & SAFE)
    # ------------------------------------------------------

    st.divider()
    st.subheader("🧠 Current state (hydration)")

    # Create a SAFE truth_df container
    truth_df = SimpleNamespace()
    truth_df.waves = {}

    # Define placeholder wave IDs (can be replaced later)
    unique_wave_ids = []

    st.write("truth_df type:", type(truth_df))
    st.write("unique_wave_ids type:", type(unique_wave_ids))
    st.write("Number of wave IDs:", len(unique_wave_ids))

    # ------------------------------------------------------
    # CONTROLLED INITIALIZATION
    # ------------------------------------------------------

    st.divider()
    st.warning(
        "Controlled execution gate\n\n"
        "Initialization will only run with explicit consent."
    )

    if st.button("🚀 Initialize WAVES (safe)"):
        try:
            result = waves.initialize_waves(
                truth_df,
                unique_wave_ids
            )

            st.success("initialize_waves() completed successfully")
            st.write("waves initialized:", result)

        except Exception as e:
            st.error("initialize_waves() failed")
            st.exception(e)
            st.code(traceback.format_exc())

    # ------------------------------------------------------
    # RECOVERY STATUS
    # ------------------------------------------------------

    st.divider()
    st.info(
        "Recovery Mode ACTIVE\n\n"
        "✔ Streamlit healthy\n"
        "✔ Environment visible\n"
        "✔ waves module loadable\n"
        "✔ truth_df explicitly hydrated\n"
        "✔ Execution gated safely\n\n"
        "Ready for full app rehydration."
    )

# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()
    