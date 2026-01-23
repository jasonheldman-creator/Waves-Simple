import streamlit as st
import sys
import os
import traceback
from types import SimpleNamespace

# ==========================================================
# WAVES — STREAMLIT RECOVERY KERNEL (CONTROLLED)
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
    # RUNTIME ENVIRONMENT
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
        st.error("❌ waves import failed — hard stop")
        st.exception(e)
        st.code(traceback.format_exc())
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
    # CURRENT STATE SNAPSHOT
    # ------------------------------------------------------

    st.divider()
    st.write("🧠 Current state")

    truth_df_existing = getattr(waves, "truth_df", None)
    unique_ids_existing = getattr(waves, "unique_wave_ids", None)

    st.write("truth_df type:", type(truth_df_existing))
    st.write("unique_wave_ids type:", type(unique_ids_existing))
    st.write(
        "Number of wave IDs:",
        len(unique_ids_existing) if isinstance(unique_ids_existing, list) else "N/A"
    )

    # ------------------------------------------------------
    # CONTROLLED INITIALIZATION GATE
    # ------------------------------------------------------

    st.divider()
    st.warning(
        "⚠️ Controlled execution gate\n\n"
        "Initialization will only run if the contract is satisfied."
    )

    if st.button("🚀 Initialize WAVES (safe)"):
        st.write("⏳ Initializing WAVES…")

        try:
            # --------------------------------------------------
            # BUILD MINIMAL VALID INPUTS
            # --------------------------------------------------

            # Create a minimal truth_df object with required structure
            truth_df = SimpleNamespace()
            truth_df.waves = {}

            # Use discovered wave IDs if present, otherwise empty list
            unique_wave_ids = (
                unique_ids_existing
                if isinstance(unique_ids_existing, list)
                else []
            )

            st.write("truth_df prepared:", truth_df)
            st.write("unique_wave_ids:", unique_wave_ids)

            # --------------------------------------------------
            # CALL INITIALIZE_WAVES CORRECTLY
            # --------------------------------------------------

            result = waves.initialize_waves(
                truth_df,
                unique_wave_ids
            )

            st.success("✅ initialize_waves() completed")
            st.write("Result:", result)
            st.write("truth_df.waves keys:", list(truth_df.waves.keys()))

        except Exception as e:
            st.error("❌ initialize_waves() failed")
            st.exception(e)
            st.code(traceback.format_exc())

    # ------------------------------------------------------
    # STATUS
    # ------------------------------------------------------

    st.divider()
    st.info(
        "Recovery Mode ACTIVE\n\n"
        "✔ Streamlit healthy\n"
        "✔ Environment validated\n"
        "✔ waves module loadable\n"
        "✔ Contract inspected\n"
        "✔ Execution gated safely\n\n"
        "Next step: full app rehydration."
    )


# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()