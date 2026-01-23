import streamlit as st
import sys
import os
import traceback
from types import ModuleType

# ==========================================================
# WAVES — STREAMLIT RECOVERY & REHYDRATION KERNEL
# Single trusted entrypoint while restoring the system.
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
    st.success("app_min.main() is now running")

    # ------------------------------------------------------
    # RUNTIME ENVIRONMENT SNAPSHOT
    # ------------------------------------------------------

    st.divider()
    st.write("🧭 Runtime environment snapshot")

    try:
        st.write("Python version:", sys.version)
        st.write("Executable:", sys.executable)
        st.write("Working directory:", os.getcwd())
        st.write("Root directory contents:", sorted(os.listdir(".")))
        st.success("Environment snapshot completed")
    except Exception as e:
        st.error("Environment snapshot failed")
        st.exception(e)

    # ------------------------------------------------------
    # WAVES IMPORT (HARD GATE)
    # ------------------------------------------------------

    st.divider()
    st.write("🔍 Import diagnostics starting…")

    try:
        import waves
        st.success("✅ waves module imported successfully")
    except Exception as e:
        st.error("❌ waves import failed — recovery halted")
        st.exception(e)
        st.code(traceback.format_exc())
        return

    # ------------------------------------------------------
    # WAVES MODULE INSPECTION (READ-ONLY)
    # ------------------------------------------------------

    st.divider()
    st.write("🧪 waves module inspection (read-only)")

    try:
        st.write("Module file:", waves.__file__)

        public_symbols = [n for n in dir(waves) if not n.startswith("_")]
        st.write("Public symbols:", public_symbols)

        st.success("waves module inspection completed safely")
    except Exception as e:
        st.error("waves inspection failed")
        st.exception(e)
        return

    # ------------------------------------------------------
    # CONTROLLED EXECUTION GATE
    # ------------------------------------------------------

    st.divider()
    st.warning(
        "⚠️ Controlled execution gate\n\n"
        "Nothing below runs automatically.\n"
        "Click only when ready."
    )

    if st.button("🚀 Initialize WAVES (controlled)"):
        st.write("⏳ Initializing WAVES…")

        try:
            # Explicit, visible wiring
            truth_df = waves.truth_df
            unique_wave_ids = waves.unique_wave_ids

            st.write("truth_df type:", type(truth_df))
            st.write("unique_wave_ids type:", type(unique_wave_ids))
            st.write("Number of wave IDs:", len(unique_wave_ids))

            result = waves.initialize_waves(truth_df, unique_wave_ids)

            st.success("✅ WAVES initialized successfully")
            st.write("Initialization result:", result)

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
        "✔ Streamlit boot confirmed\n"
        "✔ Environment visible\n"
        "✔ waves imported safely\n"
        "✔ Execution explicitly controlled\n\n"
        "Ready for full application re-enablement."
    )


# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()