import streamlit as st
import sys
import os
import traceback
from types import ModuleType

# ==========================================================
# WAVES — STREAMLIT RECOVERY KERNEL (CONTROLLED)
# Single trusted entrypoint while system is healing
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
        return  # NEVER proceed if this fails

    # ------------------------------------------------------
    # WAVES MODULE INTROSPECTION (READ-ONLY)
    # ------------------------------------------------------

    st.divider()
    st.write("🧪 waves module inspection (read-only)")

    try:
        st.write("Module file:", waves.__file__)

        public_symbols = [
            name for name in dir(waves)
            if not name.startswith("_")
        ]

        st.write("Total public symbols:", len(public_symbols))
        st.write("Public symbols:", public_symbols)

        st.success("waves module inspection completed safely")
    except Exception as e:
        st.error("waves inspection failed")
        st.exception(e)
        st.code(traceback.format_exc())

    # ------------------------------------------------------
    # CONTROLLED EXECUTION GATE
    # ------------------------------------------------------

    st.divider()
    st.warning(
        "⚠️ Controlled execution gate\n\n"
        "Nothing below runs automatically.\n"
        "Click the button only when ready."
    )

    if st.button("🚀 Initialize WAVES (controlled)"):
        st.write("⏳ Initializing WAVES…")

        try:
            result = waves.initialize_waves()
            st.success("✅ initialize_waves() completed")

            # Inspect resulting state WITHOUT mutating
            if hasattr(waves, "truth_df"):
                truth_df = waves.truth_df
                st.write("truth_df detected")

                if hasattr(truth_df, "waves"):
                    st.write("Number of waves:", len(truth_df.waves))
                    st.write("Wave IDs:", list(truth_df.waves.keys()))
                else:
                    st.warning("truth_df.waves attribute missing")
            else:
                st.warning("truth_df not present after initialization")

        except Exception as e:
            st.error("❌ initialize_waves() failed")
            st.exception(e)
            st.code(traceback.format_exc())
            return

    # ------------------------------------------------------
    # STATUS
    # ------------------------------------------------------

    st.divider()
    st.info(
        "Recovery Mode ACTIVE\n\n"
        "✔ Streamlit boot confirmed\n"
        "✔ Environment visible\n"
        "✔ waves imported safely\n"
        "✔ Execution gated behind manual control\n\n"
        "Ready for controlled system re-hydration."
    )

# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()