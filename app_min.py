import streamlit as st
import sys
import os
import traceback
from types import ModuleType

# ==========================================================
# WAVES — STREAMLIT RECOVERY KERNEL
# This file is intentionally defensive, verbose, and safe.
# It is the single trusted entrypoint while the system heals.
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
    # WAVES MODULE INTROSPECTION (READ-ONLY, SAFE)
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
        st.write("Public symbols (first 40):", public_symbols[:40])

        # Classify symbols WITHOUT touching behavior
        functions = []
        classes = []
        submodules = []

        for name in public_symbols:
            try:
                attr = getattr(waves, name)
                if isinstance(attr, ModuleType):
                    submodules.append(name)
                elif isinstance(attr, type):
                    classes.append(name)
                elif callable(attr):
                    functions.append(name)
            except Exception:
                pass  # stay purely observational

        st.divider()
        st.write("🧬 waves symbol breakdown")
        st.write("Functions (sample):", functions[:15])
        st.write("Classes (sample):", classes[:15])
        st.write("Sub-modules (sample):", submodules[:15])

        st.success("waves module inspection completed safely")

    except Exception as e:
        st.error("waves inspection failed")
        st.exception(e)
        st.code(traceback.format_exc())

    # ------------------------------------------------------
    # RECOVERY STATUS
    # ------------------------------------------------------

    st.divider()
    st.info(
        "Recovery Mode ACTIVE\n\n"
        "✔ Streamlit boot confirmed\n"
        "✔ Environment visible\n"
        "✔ waves imported safely\n"
        "✔ No execution side-effects\n\n"
        "System is ready for selective re-hydration."
    )

# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()