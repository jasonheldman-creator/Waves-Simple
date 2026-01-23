import streamlit as st
import sys
import os
import traceback
from types import ModuleType

# ==========================================================
# WAVES — STREAMLIT RECOVERY KERNEL
# Single trusted entrypoint while the system heals
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
        st.write("Public symbols (first 40):", public_symbols[:40])

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
                pass  # strictly read-only

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
    # WAVE ID DISCOVERY (READ-ONLY, NO EXECUTION)
    # ------------------------------------------------------

    st.divider()
    st.write("🧭 Wave ID discovery (read-only)")

    try:
        candidate_symbols = [
            name for name in dir(waves)
            if "wave" in name.lower() or "id" in name.lower()
        ]

        st.write("Candidate wave-related symbols:", candidate_symbols)

        for name in candidate_symbols[:10]:
            try:
                attr = getattr(waves, name)
                st.write(f"{name} →", type(attr))
            except Exception:
                pass

        st.success("Wave ID discovery completed safely")

    except Exception as e:
        st.error("Wave ID discovery failed")
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
        "✔ Wave symbols discovered (read-only)\n"
        "✔ No execution side-effects\n\n"
        "System is ready for selective re-hydration."
    )

# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()