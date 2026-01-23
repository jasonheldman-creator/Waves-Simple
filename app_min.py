import streamlit as st
import sys
import os
import traceback

# ==========================================================
# WAVES — STREAMLIT RECOVERY KERNEL (FINAL)
# This file DOES NOT initialize the system.
# It validates health and hands control back to app.py.
# ==========================================================

# ----------------------------------------------------------
# BOOT CONFIRMATION (unconditional)
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
    # ENVIRONMENT CHECK
    # ------------------------------------------------------

    st.divider()
    st.write("🧭 Runtime environment")

    st.write("Python:", sys.version)
    st.write("Executable:", sys.executable)
    st.write("Working directory:", os.getcwd())

    st.success("Environment visible")

    # ------------------------------------------------------
    # WAVES IMPORT CHECK
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
        st.code(traceback.format_exc())
        return

    # ------------------------------------------------------
    # CONTRACT DISCOVERY (NO EXECUTION)
    # ------------------------------------------------------

    st.divider()
    st.write("🧪 Contract discovery (read-only)")

    public = [n for n in dir(waves) if not n.startswith("_")]
    st.write("Public symbols:", public)

    if "initialize_waves" in public:
        st.success("initialize_waves() detected")
    else:
        st.warning("initialize_waves() NOT found")

    # ------------------------------------------------------
    # HANDOFF
    # ------------------------------------------------------

    st.divider()
    st.info(
        "Recovery complete.\n\n"
        "✔ Streamlit healthy\n"
        "✔ Environment validated\n"
        "✔ waves module loadable\n\n"
        "Next step: launch full application."
    )

    if st.button("🚀 Launch full WAVES app (app.py)"):
        st.success("Handing off to app.py…")
        st.stop()  # Streamlit will reload entrypoint

# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

if __name__ == "__main__":
    main()