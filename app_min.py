import streamlit as st
import traceback
import sys
import os
from types import ModuleType

# ─────────────────────────────────────────────
# BOOT CONFIRMATION
# ─────────────────────────────────────────────

st.error("APP_MIN EXECUTION STARTED")
st.write("🟢 STREAMLIT EXECUTION STARTED")
st.write("🟢 app_min.py reached line 1")

# ─────────────────────────────────────────────
# MAIN ENTRYPOINT
# ─────────────────────────────────────────────

def main():
    st.title("WAVES — Recovery Mode")
    st.success("app_min.main() is now running")

    # ─────────────────────────────────────────
    # ENVIRONMENT SNAPSHOT
    # ─────────────────────────────────────────

    st.divider()
    st.write("🧭 Runtime environment")

    try:
        st.write("Python version:", sys.version)
        st.write("Working directory:", os.getcwd())
        st.write("Files in root:", sorted(os.listdir(".")))
        st.success("Environment snapshot complete")
    except Exception as e:
        st.error("Environment snapshot failed")
        st.exception(e)

    # ─────────────────────────────────────────
    # WAVES IMPORT
    # ─────────────────────────────────────────

    st.divider()
    st.write("🔍 Import diagnostics starting...")

    try:
        import waves
        st.success("✅ waves module imported successfully")
    except Exception as e:
        st.error("❌ waves import failed — hard stop")
        st.exception(e)
        st.code(traceback.format_exc())
        return  # do not continue if import fails

    # ─────────────────────────────────────────
    # WAVES MODULE INTROSPECTION (READ-ONLY)
    # ─────────────────────────────────────────

    st.divider()
    st.write("🧪 waves module inspection (safe)")

    try:
        st.write("Module file:", waves.__file__)

        public_symbols = [
            name for name in dir(waves)
            if not name.startswith("_")
        ]

        st.write("Total public symbols:", len(public_symbols))
        st.write("Public symbols (first 40):", public_symbols[:40])

        # classify symbols without touching them
        functions = []
        classes = []
        modules = []

        for name in public_symbols:
            try:
                attr = getattr(waves, name)
                if isinstance(attr, ModuleType):
                    modules.append(name)
                elif callable(attr):
                    functions.append(name)
                elif isinstance(attr, type):
                    classes.append(name)
            except Exception:
                pass  # stay read-only

        st.divider()
        st.write("🧬 waves symbol breakdown")
        st.write("Functions (sample):", functions[:15])
        st.write("Classes (sample):", classes[:15])
        st.write("Sub-modules (sample):", modules[:15])

        st.success("waves module inspection completed safely")

    except Exception as e:
        st.error("waves inspection failed")
        st.exception(e)
        st.code(traceback.format_exc())

    # ─────────────────────────────────────────
    # NEXT-STAGE MARKER
    # ─────────────────────────────────────────

    st.divider()
    st.info(
        "Recovery Mode active.\n\n"
        "✔ Import stable\n"
        "✔ No execution side-effects\n"
        "✔ Ready for selective re-hydration"
    )


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()