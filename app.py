# app.py — WAVES Launcher (Safe)
# Purpose: Run a sandbox app (if present) without permanently changing your production system.
# Tries (in order):
#   1) sandbox_app.py (repo root)            -> module: sandbox_app
#   2) waves_core_v1/sandbox_app.py          -> module: waves_core_v1.sandbox_app
#   3) waves_core_v1/app.py                  -> module: waves_core_v1.app
#
# If none are found, shows a clear Streamlit error with exact next steps.

import importlib
import sys
import os

import streamlit as st


def try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        return e


def main():
    st.set_page_config(page_title="WAVES Launcher", layout="wide")
    st.title("WAVES Intelligence™ — Launcher")
    st.caption("Trying to load sandbox app…")

    candidates = [
        ("sandbox_app", "Root: sandbox_app.py"),
        ("waves_core_v1.sandbox_app", "Folder: waves_core_v1/sandbox_app.py"),
        ("waves_core_v1.app", "Folder: waves_core_v1/app.py"),
    ]

    errors = []
    for mod, label in candidates:
        result = try_import(mod)
        if not isinstance(result, Exception):
            st.success(f"Loaded: {label}")
            # If the module has a main() function, call it. Otherwise it likely runs on import.
            if hasattr(result, "main") and callable(getattr(result, "main")):
                result.main()
            return
        else:
            errors.append((label, mod, result))

    st.error("Sandbox app not found (or failed to import).")
    st.write("Here’s what I tried:")
    for label, mod, err in errors:
        st.code(f"{label}\nmodule: {mod}\nerror: {repr(err)}")

    st.markdown("### Fix (fastest)")
    st.markdown(
        "Create **one file** in the **repo root** named:\n\n"
        "`sandbox_app.py`\n\n"
        "Then paste the full sandbox code into it and commit.\n\n"
        "After that: go to Streamlit Cloud → **Reboot**."
    )

    st.markdown("### Quick check in GitHub mobile")
    st.markdown(
        "- Use repo search for: `sandbox_app.py`\n"
        "- If it exists inside a folder, move/copy it to the repo root.\n"
        "- Ensure the filename is exactly: `sandbox_app.py` (all lowercase)."
    )


if __name__ == "__main__":
    main()