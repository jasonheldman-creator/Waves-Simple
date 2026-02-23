# WAVES Intelligence Streamlit entrypoint
import streamlit as st
from pathlib import Path

if "app_min_loaded" not in st.session_state:
    st.session_state["app_min_loaded"] = True
    with open(Path(__file__).parent / "app_min.py", encoding="utf-8") as _f:
        exec(_f.read(), globals())
