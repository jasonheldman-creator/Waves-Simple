# WAVES Intelligence Streamlit entrypoint
# delegate app_min
import streamlit as st
from pathlib import Path

with open(Path(__file__).parent / "app_min.py", encoding="utf-8") as _f:
    exec(_f.read(), globals())
