import streamlit as st
import pandas as pd
import os
import traceback
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Minimal Diagnostic Lines Added Below Imports
import streamlit as st
st.write("STREAMLIT BOOT OK")
st.write("Reached top of app.py")

# ============================================================================
# CONFIGURATION
# ============================================================================

PAGE_TITLE = "WAVES Intelligence™ — Minimal Console v1"
DATA_FILE = "live_snapshot.csv"