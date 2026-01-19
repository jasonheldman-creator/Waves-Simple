import streamlit as st
import logging
import subprocess
import os
import traceback
import time
import itertools
import html
from datetime import datetime, timedelta

import pandas as pd

# -------------------------------------------------------------------
# Original content omitted for brevity...
# -------------------------------------------------------------------

# Ensure snapshot exists before referencing it
snapshot = {}

# Change 1: Success path debug block (PRICE_BOOK)
st.session_state["portfolio_snapshot_debug"] = {
    "source": "PRICE_BOOK",
    "method": "compute_portfolio_snapshot_from_price_book",
    "computed_at": snapshot.get("computed_at_utc", "N/A"),
    "periods": [1, 30, 60, 365],
}

try:
    # Some processing logic...
    # (left intentionally unchanged)
    pass

except Exception as e:
    # Change 2: Exception path debug block (TruthFrame)
    st.session_state["portfolio_snapshot_debug"] = {
        "source": "TruthFrame",
        "method": "compute_portfolio_snapshot_from_truth",
        "has_error": True,
        "exception_message": str(e),
        "exception_traceback": traceback.format_exc(),
    }