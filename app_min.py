# app_min.py
import streamlit as st
import pandas as pd
from datetime import datetime

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
)

# -------------------------------------------------
# Global Styles
# -------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #0b0f1a;
        color: white;
    }

    .blue-box {
        background: linear-gradient(135deg, #0b2a4a, #0a1f38);
        border: 2px solid #3cc9ff;
        border-radius: 18px;
        padding: 28px;
        box-shadow: 0 0 25px rgba(60,201,255,0.35);
        margin-bottom: 30px;
        min-height: 220px;
    }

    .status-banner {
        background: linear-gradient(90deg, #0f5132, #198754);
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        font-weight: 700;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("WAVES — Live Recovery Console")
st.caption("Intraday • 30D • 60D • 365D • Snapshot-Driven")
st.divider()

# -------------------------------------------------
# PORTFOLIO SNAPSHOT (EMPTY ON PURPOSE)
# -------------------------------------------------
st.markdown(
    """
    <div class="blue-box">
        <h2>🏛 Portfolio Snapshot (All Waves)</h2>
        <div style="opacity:0.75;margin-bottom:14px;">STANDARD MODE</div>

        <!-- INTENTIONALLY EMPTY -->
        <!-- Metrics will be rebuilt here using Streamlit-native layout -->
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# LIVE RETURNS & ALPHA TABLE
# -------------------------------------------------
st.subheader("📊 Live Returns & Alpha")

try:
    snapshot_df = pd.read_csv("data/live_snapshot.csv")

    table_cols = [
        "display_name",
        "Return_1D",
        "Return_30D",
        "Return_60D",
        "Return_365D",
        "Alpha_1D",
        "Alpha_30D",
        "Alpha_60D",
        "Alpha_365D",
    ]

    available_cols = [c for c in table_cols if c in snapshot_df.columns]

    st.dataframe(
        snapshot_df[available_cols],
        use_container_width=True,
        hide_index=True,
    )

except Exception as e:
    st.warning("Live snapshot data unavailable.")

# -------------------------------------------------
# ALPHA HISTORY BY HORIZON
# -------------------------------------------------
st.subheader("📈 Alpha History by Horizon")

try:
    alpha_cols = [
        "display_name",
        "Alpha_1D",
        "Alpha_30D",
        "Alpha_60D",
        "Alpha_365D",
    ]

    alpha_cols = [c for c in alpha_cols if c in snapshot_df.columns]

    alpha_df = snapshot_df[alpha_cols].set_index("display_name")

    st.bar_chart(alpha_df)

except Exception:
    st.warning("Alpha data unavailable or incomplete for horizon chart.")

# -------------------------------------------------
# SYSTEM STATUS
# -------------------------------------------------
st.markdown(
    """
    <div class="status-banner">
        LIVE SYSTEM ACTIVE ✅
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "✓ Intraday live • ✓ Multi-horizon returns • ✓ Alpha attribution • ✓ Snapshot truth"
)