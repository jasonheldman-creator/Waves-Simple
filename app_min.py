# app_min.py
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
)

# -------------------------------------------------
# Global Styles (container + glow only, NOT metrics)
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
    }

    .subtle {
        opacity: 0.75;
        font-size: 14px;
    }

    .status-banner {
        background: linear-gradient(90deg, #0f5132, #198754);
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        margin-top: 40px;
        font-size: 18px;
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
# Load Live Snapshot
# -------------------------------------------------
SNAPSHOT_PATH = Path("data/live_snapshot.csv")

if not SNAPSHOT_PATH.exists():
    st.error("❌ live_snapshot.csv not found at data/live_snapshot.csv")
    st.stop()

snapshot_df = pd.read_csv(SNAPSHOT_PATH)

# Normalize column names defensively
snapshot_df.columns = [c.strip() for c in snapshot_df.columns]

snapshot_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# -------------------------------------------------
# PORTFOLIO SNAPSHOT (VISUAL — NO HTML METRICS)
# -------------------------------------------------
st.markdown('<div class="blue-box">', unsafe_allow_html=True)

st.markdown("## 🏛 Portfolio Snapshot (All Waves)")
st.markdown('<div class="subtle">STANDARD MODE</div>', unsafe_allow_html=True)

# ---- Aggregate portfolio-level metrics safely
def safe_mean(col):
    return snapshot_df[col].mean() if col in snapshot_df.columns else None

portfolio_metrics = {
    "Return 1D": safe_mean("Return_1D"),
    "Return 30D": safe_mean("Return_30D"),
    "Return 60D": safe_mean("Return_60D"),
    "Return 365D": safe_mean("Return_365D"),
    "Alpha 1D": safe_mean("Alpha_1D"),
    "Alpha 30D": safe_mean("Alpha_30D"),
    "Alpha 60D": safe_mean("Alpha_60D"),
    "Alpha 365D": safe_mean("Alpha_365D"),
}

# ---- Render metrics visually
cols = st.columns(4)
metric_items = list(portfolio_metrics.items())

for i, (label, value) in enumerate(metric_items):
    col = cols[i % 4]
    if value is None or pd.isna(value):
        col.metric(label, "—")
    else:
        col.metric(label, f"{value:+.2f}%")

st.markdown(
    f"""
    <div class="subtle" style="margin-top:18px;">
        ⚡ Computed from live snapshot | {snapshot_time}<br/>
        ℹ Wave-specific metrics (Beta, Exposure, Cash, VIX regime) shown at wave level
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# LIVE RETURNS & ALPHA TABLE (ALL WAVES)
# -------------------------------------------------
st.subheader("📊 Live Returns & Alpha")

table_cols = [
    c for c in [
        "display_name",
        "Return_1D",
        "Return_30D",
        "Return_60D",
        "Return_365D",
        "Alpha_30D",
        "Alpha_60D",
        "Alpha_365D",
    ]
    if c in snapshot_df.columns
]

if table_cols:
    table_df = snapshot_df[table_cols].copy()
    table_df = table_df.rename(columns={"display_name": "Wave"})
    st.dataframe(table_df, use_container_width=True)
else:
    st.warning("No return/alpha columns available for table.")

# -------------------------------------------------
# ALPHA BY HORIZON (ALL WAVES — SAFE)
# -------------------------------------------------
st.subheader("📈 Alpha by Horizon")

required_alpha_cols = ["Alpha_30D", "Alpha_60D", "Alpha_365D"]

alpha_cols_present = [c for c in required_alpha_cols if c in snapshot_df.columns]

if "display_name" in snapshot_df.columns and alpha_cols_present:
    alpha_chart_df = snapshot_df[["display_name"] + alpha_cols_present]
    alpha_chart_df = alpha_chart_df.set_index("display_name")

    st.bar_chart(alpha_chart_df)
else:
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
    "✓ Intraday live • ✓ Multi-horizon alpha • ✓ Snapshot truth • ✓ No legacy dependencies"
)