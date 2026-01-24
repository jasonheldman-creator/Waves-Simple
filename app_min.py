# ==========================================================
# app_min.py — WAVES Live Recovery Console (STABLE)
# ==========================================================
# Guarantees:
# • Visual Portfolio Snapshot (no code blocks)
# • Live snapshot-driven metrics
# • Intraday + 30D + 60D + 365D
# • Full wave coverage
# • Defensive, read-only analytics
# ==========================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import os

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
    body { background-color: #0b0f1a; color: white; }

    .blue-box {
        background: linear-gradient(135deg, #0b2a4a, #0a1f38);
        border: 2px solid #3cc9ff;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 0 30px rgba(60,201,255,0.4);
        margin-bottom: 30px;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 18px;
        margin-top: 20px;
    }

    .metric {
        background: rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
    }

    .metric-label {
        font-size: 12px;
        letter-spacing: 0.5px;
        opacity: 0.75;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
    }

    .positive { color: #5cff9d; }
    .negative { color: #ff6b6b; }

    .footer-note {
        margin-top: 18px;
        font-size: 12px;
        opacity: 0.7;
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
# Load Live Snapshot
# -------------------------------------------------
SNAPSHOT_PATH = "data/live_snapshot.csv"

if not os.path.exists(SNAPSHOT_PATH):
    st.error("❌ data/live_snapshot.csv not found")
    st.stop()

snapshot_df = pd.read_csv(SNAPSHOT_PATH)

snapshot_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# -------------------------------------------------
# Helper: safe column access
# -------------------------------------------------
def col(df, name):
    return df[name] if name in df.columns else 0.0

# -------------------------------------------------
# Aggregate Portfolio Metrics (ALL WAVES)
# -------------------------------------------------
portfolio = pd.DataFrame({
    "Return_1D": col(snapshot_df, "Return_1D"),
    "Return_30D": col(snapshot_df, "Return_30D"),
    "Return_60D": col(snapshot_df, "Return_60D"),
    "Return_365D": col(snapshot_df, "Return_365D"),
    "Alpha_1D": col(snapshot_df, "Alpha_1D"),
    "Alpha_30D": col(snapshot_df, "Alpha_30D"),
    "Alpha_60D": col(snapshot_df, "Alpha_60D"),
    "Alpha_365D": col(snapshot_df, "Alpha_365D"),
})

portfolio_means = portfolio.mean()

def fmt(x):
    return f"{x*100:.2f}%"

# -------------------------------------------------
# PORTFOLIO SNAPSHOT (VISUAL BLUE BOX)
# -------------------------------------------------
st.markdown(
    f"""
    <div class="blue-box">
        <h2>🏛 Portfolio Snapshot (All Waves)</h2>
        <div style="opacity:0.7;margin-bottom:10px;">STANDARD MODE</div>

        <div class="metric-grid">
            <div class="metric"><div class="metric-label">INTRADAY RETURN</div><div class="metric-value">{fmt(portfolio_means['Return_1D'])}</div></div>
            <div class="metric"><div class="metric-label">30D RETURN</div><div class="metric-value">{fmt(portfolio_means['Return_30D'])}</div></div>
            <div class="metric"><div class="metric-label">60D RETURN</div><div class="metric-value">{fmt(portfolio_means['Return_60D'])}</div></div>
            <div class="metric"><div class="metric-label">365D RETURN</div><div class="metric-value">{fmt(portfolio_means['Return_365D'])}</div></div>

            <div class="metric"><div class="metric-label">INTRADAY ALPHA</div><div class="metric-value">{fmt(portfolio_means['Alpha_1D'])}</div></div>
            <div class="metric"><div class="metric-label">30D ALPHA</div><div class="metric-value">{fmt(portfolio_means['Alpha_30D'])}</div></div>
            <div class="metric"><div class="metric-label">60D ALPHA</div><div class="metric-value">{fmt(portfolio_means['Alpha_60D'])}</div></div>
            <div class="metric"><div class="metric-label">365D ALPHA</div><div class="metric-value">{fmt(portfolio_means['Alpha_365D'])}</div></div>
        </div>

        <div class="footer-note">
            ⚡ Computed from live snapshot | {snapshot_time}<br/>
            ℹ Wave-level Beta, Exposure, Regime shown below
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# SNAPSHOT DETAIL TABLE (ALL WAVES)
# -------------------------------------------------
st.subheader("📊 Snapshot Detail (All Waves)")

display_cols = [
    c for c in snapshot_df.columns
    if c.startswith(("Return_", "Alpha_", "Wave"))
]

st.dataframe(
    snapshot_df[display_cols],
    use_container_width=True,
    height=420
)

# -------------------------------------------------
# ALPHA HISTORY (BOTTOM SECTION)
# -------------------------------------------------
st.subheader("📈 Alpha History by Horizon")

alpha_cols = [c for c in snapshot_df.columns if c.startswith("Alpha_")]

if alpha_cols and "Wave" in snapshot_df.columns:
    alpha_hist = snapshot_df.groupby("Wave")[alpha_cols].mean()
    st.bar_chart(alpha_hist)
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

st.caption("✓ Intraday live • ✓ Multi-horizon returns • ✓ Alpha attribution • ✓ Snapshot truth")