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
# Load Live Snapshot (SOURCE OF TRUTH)
# -------------------------------------------------
SNAPSHOT_PATH = "data/live_snapshot.csv"

@st.cache_data
def load_snapshot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

snapshot_df = load_snapshot(SNAPSHOT_PATH)

# -------------------------------------------------
# Helper: Portfolio Aggregation
# -------------------------------------------------
def portfolio_metric(df, col):
    if col not in df.columns:
        return 0.0
    return df[col].mean()

# -------------------------------------------------
# Compute Portfolio Metrics
# -------------------------------------------------
portfolio = {
    "return_1d": portfolio_metric(snapshot_df, "Return_1D"),
    "return_30d": portfolio_metric(snapshot_df, "Return_30D"),
    "return_60d": portfolio_metric(snapshot_df, "Return_60D"),
    "return_365d": portfolio_metric(snapshot_df, "Return_365D"),
    "alpha_1d": portfolio_metric(snapshot_df, "Alpha_1D"),
    "alpha_30d": portfolio_metric(snapshot_df, "Alpha_30D"),
    "alpha_60d": portfolio_metric(snapshot_df, "Alpha_60D"),
    "alpha_365d": portfolio_metric(snapshot_df, "Alpha_365D"),
}

snapshot_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

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
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 18px;
        margin-top: 20px;
    }

    .metric {
        background: rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }

    .metric-label {
        font-size: 13px;
        opacity: 0.75;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
    }

    .footer-note {
        margin-top: 18px;
        font-size: 13px;
        opacity: 0.75;
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
# PORTFOLIO SNAPSHOT (VISUAL BLUE BOX — FIXED)
# -------------------------------------------------
st.markdown(
    f"""
    <div class="blue-box">
        <h2>🏛 Portfolio Snapshot (All Waves)</h2>
        <div style="opacity:0.75;margin-bottom:12px;">STANDARD MODE</div>

        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">Return 1D (Intraday)</div>
                <div class="metric-value">{portfolio['return_1d']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Return 30D</div>
                <div class="metric-value">{portfolio['return_30d']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Return 60D</div>
                <div class="metric-value">{portfolio['return_60d']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Return 365D</div>
                <div class="metric-value">{portfolio['return_365d']:.2%}</div>
            </div>

            <div class="metric">
                <div class="metric-label">Alpha 1D</div>
                <div class="metric-value">{portfolio['alpha_1d']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Alpha 30D</div>
                <div class="metric-value">{portfolio['alpha_30d']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Alpha 60D</div>
                <div class="metric-value">{portfolio['alpha_60d']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Alpha 365D</div>
                <div class="metric-value">{portfolio['alpha_365d']:.2%}</div>
            </div>
        </div>

        <div class="footer-note">
            ⚡ Computed from live snapshot | {snapshot_time}<br/>
            ℹ Wave-specific metrics (Beta, Exposure, Cash, VIX regime) shown at wave level
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# LIVE RETURNS & ALPHA (ALL WAVES)
# -------------------------------------------------
st.subheader("📊 Live Returns & Alpha")

display_cols = [
    "wave_id",
    "display_name",
    "Return_1D",
    "Return_30D",
    "Return_60D",
    "Return_365D",
    "Alpha_30D",
    "Alpha_60D",
    "Alpha_365D",
]

existing_cols = [c for c in display_cols if c in snapshot_df.columns]
st.dataframe(snapshot_df[existing_cols], use_container_width=True)

# -------------------------------------------------
# ALPHA BY HORIZON (ALL WAVES)
# -------------------------------------------------
st.subheader("📈 Alpha by Horizon")

alpha_chart_df = snapshot_df[
    ["display_name", "Alpha_30D", "Alpha_60D", "Alpha_365D"]
].set_index("display_name")

st.bar_chart(alpha_chart_df)

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

st.caption("✓ Intraday live • ✓ Multi-horizon alpha • ✓ Snapshot truth • ✓ No legacy dependencies")