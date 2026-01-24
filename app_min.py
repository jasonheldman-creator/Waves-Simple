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
# Global Styles (BOLDER EXECUTIVE MODE)
# -------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #0b0f1a;
        color: white;
    }

    .blue-box {
        background: linear-gradient(135deg, #0b2f55, #0a1e3a);
        border: 3px solid #4fd2ff;
        border-radius: 22px;
        padding: 34px;
        box-shadow: 0 0 45px rgba(79,210,255,0.45);
        margin-bottom: 40px;
    }

    .snapshot-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 6px;
    }

    .snapshot-title {
        font-size: 30px;
        font-weight: 800;
    }

    .mode-pill {
        background: #1cff9a;
        color: #00331f;
        font-size: 13px;
        font-weight: 800;
        padding: 6px 14px;
        border-radius: 999px;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin-top: 26px;
    }

    .metric {
        background: rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
    }

    .metric-label {
        font-size: 13px;
        letter-spacing: 0.04em;
        opacity: 0.7;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 34px;
        font-weight: 800;
    }

    .positive {
        color: #2dffb3;
    }

    .negative {
        color: #ff6b6b;
    }

    .footer-note {
        margin-top: 22px;
        font-size: 13px;
        opacity: 0.75;
        text-align: center;
    }

    .status-banner {
        background: linear-gradient(90deg, #0f5132, #1fbf75);
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-weight: 800;
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
# LOAD LIVE SNAPSHOT
# -------------------------------------------------
SNAPSHOT_PATH = "data/live_snapshot.csv"

snapshot_df = pd.read_csv(SNAPSHOT_PATH)

snapshot_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# -------------------------------------------------
# PORTFOLIO-LEVEL AGGREGATION (ALL WAVES)
# -------------------------------------------------
def pct(x):
    return f"{x*100:.2f}%"

returns = {
    "Return 1D": snapshot_df["Return_1D"].mean(),
    "Return 30D": snapshot_df["Return_30D"].mean(),
    "Return 60D": snapshot_df["Return_60D"].mean(),
    "Return 365D": snapshot_df["Return_365D"].mean(),
}

alphas = {
    "Alpha 1D": snapshot_df["Alpha_1D"].mean(),
    "Alpha 30D": snapshot_df["Alpha_30D"].mean(),
    "Alpha 60D": snapshot_df["Alpha_60D"].mean(),
    "Alpha 365D": snapshot_df["Alpha_365D"].mean(),
}

# -------------------------------------------------
# PORTFOLIO SNAPSHOT (BOLD BLUE BOX)
# -------------------------------------------------
st.markdown(
    f"""
    <div class="blue-box">
        <div class="snapshot-header">
            <div class="snapshot-title">🏛 Portfolio Snapshot (All Waves)</div>
            <div class="mode-pill">STANDARD</div>
        </div>

        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">1D RETURN (INTRADAY)</div>
                <div class="metric-value">{pct(returns["Return 1D"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">30D RETURN</div>
                <div class="metric-value">{pct(returns["Return 30D"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">60D RETURN</div>
                <div class="metric-value">{pct(returns["Return 60D"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">365D RETURN</div>
                <div class="metric-value">{pct(returns["Return 365D"])}</div>
            </div>

            <div class="metric">
                <div class="metric-label">ALPHA 1D</div>
                <div class="metric-value {'positive' if alphas["Alpha 1D"] >= 0 else 'negative'}">{pct(alphas["Alpha 1D"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">ALPHA 30D</div>
                <div class="metric-value {'positive' if alphas["Alpha 30D"] >= 0 else 'negative'}">{pct(alphas["Alpha 30D"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">ALPHA 60D</div>
                <div class="metric-value {'positive' if alphas["Alpha 60D"] >= 0 else 'negative'}">{pct(alphas["Alpha 60D"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">ALPHA 365D</div>
                <div class="metric-value {'positive' if alphas["Alpha 365D"] >= 0 else 'negative'}">{pct(alphas["Alpha 365D"])}</div>
            </div>
        </div>

        <div class="footer-note">
            ⚡ Live computation from snapshot | {snapshot_time}<br/>
            ℹ Wave-specific metrics (Beta, Exposure, Cash, VIX regime) available at wave level
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# LIVE RETURNS & ALPHA TABLE (ALL WAVES)
# -------------------------------------------------
st.subheader("📊 Live Returns & Alpha")

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

st.dataframe(
    snapshot_df[table_cols].rename(columns={"display_name": "Wave"}),
    use_container_width=True,
)

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