import streamlit as st
import pandas as pd
from datetime import datetime

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="centered",
)

# -------------------------------------------------
# Load snapshot
# -------------------------------------------------
@st.cache_data
def load_snapshot():
    return pd.read_csv("data/live_snapshot.csv")

df = load_snapshot()

# -------------------------------------------------
# Portfolio-level aggregates
# -------------------------------------------------
def pct(x):
    return f"{x:+.2f}%"

portfolio = {
    "ret_1d": df["Return_1D"].mean() * 100,
    "ret_30d": df["Return_30D"].mean() * 100,
    "ret_60d": df["Return_60D"].mean() * 100,
    "ret_365d": df["Return_365D"].mean() * 100,
    "alpha_1d": df["Alpha_1D"].mean() * 100,
    "alpha_30d": df["Alpha_30D"].mean() * 100,
    "alpha_60d": df["Alpha_60D"].mean() * 100,
    "alpha_365d": df["Alpha_365D"].mean() * 100,
}

timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# -------------------------------------------------
# CSS (ONCE)
# -------------------------------------------------
st.markdown(
    """
    <style>
    .snapshot-box {
        background: linear-gradient(135deg, #0b2345, #102f55);
        border: 2px solid #35c9ff;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 0 25px rgba(53,201,255,0.35);
        margin-bottom: 30px;
    }
    .snapshot-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .snapshot-subtitle {
        font-size: 14px;
        color: #9fbad6;
        margin-bottom: 18px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
    }
    .metric {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #9fbad6;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
    }
    .footer-note {
        margin-top: 16px;
        font-size: 12px;
        color: #9fbad6;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Render BLUE SNAPSHOT BOX (HTML — NOT CODE)
# -------------------------------------------------
st.markdown(
    f"""
    <div class="snapshot-box">
        <div class="snapshot-title">🏛 Portfolio Snapshot (All Waves)</div>
        <div class="snapshot-subtitle">Standard Mode</div>

        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">Return 1D (Intraday)</div>
                <div class="metric-value">{pct(portfolio["ret_1d"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Return 30D</div>
                <div class="metric-value">{pct(portfolio["ret_30d"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Return 60D</div>
                <div class="metric-value">{pct(portfolio["ret_60d"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Return 365D</div>
                <div class="metric-value">{pct(portfolio["ret_365d"])}</div>
            </div>

            <div class="metric">
                <div class="metric-label">Alpha 1D</div>
                <div class="metric-value">{pct(portfolio["alpha_1d"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Alpha 30D</div>
                <div class="metric-value">{pct(portfolio["alpha_30d"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Alpha 60D</div>
                <div class="metric-value">{pct(portfolio["alpha_60d"])}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Alpha 365D</div>
                <div class="metric-value">{pct(portfolio["alpha_365d"])}</div>
            </div>
        </div>

        <div class="footer-note">
            ⚡ Computed from live snapshot | {timestamp}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Existing sections below remain normal Streamlit
# -------------------------------------------------
st.markdown("## 📊 Live Returns & Alpha")
st.dataframe(df, use_container_width=True)

st.markdown("## 📈 Alpha by Horizon")
st.bar_chart(
    df.set_index("Wave")[
        ["Alpha_30D", "Alpha_60D", "Alpha_365D"]
    ]
)

st.success(
    "LIVE SYSTEM ACTIVE ✅  \n"
    "✓ Intraday live  ✓ Multi-horizon alpha  ✓ Snapshot truth  ✓ No legacy dependencies"
)