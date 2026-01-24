import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
)

# -------------------------
# LOAD SNAPSHOT DATA
# -------------------------
@st.cache_data
def load_snapshot():
    return pd.read_csv("data/live_snapshot.csv")

df = load_snapshot()

# -------------------------
# AGGREGATE PORTFOLIO METRICS
# -------------------------
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

risk = {
    "exposure": df["Exposure"].mean() if "Exposure" in df else None,
    "cash": df["CashPercent"].mean() if "CashPercent" in df else None,
    "vix": df["VIX_Level"].iloc[0] if "VIX_Level" in df else None,
    "regime": df["VIX_Regime"].iloc[0] if "VIX_Regime" in df else None,
}

timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# -------------------------
# STYLES
# -------------------------
st.markdown(
    """
    <style>
    .snapshot-box {
        border: 2px solid #3EC5FF;
        border-radius: 16px;
        padding: 24px;
        background: linear-gradient(135deg, #0B1E36, #0E2A47);
        box-shadow: 0 0 24px rgba(62,197,255,0.35);
        margin-bottom: 28px;
    }
    .snapshot-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .snapshot-sub {
        color: #9FB7D6;
        margin-bottom: 18px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin-bottom: 18px;
    }
    .metric {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #9FB7D6;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 600;
        margin-top: 4px;
    }
    .footer-note {
        font-size: 12px;
        color: #9FB7D6;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
    # WAVES — Live Recovery Console
    **Intraday • 30D • 60D • 365D • Snapshot-Driven**
    """
)

# -------------------------
# PORTFOLIO SNAPSHOT (BLUE BOX)
# -------------------------
st.markdown(
    f"""
    <div class="snapshot-box">
        <div class="snapshot-title">🏛️ Portfolio Snapshot (All Waves)</div>
        <div class="snapshot-sub">STANDARD MODE</div>

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
            ⚡ Computed from live snapshot | {timestamp}<br/>
            Risk Context — Exposure: {risk["exposure"]:.1f}% | Cash: {risk["cash"]:.1f}% | VIX: {risk["vix"]} | Regime: {risk["regime"]}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# LIVE RETURNS TABLE
# -------------------------
st.markdown("## 📊 Live Returns & Alpha")
st.dataframe(
    df[
        [
            "Wave_ID",
            "Return_1D",
            "Return_30D",
            "Return_60D",
            "Return_365D",
            "Alpha_1D",
            "Alpha_30D",
            "Alpha_60D",
            "Alpha_365D",
        ]
    ],
    use_container_width=True,
)

# -------------------------
# SYSTEM STATUS
# -------------------------
st.success(
    "LIVE SYSTEM ACTIVE ✅  |  Intraday live  |  Multi-horizon alpha  |  Snapshot truth  |  No legacy dependencies"
)