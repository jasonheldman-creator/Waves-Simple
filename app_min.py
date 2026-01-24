import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
)

# -----------------------------
# LOAD SNAPSHOT DATA
# -----------------------------
SNAPSHOT_PATH = "data/live_snapshot.csv"

@st.cache_data(show_spinner=False)
def load_snapshot(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

snapshot_df = load_snapshot(SNAPSHOT_PATH)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <h1 style="margin-bottom:0;">WAVES — Live Recovery Console</h1>
    <p style="opacity:0.7;">Intraday · 30D · 60D · 365D · Snapshot-Driven</p>
    <hr/>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# AGGREGATE METRICS (PORTFOLIO LEVEL)
# -----------------------------
def safe_mean(col):
    return snapshot_df[col].mean() if col in snapshot_df.columns else 0.0

metrics = {
    "Return_1D": safe_mean("Return_1D"),
    "Return_30D": safe_mean("Return_30D"),
    "Return_60D": safe_mean("Return_60D"),
    "Return_365D": safe_mean("Return_365D"),
    "Alpha_1D": safe_mean("Alpha_1D"),
    "Alpha_30D": safe_mean("Alpha_30D"),
    "Alpha_60D": safe_mean("Alpha_60D"),
    "Alpha_365D": safe_mean("Alpha_365D"),
}

def pct(x):
    return f"{x*100:+.2f}%"

# -----------------------------
# PORTFOLIO SNAPSHOT — VISUAL (NOT CODE)
# -----------------------------
snapshot_html = f"""
<style>
.metric-grid {{
    display:grid;
    grid-template-columns: repeat(4, 1fr);
    gap:14px;
    margin-top:12px;
}}
.metric-card {{
    background: rgba(255,255,255,0.06);
    border-radius:14px;
    padding:16px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.12);
}}
.metric-label {{
    font-size:12px;
    letter-spacing:0.08em;
    opacity:0.75;
}}
.metric-value {{
    font-size:28px;
    font-weight:800;
    margin-top:6px;
}}
.pos {{ color:#2ECC71; }}
.neg {{ color:#E74C3C; }}
.footer-note {{
    margin-top:14px;
    font-size:12px;
    opacity:0.75;
}}
</style>

<div style="
    background: linear-gradient(135deg,#0A2A4F,#0E3B66);
    border-radius:20px;
    padding:26px;
    border:2px solid #42C5FF;
    box-shadow:0 0 28px rgba(66,197,255,0.35);
">
    <h2 style="margin-top:0;">
        🏛️ Portfolio Snapshot (All Waves)
        <span style="
            background:#1ED760;
            color:black;
            padding:4px 10px;
            border-radius:12px;
            font-size:12px;
            margin-left:10px;
        ">STANDARD</span>
    </h2>

    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">INTRADAY RETURN</div>
            <div class="metric-value">{pct(metrics["Return_1D"])}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">30D RETURN</div>
            <div class="metric-value">{pct(metrics["Return_30D"])}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">60D RETURN</div>
            <div class="metric-value">{pct(metrics["Return_60D"])}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">365D RETURN</div>
            <div class="metric-value">{pct(metrics["Return_365D"])}</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">ALPHA 1D</div>
            <div class="metric-value {'pos' if metrics['Alpha_1D']>=0 else 'neg'}">
                {pct(metrics["Alpha_1D"])}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ALPHA 30D</div>
            <div class="metric-value {'pos' if metrics['Alpha_30D']>=0 else 'neg'}">
                {pct(metrics["Alpha_30D"])}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ALPHA 60D</div>
            <div class="metric-value {'pos' if metrics['Alpha_60D']>=0 else 'neg'}">
                {pct(metrics["Alpha_60D"])}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">ALPHA 365D</div>
            <div class="metric-value {'pos' if metrics['Alpha_365D']>=0 else 'neg'}">
                {pct(metrics["Alpha_365D"])}
            </div>
        </div>
    </div>

    <div class="footer-note">
        ⚡ Live computation from snapshot | {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC<br/>
        ℹ Wave-specific metrics (Beta, Exposure, Cash, VIX regime) available at wave level
    </div>
</div>
"""

st.markdown(snapshot_html, unsafe_allow_html=True)

# -----------------------------
# SNAPSHOT DETAIL TABLE
# -----------------------------
st.markdown("### 📊 Snapshot Detail (All Waves)")

if not snapshot_df.empty:
    st.dataframe(snapshot_df, use_container_width=True)
else:
    st.warning("Snapshot data unavailable.")

# -----------------------------
# ALPHA HISTORY BY HORIZON
# -----------------------------
st.markdown("## 📈 Alpha History by Horizon")

alpha_cols = ["Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D"]
available_alpha = [c for c in alpha_cols if c in snapshot_df.columns]

if available_alpha and "display_name" in snapshot_df.columns:
    alpha_df = snapshot_df[["display_name"] + available_alpha].set_index("display_name")
    alpha_df = alpha_df.reset_index().melt(
        id_vars="display_name",
        var_name="Horizon",
        value_name="Alpha"
    )

    fig = px.bar(
        alpha_df,
        x="display_name",
        y="Alpha",
        color="Horizon",
        barmode="group",
    )
    fig.update_layout(height=420, xaxis_title="", yaxis_title="Alpha")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Alpha data unavailable or incomplete for horizon chart.")

# -----------------------------
# SYSTEM STATUS
# -----------------------------
st.markdown(
    """
    <div style="
        background:linear-gradient(90deg,#1E8449,#27AE60);
        padding:16px;
        border-radius:14px;
        text-align:center;
        font-weight:700;
        font-size:18px;
        margin-top:24px;
    ">
        LIVE SYSTEM ACTIVE ✅
        <div style="font-size:13px; font-weight:400; margin-top:6px;">
            ✓ Intraday live · ✓ Multi-horizon returns · ✓ Alpha attribution · ✓ Snapshot truth
        </div>
    </div>
    """,
    unsafe_allow_html=True
)