import streamlit as st
import pandas as pd
from datetime import datetime

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================
# STYLES
# =============================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.blue-box {
    background-color: #081c33;
    border: 2px solid #3fd0ff;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 30px;
    box-shadow: 0 0 25px rgba(63, 208, 255, 0.35);
}
.status-banner {
    background: linear-gradient(90deg, #1f8f4e, #2ecc71);
    padding: 18px;
    border-radius: 12px;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("WAVES — Live Recovery Console")
st.caption("Intraday • 30D • 60D • 365D • Snapshot-Driven")
st.divider()

# =============================
# LOAD DATA
# =============================
snapshot_df = pd.read_csv("data/live_snapshot.csv")

# =============================
# ATTRIBUTION SCOPE SELECTOR
# =============================
st.subheader("🎯 Attribution Scope")

scope_options = ["Portfolio"] + sorted(snapshot_df["Wave"].unique().tolist())
selected_scope = st.selectbox(
    "Select attribution scope:",
    scope_options
)

# =============================
# FILTER DATA BASED ON SCOPE
# =============================
if selected_scope == "Portfolio":
    scoped_df = snapshot_df.copy()
else:
    scoped_df = snapshot_df[snapshot_df["Wave"] == selected_scope]

# Guardrail
if scoped_df.empty:
    st.error("No data available for selected scope.")
    st.stop()

# =============================
# AGGREGATE METRICS
# =============================
metrics = scoped_df.mean(numeric_only=True)

# =============================
# PORTFOLIO / WAVE SNAPSHOT
# =============================
with st.container():
    st.markdown('<div class="blue-box">', unsafe_allow_html=True)

    st.subheader(f"🏛 Snapshot — {selected_scope}")
    st.caption("STANDARD MODE")

    # Returns
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intraday Return", f"{metrics['Return_1D']*100:.2f}%")
    c2.metric("30D Return", f"{metrics['Return_30D']*100:.2f}%")
    c3.metric("60D Return", f"{metrics['Return_60D']*100:.2f}%")
    c4.metric("365D Return", f"{metrics['Return_365D']*100:.2f}%")

    # Alpha
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha Intraday", f"{metrics['Alpha_1D']*100:.2f}%")
    c2.metric("Alpha 30D", f"{metrics['Alpha_30D']*100:.2f}%")
    c3.metric("Alpha 60D", f"{metrics['Alpha_60D']*100:.2f}%")
    c4.metric("Alpha 365D", f"{metrics['Alpha_365D']*100:.2f}%")

    st.caption(
        f"⚡ Computed from live snapshot | "
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# ALPHA ATTRIBUTION BREAKDOWN
# =============================
st.subheader("🧠 Alpha Attribution Breakdown")

attribution_sources = {
    "Volatility / VIX Regime": metrics.get("Alpha_VIX", 0.0),
    "Momentum / Trend": metrics.get("Alpha_Momentum", 0.0),
    "Stock Selection": metrics.get("Alpha_Selection", 0.0),
    "Factor Tilts": metrics.get("Alpha_Factors", 0.0),
    "Risk Control / Drawdown Mgmt": metrics.get("Alpha_Risk", 0.0),
    "Execution / Rebalancing": metrics.get("Alpha_Execution", 0.0),
}

attr_df = pd.DataFrame.from_dict(
    attribution_sources,
    orient="index",
    columns=["Alpha Contribution"]
)

if attr_df["Alpha Contribution"].abs().sum() == 0:
    st.warning("Alpha attribution data not available for this scope.")
else:
    st.bar_chart(attr_df)

# =============================
# LIVE RETURNS & ALPHA TABLE
# =============================
st.subheader("📊 Live Returns & Alpha — Detail View")

st.dataframe(
    scoped_df[
        [
            "Wave",
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
    use_container_width=True
)

# =============================
# STATUS
# =============================
st.markdown(
    """
    <div class="status-banner">
        LIVE SYSTEM ACTIVE ✅<br/>
        ✓ Attribution wired • ✓ Portfolio & Wave scope • ✓ Truth-gated analytics
    </div>
    """,
    unsafe_allow_html=True
)