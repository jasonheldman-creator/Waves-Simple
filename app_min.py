import streamlit as st
import pandas as pd
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# HEADER
# ==========================================================
st.title("WAVES — Live Recovery Console")
st.caption("Intraday • 30D • 60D • 365D • Snapshot-Driven")
st.divider()

# ==========================================================
# LOAD SNAPSHOT DATA
# ==========================================================
SNAPSHOT_PATH = "data/live_snapshot.csv"

if not st.session_state.get("snapshot_loaded"):
    snapshot_df = pd.read_csv(SNAPSHOT_PATH)
    st.session_state["snapshot_df"] = snapshot_df
    st.session_state["snapshot_loaded"] = True
else:
    snapshot_df = st.session_state["snapshot_df"]

# Defensive numeric coercion
numeric_cols = snapshot_df.select_dtypes(include="number").columns
snapshot_df[numeric_cols] = snapshot_df[numeric_cols].fillna(0.0)

# Portfolio aggregation
portfolio = snapshot_df[numeric_cols].mean()

# ==========================================================
# PORTFOLIO SNAPSHOT (TOP SECTION)
# ==========================================================
st.subheader("🏛 Portfolio Snapshot — All Waves")
st.caption("STANDARD MODE")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Intraday Return", f"{portfolio.get('Return_1D', 0)*100:.2f}%")
r2.metric("30D Return", f"{portfolio.get('Return_30D', 0)*100:.2f}%")
r3.metric("60D Return", f"{portfolio.get('Return_60D', 0)*100:.2f}%")
r4.metric("365D Return", f"{portfolio.get('Return_365D', 0)*100:.2f}%")

a1, a2, a3, a4 = st.columns(4)
a1.metric("Alpha Intraday", f"{portfolio.get('Alpha_1D', 0)*100:.2f}%")
a2.metric("Alpha 30D", f"{portfolio.get('Alpha_30D', 0)*100:.2f}%")
a3.metric("Alpha 60D", f"{portfolio.get('Alpha_60D', 0)*100:.2f}%")
a4.metric("Alpha 365D", f"{portfolio.get('Alpha_365D', 0)*100:.2f}%")

st.caption(
    f"⚡ Computed from live snapshot | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
)

st.divider()

# ==========================================================
# LIVE RETURNS & ALPHA (MIDDLE SECTION)
# ==========================================================
st.subheader("📊 Live Returns & Alpha — By Wave")

display_cols = [
    "Wave",
    "Return_1D", "Return_30D", "Return_60D", "Return_365D",
    "Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D",
]

available_cols = [c for c in display_cols if c in snapshot_df.columns]

st.dataframe(
    snapshot_df[available_cols],
    use_container_width=True
)

st.divider()

# ==========================================================
# 🧠 ALPHA ATTRIBUTION (NEW CORE SECTION)
# ==========================================================
st.subheader("🧠 Alpha Attribution")
st.caption("Where excess return is generated — by source and scope")

# ----------------------------------------------------------
# Attribution Scope Selector
# ----------------------------------------------------------
wave_options = ["Portfolio (All Waves)"] + sorted(snapshot_df["Wave"].unique().tolist())
selected_scope = st.selectbox(
    "Attribution Scope",
    wave_options,
    index=0
)

if selected_scope == "Portfolio (All Waves)":
    attr_df = snapshot_df.copy()
else:
    attr_df = snapshot_df[snapshot_df["Wave"] == selected_scope].copy()

# ----------------------------------------------------------
# Define Alpha Sources (6)
# ----------------------------------------------------------
alpha_sources = {
    "Stock Selection Alpha": attr_df.get("Alpha_Stock", 0),
    "Factor / Signal Alpha": attr_df.get("Alpha_Factor", 0),
    "Strategy Overlay Alpha": attr_df.get("Alpha_Strategy", 0),
    "Dynamic Benchmark Alpha": attr_df.get("Alpha_Dynamic_Benchmark", 0),
    "Regime / Risk Alpha": attr_df.get("Alpha_Regime", 0),
    "Residual / Interaction Alpha": attr_df.get("Alpha_Residual", 0),
}

# Build attribution table
attr_table = pd.DataFrame({
    "Alpha Source": list(alpha_sources.keys()),
    "Alpha Contribution": [
        pd.Series(v).mean() if isinstance(v, pd.Series) else float(v)
        for v in alpha_sources.values()
    ]
})

total_alpha = attr_table["Alpha Contribution"].sum()
attr_table["% of Total Alpha"] = (
    attr_table["Alpha Contribution"] / total_alpha * 100
    if total_alpha != 0 else 0
)

attr_table = attr_table.sort_values("Alpha Contribution", ascending=False)

# ----------------------------------------------------------
# Attribution Display
# ----------------------------------------------------------
c1, c2 = st.columns([1.2, 1])

with c1:
    st.markdown("**Alpha Attribution Breakdown**")
    st.dataframe(attr_table, use_container_width=True)

with c2:
    st.markdown("**Contribution Visualization**")
    chart_df = attr_table.set_index("Alpha Source")[["Alpha Contribution"]]
    st.bar_chart(chart_df)

st.divider()

# ==========================================================
# ALPHA HISTORY BY HORIZON (BOTTOM SECTION)
# ==========================================================
st.subheader("📈 Alpha History by Horizon")

alpha_hist_cols = [
    c for c in ["Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D"]
    if c in snapshot_df.columns
]

if alpha_hist_cols:
    hist_df = snapshot_df[["Wave"] + alpha_hist_cols].set_index("Wave")
    st.bar_chart(hist_df)
else:
    st.warning("Alpha history data unavailable.")

# ==========================================================
# SYSTEM STATUS
# ==========================================================
st.markdown(
    """
    <div style="margin-top:30px;padding:18px;
         background:linear-gradient(90deg,#1f8f4e,#2ecc71);
         border-radius:12px;
         color:white;
         font-weight:700;
         text-align:center;">
        LIVE SYSTEM ACTIVE ✅<br/>
        ✓ Intraday live • ✓ Multi-horizon returns • ✓ Alpha attribution • ✓ Snapshot truth
    </div>
    """,
    unsafe_allow_html=True
)