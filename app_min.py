import streamlit as st
import pandas as pd
from datetime import datetime

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="WAVES — Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD DATA
# =====================================================
DATA_PATH = "data/live_snapshot.csv"

@st.cache_data
def load_snapshot():
    return pd.read_csv(DATA_PATH)

snapshot_df = load_snapshot()

# =====================================================
# SIDEBAR — GLOBAL STATE
# =====================================================
st.sidebar.title("WAVES Control Panel")

scope = st.sidebar.radio(
    "Attribution Scope",
    ["Portfolio", "Wave"],
    index=0
)

all_waves = sorted(snapshot_df["Wave"].dropna().unique().tolist())

selected_wave = None
if scope == "Wave":
    selected_wave = st.sidebar.selectbox(
        "Select Wave",
        all_waves
    )

st.sidebar.divider()
st.sidebar.caption("All analytics below respond to this selection.")

# =====================================================
# ACTIVE DATAFRAME (SINGLE SOURCE OF TRUTH)
# =====================================================
if scope == "Portfolio":
    active_df = snapshot_df.copy()
else:
    active_df = snapshot_df[snapshot_df["Wave"] == selected_wave].copy()

# =====================================================
# HEADER
# =====================================================
st.title("WAVES — Institutional Recovery Console")
st.caption("Live • Multi-Horizon • Attribution-First")
st.divider()

# =====================================================
# PORTFOLIO / WAVE SNAPSHOT
# =====================================================
st.subheader("🏛 Portfolio Snapshot" if scope == "Portfolio" else f"🏛 Wave Snapshot — {selected_wave}")

agg = active_df.mean(numeric_only=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Intraday Return", f"{agg.get('Return_1D', 0)*100:.2f}%")
c2.metric("30D Return", f"{agg.get('Return_30D', 0)*100:.2f}%")
c3.metric("60D Return", f"{agg.get('Return_60D', 0)*100:.2f}%")
c4.metric("365D Return", f"{agg.get('Return_365D', 0)*100:.2f}%")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Alpha 1D", f"{agg.get('Alpha_1D', 0)*100:.2f}%")
c2.metric("Alpha 30D", f"{agg.get('Alpha_30D', 0)*100:.2f}%")
c3.metric("Alpha 60D", f"{agg.get('Alpha_60D', 0)*100:.2f}%")
c4.metric("Alpha 365D", f"{agg.get('Alpha_365D', 0)*100:.2f}%")

st.caption(f"⚡ Computed from live snapshot | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

st.divider()

# =====================================================
# LIVE RETURNS & ALPHA TABLE
# =====================================================
st.subheader("📊 Live Returns & Alpha")

table_cols = [
    "Wave",
    "Return_1D", "Return_30D", "Return_60D", "Return_365D",
    "Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D"
]

existing_cols = [c for c in table_cols if c in active_df.columns]

st.dataframe(
    active_df[existing_cols],
    use_container_width=True
)

st.divider()

# =====================================================
# ALPHA ATTRIBUTION BREAKDOWN
# =====================================================
st.subheader("🧠 Alpha Attribution Breakdown")

ATTR_SOURCES = [
    "Stock_Selection_Alpha",
    "Strategy_Overlay_Alpha",
    "VIX_Regime_Alpha",
    "Dynamic_Benchmark_Alpha",
    "Timing_Alpha",
    "Residual_Alpha"
]

attr_data = {}
for src in ATTR_SOURCES:
    if src in active_df.columns:
        attr_data[src] = active_df[src].mean()

if attr_data:
    attr_df = pd.DataFrame.from_dict(
        attr_data,
        orient="index",
        columns=["Alpha Contribution"]
    ).sort_values("Alpha Contribution", ascending=False)

    st.bar_chart(attr_df)
else:
    st.info("Attribution columns not yet populated in snapshot.")

st.divider()

# =====================================================
# ALPHA HISTORY BY HORIZON
# =====================================================
st.subheader("📈 Alpha History by Horizon")

horizon_cols = ["Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D"]
existing_horizons = [c for c in horizon_cols if c in active_df.columns]

if existing_horizons:
    hist_df = active_df[["Wave"] + existing_horizons].set_index("Wave")
    st.line_chart(hist_df)
else:
    st.warning("Horizon alpha data unavailable.")

st.divider()

# =====================================================
# STATUS
# =====================================================
st.success("LIVE SYSTEM ACTIVE ✅")

st.caption(
    "✓ Snapshot-driven • ✓ Scope-aware • ✓ Attribution-ready • ✓ Institutional foundation established"
)

# =====================================================
# PLACEHOLDERS (INTENTIONAL)
# =====================================================
st.divider()
st.subheader("🚧 Coming Next")
st.markdown("""
- Factor-level attribution  
- Regime-conditioned attribution  
- Risk-adjusted alpha decomposition  
- Per-wave drilldown pages  
- Exportable institutional reports  
""")