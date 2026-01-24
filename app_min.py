# ==========================================================
# app_min.py — WAVES Live Recovery Console (Stable Rewrite)
# ==========================================================
# • Snapshot-driven
# • Intraday + 30D + 60D + 365D
# • No legacy wave init
# • Streamlit-native layout (mobile safe)
# ==========================================================

import streamlit as st
import pandas as pd
import os
from datetime import datetime, timezone

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------

st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------------
# BOOT CONFIRMATION
# ----------------------------------------------------------

st.success("STREAMLIT EXECUTION STARTED")
st.caption("app_min.py reached line 1")

# ----------------------------------------------------------
# LOAD SNAPSHOT
# ----------------------------------------------------------

SNAPSHOT_PATH = "data/live_snapshot.csv"

if not os.path.exists(SNAPSHOT_PATH):
    st.error("❌ live_snapshot.csv not found")
    st.stop()

df = pd.read_csv(SNAPSHOT_PATH)

st.success("Live snapshot loaded")
st.caption(f"Rows: {len(df)}")

# ----------------------------------------------------------
# BASIC VALIDATION
# ----------------------------------------------------------

required_cols = [
    "Return_1D", "Return_30D", "Return_60D", "Return_365D",
    "Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ----------------------------------------------------------
# AGGREGATE PORTFOLIO METRICS
# ----------------------------------------------------------

def mean_safe(series):
    return float(series.dropna().mean()) if len(series.dropna()) > 0 else 0.0

portfolio = {
    "ret_1d": mean_safe(df["Return_1D"]),
    "ret_30d": mean_safe(df["Return_30D"]),
    "ret_60d": mean_safe(df["Return_60D"]),
    "ret_365d": mean_safe(df["Return_365D"]),
    "alpha_1d": mean_safe(df["Alpha_1D"]),
    "alpha_30d": mean_safe(df["Alpha_30D"]),
    "alpha_60d": mean_safe(df["Alpha_60D"]),
    "alpha_365d": mean_safe(df["Alpha_365D"]),
}

# Optional risk metrics (safe fallback)
exposure = mean_safe(df["Exposure"]) if "Exposure" in df.columns else None
cash = mean_safe(df["CashPercent"]) if "CashPercent" in df.columns else None
vix = mean_safe(df["VIX_Level"]) if "VIX_Level" in df.columns else None
regime = df["VIX_Regime"].mode()[0] if "VIX_Regime" in df.columns and len(df) else "N/A"

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------

st.title("WAVES — Live Recovery Console")
st.caption("Intraday • 30D • 60D • 365D • Snapshot-Driven")

# ----------------------------------------------------------
# BLUE PORTFOLIO SNAPSHOT BOX
# ----------------------------------------------------------

with st.container():
    st.markdown("### 🏛️ Portfolio Snapshot (All Waves)")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Return 1D (Intraday)", f"{portfolio['ret_1d']:+.2%}")
    col2.metric("Return 30D", f"{portfolio['ret_30d']:+.2%}")
    col3.metric("Return 60D", f"{portfolio['ret_60d']:+.2%}")
    col4.metric("Return 365D", f"{portfolio['ret_365d']:+.2%}")

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Alpha 1D", f"{portfolio['alpha_1d']:+.2%}")
    col6.metric("Alpha 30D", f"{portfolio['alpha_30d']:+.2%}")
    col7.metric("Alpha 60D", f"{portfolio['alpha_60d']:+.2%}")
    col8.metric("Alpha 365D", f"{portfolio['alpha_365d']:+.2%}")

    st.caption(
        f"⚡ Computed from live snapshot | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    if exposure is not None:
        st.caption(
            f"Risk Context — Exposure: {exposure:.1f}% | Cash: {cash:.1f}% | VIX: {vix:.1f} | Regime: {regime}"
        )

# ----------------------------------------------------------
# LIVE RETURNS TABLE
# ----------------------------------------------------------

st.markdown("### 📊 Live Returns & Alpha")

table_cols = [
    "Wave_ID",
    "Return_1D", "Alpha_1D",
    "Return_30D", "Alpha_30D",
    "Return_60D", "Alpha_60D",
    "Return_365D", "Alpha_365D",
]

visible_cols = [c for c in table_cols if c in df.columns]

st.dataframe(
    df[visible_cols].sort_values("Alpha_365D", ascending=False),
    use_container_width=True
)

# ----------------------------------------------------------
# ALPHA BY HORIZON CHART
# ----------------------------------------------------------

st.markdown("### 📈 Alpha by Horizon")

chart_df = (
    df.groupby("Wave_ID")[["Alpha_30D", "Alpha_60D", "Alpha_365D"]]
    .mean()
    .reset_index()
)

st.bar_chart(
    chart_df.set_index("Wave_ID"),
    use_container_width=True
)

# ----------------------------------------------------------
# FINAL STATUS
# ----------------------------------------------------------

st.success("LIVE SYSTEM ACTIVE ✅")
st.caption(
    "✓ Intraday live ✓ Multi-horizon alpha ✓ Snapshot truth ✓ No legacy dependencies"
)

with st.expander("Preview snapshot (first 10 rows)"):
    st.dataframe(df.head(10))