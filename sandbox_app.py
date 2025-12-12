# sandbox_app.py
# WAVES Intelligence™ — Sandbox
# Small–Mid Cap Value Acceleration Wave (Mock)

import streamlit as st
import pandas as pd
import random

st.set_page_config(
    page_title="WAVES Intelligence™ — Sandbox",
    layout="wide"
)

st.title("WAVES Intelligence™ — Sandbox")
st.caption("Safe testing environment — production app.py untouched")

st.success("Sandbox loaded successfully.")

st.markdown("---")

# ============================================================
# Small–Mid Cap Value Acceleration Wave (Mock)
# ============================================================

st.subheader("📈 Small–Mid Cap Value Acceleration Wave")
st.caption(
    "Filters: 20%+ QoQ Revenue • 25%+ QoQ Earnings • P/E ≤ 12"
)

# Mock universe
tickers = [
    "ALIT", "CNMD", "DXLG", "HBB", "HZO",
    "MLKN", "PRTS", "RCMT", "TUP", "VIRC"
]

data = []

for t in tickers:
    data.append({
        "Ticker": t,
        "QoQ Revenue Growth (%)": round(random.uniform(20, 45), 1),
        "QoQ Earnings Growth (%)": round(random.uniform(25, 70), 1),
        "P/E Ratio": round(random.uniform(6, 12), 1),
        "Weight (%)": round(100 / len(tickers), 2)
    })

df = pd.DataFrame(data)

# Apply filters
filtered = df[
    (df["QoQ Revenue Growth (%)"] >= 20) &
    (df["QoQ Earnings Growth (%)"] >= 25) &
    (df["P/E Ratio"] <= 12)
].reset_index(drop=True)

st.markdown("### Selected Holdings")
st.dataframe(filtered, use_container_width=True)

# ============================================================
# Performance Snapshot (Mock)
# ============================================================

st.markdown("### Performance Snapshot (Mock)")

perf = pd.DataFrame({
    "Metric": ["30D Return", "60D Return", "365D Return", "Alpha (vs Russell 2000)"],
    "Value": ["+6.4%", "+11.2%", "+28.9%", "+9.6%"]
})

st.table(perf)

st.info(
    "ℹ️ Returns and alpha shown here are placeholders. "
    "This sandbox is for structure, math logic, and Wave design only."
)

st.markdown("---")

st.subheader("Next Steps")
st.markdown("""
- Tune filters (tighten / loosen thresholds)
- Add ranking logic (rev + earnings acceleration score)
- Plug into real data later
- Graduate Wave into production when ready
""")