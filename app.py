# sandbox_app.py
# WAVES Intelligence™ — Sandbox
# Small–Mid Cap Value Acceleration Wave (Interactive Mock)

import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="WAVES Intelligence™ — Sandbox", layout="wide")

st.title("WAVES Intelligence™ — Sandbox")
st.caption("Safe testing environment — production app.py untouched ✅")
st.success("Sandbox loaded successfully.")

st.markdown("---")

st.subheader("📈 Small–Mid Cap Value Acceleration Wave")
st.caption("Goal: capture SMID value + acceleration using simple, explainable filters + ranking.")

# -----------------------------
# Controls (Jason-friendly)
# -----------------------------
with st.expander("🔧 Filters & Construction Controls", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        min_qoq_rev = st.slider("Min QoQ Revenue Growth (%)", 0, 80, 20, 1)
        min_qoq_eps = st.slider("Min QoQ Earnings Growth (%)", 0, 120, 25, 1)
    with c2:
        max_pe = st.slider("Max P/E", 5, 30, 12, 1)
        max_names = st.slider("Max holdings", 5, 30, 10, 1)
    with c3:
        seed = st.number_input("Random seed (stability)", min_value=1, max_value=9999, value=7, step=1)

st.markdown("")

# -----------------------------
# Mock Universe (still mock but richer)
# -----------------------------
rng = random.Random(int(seed))
tickers = [
    "ALIT","CNMD","DXLG","HBB","HZO","MLKN","PRTS","RCMT","TUP","VIRC",
    "SMCI","APLD","GDYN","CLMB","RAMP","SSTK","BLKB","EVCM","ARLO","OSPN",
    "SGH","CALX","ACLS","KN","CMTL","SAFT","HDSN","LZB","GIII","KTB"
]

rows = []
for t in tickers:
    rev = round(rng.uniform(5, 60), 1)
    eps = round(rng.uniform(-10, 120), 1)
    pe = round(rng.uniform(6, 26), 1)
    # add a couple “quality/risk” helpers (still mock)
    ocf = round(rng.uniform(-50, 250), 1)     # $M
    vol = round(rng.uniform(18, 65), 1)       # %
    rows.append({
        "Ticker": t,
        "QoQ Revenue Growth (%)": rev,
        "QoQ Earnings Growth (%)": eps,
        "P/E Ratio": pe,
        "Operating Cash Flow ($M)": ocf,
        "Volatility (60D, %)": vol,
    })

df = pd.DataFrame(rows)

# -----------------------------
# Filter
# -----------------------------
filtered = df[
    (df["QoQ Revenue Growth (%)"] >= min_qoq_rev) &
    (df["QoQ Earnings Growth (%)"] >= min_qoq_eps) &
    (df["P/E Ratio"] <= max_pe)
].copy()

# If nothing passes, show guidance
if filtered.empty:
    st.warning("No stocks pass the current filters. Try lowering growth thresholds or increasing Max P/E.")
    st.dataframe(df, use_container_width=True)
    st.stop()

# -----------------------------
# Ranking / Score (explainable)
# -----------------------------
# Higher rev + higher eps + lower P/E + positive OCF + lower vol wins
filtered["RevScore"] = filtered["QoQ Revenue Growth (%)"].rank(pct=True)
filtered["EpsScore"] = filtered["QoQ Earnings Growth (%)"].rank(pct=True)
filtered["ValueScore"] = (1.0 / filtered["P/E Ratio"]).rank(pct=True)
filtered["OCFScore"] = filtered["Operating Cash Flow ($M)"].rank(pct=True)
filtered["StabilityScore"] = (1.0 / filtered["Volatility (60D, %)"]).rank(pct=True)

filtered["CompositeScore"] = (
    0.35 * filtered["RevScore"] +
    0.35 * filtered["EpsScore"] +
    0.20 * filtered["ValueScore"] +
    0.05 * filtered["OCFScore"] +
    0.05 * filtered["StabilityScore"]
)

filtered = filtered.sort_values("CompositeScore", ascending=False).head(int(max_names)).reset_index(drop=True)

# -----------------------------
# Weights (score-weighted, capped)
# -----------------------------
weights_raw = filtered["CompositeScore"].clip(lower=0.0001)
weights = (weights_raw / weights_raw.sum()) * 100.0

# cap each weight at 15%
# cap and ship constraint