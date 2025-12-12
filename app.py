# app.py — WAVES Intelligence™ Institutional Console
#
# Private Logic Phase-1 Refactor:
# - Shows "—" when PL is OFF (regime gate closed)
# - Shows "—" for excluded defensive waves in PL (SmartSafe + ladders)
# - Adds an explicit PL status banner with reasons
#
# This app calls waves_engine.run_engine() each refresh.

import math
import pandas as pd
import streamlit as st

from waves_engine import (
    run_engine,
    MODE_STANDARD,
    MODE_AMB,
    MODE_PL,
)

st.set_page_config(page_title="WAVES Intelligence™ Console", layout="wide")

def fmt_pct(x):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "—"
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "—"

st.title("WAVES Intelligence™ — Institutional Console")
st.caption("Portfolio-Level Overview (All Waves)")

mode = st.radio(
    "Mode",
    options=[MODE_STANDARD, MODE_AMB, MODE_PL],
    horizontal=True,
    index=1
)

with st.spinner("Running engine…"):
    positions, metrics, diagnostics = run_engine(mode=mode)

# Private Logic status banner
if mode == MODE_PL:
    pl = diagnostics.get("pl_gate") or {}
    is_on = bool(pl.get("is_on", False))
    reasons = pl.get("reasons", [])
    vix_last = pl.get("vix_last", None)
    vix_pct = pl.get("vix_percentile_1y", None)
    corr = pl.get("corr_spy_tlt_60d", None)

    colA, colB, colC, colD = st.columns([1.2, 1.0, 1.0, 2.0])
    with colA:
        st.metric("Private Logic Status", "ON ✅" if is_on else "OFF ⛔️")
    with colB:
        st.metric("VIX", "—" if vix_last is None else f"{float(vix_last):.2f}")
    with colC:
        st.metric("VIX 1Y pct", "—" if vix_pct is None else f"{float(vix_pct):.2f}")
    with colD:
        st.metric("SPY/TLT corr60", "—" if corr is None else f"{float(corr):.2f}")

    if reasons:
        if is_on:
            st.success("Gate reasons: " + " | ".join([str(r) for r in reasons]))
        else:
            st.warning("Gate reasons: " + " | ".join([str(r) for r in reasons]))

# Build overview table
rows = []
for m in metrics:
    rows.append({
        "Wave": m.wave,
        "365D Return": fmt_pct(m.ret_365),
        "365D Alpha": fmt_pct(m.alpha_365),
        "Benchmark": m.benchmark,
    })

df = pd.DataFrame(rows)

# Sort: show best alpha at top (when numbers exist), keep dashes at bottom
def alpha_sort_key(a_str):
    if a_str == "—" or not isinstance(a_str, str):
        return -999999
    try:
        return float(a_str.replace("%", ""))
    except Exception:
        return -999999

df["_alpha_sort"] = df["365D Alpha"].apply(alpha_sort_key)
df = df.sort_values("_alpha_sort", ascending=False).drop(columns=["_alpha_sort"])

st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()

# Optional: quick notes
if mode == MODE_PL:
    st.markdown(
        """
**How to read this screen (Private Logic):**
- `—` means **Private Logic is OFF** (regime gate closed) *or* the Wave is **excluded** (SmartSafe / ladders / defensive).
- When the gate is ON, eligible Waves will populate with live metrics.
"""
    )
else:
    st.markdown(
        """
**How to read this screen:**
- This is the consolidated portfolio overview for the selected mode.
"""
    )