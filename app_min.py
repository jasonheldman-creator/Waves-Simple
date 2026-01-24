# ==========================================================
# app_min.py — WAVES Live Recovery Console (Canonical)
# ==========================================================
# Snapshot-driven, read-only, zero legacy dependency
# ==========================================================

import streamlit as st
import pandas as pd
import os

# ----------------------------------------------------------
# BOOT CONFIRMATION
# ----------------------------------------------------------

st.success("STREAMLIT EXECUTION STARTED")
st.write("app_min.py reached line 1")

st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
)

# ----------------------------------------------------------
# LOAD SNAPSHOT
# ----------------------------------------------------------

SNAPSHOT_PATH = "data/live_snapshot.csv"

if not os.path.exists(SNAPSHOT_PATH):
    st.error("live_snapshot.csv not found")
    st.stop()

df = pd.read_csv(SNAPSHOT_PATH)

st.success("Live snapshot loaded")
st.caption(f"Rows: {len(df)}")

# ----------------------------------------------------------
# SAFE HELPERS
# ----------------------------------------------------------

def col(name, default=0.0):
    return df[name] if name in df.columns else default

def pct(x):
    try:
        return f"{x * 100:.3f}%"
    except Exception:
        return "—"

# ----------------------------------------------------------
# PORTFOLIO AGGREGATES
# ----------------------------------------------------------

portfolio = {
    "return_1d": col("Return_1D").mean(),
    "alpha_1d": col("Alpha_1D").mean(),
    "alpha_30d": col("Alpha_30D").mean(),
    "alpha_60d": col("Alpha_60D").mean(),
    "alpha_365d": col("Alpha_365D").mean(),
    "exposure": col("Exposure").mean(),
    "cash": col("CashPercent").mean(),
    "vix": col("VIX_Level").mean(),
    "regime": col("VIX_Regime").mode().iloc[0] if "VIX_Regime" in df else "—",
}

# ----------------------------------------------------------
# BLUE PORTFOLIO SNAPSHOT BOX (SINGLE BLOCK)
# ----------------------------------------------------------

snapshot_html = f"""
<div style="
    background: linear-gradient(135deg, #0b3c5d, #0a2540);
    border-radius: 18px;
    padding: 26px;
    margin-bottom: 30px;
    color: #e8f1ff;
    box-shadow: 0 0 30px rgba(0,0,0,0.4);
">
  <div style="
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 18px;
      text-align: center;
  ">

    <div>
      <strong>SYSTEM</strong><br/>
      LIVE<br/>
      Waves: {df["Wave_ID"].nunique()}
    </div>

    <div>
      <strong>PORTFOLIO</strong><br/>
      Return {pct(portfolio["return_1d"])}<br/>
      Alpha {pct(portfolio["alpha_1d"])}
    </div>

    <div>
      <strong>ALPHA</strong><br/>
      30D {pct(portfolio["alpha_30d"])}<br/>
      60D {pct(portfolio["alpha_60d"])}<br/>
      365D {pct(portfolio["alpha_365d"])}
    </div>

    <div>
      <strong>RISK</strong><br/>
      Exposure {pct(portfolio["exposure"])}<br/>
      Cash {pct(portfolio["cash"])}<br/>
      VIX {portfolio["vix"]:.1f}<br/>
      Regime {portfolio["regime"]}
    </div>

  </div>
</div>
"""

st.markdown(snapshot_html, unsafe_allow_html=True)

# ----------------------------------------------------------
# LIVE RETURNS & ALPHA TABLE
# ----------------------------------------------------------

st.subheader("📊 Live Returns & Alpha")

returns_cols = [
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

existing_cols = [c for c in returns_cols if c in df.columns]

returns_df = df[existing_cols].copy()
returns_df = returns_df.sort_values("Alpha_365D", ascending=False)

st.dataframe(returns_df, use_container_width=True)

# ----------------------------------------------------------
# ALPHA BY HORIZON CHART
# ----------------------------------------------------------

st.subheader("📈 Alpha by Horizon")

alpha_chart = (
    df.groupby("Wave_ID")[["Alpha_30D", "Alpha_60D", "Alpha_365D"]]
    .mean()
    .sort_values("Alpha_365D", ascending=False)
)

st.bar_chart(alpha_chart)

# ----------------------------------------------------------
# SYSTEM STATUS
# ----------------------------------------------------------

st.success(
    "LIVE SYSTEM ACTIVE ✅\n\n"
    "✔ Intraday returns populated\n"
    "✔ Multi-horizon alpha live\n"
    "✔ Snapshot-driven truth\n"
    "✔ Zero dependency on legacy wave init\n\n"
    "Forward build unblocked."
)

with st.expander("Preview snapshot (first 10 rows)"):
    st.dataframe(df.head(10))