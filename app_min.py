import streamlit as st
import pandas as pd
from datetime import datetime, timezone

st.set_page_config(
    page_title="WAVES — Live Recovery Console",
    layout="wide",
)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("WAVES — Live Recovery Console")
st.caption("Intraday • 30D • 60D • 365D • Snapshot-Driven")

st.divider()

# ------------------------------------------------------------
# LOAD LIVE SNAPSHOT
# ------------------------------------------------------------
@st.cache_data(ttl=60)
def load_snapshot():
    return pd.read_csv("data/live_snapshot.csv")

df = load_snapshot()

# ------------------------------------------------------------
# PORTFOLIO AGGREGATES
# ------------------------------------------------------------
portfolio = {
    "ret_1d": df["Return_1D"].mean(),
    "ret_30d": df["Return_30D"].mean(),
    "ret_60d": df["Return_60D"].mean(),
    "ret_365d": df["Return_365D"].mean(),
    "alpha_1d": df["Alpha_1D"].mean(),
    "alpha_30d": df["Alpha_30D"].mean(),
    "alpha_60d": df["Alpha_60D"].mean(),
    "alpha_365d": df["Alpha_365D"].mean(),
}

now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ------------------------------------------------------------
# ENHANCED BLUE SNAPSHOT BOX
# ------------------------------------------------------------
with st.container():
    st.markdown(
        """
        <style>
        .snapshot-box {
            background: linear-gradient(135deg, #0b1f3a, #0e2b52);
            border-radius: 16px;
            padding: 24px;
            border: 2px solid #1fd1ff;
            box-shadow: 0 0 24px rgba(31,209,255,0.25);
            margin-bottom: 24px;
        }
        .snapshot-title {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
        }
        .snapshot-sub {
            opacity: 0.85;
            margin-bottom: 20px;
        }
        .metric {
            font-size: 28px;
            font-weight: 700;
        }
        .label {
            opacity: 0.7;
            font-size: 13px;
            letter-spacing: 0.04em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="snapshot-box">
            <div class="snapshot-title">🏛️ Portfolio Snapshot (All Waves)</div>
            <div class="snapshot-sub">STANDARD MODE</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Return 1D (Intraday)", f"{portfolio['ret_1d']:+.2%}")
    c2.metric("Return 30D", f"{portfolio['ret_30d']:+.2%}")
    c3.metric("Return 60D", f"{portfolio['ret_60d']:+.2%}")
    c4.metric("Return 365D", f"{portfolio['ret_365d']:+.2%}")

    st.divider()

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Alpha 1D", f"{portfolio['alpha_1d']:+.2%}")
    a2.metric("Alpha 30D", f"{portfolio['alpha_30d']:+.2%}")
    a3.metric("Alpha 60D", f"{portfolio['alpha_60d']:+.2%}")
    a4.metric("Alpha 365D", f"{portfolio['alpha_365d']:+.2%}")

    st.caption(f"⚡ Computed from live snapshot | {now_utc}")
    st.caption(
        "ℹ️ Wave-specific metrics (Beta, Exposure, Cash, VIX regime) shown at wave level"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# LIVE RETURNS TABLE
# ------------------------------------------------------------
st.subheader("📊 Live Returns & Alpha")

cols = [
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

st.dataframe(
    df[cols].sort_values("Alpha_365D", ascending=False),
    use_container_width=True,
)

# ------------------------------------------------------------
# SYSTEM STATUS
# ------------------------------------------------------------
st.success(
    "LIVE SYSTEM ACTIVE ✅  —  Intraday live • Multi-horizon alpha • Snapshot truth"
)