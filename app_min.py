# app_min.py
# WAVES Intelligence™ Console (Minimal)
# INSTITUTIONAL OVERVIEW SNAPSHOT — FINAL STRUCTURE

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import render_alpha_quality_and_confidence

# ===========================
# Page Config
# ===========================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Paths & Constants
# ===========================
DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"

RETURN_COLS = {
    "1D": "return_1d",
    "30D": "return_30d",
    "60D": "return_60d",
    "365D": "return_365d",
}

ALPHA_COLS = {
    "1D": "alpha_1d",
    "30D": "alpha_30d",
    "60D": "alpha_60d",
    "365D": "alpha_365d",
}

# ===========================
# Load Snapshot
# ===========================
def load_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, "Live snapshot file not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    if "display_name" not in df.columns:
        if "wave_name" in df.columns:
            df["display_name"] = df["wave_name"]
        else:
            df["display_name"] = df["wave_id"]

    for col in list(RETURN_COLS.values()) + list(ALPHA_COLS.values()):
        if col not in df.columns:
            df[col] = np.nan

    return df, None


snapshot_df, snapshot_error = load_snapshot()

# ===========================
# Global CSS (Injected Once)
# ===========================
st.markdown(
    """
    <style>
    .snapshot-card {
        background: linear-gradient(145deg, #0f1220, #0b0e1a);
        border: 1px solid rgba(0, 255, 255, 0.25);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 28px;
        box-shadow: 0 0 28px rgba(0, 255, 255, 0.12);
    }

    .snapshot-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .snapshot-subtitle {
        color: #9aa4bf;
        margin-bottom: 18px;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 14px;
    }

    .metric-cell {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #8fa3c8;
        margin-bottom: 4px;
        letter-spacing: 0.04em;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
    }

    .positive { color: #35f2a6; }
    .negative { color: #ff6b6b; }
    .neutral  { color: #cfd6e6; }

    .row-label {
        margin: 10px 0 6px;
        font-size: 0.85rem;
        color: #6f7fa8;
        letter-spacing: 0.08em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===========================
# Snapshot Card Renderer
# ===========================
def render_snapshot_card(title, subtitle, returns, alphas):
    def fmt(val):
        if pd.isna(val):
            return "—", "neutral"
        cls = "positive" if val > 0 else "negative" if val < 0 else "neutral"
        return f"{val:.2%}", cls

    html = f"""
    <div class="snapshot-card">
        <div class="snapshot-title">{title}</div>
        <div class="snapshot-subtitle">{subtitle}</div>

        <div class="row-label">RETURNS</div>
        <div class="metric-grid">
    """

    for k in ["1D", "30D", "60D", "365D"]:
        v, cls = fmt(returns.get(k))
        html += f"""
        <div class="metric-cell">
            <div class="metric-label">{k}</div>
            <div class="metric-value {cls}">{v}</div>
        </div>
        """

    html += "</div><div class='row-label'>ALPHA</div><div class='metric-grid'>"

    for k in ["1D", "30D", "60D", "365D"]:
        v, cls = fmt(alphas.get(k))
        html += f"""
        <div class="metric-cell">
            <div class="metric-label">{k}</div>
            <div class="metric-value {cls}">{v}</div>
        </div>
        """

    html += "</div></div>"

    st.markdown(html, unsafe_allow_html=True)

# ===========================
# Sidebar
# ===========================
st.sidebar.title("Wave Selection")

if snapshot_error:
    st.sidebar.error("Snapshot unavailable")
    selected_wave = None
else:
    selected_wave = st.sidebar.selectbox(
        "Select Wave",
        snapshot_df["display_name"].tolist()
    )

# ===========================
# Tabs
# ===========================
tabs = st.tabs([
    "Overview",
    "Alpha Attribution",
    "Adaptive Intelligence",
    "Operations"
])

# ===========================
# OVERVIEW TAB
# ===========================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # ---- Portfolio (Equal-Weighted)
        portfolio_returns = {
            k: df[v].mean(skipna=True) for k, v in RETURN_COLS.items()
        }
        portfolio_alphas = {
            k: df[ALPHA_COLS[k]].mean(skipna=True) for k in ALPHA_COLS
        }

        render_snapshot_card(
            "🏛️ Portfolio Snapshot",
            "Equal-Weighted Diagnostic Portfolio · Live Data",
            portfolio_returns,
            portfolio_alphas
        )

        # ---- Selected Wave
        wave_row = df[df["display_name"] == selected_wave].iloc[0]

        wave_returns = {k: wave_row[v] for k, v in RETURN_COLS.items()}
        wave_alphas = {k: wave_row[ALPHA_COLS[k]] for k in ALPHA_COLS}

        render_snapshot_card(
            f"📈 Wave Snapshot — {selected_wave}",
            "Wave-Level Diagnostic Snapshot",
            wave_returns,
            wave_alphas
        )

# ===========================
# ALPHA ATTRIBUTION TAB
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        source_df = pd.DataFrame({
            "Alpha Source": [
                "Selection Alpha",
                "Momentum Alpha",
                "Regime Alpha",
                "Exposure Alpha",
                "Residual Alpha",
            ],
            "Contribution": [0.012, 0.008, -0.003, 0.004, 0.001],
        })

        st.dataframe(source_df, use_container_width=True, hide_index=True)

        render_alpha_quality_and_confidence(
            snapshot_df,
            source_df,
            selected_wave,
            RETURN_COLS,
            {},
        )

# ===========================
# ADAPTIVE INTELLIGENCE TAB
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Derived from Alpha Attribution (read-only)")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            {},
        )

# ===========================
# OPERATIONS TAB
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Operations control center coming next.")