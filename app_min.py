# app_min.py
# WAVES Intelligence™ Console — Institutional Preview
# Equal-Weighted Diagnostic Portfolio Snapshot

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from intelligence.adaptive_intelligence import (
    render_alpha_quality_and_confidence,
    render_adaptive_intelligence_preview,
)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Constants
# ---------------------------
DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"

RETURN_COLS = {
    "1D": "return_1d",
    "30D": "return_30d",
    "60D": "return_60d",
    "365D": "return_365d",
}

BENCHMARK_COLS = {
    "30D": "benchmark_return_30d",
    "60D": "benchmark_return_60d",
    "365D": "benchmark_return_365d",
}

# ---------------------------
# Load Snapshot
# ---------------------------
def load_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, "Live snapshot file not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    if "display_name" not in df.columns:
        if "wave_name" in df.columns:
            df["display_name"] = df["wave_name"]
        elif "wave_id" in df.columns:
            df["display_name"] = df["wave_id"]
        else:
            df["display_name"] = "Unnamed Wave"

    for col in list(RETURN_COLS.values()) + list(BENCHMARK_COLS.values()):
        if col not in df.columns:
            df[col] = np.nan

    return df, None


snapshot_df, snapshot_error = load_snapshot()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("System Status")
st.sidebar.markdown(
    f"""
    **Live Snapshot:** {'✅ Loaded' if snapshot_error is None else '❌ Missing'}  
    **Portfolio Mode:** Equal-Weighted Diagnostic  
    **Alpha Attribution:** Active  
    **Adaptive Intelligence:** Preview  
    """
)
st.sidebar.divider()

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(
    [
        "Overview",
        "Alpha Attribution",
        "Adaptive Intelligence",
        "Operations",
    ]
)

# ===========================
# OVERVIEW TAB (INSTITUTIONAL)
# ===========================
with tabs[0]:
    st.header("Portfolio Overview")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        df = snapshot_df.copy()

        # ---------------------------
        # Equal-Weighted Portfolio Math
        # ---------------------------
        portfolio_returns = {
            label: df[col].mean(skipna=True)
            for label, col in RETURN_COLS.items()
        }

        portfolio_alpha = {}
        for label in ["30D", "60D", "365D"]:
            r_col = RETURN_COLS[label]
            b_col = BENCHMARK_COLS[label]
            portfolio_alpha[label] = (
                (df[r_col] - df[b_col]).mean(skipna=True)
                if b_col in df.columns
                else np.nan
            )

        # ---------------------------
        # HERO SNAPSHOT CARD
        # ---------------------------
        st.markdown(
            """
            <style>
            .snapshot-card {
                background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
                border: 1px solid rgba(148,163,184,0.25);
                border-radius: 14px;
                padding: 28px;
                margin-bottom: 28px;
                box-shadow: 0 0 0 1px rgba(148,163,184,0.1),
                            0 20px 40px rgba(0,0,0,0.6);
            }
            .snapshot-title {
                font-size: 20px;
                font-weight: 700;
                color: #e5e7eb;
                margin-bottom: 6px;
            }
            .snapshot-subtitle {
                font-size: 13px;
                color: #94a3b8;
                margin-bottom: 22px;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
            }
            .metric-tile {
                background: rgba(15,23,42,0.85);
                border: 1px solid rgba(148,163,184,0.18);
                border-radius: 10px;
                padding: 16px;
            }
            .metric-label {
                font-size: 12px;
                color: #94a3b8;
                margin-bottom: 6px;
            }
            .metric-value {
                font-size: 22px;
                font-weight: 700;
                color: #f8fafc;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="snapshot-card">
                <div class="snapshot-title">
                    Equal-Weighted Diagnostic Portfolio Snapshot
                </div>
                <div class="snapshot-subtitle">
                    Hypothetical equal allocation across all Waves · LIVE data
                </div>

                <div class="metric-grid">
                    <div class="metric-tile">
                        <div class="metric-label">Intraday Return</div>
                        <div class="metric-value">{portfolio_returns["1D"]:.2%}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">30-Day Return</div>
                        <div class="metric-value">{portfolio_returns["30D"]:.2%}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">60-Day Return</div>
                        <div class="metric-value">{portfolio_returns["60D"]:.2%}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">365-Day Return</div>
                        <div class="metric-value">{portfolio_returns["365D"]:.2%}</div>
                    </div>

                    <div class="metric-tile">
                        <div class="metric-label">30-Day Alpha</div>
                        <div class="metric-value">{portfolio_alpha["30D"]:.2%}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">60-Day Alpha</div>
                        <div class="metric-value">{portfolio_alpha["60D"]:.2%}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">365-Day Alpha</div>
                        <div class="metric-value">{portfolio_alpha["365D"]:.2%}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------
        # Wave Table (Context)
        # ---------------------------
        st.subheader("Wave Performance Context")

        view = df[
            ["display_name"] + list(RETURN_COLS.values())
        ].rename(
            columns={
                "display_name": "Wave",
                "return_1d": "1D",
                "return_30d": "30D",
                "return_60d": "60D",
                "return_365d": "365D",
            }
        )

        view = view.replace({np.nan: "—"})
        st.dataframe(view, use_container_width=True, hide_index=True)

# ===========================
# ALPHA ATTRIBUTION
# ===========================
with tabs[1]:
    st.header("Alpha Attribution")
    st.caption("Explains where performance came from")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()

        selected_wave = st.selectbox(
            "Select Wave",
            waves,
            key="alpha_attr_wave_select",
        )

        source_df = pd.DataFrame(
            {
                "Alpha Source": [
                    "Selection Alpha",
                    "Momentum Alpha",
                    "Regime Alpha",
                    "Exposure Alpha",
                    "Residual Alpha",
                ],
                "Contribution": [
                    snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave,
                        "selection_alpha",
                    ].values[0]
                    if "selection_alpha" in snapshot_df.columns
                    else None,
                    snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave,
                        "momentum_alpha",
                    ].values[0]
                    if "momentum_alpha" in snapshot_df.columns
                    else None,
                    snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave,
                        "regime_alpha",
                    ].values[0]
                    if "regime_alpha" in snapshot_df.columns
                    else None,
                    snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave,
                        "exposure_alpha",
                    ].values[0]
                    if "exposure_alpha" in snapshot_df.columns
                    else None,
                    snapshot_df.loc[
                        snapshot_df["display_name"] == selected_wave,
                        "residual_alpha",
                    ].values[0]
                    if "residual_alpha" in snapshot_df.columns
                    else None,
                ],
            }
        )

        st.subheader("Source Breakdown")
        st.dataframe(source_df, use_container_width=True, hide_index=True)

# ===========================
# ADAPTIVE INTELLIGENCE
# ===========================
with tabs[2]:
    st.header("Adaptive Intelligence")
    st.caption("Interpretive intelligence layer (no execution)")

    if snapshot_error:
        st.error(snapshot_error)
    else:
        waves = snapshot_df["display_name"].tolist()

        selected_wave = st.selectbox(
            "Select Wave",
            waves,
            key="adaptive_intel_wave_select",
        )

        render_alpha_quality_and_confidence(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

        render_adaptive_intelligence_preview(
            snapshot_df,
            None,
            selected_wave,
            RETURN_COLS,
            BENCHMARK_COLS,
        )

# ===========================
# OPERATIONS
# ===========================
with tabs[3]:
    st.header("Operations")
    st.info("Human-in-the-loop controls coming next.")