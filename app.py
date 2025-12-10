# app.py

import streamlit as st
import pandas as pd
import numpy as np
from waves_engine import WavesEngine

# Hard reset of Streamlit caches on each rerun so we never see stale logic
st.cache_data.clear()
st.cache_resource.clear()

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Wave Engine • Alpha Capture • Benchmark-Relative Performance")

# Sidebar controls
st.sidebar.header("Engine Controls")
mode = st.sidebar.selectbox(
    "Mode (label only for now)",
    ["standard", "alpha_minus_beta", "private_logic"],
    index=0,
)

lookback_days = st.sidebar.slider(
    "Lookback window (days)",
    min_value=120,
    max_value=730,
    value=365,
    step=30,
    help="History used for beta and 1-Year alpha estimation.",
)

force_reload = st.sidebar.button("Force Reload Engine & Data")

@st.cache_resource(show_spinner=True)
def get_engine(lookback: int, mode_label: str) -> WavesEngine:
    engine = WavesEngine(lookback_days=lookback, mode=mode_label)
    engine.compute_all_metrics()
    return engine

if force_reload:
    # Clearing cache and rebuilding engine
    st.cache_resource.clear()

engine = get_engine(lookback_days, mode)
metrics_df = engine.metrics_dataframe()

tabs = st.tabs(["Dashboard", "Wave Explorer", "Alpha Matrix", "History (30-Day)", "About / Diagnostics"])

# ------------------------------
# Dashboard
# ------------------------------

with tabs[0]:
    st.subheader(f"Dashboard — Mode: {mode}")

    # Summary metrics (ignoring NaNs)
    valid = metrics_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["30D Alpha (%)", "60D Alpha (%)", "1Y Alpha (%)"], how="all"
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        avg_30d_alpha = valid["30D Alpha (%)"].mean()
        st.metric("Avg 30-Day Alpha (All Waves)", f"{avg_30d_alpha:,.2f}%" if pd.notna(avg_30d_alpha) else "—")
    with col_b:
        avg_60d_alpha = valid["60D Alpha (%)"].mean()
        st.metric("Avg 60-Day Alpha (All Waves)", f"{avg_60d_alpha:,.2f}%" if pd.notna(avg_60d_alpha) else "—")
    with col_c:
        avg_1y_alpha = valid["1Y Alpha (%)"].mean()
        st.metric("Avg 1-Year Alpha (All Waves)", f"{avg_1y_alpha:,.2f}%" if pd.notna(avg_1y_alpha) else "—")

    st.markdown("### All Waves Snapshot")
    st.dataframe(
        metrics_df[
            [
                "Wave",
                "Benchmark",
                "Beta (≈)",
                "Intraday Return (%)",
                "Intraday Alpha (%)",
                "30D Return (%)",
                "30D Alpha (%)",
                "60D Return (%)",
                "60D Alpha (%)",
                "1Y Return (%)",
                "1Y Alpha (%)",
            ]
        ],
        use_container_width=True,
    )

# ------------------------------
# Wave Explorer
# ------------------------------

with tabs[1]:
    st.subheader("Wave Explorer")

    selected_wave = st.selectbox("Select a Wave", metrics_df["Wave"].tolist())

    w_row = metrics_df[metrics_df["Wave"] == selected_wave].iloc[0]

    top_cols = st.columns(4)
    with top_cols[0]:
        st.metric("Benchmark", w_row["Benchmark"])
    with top_cols[1]:
        st.metric("Beta (≈)", f"{w_row['Beta (≈)']:.2f}" if pd.notna(w_row["Beta (≈)"]) else "—")
    with top_cols[2]:
        st.metric("30-Day Alpha", f"{w_row['30D Alpha (%)']:.2f}%" if pd.notna(w_row["30D Alpha (%)"]) else "—")
    with top_cols[3]:
        st.metric("1-Year Alpha", f"{w_row['1Y Alpha (%)']:.2f}%" if pd.notna(w_row["1Y Alpha (%)"]) else "—")

    # History chart
    hist = engine.wave_history(selected_wave)
    if hist is not None and not hist.empty:
        st.markdown("#### Wave vs Benchmark (Index = 1.0)")
        st.line_chart(hist)
    else:
        st.info("No history available for this Wave (check data coverage or tickers).")

# ------------------------------
# Alpha Matrix
# ------------------------------

with tabs[2]:
    st.subheader("Alpha Matrix (All Waves)")

    sort_key = st.selectbox(
        "Sort Waves by",
        ["Alpha 30D (%)", "Alpha 60D (%)", "Alpha 1Y (%)"],
        index=0,
    )

    mapping = {
        "Alpha 30D (%)": "30D Alpha (%)",
        "Alpha 60D (%)": "60D Alpha (%)",
        "Alpha 1Y (%)": "1Y Alpha (%)",
    }
    sort_col = mapping[sort_key]

    display_df = metrics_df[
        [
            "Wave",
            "Benchmark",
            "Beta (≈)",
            "30D Return (%)",
            "30D Alpha (%)",
            "60D Return (%)",
            "60D Alpha (%)",
            "1Y Return (%)",
            "1Y Alpha (%)",
        ]
    ].sort_values(sort_col, ascending=False, na_position="last")

    st.dataframe(display_df, use_container_width=True)

# ------------------------------
# History (30-Day)
# ------------------------------

with tabs[3]:
    st.subheader("History (30-Day Alpha)")

    wave_for_hist = st.selectbox(
        "Select Wave for 30-Day history",
        metrics_df["Wave"].tolist(),
        key="hist_wave",
    )
    hist = engine.wave_history(wave_for_hist)
    if hist is not None and not hist.empty:
        # Compute daily alpha over last 30 trading days
        returns = hist.pct_change().dropna()
        if not returns.empty:
            # simple beta =1 for short window to avoid noise
            alpha_series = returns["Wave Index"] - returns["Benchmark Index"]
            alpha_30 = alpha_series.tail(30)
            st.line_chart(alpha_30)
        else:
            st.info("Not enough data to compute 30-Day history.")
    else:
        st.info("No history available for this Wave.")

# ------------------------------
# About / Diagnostics
# ------------------------------

with tabs[4]:
    st.subheader("About / Diagnostics")

    st.markdown(
        """
        **Engine:** WAVES Intelligence™ alpha capture prototype  
        **Alpha Definition:** Wave return − β × Benchmark return  
        **Benchmarks:** ETF and blended indexes mapped per Wave using the latest spec  
        **Lookback:** Configurable (default 365 days) for beta and 1-Year metrics.  

        If a Wave shows `ERROR:` in the Benchmark column, it means:
        - Missing price data for one or more tickers, OR  
        - No overlap between Wave tickers and price history, OR  
        - Issues fetching ETF history via yfinance.  

        In those cases, metrics are left as blanks so they don't pollute averages.
        """
    )

    if st.checkbox("Show raw metrics dataframe"):
        st.dataframe(metrics_df, use_container_width=True)