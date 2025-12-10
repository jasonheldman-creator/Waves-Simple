# app.py
import streamlit as st
import pandas as pd

from waves_engine import WavesEngine

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

# ---------------------------------------------------------------------
# Cached engine + metrics loader
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_engine_and_metrics():
    eng = WavesEngine()
    metrics_df, warnings = eng.compute_all_metrics()
    return metrics_df, warnings


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------
def format_percent(x):
    if pd.isna(x):
        return "—"
    return f"{x*100:0.2f}%"


# ---------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------
st.title("WAVES Intelligence™ Institutional Console")
st.caption("**Live Wave Engine • Alpha Capture • Benchmarks • Diagnostics**")

metrics_df, warnings = load_engine_and_metrics()

tab_dashboard, tab_explorer, tab_diag = st.tabs(
    ["Dashboard", "Wave Explorer", "Diagnostics"]
)

# ---------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------
with tab_dashboard:
    st.subheader("All Waves Snapshot")

    if metrics_df.empty:
        st.error(
            "No Wave metrics available. This usually means **all** Wave tickers "
            "failed to load price data. Check the Diagnostics tab for details."
        )
    else:
        display_df = metrics_df.copy()
        display_df["Return 60D"] = display_df["Return 60D"].apply(format_percent)
        display_df["Alpha 60D"] = display_df["Alpha 60D"].apply(format_percent)
        display_df["Return 1Y"] = display_df["Return 1Y"].apply(format_percent)
        display_df["Alpha 1Y"] = display_df["Alpha 1Y"].apply(format_percent)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Alpha is Wave return minus its benchmark blend over the same window. "
            "Safe Mode: any holdings with missing prices are dropped per-Wave; "
            "the engine keeps running even if some tickers fail."
        )

# ---------------------------------------------------------------------
# WAVE EXPLORER
# ---------------------------------------------------------------------
with tab_explorer:
    st.subheader("Wave Explorer")

    if metrics_df.empty:
        st.info("No Waves to explore – fix data issues first (see Diagnostics).")
    else:
        wave_names = metrics_df["Wave"].tolist()
        selected_wave = st.selectbox("Choose a Wave", wave_names)

        row = metrics_df[metrics_df["Wave"] == selected_wave].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Return 60D", format_percent(row["Return 60D"]))
            st.metric("Return 1Y", format_percent(row["Return 1Y"]))
        with col2:
            st.metric("Alpha 60D", format_percent(row["Alpha 60D"]))
            st.metric("Alpha 1Y", format_percent(row["Alpha 1Y"]))

        st.markdown(
            f"- **# Holdings Defined:** {int(row['# Holdings Defined'])}  \n"
            f"- **# Holdings Used (with price data):** {int(row['# Holdings Used'])}"
        )

        # Simple explanation
        st.markdown(
            """
            **How this is calculated (Safe Mode)**  
            - Take all tickers for this Wave from `wave_weights.csv`.  
            - Drop any ticker that has no price history from yfinance.  
            - Normalize the remaining weights and compute a daily Wave return series.  
            - Build the benchmark blend you specified (per-Wave ETF mix).  
            - Compute 60-day and 1-year total returns for both Wave and benchmark.  
            - Alpha = Wave return − Benchmark return.
            """
        )

# ---------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------
with tab_diag:
    st.subheader("Diagnostics & Warnings")

    if not warnings:
        st.success("No warnings reported by the engine. All Waves computed successfully.")
    else:
        for wave, msgs in warnings.items():
            with st.expander(f"{wave} — {len(msgs)} warning(s)"):
                for m in msgs:
                    st.write("- ", m)

    st.markdown("---")
    st.caption(
        "Safe Mode is enabled. The engine will **never** crash because of a single "
        "invalid or missing ticker. Instead, it drops problem tickers per Wave and "
        "reports them here."
    )