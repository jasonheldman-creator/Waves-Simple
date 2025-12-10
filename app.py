# app.py

import streamlit as st
import pandas as pd

from waves_engine import build_engine

# ------------------------------------------------------------
# Page config / styling
# ------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #05070D;
        color: #FFFFFF;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("WAVES Intelligence\u2122 Institutional Console")
st.caption("Live Wave Engine • Alpha Capture • Benchmarks • Diagnostics")

# ------------------------------------------------------------
# Data loading (cached)
# ------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_metrics() -> pd.DataFrame:
    return build_engine()


# ------------------------------------------------------------
# Try to build engine
# ------------------------------------------------------------

try:
    metrics_df = load_metrics()
    engine_error = None
except Exception as e:
    metrics_df = pd.DataFrame()
    engine_error = str(e)

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

tab_dashboard, tab_explorer, tab_diag = st.tabs(
    ["Dashboard", "Wave Explorer", "Diagnostics"]
)

# ---------------- Dashboard ----------------

with tab_dashboard:
    st.subheader("All Waves Snapshot")

    if engine_error:
        st.error("Engine failed while computing metrics. See Diagnostics tab for details.")
    elif metrics_df.empty:
        st.warning("No metrics available yet. Check Diagnostics for details.")
    else:
        display_df = metrics_df.copy()

        # Convert to percentage strings for nicer display
        for col in ["Return 60D", "Alpha 60D", "Return 1Y", "Alpha 1Y"]:
            if col in display_df.columns:
                display_df[col] = (display_df[col] * 100).map(lambda x: f"{x:0.2f}%" if pd.notnull(x) else "—")

        st.dataframe(
            display_df[
                ["Return 60D", "Alpha 60D", "Return 1Y", "Alpha 1Y", "Benchmark", "Notes"]
            ],
            use_container_width=True,
        )

# ---------------- Wave Explorer ----------------

with tab_explorer:
    st.subheader("Wave Explorer")

    if engine_error or metrics_df.empty:
        st.info("Engine is not currently available. See Diagnostics.")
    else:
        wave_names = list(metrics_df.index)
        default_wave = "S&P Wave" if "S&P Wave" in wave_names else wave_names[0]
        wave_choice = st.selectbox("Select Wave", wave_names, index=wave_names.index(default_wave))

        row = metrics_df.loc[wave_choice]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="60D Total Return",
                value=f"{row['Return 60D']*100:0.2f}%" if pd.notnull(row['Return 60D']) else "—",
            )
            st.metric(
                label="1Y Total Return",
                value=f"{row['Return 1Y']*100:0.2f}%" if pd.notnull(row['Return 1Y']) else "—",
            )

        with col2:
            st.metric(
                label="60D Alpha vs Benchmark",
                value=f"{row['Alpha 60D']*100:0.2f}%" if pd.notnull(row['Alpha 60D']) else "—",
            )
            st.metric(
                label="1Y Alpha vs Benchmark",
                value=f"{row['Alpha 1Y']*100:0.2f}%" if pd.notnull(row['Alpha 1Y']) else "—",
            )

        st.markdown("**Benchmark Blend**")
        st.write(row.get("Benchmark", "—"))

        if isinstance(row.get("Notes", None), str) and row["Notes"].strip():
            st.markdown("**Engine Notes**")
            st.info(row["Notes"])

# ---------------- Diagnostics ----------------

with tab_diag:
    st.subheader("Engine Diagnostics")

    if engine_error:
        st.error("Engine error:")
        st.code(engine_error)
    else:
        st.success("Engine loaded successfully.")
        st.write(
            "If you want deeper diagnostics (per-ticker price gaps, "
            "dropped symbols, etc.), we can extend the engine to return "
            "a richer diagnostics table."
        )