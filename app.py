# app.py – WAVES Intelligence™ Institutional Console
# Compatible with the NEW waves_engine.py (WavesEngine class)

import streamlit as st
from waves_engine import WavesEngine

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

# ----------------------------------------------------
# Cache the engine + metrics for performance
# ----------------------------------------------------

@st.cache_resource
def load_engine():
    return WavesEngine()   # NEW — replaces build_engine()


@st.cache_data
def load_metrics():
    eng = load_engine()
    return eng.metrics_snapshot_df()


# ----------------------------------------------------
# UI Layout
# ----------------------------------------------------

st.title("WAVES Intelligence™ Institutional Console")
st.caption("**Live Wave Engine • Alpha Capture • Benchmarks • Diagnostics**")

# Load metrics table
try:
    metrics_df = load_metrics()
except Exception as e:
    st.error("Engine failed while computing metrics.")
    with st.expander("Full error traceback"):
        st.write(e)
    st.stop()

st.subheader("All Waves Snapshot")
st.dataframe(metrics_df.style.format("{:.2%}"))

# ----------------------------------------------------
# Per-Wave Explorer
# ----------------------------------------------------

st.subheader("Wave Explorer")

eng = load_engine()
wave_list = sorted(list(eng.wave_weights.keys()))
selected_wave = st.selectbox("Select Wave", wave_list)

if selected_wave:
    try:
        metrics = eng.compute_wave_metrics(selected_wave)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("60D Alpha", f"{metrics.alpha_60d:.2%}")
            st.metric("60D Return", f"{metrics.total_return_60d:.2%}")
            st.metric("60D Benchmark", f"{metrics.bench_return_60d:.2%}")

        with col2:
            st.metric("1Y Alpha", f"{metrics.alpha_1y:.2%}")
            st.metric("1Y Return", f"{metrics.total_return_1y:.2%}")
            st.metric("1Y Benchmark", f"{metrics.bench_return_1y:.2%}")

        st.line_chart(metrics.nav_series, height=300)
        st.caption("Wave NAV")

        st.line_chart(metrics.benchmark_nav, height=300)
        st.caption("Benchmark NAV")

    except Exception as e:
        st.error(f"Failed to compute metrics for {selected_wave}.")
        with st.expander("Diagnostics"):
            st.write(e)


# ----------------------------------------------------
# Diagnostics – optional helper
# ----------------------------------------------------

with st.expander("Engine Diagnostics"):
    st.write("### Loaded Waves:")
    st.write(wave_list)

    st.write("### Raw Wave Weights:")
    st.write(eng.wave_weights)

    if eng.price_data is not None:
        st.write("### Available Price Data (tickers):")
        st.write(list(eng.price_data.columns))