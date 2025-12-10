# app.py

import traceback

import pandas as pd
import streamlit as st

from waves_engine import WavesEngine


st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------


@st.cache_resource(show_spinner=True)
def load_engine() -> WavesEngine:
    return WavesEngine(weights_path="wave_weights.csv", lookback_years=5)


@st.cache_data(show_spinner=True)
def load_metrics_df(engine: WavesEngine) -> pd.DataFrame:
    return engine.all_metrics_df()


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------


def format_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x*100:0.2f}%"


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------

st.title("WAVES Intelligence™ Institutional Console")
st.subheader("Live Wave Engine • Alpha Capture • Benchmarks • Diagnostics")

tabs = st.tabs(["Dashboard", "Wave Explorer", "Diagnostics"])

engine: WavesEngine | None = None
metrics_df: pd.DataFrame | None = None
engine_error: Exception | None = None
traceback_str: str | None = None

try:
    engine = load_engine()
    metrics_df = load_metrics_df(engine)
except Exception as e:
    engine_error = e
    traceback_str = traceback.format_exc()

# ----------------------- DASHBOARD TAB --------------------------------
with tabs[0]:
    st.header("All Waves Snapshot")

    if engine_error is not None:
        st.error(
            "Engine failed while computing metrics. "
            "See Diagnostics tab for details."
        )
    else:
        # Display metrics table
        if metrics_df is not None and not metrics_df.empty:
            display_df = metrics_df.copy()
            for col in [
                "Intraday Alpha",
                "Alpha 30D",
                "Alpha 60D",
                "Alpha 1Y",
                "Wave 1Y Return",
                "Benchmark 1Y Return",
            ]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(format_pct)

            st.dataframe(
                display_df,
                use_container_width=True,
            )
        else:
            st.warning("No metrics available. Check Diagnostics tab.")

# --------------------- WAVE EXPLORER TAB ------------------------------
with tabs[1]:
    st.header("Wave Explorer")

    if engine_error is not None:
        st.error(
            "Engine is currently unavailable. "
            "Fix the underlying error first (see Diagnostics)."
        )
    elif engine is None:
        st.warning("Engine not initialised.")
    else:
        wave_names = sorted(engine.metrics.keys())
        if not wave_names:
            st.warning("No Waves with computed metrics.")
        else:
            wave = st.selectbox("Select Wave", wave_names, index=0)

            try:
                port, bench = engine.wave_series_pair(wave)

                # Build cumulative NAV series (starting at 1.0)
                port_nav = (1 + port).cumprod()
                bench_nav = (1 + bench).cumprod()

                nav_df = pd.DataFrame(
                    {
                        "Portfolio (NAV)": port_nav,
                        "Benchmark (NAV)": bench_nav,
                    }
                )

                st.line_chart(nav_df, use_container_width=True)

                st.caption(
                    "Cumulative NAV based on daily returns for portfolio vs. benchmark."
                )

                # Show numeric snapshot for this wave
                m = engine.metrics[wave]
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Alpha 30D",
                    format_pct(m.alpha_30d),
                )
                col2.metric(
                    "Alpha 60D",
                    format_pct(m.alpha_60d),
                )
                col3.metric(
                    "Alpha 1Y",
                    format_pct(m.alpha_1y),
                )

            except Exception as ex:
                st.error(f"Failed to render Wave Explorer for '{wave}': {ex}")

# ------------------------ DIAGNOSTICS TAB -----------------------------
with tabs[2]:
    st.header("Diagnostics")

    if engine_error is not None:
        st.subheader("Engine start-up error")
        st.error(str(engine_error))
        with st.expander("Full traceback"):
            st.code(traceback_str or "", language="python")

    if engine is not None:
        if engine.diagnostics:
            st.subheader("Engine diagnostics log")
            for msg in engine.diagnostics:
                if "[ERROR]" in msg:
                    st.error(msg)
                elif "[WARN]" in msg:
                    st.warning(msg)
                else:
                    st.info(msg)
        else:
            st.info("No diagnostics messages recorded.")

        if engine.missing_tickers:
            st.subheader("Tickers with missing or invalid data")
            uniq = sorted(set(engine.missing_tickers))
            st.write(", ".join(uniq))
    else:
        if engine_error is None:
            st.info("Engine has not been initialised yet.")