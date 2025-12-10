# app.py

import streamlit as st
import pandas as pd

from waves_engine import WavesEngine, WaveMetrics


st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)


@st.cache_resource
def load_engine() -> WavesEngine:
    # wave_weights.csv is in the same directory on Streamlit Cloud
    return WavesEngine("wave_weights.csv")


def format_pct(x: float) -> str:
    if x != x or x is None:  # NaN check
        return "—"
    return f"{x * 100:.2f}%"


def dashboard_tab(engine: WavesEngine) -> None:
    st.subheader("All Waves Snapshot")

    try:
        metrics_dict = engine.compute_all_metrics()
    except Exception as e:
        st.error("Engine failed while computing metrics.")
        st.caption("See Diagnostics tab or Streamlit logs for details.")
        # Show the raw exception here – Cloud will still redact sensitive bits
        st.exception(e)
        return

    if not metrics_dict:
        st.warning("No Wave metrics available. Check your wave_weights.csv file.")
        return

    # Build DataFrame for display
    rows = []
    for wave_name, m in metrics_dict.items():
        rows.append(
            {
                "Wave": wave_name,
                "60D Return": format_pct(m.ret_60d),
                "1Y Return": format_pct(m.ret_1y),
                "1Y Alpha vs Benchmark": format_pct(m.alpha_1y),
            }
        )

    df = pd.DataFrame(rows).set_index("Wave").sort_index()
    st.dataframe(df, use_container_width=True)


def explorer_tab(engine: WavesEngine) -> None:
    st.subheader("Wave Explorer (basic placeholder)")
    st.caption(
        "This tab can be expanded later with per-wave charts, top holdings, "
        "and mode views. For now, use the Dashboard for core metrics."
    )
    waves = sorted(engine.wave_weights["wave"].unique())
    st.write("Detected Waves:")
    st.write(waves)


def diagnostics_tab(engine: WavesEngine) -> None:
    st.subheader("Diagnostics")

    st.markdown("### wave_weights.csv preview")
    st.dataframe(engine.wave_weights.head(50), use_container_width=True)

    st.markdown("### Benchmark mapping in engine")
    bench_rows = []
    for wave, comp in engine.benchmarks.items():
        for t, w in comp.items():
            bench_rows.append({"Wave": wave, "Ticker": t, "Weight": w})
    bench_df = pd.DataFrame(bench_rows)
    st.dataframe(bench_df, use_container_width=True)

    st.caption(
        "If a Wave has no benchmark mapping, 1Y alpha will show as '—'. "
        "If prices fail to download for some ticker, it will be skipped."
    )


def main() -> None:
    st.title("WAVES Intelligence™ Institutional Console")

    # Load engine once per session
    engine = load_engine()

    tab_dashboard, tab_explorer, tab_diag = st.tabs(
        ["Dashboard", "Wave Explorer", "Diagnostics"]
    )

    with tab_dashboard:
        dashboard_tab(engine)

    with tab_explorer:
        explorer_tab(engine)

    with tab_diag:
        diagnostics_tab(engine)


if __name__ == "__main__":
    main()