# app.py

from __future__ import annotations

import traceback
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from waves_engine import WavesEngine, WaveMetrics, build_engine


st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

# ----------------- Helpers & Caching ----------------- #


@st.cache_data(show_spinner=False)
def load_metrics_cached() -> Tuple[Dict[str, WaveMetrics], List[str]]:
    engine = build_engine()
    metrics_dict, warnings = engine.compute_all_metrics()
    return metrics_dict, warnings


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


# ----------------- UI Layout ----------------- #

st.markdown(
    """
    # WAVES Intelligence™ Institutional Console

    **Live Wave Engine • Alpha Capture • Benchmarks • Diagnostics**
    """.strip()
)

tab_dashboard, tab_wave_explorer, tab_diag = st.tabs(
    ["Dashboard", "Wave Explorer", "Diagnostics"]
)

# ----------------- Dashboard Tab ----------------- #

with tab_dashboard:
    st.subheader("All Waves Snapshot")

    try:
        metrics_dict, warnings = load_metrics_cached()
    except Exception as e:
        st.error("Engine failed while computing metrics. See Diagnostics tab for details.")
        with st.expander("Full error traceback"):
            st.code("".join(traceback.format_exc()))
        st.stop()

    if not metrics_dict:
        st.error("No Wave metrics available. Check Diagnostics tab for details.")
    else:
        rows = []
        for wave_name, wm in sorted(metrics_dict.items(), key=lambda kv: kv[0]):
            stats = wm.stats
            rows.append(
                {
                    "Wave": wave_name,
                    "60D": stats.get("60D"),
                    "Alpha 1Y": stats.get("Alpha 1Y"),
                }
            )

        df = pd.DataFrame(rows).set_index("Wave")
        df_display = df.copy()
        df_display["60D"] = df_display["60D"].map(format_pct)
        df_display["Alpha 1Y"] = df_display["Alpha 1Y"].map(format_pct)

        st.dataframe(
            df_display,
            use_container_width=True,
        )

        st.caption(
            "• 60D = cumulative Wave return over the last 60 trading days.  "
            "• Alpha 1Y = Wave return minus benchmark over the last ~252 trading days."
        )

# ----------------- Wave Explorer Tab ----------------- #

with tab_wave_explorer:
    st.subheader("Wave Explorer")

    metrics_dict, warnings = load_metrics_cached()

    if not metrics_dict:
        st.info("No Waves available to explore.")
    else:
        wave_names = sorted(metrics_dict.keys())
        selected = st.selectbox("Select Wave", wave_names)

        wm = metrics_dict[selected]

        col_a, col_b = st.columns(2)

        # Cumulative growth of $1
        port_cum = (1.0 + wm.portfolio_returns).cumprod()
        bench_cum = (1.0 + wm.benchmark_returns).cumprod()
        alpha_cum = (1.0 + wm.alpha_daily).cumprod()

        with col_a:
            st.markdown("#### Cumulative Return vs Benchmark")
            chart_df = pd.DataFrame(
                {
                    "Wave": port_cum,
                    "Benchmark": bench_cum,
                }
            )
            st.line_chart(chart_df)

        with col_b:
            st.markdown("#### Cumulative Alpha (Wave − Benchmark)")
            st.line_chart(alpha_cum.to_frame(name="Alpha"))

        with st.expander("Raw Daily Series"):
            st.write("Daily returns (Wave, Benchmark, Alpha):")
            daily_df = pd.DataFrame(
                {
                    "Wave": wm.portfolio_returns,
                    "Benchmark": wm.benchmark_returns,
                    "Alpha": wm.alpha_daily,
                }
            )
            st.dataframe(daily_df.tail(252), use_container_width=True)

# ----------------- Diagnostics Tab ----------------- #

with tab_diag:
    st.subheader("Diagnostics")

    metrics_dict, warnings = load_metrics_cached()

    if warnings:
        st.markdown("### Engine Warnings")
        for w in warnings:
            st.warning(w)
    else:
        st.success("No engine warnings recorded.")

    st.markdown("### Internal Status")
    st.write(f"Total Waves computed: **{len(metrics_dict)}**")

    st.markdown(
        """
        **Notes**

        - This console is running the *benchmark-aware* WAVES Engine with:
          - 10 Waves (including **AI Wave** and **SmartSafe Wave**).
          - ETF benchmark blends per Wave.
          - Defensive handling of missing tickers / missing price data (waves with no data are skipped, not fatal).
        - To adjust benchmarks, edit `BENCHMARK_MAP` in `waves_engine.py`.
        - To adjust holdings, edit `wave_weights.csv` (wave name, ticker, weight) and redeploy.
        """
    )