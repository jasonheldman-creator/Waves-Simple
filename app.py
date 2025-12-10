"""
app.py

WAVES Intelligence™ Institutional Console
- Uses WavesEngine from waves_engine.py
- 4 tabs: Dashboard, Wave Explorer, Alpha Matrix, History (30D)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from waves_engine import WavesEngine, WaveMetrics


# ---------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)


# ---------------------------------------------------------
# Engine loader (cached)
# ---------------------------------------------------------

@st.cache_resource(show_spinner=True)
def load_engine() -> WavesEngine:
    return WavesEngine(
        list_path="list.csv",
        weights_path="wave_weights.csv",
    )


@st.cache_data(show_spinner=True)
def load_metrics() -> pd.DataFrame:
    eng = load_engine()
    metrics_dict: Dict[str, WaveMetrics] = eng.compute_all_metrics()
    rows = []
    for wave, m in metrics_dict.items():
        rows.append(
            {
                "Wave": m.wave,
                "Intraday Return": m.intraday_return,
                "Intraday Alpha": m.intraday_alpha,
                "30D Return": m.return_30d,
                "30D Alpha": m.alpha_30d,
                "60D Return": m.return_60d,
                "60D Alpha": m.alpha_60d,
                "1Y Return": m.return_1y,
                "1Y Alpha": m.alpha_1y,
            }
        )
    df = pd.DataFrame(rows).set_index("Wave").sort_index()
    return df


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


def google_quote_link(ticker: str) -> str:
    base = "https://www.google.com/finance/quote/"
    # Let Google auto-detect exchange if we don't know it
    return f"[{ticker}]({base}{ticker})"


# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Wave Engine • Alpha Capture • Benchmark-Relative Performance")

metrics_df = load_metrics()
engine = load_engine()

tab_dash, tab_explorer, tab_matrix, tab_history = st.tabs(
    ["Dashboard", "Wave Explorer", "Alpha Matrix", "History (30-Day)"]
)

# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------

with tab_dash:
    st.subheader("Dashboard — Mode: standard")

    # High-level averages
    avg_30d_alpha = metrics_df["30D Alpha"].mean()
    avg_60d_alpha = metrics_df["60D Alpha"].mean()
    avg_1y_alpha = metrics_df["1Y Alpha"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg 30D Alpha", fmt_pct(avg_30d_alpha))
    c2.metric("Avg 60D Alpha", fmt_pct(avg_60d_alpha))
    c3.metric("Avg 1Y Alpha", fmt_pct(avg_1y_alpha))

    st.markdown("### All Waves Snapshot")

    display = metrics_df.copy()
    display["Intraday Return"] = display["Intraday Return"].apply(fmt_pct)
    display["Intraday Alpha"] = display["Intraday Alpha"].apply(fmt_pct)
    display["30D Return"] = display["30D Return"].apply(fmt_pct)
    display["30D Alpha"] = display["30D Alpha"].apply(fmt_pct)
    display["60D Return"] = display["60D Return"].apply(fmt_pct)
    display["60D Alpha"] = display["60D Alpha"].apply(fmt_pct)
    display["1Y Return"] = display["1Y Return"].apply(fmt_pct)
    display["1Y Alpha"] = display["1Y Alpha"].apply(fmt_pct)

    st.dataframe(display, use_container_width=True)

    st.markdown(
        """
**Engine:** WAVES Intelligence™ • `list.csv` = total market universe • `wave_weights.csv` = Wave definitions  
Alpha = benchmark-relative cumulative return over each window using your blended ETF benchmarks.
        """
    )


# ---------------------------------------------------------
# Wave Explorer
# ---------------------------------------------------------

with tab_explorer:
    st.subheader("Wave Explorer")

    wave_names = list(engine.wave_weights.keys())
    selected_wave = st.selectbox("Select Wave", wave_names)

    weights = engine.wave_weights[selected_wave].copy()
    weights = weights.sort_values(ascending=False)

    st.markdown(f"#### {selected_wave} — Holdings")

    top_n = min(25, len(weights))
    top_weights = weights.head(top_n)

    holdings_df = pd.DataFrame(
        {
            "Ticker": top_weights.index,
            "Weight": top_weights.values,
            "Quote": [google_quote_link(t) for t in top_weights.index],
        }
    )
    holdings_df["Weight"] = holdings_df["Weight"].apply(fmt_pct)

    st.dataframe(holdings_df, use_container_width=True, hide_index=True)

    st.caption(
        "Click any ticker symbol to open its Google Finance quote screen."
    )


# ---------------------------------------------------------
# Alpha Matrix
# ---------------------------------------------------------

with tab_matrix:
    st.subheader("Alpha Matrix (All Waves)")

    matrix_df = metrics_df[
        ["30D Alpha", "60D Alpha", "1Y Alpha"]
    ].copy()
    matrix_df.columns = ["Alpha 30D", "Alpha 60D", "Alpha 1Y"]

    display_matrix = matrix_df.applymap(fmt_pct)

    st.dataframe(display_matrix, use_container_width=True)

    st.caption(
        "Sort by any column using the header menu to rank Waves by near-term vs long-term alpha capture."
    )


# ---------------------------------------------------------
# History (30-Day)
# ---------------------------------------------------------

with tab_history:
    st.subheader("History (30-Day Rolling)")

    wave_names = list(engine.wave_weights.keys())
    hist_wave = st.selectbox("Choose Wave for 30D history", wave_names, key="hist")

    prices = engine.get_price_history()
    wave_rets = engine._wave_return_series(hist_wave, prices)

    if wave_rets.empty:
        st.warning("Not enough data to plot 30-day history for this Wave yet.")
    else:
        # 30D rolling cumulative return
        rolling = (1.0 + wave_rets).rolling(30).apply(
            lambda x: (1.0 + x).prod() - 1.0, raw=False
        )
        rolling = rolling.dropna()
        plot_df = pd.DataFrame({"30D Rolling Return": rolling})

        st.line_chart(plot_df)

        st.caption(
            "Rolling 30-day cumulative return for the selected Wave (not benchmark-relative)."
        )