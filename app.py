"""
app.py

WAVES Intelligence™ Institutional Console — Clean Version
Works with the simplified waves_engine.py (no circular imports).

Features:
  • Multi-tab Streamlit app:
      1) Dashboard (all Waves snapshot)
      2) Wave Explorer (detail view)
      3) Alpha Matrix (comparison table)
      4) History (30-day return/alpha table)
      5) About / Diagnostics
  • Intraday, 30-Day, 60-Day, 1-Year Alpha Captured
  • 30-Day price/alpha chart for the selected Wave
  • Top-10 holdings with clickable tickers (Google Finance quotes)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from waves_engine import WavesEngine


# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------
def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


def fmt_beta(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:0.2f}"


def fmt_pts_diff(wave_ret, bm_ret):
    if wave_ret is None or bm_ret is None or pd.isna(wave_ret) or pd.isna(bm_ret):
        return "—"
    diff = (wave_ret - bm_ret) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:0.2f} pts vs BM"


# ------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Wave Engine • Alpha Capture • Benchmark-Relative Performance")


# ------------------------------------------------------------
# Initialise engine
# ------------------------------------------------------------
try:
    engine = WavesEngine(list_path="list.csv", weights_path="wave_weights.csv")
except Exception as e:
    st.error(f"Engine failed to initialise: {e}")
    st.stop()

wave_names = engine.get_wave_names()
if not wave_names:
    st.error("No Waves found in wave_weights.csv.")
    st.stop()


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.header("Controls")

selected_wave = st.sidebar.selectbox("Wave", wave_names, index=0)
mode_label = st.sidebar.selectbox(
    "Mode (display only)",
    ["standard", "alpha-minus-beta", "private_logic"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Files**")
st.sidebar.code("list.csv\nwave_weights.csv", language="text")


# ------------------------------------------------------------
# Helper: get performance safely
# ------------------------------------------------------------
def get_wave_perf(wave: str, mode: str = "standard"):
    try:
        return engine.get_wave_performance(wave, mode=mode, days=30, log=False)
    except Exception as e:
        st.warning(f"Could not compute performance for {wave}: {e}")
        return None


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab_dash, tab_wave, tab_matrix, tab_history, tab_about = st.tabs(
    [
        "Dashboard",
        "Wave Explorer",
        "Alpha Matrix",
        "History (30-Day)",
        "About / Diagnostics",
    ]
)


# ------------------------------------------------------------
# TAB 1 — DASHBOARD
# ------------------------------------------------------------
with tab_dash:
    st.subheader(f"Dashboard — Mode: {mode_label}")

    rows = []
    for w in wave_names:
        perf = get_wave_perf(w, mode_label)
        if perf is None:
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": "—",
                    "Beta (≈60d)": np.nan,
                    "Intraday Alpha": np.nan,
                    "Alpha 30D": np.nan,
                    "Alpha 60D": np.nan,
                    "Alpha 1Y": np.nan,
                    "1Y Wave Return": np.nan,
                    "1Y BM Return": np.nan,
                }
            )
            continue

        rows.append(
            {
                "Wave": w,
                "Benchmark": perf["benchmark"],
                "Beta (≈60d)": perf["beta_realized"],
                "Intraday Alpha": perf["intraday_alpha_captured"],
                "Alpha 30D": perf["alpha_30d"],
                "Alpha 60D": perf["alpha_60d"],
                "Alpha 1Y": perf["alpha_1y"],
                "1Y Wave Return": perf["return_1y_wave"],
                "1Y BM Return": perf["return_1y_benchmark"],
            }
        )

    dash_df = pd.DataFrame(rows)

    # Summary metrics across all Waves
    for col in ["Intraday Alpha", "Alpha 30D", "Alpha 60D", "Alpha 1Y"]:
        dash_df[col] = pd.to_numeric(dash_df[col], errors="coerce")

    avg_intraday = dash_df["Intraday Alpha"].mean()
    avg_30 = dash_df["Alpha 30D"].mean()
    avg_60 = dash_df["Alpha 60D"].mean()
    avg_1y = dash_df["Alpha 1Y"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Intraday Alpha", fmt_pct(avg_intraday))
    c2.metric("Avg 30D Alpha", fmt_pct(avg_30))
    c3.metric("Avg 60D Alpha", fmt_pct(avg_60))
    c4.metric("Avg 1Y Alpha", fmt_pct(avg_1y))

    # Detailed table
    table = dash_df.copy()
    for col in ["Intraday Alpha", "Alpha 30D", "Alpha 60D", "Alpha 1Y", "1Y Wave Return", "1Y BM Return"]:
        table[col] = (pd.to_numeric(table[col], errors="coerce") * 100).round(2)

    table = table.rename(
        columns={
            "Beta (≈60d)": "Beta (≈60d)",
            "Intraday Alpha": "Intraday Alpha (%)",
            "Alpha 30D": "Alpha 30D (%)",
            "Alpha 60D": "Alpha 60D (%)",
            "Alpha 1Y": "Alpha 1Y (%)",
            "1Y Wave Return": "1Y Wave Return (%)",
            "1Y BM Return": "1Y BM Return (%)",
        }
    )

    st.markdown("### All Waves Snapshot")
    st.dataframe(table, hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# TAB 2 — WAVE EXPLORER
# ------------------------------------------------------------
with tab_wave:
    st.subheader(f"Wave Explorer — {selected_wave} (mode: {mode_label})")

    perf = get_wave_perf(selected_wave, mode_label)
    if perf is None:
        st.stop()

    benchmark = perf["benchmark"]

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Intraday Alpha Captured", fmt_pct(perf["intraday_alpha_captured"]))
    m2.metric("30D Alpha Captured", fmt_pct(perf["alpha_30d"]))
    m3.metric("60D Alpha Captured", fmt_pct(perf["alpha_60d"]))
    m4.metric("1Y Alpha Captured", fmt_pct(perf["alpha_1y"]))

    r1, r2, r3 = st.columns(3)
    r1.metric(
        "30D Wave Return",
        fmt_pct(perf["return_30d_wave"]),
        fmt_pts_diff(perf["return_30d_wave"], perf["return_30d_benchmark"]),
    )
    r2.metric(
        "60D Wave Return",
        fmt_pct(perf["return_60d_wave"]),
        fmt_pts_diff(perf["return_60d_wave"], perf["return_60d_benchmark"]),
    )
    r3.metric(
        "1Y Wave Return",
        fmt_pct(perf["return_1y_wave"]),
        fmt_pts_diff(perf["return_1y_wave"], perf["return_1y_benchmark"]),
    )

    st.markdown(
        f"**Benchmark:** {benchmark} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Realised Beta (≈60d):** {fmt_beta(perf['beta_realized'])}"
    )

    # Layout: chart + table + top 10 holdings
    chart_col, table_col = st.columns([2, 1.4])

    # 30-Day curve chart & daily alpha
    with chart_col:
        st.markdown("#### 30-Day Curve — Wave vs Benchmark")
        hist = perf["history_30d"].copy()

        if "wave_value" in hist.columns and "benchmark_value" in hist.columns:
            st.line_chart(hist[["wave_value", "benchmark_value"]])
        else:
            st.warning("History data missing value columns for chart.")

        # Daily returns / alpha table
        hist_display = hist.copy().reset_index()
        date_col = hist_display.columns[0]
        hist_display = hist_display.rename(columns={date_col: "Date"})
        hist_display["Date"] = pd.to_datetime(hist_display["Date"]).dt.date

        for col in ["wave_return", "benchmark_return", "alpha_captured"]:
            if col in hist_display.columns:
                hist_display[col] = (hist_display[col] * 100).round(3)

        cols_to_show = [
            c for c in ["Date", "wave_return", "benchmark_return", "alpha_captured"]
            if c in hist_display.columns
        ]

        if cols_to_show:
            hist_display = hist_display[cols_to_show].tail(15).iloc[::-1]
            hist_display = hist_display.rename(
                columns={
                    "wave_return": "Wave Return (%)",
                    "benchmark_return": "BM Return (%)",
                    "alpha_captured": "Alpha Captured (%)",
                }
            )
            st.markdown("#### Recent Daily Returns & Alpha (Last 15 Days)")
            st.dataframe(hist_display, hide_index=True, use_container_width=True)

    # Top 10 holdings with clickable tickers
    with table_col:
        st.markdown("#### Top 10 Holdings")

        try:
            top10 = engine.get_top_holdings(selected_wave, n=10)
        except Exception as e:
            st.error(f"Error loading holdings: {e}")
            top10 = None

        if top10 is not None and not top10.empty:

            def google_url(ticker: str) -> str:
                return f"https://www.google.com/finance/quote/{ticker}"

            md_lines = [
                "| Ticker | Weight |",
                "|:------:|-------:|",
            ]
            for _, row in top10.iterrows():
                tkr = str(row["ticker"])
                w = float(row["weight"])
                md_lines.append(f"| [{tkr}]({google_url(tkr)}) | {w:.2%} |")

            st.markdown("\n".join(md_lines), unsafe_allow_html=True)
        else:
            st.write("No holdings found for this Wave.")


# ------------------------------------------------------------
# TAB 3 — ALPHA MATRIX
# ------------------------------------------------------------
with tab_matrix:
    st.subheader(f"Alpha Matrix — Mode: {mode_label}")

    rows = []
    for w in wave_names:
        perf = get_wave_perf(w, mode_label)
        if perf is None:
            continue
        rows.append(
            {
                "Wave": w,
                "Benchmark": perf["benchmark"],
                "Beta (≈60d)": perf["beta_realized"],
                "Intraday Alpha (%)": perf["intraday_alpha_captured"] * 100,
                "Alpha 30D (%)": perf["alpha_30d"] * 100,
                "Alpha 60D (%)": perf["alpha_60d"] * 100,
                "Alpha 1Y (%)": perf["alpha_1y"] * 100,
                "1Y Wave Return (%)": (perf["return_1y_wave"] or 0) * 100
                if perf["return_1y_wave"] is not None
                else np.nan,
                "1Y BM Return (%)": (perf["return_1y_benchmark"] or 0) * 100
                if perf["return_1y_benchmark"] is not None
                else np.nan,
            }
        )

    if not rows:
        st.info("No alpha data available.")
    else:
        matrix = pd.DataFrame(rows)

        sort_by = st.selectbox(
            "Sort by",
            ["Alpha 30D (%)", "Alpha 60D (%)", "Alpha 1Y (%)", "Intraday Alpha (%)"],
            index=0,
        )

        matrix = matrix.sort_values(sort_by, ascending=False)
        matrix["Beta (≈60d)"] = pd.to_numeric(matrix["Beta (≈60d)"], errors="coerce").round(2)

        st.dataframe(matrix, hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# TAB 4 — HISTORY (30-DAY TABLE)
# ------------------------------------------------------------
with tab_history:
    st.subheader(f"30-Day History — {selected_wave}")

    perf = get_wave_perf(selected_wave, mode_label)
    if perf is None:
        st.stop()

    hist = perf["history_30d"].copy().reset_index()
    date_col = hist.columns[0]
    hist = hist.rename(columns={date_col: "Date"})
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.date

    for col in ["wave_return", "benchmark_return", "alpha_captured"]:
        if col in hist.columns:
            hist[col] = (hist[col] * 100).round(3)

    cols = [
        c
        for c in [
            "Date",
            "wave_return",
            "benchmark_return",
            "alpha_captured",
            "wave_value",
            "benchmark_value",
        ]
        if c in hist.columns
    ]

    hist = hist[cols]
    hist = hist.rename(
        columns={
            "wave_return": "Wave Return (%)",
            "benchmark_return": "BM Return (%)",
            "alpha_captured": "Alpha Captured (%)",
            "wave_value": "Wave Value (Index)",
            "benchmark_value": "BM Value (Index)",
        }
    )

    st.dataframe(hist, hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# TAB 5 — ABOUT / DIAGNOSTICS
# ------------------------------------------------------------
with tab_about:
    st.subheader("About / Diagnostics")

    st.markdown(
        """
        **WAVES Intelligence™ Engine (Current Session)**  

        • `list.csv` provides the total market universe (tickers + optional metadata).  
        • `wave_weights.csv` defines each Wave via `wave,ticker,weight`.  
        • The engine:
          - pulls up to 1 year of history from `yfinance`,
          - computes Wave returns by weighting constituent returns,
          - compares them to a benchmark (default `SPY`),
          - derives intraday, 30-day, 60-day, and full-period (≈1-year) **alpha captured**,
          - estimates a realised beta over the last ~60 trading days.
        """
    )

    st.markdown("#### Files Present")
    for p in ["list.csv", "wave_weights.csv"]:
        exists = Path(p).exists()
        st.write(f"- `{p}` : {'✅ found' if exists else '❌ missing'}")

    st.markdown("---")
    st.caption(
        "Engine: WAVES Intelligence™ • Alpha = Wave return minus benchmark return • "
        "All metrics are for illustration/testing only and not investment advice."
    )