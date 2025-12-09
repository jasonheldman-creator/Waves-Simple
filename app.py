"""
app.py

WAVES Intelligence™ Institutional Console — Phase 2.5

- Mode selector: Standard / Alpha-Minus-Beta / Private Logic
- Tabs:
    1) Overview (All Waves): Intraday / 30D / 60D / 1Y Alpha Captured for ALL *active* Waves
    2) Selected Wave Detail: full dashboard for the chosen Wave
- Uses mode-aware + VIX-gated WavesEngine.
- Explicitly excludes Crypto Income variants from the console.
"""

import streamlit as st
import pandas as pd

from waves_engine import WavesEngine


# ----------------------------------------------------------------------
# Formatting Helpers
# ----------------------------------------------------------------------
def _fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


def _fmt_pct_diff(wave, bm):
    if wave is None or bm is None or pd.isna(wave) or pd.isna(bm):
        return "—"
    diff = (wave - bm) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:0.2f} pts vs BM"


# ----------------------------------------------------------------------
# Hard cache reset on app start
# ----------------------------------------------------------------------
def clear_streamlit_cache_once():
    if "cache_cleared" in st.session_state:
        return

    try:
        if hasattr(st, "cache_data"):
            st.cache_data.clear()
        if hasattr(st, "cache_resource"):
            st.cache_resource.clear()
    except Exception:
        pass

    st.session_state["cache_cleared"] = True


clear_streamlit_cache_once()

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console")
st.caption(
    "Live Wave Engine • Mode-Aware + VIX-Gated Beta-Adjusted Alpha Captured • "
    "Intraday + 30/60/1-Year • S&P Wave + Full Lineup"
)

# ----------------------------------------------------------------------
# Initialize engine
# ----------------------------------------------------------------------
try:
    engine = WavesEngine(list_path="list.csv", weights_path="wave_weights.csv")
except Exception as e:
    st.error(f"Engine failed to initialize: {e}")
    st.stop()

# Filter out unwanted waves (Crypto Income, etc.)
EXCLUDED_WAVES = {
    "Crypto Income Wave",
    "Crypto Income",
    "Crypto Wave",
    "Crypto Income APW",
    "Crypto Income AIW",
}

all_waves = engine.get_wave_names()
waves = [w for w in all_waves if w not in EXCLUDED_WAVES]

if not waves:
    st.error("No active Waves detected (after exclusions).")
    st.stop()

# Sidebar
st.sidebar.header("Wave Selector")
selected_wave = st.sidebar.selectbox("Select Wave", waves, index=0)

st.sidebar.header("Mode")
mode_label = st.sidebar.selectbox(
    "Select Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic"],
    index=0,
)

mode_map = {
    "Standard": "standard",
    "Alpha-Minus-Beta": "alpha-minus-beta",
    "Private Logic": "private_logic",
}
selected_mode_key = mode_map[mode_label]

st.sidebar.markdown("---")
st.sidebar.markdown("**Files in use:**")
st.sidebar.code("list.csv\nwave_weights.csv", language="text")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_overview, tab_detail = st.tabs(
    ["Overview — All Waves Alpha Capture", "Selected Wave Detail"]
)

# ----------------------------------------------------------------------
# TAB 1: Overview — All Waves Alpha Capture
# ----------------------------------------------------------------------
with tab_overview:
    st.subheader(f"All Waves — Alpha Captured (Mode: {mode_label})")

    overview_rows = []
    for wave_name in waves:
        try:
            perf = engine.get_wave_performance(
                wave_name, mode=selected_mode_key, days=30, log=False
            )
        except Exception:
            perf = None

        if perf is not None:
            row = {
                "Wave": wave_name,
                "Benchmark": perf["benchmark"],
                "Realized Beta (≈60d)": perf["beta_realized"],
                "Exposure (Net)": perf.get("exposure_final", None),
                "Intraday Alpha Captured (%)": perf["intraday_alpha_captured"] * 100
                if perf["intraday_alpha_captured"] is not None
                else None,
                "30-Day Alpha Captured (%)": perf["alpha_30d"] * 100
                if perf["alpha_30d"] is not None
                else None,
                "60-Day Alpha Captured (%)": perf["alpha_60d"] * 100
                if perf["alpha_60d"] is not None
                else None,
                "1-Year Alpha Captured (%)": perf["alpha_1y"] * 100
                if perf["alpha_1y"] is not None
                else None,
            }
        else:
            row = {
                "Wave": wave_name,
                "Benchmark": "—",
                "Realized Beta (≈60d)": None,
                "Exposure (Net)": None,
                "Intraday Alpha Captured (%)": None,
                "30-Day Alpha Captured (%)": None,
                "60-Day Alpha Captured (%)": None,
                "1-Year Alpha Captured (%)": None,
            }

        overview_rows.append(row)

    overview_df = pd.DataFrame(overview_rows)

    for col in [
        "Realized Beta (≈60d)",
        "Exposure (Net)",
        "Intraday Alpha Captured (%)",
        "30-Day Alpha Captured (%)",
        "60-Day Alpha Captured (%)",
        "1-Year Alpha Captured (%)",
    ]:
        if col in overview_df.columns:
            overview_df[col] = overview_df[col].astype(float).round(2)

    st.dataframe(overview_df, hide_index=True, use_container_width=True)

    st.caption(
        "Alpha Captured is beta-adjusted residual performance vs benchmark. "
        "Exposure (Net) reflects combined Mode + VIX overlay."
    )

# ----------------------------------------------------------------------
# TAB 2: Selected Wave Detail
# ----------------------------------------------------------------------
with tab_detail:
    top_row = st.columns([2.2, 1.2])
    perf_col, snapshot_col = top_row

    bottom_row = st.columns([2.0, 1.4])
    chart_col, holdings_col = bottom_row

    # ----- Wave Snapshot -----
    with snapshot_col:
        st.subheader("Wave Snapshot")

        try:
            holdings_df = engine.get_wave_holdings(selected_wave)
            num_holdings = len(holdings_df)
            benchmark = engine.get_benchmark(selected_wave)
        except Exception as e:
            st.error(f"Could not load snapshot for {selected_wave}: {e}")
            holdings_df = None
            num_holdings = 0
            benchmark = "SPY"

        snapshot1, snapshot2 = st.columns(2)
        snapshot1.metric("Wave", selected_wave)
        snapshot2.metric("Benchmark", benchmark)

        st.write("")
        st.write(f"**Mode:** {mode_label}")
        st.write(f"**Holdings:** {num_holdings:,}")

        if holdings_df is not None and "sector" in holdings_df.columns:
            sector_weights = (
                holdings_df.groupby("sector")["weight"]
                .sum()
                .sort_values(ascending=False)
            )
            if not sector_weights.empty:
                top_sector = sector_weights.index[0]
                top_sector_weight = float(sector_weights.iloc[0])
                st.write(f"**Top Sector:** {top_sector} ({top_sector_weight:.1%})")

    # ----- Performance Panel -----
    with perf_col:
        st.subheader(f"{selected_wave} — Alpha Captured (β-Adjusted)")

        try:
            perf = engine.get_wave_performance(
                selected_wave, mode=selected_mode_key, days=30, log=True
            )
        except Exception as e:
            st.error(f"Could not compute performance for {selected_wave}: {e}")
            perf = None

        if perf is not None:
            beta = perf["beta_realized"]
            exposure = perf.get("exposure_final", None)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Intraday Alpha Captured",
                _fmt_pct(perf["intraday_alpha_captured"]),
            )
            c2.metric("30-Day Alpha Captured", _fmt_pct(perf["alpha_30d"]))
            c3.metric("60-Day Alpha Captured", _fmt_pct(perf["alpha_60d"]))
            c4.metric("1-Year Alpha Captured", _fmt_pct(perf["alpha_1y"]))

            st.markdown(f"**Realized Beta (≈60d):** {beta:0.2f}")
            if exposure is not None:
                st.markdown(f"**Net Exposure (Mode + VIX):** {exposure:0.2f}×")
            st.markdown(f"**Benchmark:** {perf['benchmark']}")

            st.write("")
            r1, r2, r3 = st.columns(3)
            r1.metric(
                "30-Day Wave Return",
                _fmt_pct(perf["return_30d_wave"]),
                delta=_fmt_pct_diff(
                    perf["return_30d_wave"], perf["return_30d_benchmark"]
                ),
            )
            r2.metric(
                "60-Day Wave Return",
                _fmt_pct(perf["return_60d_wave"]),
                delta=_fmt_pct_diff(
                    perf["return_60d_wave"], perf["return_60d_benchmark"]
                ),
            )
            r3.metric(
                "1-Year Wave Return",
                _fmt_pct(perf["return_1y_wave"]),
                delta=_fmt_pct_diff(
                    perf["return_1y_wave"], perf["return_1y_benchmark"]
                ),
            )

    # ----- Chart + History -----
    with chart_col:
        if perf is not None:
            st.markdown(
                f"### {selected_wave} vs {perf['benchmark']} — 30-Day Curve (β-Adj Alpha)"
            )

            history = perf["history_30d"]
            chart_data = history[["wave_value", "benchmark_value"]]
            st.line_chart(chart_data)

            hist_df = history.copy()
            hist_df = hist_df.reset_index()
            date_col = hist_df.columns[0]
            hist_df = hist_df.rename(columns={date_col: "date"})
            hist_df["date"] = pd.to_datetime(hist_df["date"]).dt.date

            hist_df["wave_return_pct"] = hist_df["wave_return"] * 100
            hist_df["benchmark_return_pct"] = hist_df["benchmark_return"] * 100
            hist_df["alpha_captured_pct"] = hist_df["alpha_captured"] * 100

            display_cols = [
                "date",
                "wave_return_pct",
                "benchmark_return_pct",
                "alpha_captured_pct",
            ]
            hist_display = hist_df[display_cols].tail(15).iloc[::-1]
            hist_display = hist_display.rename(
                columns={
                    "date": "Date",
                    "wave_return_pct": "Wave Return (%)",
                    "benchmark_return_pct": "Benchmark Return (%)",
                    "alpha_captured_pct": "Alpha Captured (%)",
                }
            )
            hist_display = hist_display.round(3)

            st.markdown("#### Recent Daily Returns & Alpha Captured (Last 15 Days)")
            st.dataframe(hist_display, hide_index=True, use_container_width=True)

    # ----- Holdings -----
    with holdings_col:
        st.subheader(f"{selected_wave} — Top 10 Holdings")

        try:
            top10 = engine.get_top_holdings(selected_wave, n=10)
        except Exception as e:
            st.error(f"Could not load holdings for {selected_wave}: {e}")
            top10 = None

        if top10 is not None and not top10.empty:

            def google_finance_url(ticker: str) -> str:
                return f"https://www.google.com/finance/quote/{ticker}:NASDAQ"

            display_df = top10.copy()

            if "company" not in display_df.columns:
                display_df["company"] = ""

            display_df = display_df[["ticker", "company", "weight"]].copy()
            display_df["weight"] = display_df["weight"].round(4)
            display_df["Google Finance"] = display_df["ticker"].apply(
                google_finance_url
            )

            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.write("No holdings found for this Wave.")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Engine: WAVES Intelligence™ • list.csv = total market universe • "
    "wave_weights.csv = Wave definitions • Alpha = Mode-Aware + VIX-Gated, "
    "Beta-Adjusted Alpha Captured • Modes: Standard / Alpha-Minus-Beta / "
    "Private Logic fully handled in engine logic."
)