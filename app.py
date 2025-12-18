import streamlit as st
import pandas as pd
import numpy as np

import waves_engine as engine

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console")
st.caption(
    "Stage 8+ — Wave-specific momentum, engineered concentration, "
    "SmartSafe 2.0 & 3.0 LIVE, blended benchmarks, and execution telemetry."
)

# ---------------------------------------------------------
# Mode selector
# ---------------------------------------------------------
mode_label_to_token = {
    "Standard": "standard",
    "Alpha-Minus-Beta (AMB)": "amb",
    "Private Logic™": "pl",
}

with st.sidebar:
    st.subheader("Mode")
    mode_label = st.radio(
        "Wave Mode",
        list(mode_label_to_token.keys()),
        index=0,
    )
    mode_token = mode_label_to_token[mode_label]

    st.markdown("---")
    if st.button("Force Refresh Data", use_container_width=True):
        # Clear cache & rerun to force new engine snapshots
        load_all_snapshots.clear()
        st.experimental_rerun()

# ---------------------------------------------------------
# Data loader (cached)
# ---------------------------------------------------------
@st.cache_data(ttl=120)
def load_all_snapshots(mode: str):
    """Load snapshots for all Waves from the engine for the selected mode."""
    waves = engine.get_available_waves()
    snapshots = {}
    for w in waves:
        try:
            snapshot = engine.get_wave_snapshot(w, mode=mode)
            snapshots[w] = snapshot
        except Exception as e:
            # Keep going even if one Wave fails
            snapshots[w] = {"error": str(e)}
    return snapshots


snapshots = load_all_snapshots(mode_token)

# Only keep Waves that successfully returned a snapshot
valid_waves = sorted(
    [w for w, snap in snapshots.items() if isinstance(snap, dict) and "metrics" in snap],
)

if not valid_waves:
    st.error("No Waves could be loaded. Check the engine / logs.")
    st.stop()

# ---------------------------------------------------------
# Build Overview DataFrame
# ---------------------------------------------------------
rows = []
for wave in valid_waves:
    snap = snapshots[wave]
    metrics = snap["metrics"]
    row = {
        "Wave": wave,
        "Benchmark": snap.get("benchmark", ""),
        "Intraday Return": metrics.get("intraday_return", 0.0),
        "30D Return": metrics.get("ret_30d", 0.0),
        "30D Alpha": metrics.get("alpha_30d", 0.0),
        "60D Return": metrics.get("ret_60d", 0.0),
        "60D Alpha": metrics.get("alpha_60d", 0.0),
        "1Y Return": metrics.get("ret_1y", 0.0),
        "1Y Alpha": metrics.get("alpha_1y", 0.0),
        "SI Return": metrics.get("ret_si", 0.0),
        "SI Alpha": metrics.get("alpha_si", 0.0),
        "1Y Vol": metrics.get("vol_1y", 0.0),
        "Max Drawdown": metrics.get("maxdd", 0.0),
        "1Y Info Ratio": metrics.get("ir_1y", 0.0),
        "1Y Hit Rate": metrics.get("hit_rate_1y", 0.0),
        "1Y Beta": metrics.get("beta_1y", 0.0),
        "Beta Target": metrics.get("beta_target", 0.0),
        "Beta Drift": metrics.get("beta_drift", 0.0),
        "SmartSafe 2.0 Sweep": metrics.get("smartsafe_sweep_fraction", 0.0),
        "SmartSafe 3.0 Regime": metrics.get("smartsafe3_state", ""),
        "SmartSafe 3.0 Extra": metrics.get("smartsafe3_extra_fraction", 0.0),
        "Turnover 1D": metrics.get("turnover_1d", 0.0),
        "Execution Regime": metrics.get("execution_regime", ""),
        "Mode": metrics.get("mode", mode_token),
    }
    rows.append(row)

overview_df = pd.DataFrame(rows)

# Helper to pretty-format % columns for display (but keep raw for CSV)
percent_cols = [
    "Intraday Return",
    "30D Return",
    "30D Alpha",
    "60D Return",
    "60D Alpha",
    "1Y Return",
    "1Y Alpha",
    "SI Return",
    "SI Alpha",
    "1Y Vol",
    "Max Drawdown",
    "Turnover 1D",
    "SmartSafe 2.0 Sweep",
    "SmartSafe 3.0 Extra",
]

def format_percent(x):
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "0.00%"

# ---------------------------------------------------------
# Layout: Overview + Wave Detail tabs
# ---------------------------------------------------------
tab_overview, tab_detail = st.tabs(["Overview — All Waves", "Wave Detail"])

with tab_overview:
    st.subheader("Overview — All Waves")

    col1, col2 = st.columns([1, 2])
    with col1:
        sort_direction = st.radio(
            "Direction",
            ["High → Low", "Low → High"],
            index=0,
        )
        ascending = sort_direction == "Low → High"

    with col2:
        sort_by = st.selectbox(
            "Sort by",
            [
                "1Y Alpha",
                "1Y Return",
                "SI Alpha",
                "SI Return",
                "60D Alpha",
                "60D Return",
                "30D Alpha",
                "30D Return",
                "Intraday Return",
            ],
            index=0,
        )

    display_cols = [
        "Wave",
        "Benchmark",
        "Intraday Return",
        "30D Return",
        "30D Alpha",
        "60D Return",
        "60D Alpha",
        "1Y Return",
        "1Y Alpha",
        "SI Return",
        "SI Alpha",
        "1Y Vol",
        "Max Drawdown",
        "1Y Beta",
        "Beta Target",
        "SmartSafe 2.0 Sweep",
        "SmartSafe 3.0 Regime",
        "SmartSafe 3.0 Extra",
        "Turnover 1D",
        "Execution Regime",
    ]

    df_sorted = overview_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    # Pretty version for on-screen display
    df_display = df_sorted.copy()
    for col in percent_cols:
        df_display[col] = df_display[col].apply(format_percent)

    st.dataframe(
        df_display[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    # CSV download uses raw numeric values
    csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Overview as CSV",
        data=csv_bytes,
        file_name="waves_overview.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Benchmark Set — All Waves")

    bench_df = df_sorted[["Wave", "Benchmark", "1Y Return", "1Y Alpha", "SI Return", "SI Alpha"]].copy()
    # Format returns / alpha for readability
    for col in ["1Y Return", "1Y Alpha", "SI Return", "SI Alpha"]:
        bench_df[col] = bench_df[col].apply(format_percent)

    st.dataframe(
        bench_df.rename(
            columns={
                "Wave": "Wave",
                "Benchmark": "Benchmark",
                "1Y Return": "1Y Return",
                "1Y Alpha": "1Y Alpha",
                "SI Return": "Since Inception Return",
                "SI Alpha": "Since Inception Alpha",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab_detail:
    st.subheader("Wave Detail")

    wave_choice = st.selectbox("Select Wave", valid_waves, index=0)
    snap = snapshots[wave_choice]
    metrics = snap["metrics"]
    positions = snap["positions"]

    st.markdown(f"### {wave_choice}")
    st.caption(f"Benchmark: **{snap.get('benchmark', '')}**  |  Mode: **{metrics.get('mode', mode_token)}**")

    # --- Performance metrics ---
    perf_cols_left = [
        ("Intraday Return", "intraday_return"),
        ("30-Day Return", "ret_30d"),
        ("30-Day Alpha", "alpha_30d"),
        ("60-Day Return", "ret_60d"),
        ("60-Day Alpha", "alpha_60d"),
    ]
    perf_cols_right = [
        ("1-Year Return", "ret_1y"),
        ("1-Year Alpha", "alpha_1y"),
        ("SI Return", "ret_si"),
        ("SI Alpha", "alpha_si"),
        ("1-Year Volatility", "vol_1y"),
    ]

    c1, c2 = st.columns(2)
    with c1:
        for label, key in perf_cols_left:
            st.metric(label, format_percent(metrics.get(key, 0.0)))
    with c2:
        for label, key in perf_cols_right:
            st.metric(label, format_percent(metrics.get(key, 0.0)))

    st.markdown("#### Risk & Discipline")
    c3, c4, c5 = st.columns(3)
    with c3:
        st.metric("Max Drawdown", format_percent(metrics.get("maxdd", 0.0)))
    with c4:
        st.metric("1Y Info Ratio", f"{metrics.get('ir_1y', 0.0):.2f}")
    with c5:
        st.metric("1Y Hit Rate", format_percent(metrics.get("hit_rate_1y", 0.0)))

    c6, c7, c8 = st.columns(3)
    with c6:
        st.metric("1Y Beta vs Benchmark", f"{metrics.get('beta_1y', 0.0):.2f}")
    with c7:
        st.metric("Beta Target", f"{metrics.get('beta_target', 0.0):.2f}")
    with c8:
        st.metric("Beta Drift", f"{metrics.get('beta_drift', 0.0):+.2f}")

    # --- SmartSafe status ---
    st.markdown("#### SmartSafe / Execution Status")
    c9, c10, c11 = st.columns(3)
    with c9:
        vix_val = metrics.get("vix_level", None)
        st.metric("VIX Level", f"{vix_val:.2f}" if vix_val is not None else "n/a")
    with c10:
        st.metric("SmartSafe 2.0 Sweep to BIL (Base)", format_percent(metrics.get("smartsafe_sweep_fraction", 0.0)))
    with c11:
        st.metric("Turnover Since Last Log", format_percent(metrics.get("turnover_1d", 0.0)))

    c12, c13 = st.columns(2)
    with c12:
        st.metric("SmartSafe 3.0 Regime (LIVE)", metrics.get("smartsafe3_state", "Idle"))
    with c13:
        st.metric("SmartSafe 3.0 Extra Sweep (LIVE)", format_percent(metrics.get("smartsafe3_extra_fraction", 0.0)))

    st.markdown("#### Top 10 Holdings")

    if not positions.empty:
        pos_df = positions.copy().sort_values("weight", ascending=False).head(10)
        # Google Finance links
        def google_link(tkr: str) -> str:
            tkr_str = str(tkr).upper()
            return f"[Quote](https://www.google.com/finance/quote/{tkr_str}:NASDAQ)"

        pos_df["Weight"] = pos_df["weight"].apply(format_percent)
        pos_df["Last Price"] = pos_df["last_price"].fillna(0.0).map(lambda x: f"{x:.2f}" if x else "-")
        pos_df["Google Finance"] = pos_df["ticker"].apply(google_link)

        holdings_display = pos_df[["ticker", "Weight", "Last Price", "Google Finance"]]
        holdings_display = holdings_display.rename(
            columns={
                "ticker": "Ticker",
            }
        )
        st.dataframe(holdings_display, use_container_width=True, hide_index=True)
    else:
        st.info("No holdings available for this Wave snapshot.")