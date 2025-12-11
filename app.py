"""
app.py — WAVES Intelligence™ Institutional Console
Stage 7: Mode-aware + Momentum + Engineered Concentration
         + SmartSafe 2.0 + SmartSafe 3.0 (LIVE overlay)

Features:
- Mode selector in sidebar (Standard / AMB / Private Logic™)
- Overview leaderboard for selected mode
- Wave Detail panel with:
    * Intraday, 30D, 60D, 1Y, SI returns & alpha
    * 1Y Info Ratio
    * 1Y Hit Rate
    * Max Drawdown
    * 1Y Beta vs Benchmark, Beta Target, Beta Drift
    * SmartSafe 2.0 sweep fraction (VIX ladder, live)
    * SmartSafe 3.0 regime + extra sweep fraction (LIVE overlay)
- Top 10 holdings with Google Finance links
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

import waves_engine as we

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODE_LABEL_TO_TOKEN = {
    "Standard": "standard",
    "Alpha-Minus-Beta": "amb",
    "Private Logic™": "pl",
}

# ---------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_wave_list():
    try:
        return we.get_available_waves()
    except Exception as e:
        st.error(f"Error loading Waves from wave_weights.csv: {e}")
        return []


def load_wave_snapshot(wave_name: str, mode_token: str):
    return we.get_wave_snapshot(wave_name, mode=mode_token)


@st.cache_data(show_spinner=True)
def load_all_snapshots(waves: list[str], mode_token: str) -> pd.DataFrame:
    rows = []
    for w in waves:
        try:
            snap = we.get_wave_snapshot(w, mode=mode_token)
            m = snap["metrics"]
            beta_1y = m.get("beta_1y", 0.0)
            beta_target = m.get("beta_target", 0.0)
            beta_drift = beta_1y - beta_target
            ss3_state = m.get("smartsafe3_state", "")
            ss3_extra = m.get("smartsafe3_extra_fraction", 0.0)
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": snap["benchmark"],
                    "Mode": mode_token,
                    "Intraday Return": m.get("intraday_return", 0.0),
                    "30D Return": m.get("ret_30d", 0.0),
                    "30D Alpha": m.get("alpha_30d", 0.0),
                    "60D Return": m.get("ret_60d", 0.0),
                    "60D Alpha": m.get("alpha_60d", 0.0),
                    "1Y Return": m.get("ret_1y", 0.0),
                    "1Y Alpha": m.get("alpha_1y", 0.0),
                    "SI Return": m.get("ret_si", 0.0),
                    "SI Alpha": m.get("alpha_si", 0.0),
                    "1Y IR": m.get("ir_1y", 0.0),
                    "1Y Hit Rate": m.get("hit_rate_1y", 0.0),
                    "Max Drawdown": m.get("maxdd", 0.0),
                    "Beta 1Y": beta_1y,
                    "Beta Drift": beta_drift,
                    "SS3 State": ss3_state,
                    "SS3 Extra Sweep": ss3_extra,
                }
            )
        except Exception:
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": "",
                    "Mode": mode_token,
                    "Intraday Return": 0.0,
                    "30D Return": 0.0,
                    "30D Alpha": 0.0,
                    "60D Return": 0.0,
                    "60D Alpha": 0.0,
                    "1Y Return": 0.0,
                    "1Y Alpha": 0.0,
                    "SI Return": 0.0,
                    "SI Alpha": 0.0,
                    "1Y IR": 0.0,
                    "1Y Hit Rate": 0.0,
                    "Max Drawdown": 0.0,
                    "Beta 1Y": 0.0,
                    "Beta Drift": 0.0,
                    "SS3 State": "",
                    "SS3 Extra Sweep": 0.0,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Wave",
                "Benchmark",
                "Mode",
                "Intraday Return",
                "30D Return",
                "30D Alpha",
                "60D Return",
                "60D Alpha",
                "1Y Return",
                "1Y Alpha",
                "SI Return",
                "SI Alpha",
                "1Y IR",
                "1Y Hit Rate",
                "Max Drawdown",
                "Beta 1Y",
                "Beta Drift",
                "SS3 State",
                "SS3 Extra Sweep",
            ]
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------

def format_pct(x: float) -> str:
    try:
        return f"{x * 100:,.2f}%"
    except Exception:
        return "-"


def format_ratio(x: float) -> str:
    try:
        return f"{x:,.2f}"
    except Exception:
        return "-"


def top_holdings_table(positions: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()

    df = positions.copy()
    df = df.sort_values("weight", ascending=False).head(max_rows)

    df["Weight"] = df["weight"].apply(lambda w: f"{w * 100:,.2f}%")
    df["Last Price"] = df["last_price"].apply(
        lambda p: f"${p:,.2f}" if pd.notnull(p) else "-"
    )
    df["Google Finance"] = df["ticker"].apply(
        lambda t: f"[Quote](https://www.google.com/finance/quote/{t})"
    )

    df = df[["ticker", "Weight", "Last Price", "Google Finance"]].rename(
        columns={"ticker": "Ticker"}
    )
    return df


def render_header():
    st.markdown(
        """
        <h1 style="margin-bottom:0;">WAVES Intelligence™ Institutional Console</h1>
        <p style="margin-top:0.25rem; font-size:0.9rem; opacity:0.7;">
            Stage 7 — Mode-aware, momentum-aware, engineered concentration, SmartSafe 2.0,
            blended benchmarks, multi-horizon alpha & beta discipline, and SmartSafe 3.0
            <strong>LIVE overlay</strong> on portfolio exposures.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(waves: list[str]) -> tuple[str, str]:
    st.sidebar.markdown("### Waves Intelligence™")

    mode_label = st.sidebar.radio(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )
    mode_token = MODE_LABEL_TO_TOKEN[mode_label]
    st.sidebar.caption("Mode applies to Overview and Wave Detail.")

    if not waves:
        st.sidebar.error("No Waves found in wave_weights.csv.")
        return "", mode_token

    st.sidebar.markdown("---")
    st.sidebar.write("Select a Wave for the **Wave Detail** tab:")
    selected_wave = st.sidebar.selectbox("Wave", waves, index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "SmartSafe 2.0 sweeps and SmartSafe 3.0 extra sweeps are both live in this build."
    )

    return selected_wave, mode_token


def render_metrics(snapshot: dict):
    metrics = snapshot["metrics"]
    benchmark = snapshot["benchmark"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Intraday Return", format_pct(metrics.get("intraday_return", 0.0)))
    with col2:
        st.metric("30-Day Return", format_pct(metrics.get("ret_30d", 0.0)))
    with col3:
        st.metric("60-Day Return", format_pct(metrics.get("ret_60d", 0.0)))
    with col4:
        st.metric("30-Day Alpha vs " + str(benchmark), format_pct(metrics.get("alpha_30d", 0.0)))

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("60-Day Alpha vs " + str(benchmark), format_pct(metrics.get("alpha_60d", 0.0)))
    with col6:
        st.metric("1-Year Return", format_pct(metrics.get("ret_1y", 0.0)))
    with col7:
        st.metric("1-Year Alpha vs " + str(benchmark), format_pct(metrics.get("alpha_1y", 0.0)))
    with col8:
        st.metric("Since Inception Return", format_pct(metrics.get("ret_si", 0.0)))

    col9, col10, col11, col12 = st.columns(4)
    with col9:
        st.metric(
            "Since Inception Alpha vs " + str(benchmark),
            format_pct(metrics.get("alpha_si", 0.0)),
        )
    with col10:
        st.metric("1-Year Info Ratio", format_ratio(metrics.get("ir_1y", 0.0)))
    with col11:
        st.metric("1-Year Hit Rate", format_pct(metrics.get("hit_rate_1y", 0.0)))
    with col12:
        st.metric("Max Drawdown", format_pct(metrics.get("maxdd", 0.0)))

    # Beta telemetry row
    beta_1y = metrics.get("beta_1y", 0.0)
    beta_target = metrics.get("beta_target", 0.0)
    beta_drift = metrics.get("beta_drift", 0.0)

    colb1, colb2, colb3, _ = st.columns(4)
    with colb1:
        st.metric("1-Year Beta vs Benchmark", format_ratio(beta_1y))
    with colb2:
        st.metric("Beta Target", format_ratio(beta_target))
    with colb3:
        st.metric("Beta Drift (Actual - Target)", format_ratio(beta_drift))


def render_top_holdings(snapshot: dict):
    positions = snapshot["positions"]
    holdings_df = top_holdings_table(positions, max_rows=10)

    st.subheader("Top 10 Holdings")
    if holdings_df.empty:
        st.info("No holdings data available for this Wave.")
        return

    st.markdown("Click **Quote** to open the Google Finance page.", unsafe_allow_html=True)

    st.dataframe(
        holdings_df,
        use_container_width=True,
        hide_index=True,
    )


def render_positions_raw(snapshot: dict):
    st.subheader("Underlying Positions (Post SmartSafe 3.0 Overlay)")
    positions = snapshot["positions"]
    if positions.empty:
        st.info("No underlying positions available.")
        return

    display_cols = ["ticker", "weight", "last_price", "intraday_return"]
    for c in display_cols:
        if c not in positions.columns:
            positions[c] = None

    df = positions[display_cols].copy()
    df = df.rename(
        columns={
            "ticker": "Ticker",
            "weight": "Weight",
            "last_price": "Last Price",
            "intraday_return": "Intraday Return",
        }
    )
    df["Weight"] = df["Weight"].apply(lambda w: f"{w * 100:,.2f}%")
    df["Last Price"] = df["Last Price"].apply(
        lambda p: f"${p:,.2f}" if pd.notnull(p) else "-"
    )
    df["Intraday Return"] = df["Intraday Return"].apply(format_pct)

    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------

def _color_metric(val: str) -> str:
    if not isinstance(val, str):
        return ""
    if val.startswith("-"):
        return "color: #ff4d4f;"
    if val.endswith("%") and not val.startswith("0.00"):
        return "color: #16c784;"
    return ""


def render_overview_tab(waves: list[str], mode_token: str):
    st.subheader("Overview — All Waves")

    if not waves:
        st.info("No Waves available. Check wave_weights.csv.")
        return

    with st.spinner(f"Calculating metrics for mode: {mode_token}"):
        df = load_all_snapshots(waves, mode_token)

    if df.empty:
        st.info("No metrics available yet.")
        return

    st.markdown(
        """
        This grid shows Intraday, 30D, 60D, 1Y, and Since-Inception
        performance and alpha for each Wave in the **selected mode**, plus
        1Y Info Ratio, 1Y Hit Rate, Max Drawdown, Beta discipline, and
        SmartSafe 3.0 regime + extra sweep (LIVE overlay on top of 2.0).
        """,
        unsafe_allow_html=True,
    )

    metric_options = [
        "Intraday Return",
        "30D Return",
        "30D Alpha",
        "60D Return",
        "60D Alpha",
        "1Y Return",
        "1Y Alpha",
        "SI Return",
        "SI Alpha",
    ]
    col_sort, col_dir, _ = st.columns([2, 1.5, 1])
    with col_sort:
        sort_col = st.selectbox("Sort by metric", metric_options, index=2)
    with col_dir:
        sort_dir = st.radio("Direction", ["High → Low", "Low → High"], index=0)

    ascending = sort_dir == "Low → High"
    df_sorted = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    df_sorted.insert(0, "Rank", range(1, len(df_sorted) + 1))

    df_display = df_sorted.copy()

    # Format percentage-type columns
    pct_cols = [
        "Intraday Return",
        "30D Return",
        "30D Alpha",
        "60D Return",
        "60D Alpha",
        "1Y Return",
        "1Y Alpha",
        "SI Return",
        "SI Alpha",
        "1Y Hit Rate",
        "Max Drawdown",
        "SS3 Extra Sweep",
    ]
    for col in pct_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(format_pct)

    # Ratio columns
    if "1Y IR" in df_display.columns:
        df_display["1Y IR"] = df_display["1Y IR"].apply(format_ratio)
    if "Beta 1Y" in df_display.columns:
        df_display["Beta 1Y"] = df_display["Beta 1Y"].apply(format_ratio)
    if "Beta Drift" in df_display.columns:
        df_display["Beta Drift"] = df_display["Beta Drift"].apply(format_ratio)

    styler = df_display.style.applymap(
        _color_metric,
        subset=metric_options,
    )

    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = df_sorted.to_csv(index=False)
    st.download_button(
        label="Download Overview as CSV",
        data=csv_data,
        file_name=f"waves_overview_{mode_token}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# Wave Detail tab
# ---------------------------------------------------------------------

def render_smartsafe_panel(metrics: dict):
    st.subheader("SmartSafe Status")

    vix_level = metrics.get("vix_level", None)
    sweep_frac = metrics.get("smartsafe_sweep_fraction", 0.0)
    ss3_state = metrics.get("smartsafe3_state", "Normal")
    ss3_extra = metrics.get("smartsafe3_extra_fraction", 0.0)

    col1, col2 = st.columns(2)
    with col1:
        if vix_level is not None:
            st.metric("VIX Level", f"{vix_level:,.2f}")
    with col2:
        if sweep_frac and sweep_frac > 0:
            st.metric("SmartSafe 2.0 Sweep to BIL (Base)", format_pct(sweep_frac))
        else:
            st.metric("SmartSafe 2.0 Sweep to BIL (Base)", "0.00%")

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.metric("SmartSafe 3.0 Regime (LIVE)", ss3_state)
    with col4:
        st.metric("SmartSafe 3.0 Extra Sweep (LIVE)", format_pct(ss3_extra))

    st.caption(
        "SmartSafe 3.0 applies an additional volatility-based sweep into BIL on top of "
        "SmartSafe 2.0, driven by VIX, 60D returns, max drawdown, and beta drift."
    )


def render_wave_detail_tab(selected_wave: str, mode_token: str):
    if not selected_wave:
        st.info("Select a Wave in the sidebar to view details.")
        return

    with st.spinner(f"Loading {selected_wave} in mode: {mode_token}"):
        snapshot = load_wave_snapshot(selected_wave, mode_token)

    st.markdown(f"## {selected_wave}")
    st.caption(f"Benchmark: {snapshot['benchmark']}")

    render_metrics(snapshot)

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_top_holdings(snapshot)
    with col_right:
        render_smartsafe_panel(snapshot["metrics"])

    st.markdown("---")
    render_positions_raw(snapshot)

    st.markdown(
        """
        <div style="font-size:0.75rem; opacity:0.6; margin-top:1rem;">
        WAVES Intelligence™ — Stage 7 adaptive momentum + engineered concentration +
        beta discipline, VIX-aware SmartSafe 2.0, and SmartSafe 3.0 LIVE overlay.
        Blended benchmarks, multi-horizon alpha & risk. For internal / demo use only.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main():
    render_header()

    waves = load_wave_list()
    selected_wave, mode_token = render_sidebar(waves)

    tab_overview, tab_detail = st.tabs(["Overview", "Wave Detail"])

    with tab_overview:
        render_overview_tab(waves, mode_token)

    with tab_detail:
        render_wave_detail_tab(selected_wave, mode_token)


if __name__ == "__main__":
    main()