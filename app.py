"""
app.py — WAVES Intelligence™ Institutional Console
(Leaderboard + 1Y & Since Inception Metrics)

Key behaviors:
- Auto-discovers Waves from wave_weights.csv via waves_engine.get_available_waves()
- No SmartSafe 3.0 logic. Only SmartSafe 2.0 hooks via engine.
- Tabs:
    1) Overview (all Waves grid, sortable, color-coded, ranked, CSV export)
    2) Wave Detail (single-Wave dashboard)

Metrics:
- Intraday Return
- 30-Day Return & Alpha
- 60-Day Return & Alpha
- 1-Year Return & Alpha
- Since-Inception Return & Alpha

Per-Wave view:
- Metrics (all windows)
- Top 10 holdings with clickable Google Finance quote links
- Raw positions
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


# ---------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_wave_list():
    """Load all Waves once (names come from wave_weights.csv)."""
    try:
        waves = we.get_available_waves()
        return waves
    except Exception as e:
        st.error(f"Error loading Waves from wave_weights.csv: {e}")
        return []


def load_wave_snapshot(wave_name: str):
    """
    Wrapper around we.get_wave_snapshot.

    Not cached so intraday data is always fresh when the page reloads.
    """
    return we.get_wave_snapshot(wave_name)


@st.cache_data(show_spinner=True)
def load_all_snapshots(waves: list[str]) -> pd.DataFrame:
    """
    Build an overview DataFrame with metrics for all Waves.
    """
    rows = []
    for w in waves:
        try:
            snap = we.get_wave_snapshot(w)
            m = snap["metrics"]
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": snap["benchmark"],
                    "Intraday Return": m.get("intraday_return", 0.0),
                    "30D Return": m.get("ret_30d", 0.0),
                    "30D Alpha": m.get("alpha_30d", 0.0),
                    "60D Return": m.get("ret_60d", 0.0),
                    "60D Alpha": m.get("alpha_60d", 0.0),
                    "1Y Return": m.get("ret_1y", 0.0),
                    "1Y Alpha": m.get("alpha_1y", 0.0),
                    "SI Return": m.get("ret_si", 0.0),
                    "SI Alpha": m.get("alpha_si", 0.0),
                }
            )
        except Exception:
            # Non-fatal; still show row with zeros so the grid is complete
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": "",
                    "Intraday Return": 0.0,
                    "30D Return": 0.0,
                    "30D Alpha": 0.0,
                    "60D Return": 0.0,
                    "60D Alpha": 0.0,
                    "1Y Return": 0.0,
                    "1Y Alpha": 0.0,
                    "SI Return": 0.0,
                    "SI Alpha": 0.0,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------

def format_pct(x: float) -> str:
    try:
        return f"{x * 100:,.2f}%"
    except Exception:
        return "-"


def top_holdings_table(positions: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
    """Prepare Top 10 holdings table with Google Finance links."""
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
            Restored baseline — VIX-aware, SmartSafe 2.0 sweeps, blended benchmarks, multi-horizon alpha.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(waves: list) -> str:
    """
    Sidebar:
    - Wave selector (for detail tab)
    - Mode selector (visual only)
    """
    st.sidebar.markdown("### Waves Intelligence™")
    st.sidebar.write("Select a Wave to inspect in the **Wave Detail** tab:")

    if not waves:
        st.sidebar.error("No Waves found in wave_weights.csv.")
        return ""

    selected_wave = st.sidebar.selectbox("Wave", waves, index=0)
    st.sidebar.markdown("---")
    st.sidebar.caption("Modes shown are cosmetic only in this restored baseline.")
    _ = st.sidebar.radio(
        "Mode (visual only)",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("SmartSafe 2.0 sweep is wired but conservative and non-invasive.")
    return selected_wave


def render_metrics(snapshot: dict):
    metrics = snapshot["metrics"]
    benchmark = snapshot["benchmark"]

    # Row 1: Short-horizon
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Intraday Return", format_pct(metrics.get("intraday_return", 0.0)))
    with col2:
        st.metric("30-Day Return", format_pct(metrics.get("ret_30d", 0.0)))
    with col3:
        st.metric("60-Day Return", format_pct(metrics.get("ret_60d", 0.0)))
    with col4:
        st.metric("30-Day Alpha vs " + str(benchmark), format_pct(metrics.get("alpha_30d", 0.0)))

    # Row 2: 60D alpha + 1Y
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("60-Day Alpha vs " + str(benchmark), format_pct(metrics.get("alpha_60d", 0.0)))
    with col6:
        st.metric("1-Year Return", format_pct(metrics.get("ret_1y", 0.0)))
    with col7:
        st.metric("1-Year Alpha vs " + str(benchmark), format_pct(metrics.get("alpha_1y", 0.0)))
    with col8:
        st.empty()

    # Row 3: Since inception
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        st.metric("Since Inception Return", format_pct(metrics.get("ret_si", 0.0)))
    with col10:
        st.metric("Since Inception Alpha vs " + str(benchmark), format_pct(metrics.get("alpha_si", 0.0)))
    with col11:
        vix_level = metrics.get("vix_level", None)
        if vix_level is not None:
            st.metric("VIX Level", f"{vix_level:,.2f}")
    with col12:
        sweep_frac = metrics.get("smartsafe_sweep_fraction", 0.0)
        if sweep_frac and sweep_frac > 0:
            st.metric("SmartSafe Sweep %", format_pct(sweep_frac))


def render_top_holdings(snapshot: dict):
    positions = snapshot["positions"]
    holdings_df = top_holdings_table(positions, max_rows=10)

    st.subheader("Top 10 Holdings")
    if holdings_df.empty:
        st.info("No holdings data available for this Wave.")
        return

    st.markdown(
        "Click **Quote** to open the Google Finance page for each holding.",
        unsafe_allow_html=True,
    )

    st.dataframe(
        holdings_df,
        use_container_width=True,
        hide_index=True,
    )


def render_positions_raw(snapshot: dict):
    st.subheader("Underlying Positions (Raw)")
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
    """
    Apply red/green coloring based on sign of the percent string, e.g. "1.23%".
    """
    if not isinstance(val, str):
        return ""
    if val.startswith("-"):
        return "color: #ff4d4f;"  # red-ish
    if val.endswith("%") and not val.startswith("0.00"):
        return "color: #16c784;"  # green-ish
    return ""


def render_overview_tab(waves: list[str]):
    st.subheader("Overview — All Waves")

    if not waves:
        st.info("No Waves available. Check wave_weights.csv.")
        return

    with st.spinner("Calculating Wave metrics..."):
        df = load_all_snapshots(waves)

    if df.empty:
        st.info("No metrics available yet.")
        return

    st.markdown(
        """
        This grid shows Intraday, 30-Day, 60-Day, 1-Year, and Since-Inception
        performance and alpha for each Wave. Rank is applied after sorting.
        """,
        unsafe_allow_html=True,
    )

    # Sorting controls
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
    col_sort, col_dir, col_spacer = st.columns([2, 1.5, 1])
    with col_sort:
        sort_col = st.selectbox("Sort by metric", metric_options, index=2)  # default: 30D Alpha
    with col_dir:
        sort_dir = st.radio("Direction", ["High → Low", "Low → High"], index=0)
    with col_spacer:
        st.write("")

    ascending = sort_dir == "Low → High"
    df_sorted = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    # Add Rank column after sorting
    df_sorted.insert(0, "Rank", range(1, len(df_sorted) + 1))

    # Create formatted copy for display
    df_display = df_sorted.copy()
    for col in metric_options:
        df_display[col] = df_display[col].apply(format_pct)

    # Styler to color-code positives/negatives
    styler = df_display.style.applymap(
        _color_metric,
        subset=metric_options,
    )

    st.dataframe(
        styler,
        use_container_width=True,
        hide_index=True,
    )

    # CSV export (raw numeric values, including Rank and all metrics)
    csv_data = df_sorted.to_csv(index=False)
    st.download_button(
        label="Download Overview as CSV",
        data=csv_data,
        file_name="waves_overview_leaderboard.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------
# Wave Detail tab
# ---------------------------------------------------------------------

def render_smartsafe_panel():
    st.subheader("SmartSafe 2.0 Status")
    st.write(
        """
        SmartSafe 2.0 sweep hooks are active but conservative and non-invasive
        in this restored baseline.

        - No SmartSafe 3.0 logic is running.
        - Sweeps are driven by the VIX ladder and applied to highest-vol holdings first.
        """
    )


def render_wave_detail_tab(selected_wave: str):
    if not selected_wave:
        st.info("Select a Wave in the sidebar to view details.")
        return

    with st.spinner(f"Loading Wave: {selected_wave}"):
        snapshot = load_wave_snapshot(selected_wave)

    st.markdown(f"## {selected_wave}")
    st.caption(f"Benchmark: {snapshot['benchmark']}")

    render_metrics(snapshot)

    st.markdown("---")
    col_left, col_right = st.columns([2, 1])
    with col_left:
        render_top_holdings(snapshot)
    with col_right:
        render_smartsafe_panel()

    st.markdown("---")
    render_positions_raw(snapshot)

    st.markdown(
        """
        <div style="font-size:0.75rem; opacity:0.6; margin-top:1rem;">
        WAVES Intelligence™ — VIX-aware, SmartSafe 2.0, blended benchmarks, multi-horizon alpha.
        For internal / demo use only.
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
    selected_wave = render_sidebar(waves)

    tab_overview, tab_detail = st.tabs(["Overview", "Wave Detail"])

    with tab_overview:
        render_overview_tab(waves)

    with tab_detail:
        render_wave_detail_tab(selected_wave)


if __name__ == "__main__":
    main()