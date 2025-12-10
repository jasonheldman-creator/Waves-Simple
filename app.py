"""
app.py — WAVES Intelligence™ Institutional Console
(Restored Baseline + Overview Grid)

Key behaviors:
- Auto-discovers Waves from wave_weights.csv via waves_engine.get_available_waves()
- No SmartSafe 3.0 logic. Only SmartSafe 2.0 hooks via engine (no-op).
- Tabs:
    1) Overview (all Waves grid)
    2) Wave Detail (single-Wave dashboard)
- Metrics:
    Intraday, 30-Day, 60-Day returns and alpha vs benchmark.
- Per-Wave view:
    Top 10 holdings with clickable Google Finance quote links.
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
    try:
        waves = we.get_available_waves()
        return waves
    except Exception as e:
        st.error(f"Error loading wave list: {e}")
        return []


@st.cache_data(show_spinner=False)
def load_wave_snapshot(wave_name: str):
    """
    Snapshot for a single Wave (used in detail tab).
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
                }
            )
        except Exception:
            # Non-fatal; we still show other waves
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": "",
                    "Intraday Return": 0.0,
                    "30D Return": 0.0,
                    "30D Alpha": 0.0,
                    "60D Return": 0.0,
                    "60D Alpha": 0.0,
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
            ]
        )

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------

def format_pct(x: float) -> str:
    try:
        return f"{x * 100:,.2f}%"
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

    return df[["ticker", "Weight", "Last Price", "Google Finance"]].rename(
        columns={"ticker": "Ticker"}
    )


def render_header():
    st.markdown(
        """
        <h1 style="margin-bottom:0;">WAVES Intelligence™ Institutional Console</h1>
        <p style="margin-top:0.25rem; font-size:0.9rem; opacity:0.7;">
            Restored baseline — 12 Waves, SmartSafe 2.0 hooks only, no SmartSafe 3.0 logic.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(waves: list) -> str:
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
    st.sidebar.caption("SmartSafe 2.0 sweep is wired but non-invasive in this version.")
    return selected_wave


def render_metrics(snapshot: dict):
    metrics = snapshot["metrics"]
    benchmark = snapshot["benchmark"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Intraday Return", format_pct(metrics["intraday_return"]))
    with col2:
        st.metric("30-Day Return", format_pct(metrics["ret_30d"]))
    with col3:
        st.metric("60-Day Return", format_pct(metrics["ret_60d"]))
    with col4:
        st.metric("30-Day Alpha vs " + benchmark, format_pct(metrics["alpha_30d"]))

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("60-Day Alpha vs " + benchmark, format_pct(metrics["alpha_60d"]))
    with col6:
        st.empty()
    with col7:
        st.empty()
    with col8:
        st.empty()


def render_top_holdings(snapshot: dict):
    positions = snapshot["positions"]
    holdings_df = top_holdings_table(positions, max_rows=10)

    st.subheader("Top 10 Holdings")
    if holdings_df.empty:
        st.info("No holdings data available for this Wave.")
        return

    st.markdown(
        """
        Click **Quote** to open the Google Finance page for each holding.
        """,
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


def render_overview_tab(waves: list[str]):
    st.subheader("Overview — All Waves")

    if not waves:
        st.info("No Waves found.")
        return

    df = load_all_snapshots(waves)
    if df.empty:
        st.info("No metrics available yet.")
        return

    # Format for display
    df_display = df.copy()
    for col in [
        "Intraday Return",
        "30D Return",
        "30D Alpha",
        "60D Return",
        "60D Alpha",
    ]:
        df_display[col] = df_display[col].apply(format_pct)

    st.markdown(
        """
        This grid shows Intraday, 30-Day, and 60-Day performance and alpha for each Wave.
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
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
        st.subheader("SmartSafe 2.0 Status")
        st.write(
            "SmartSafe 2.0 sweep hooks are active but non-invasive in this restored baseline.\n\n"
            "- No SmartSafe 3.0 logic is running.\n"
            "- Sweeps, if any, will be simple and conservative."
        )

    st.markdown("---")
    render_positions_raw(snapshot)

    st.markdown(
        """
        <div style="font-size:0.75rem; opacity:0.6; margin-top:1rem;">
        WAVES Intelligence™ — Restored baseline console. For internal / demo use only.
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