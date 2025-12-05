"""
app.py

WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE
Streamlit UI that sits on top of waves_equity_universe_v2.py.
"""

from __future__ import annotations

import textwrap

import pandas as pd
import streamlit as st

from waves_equity_universe_v2 import (
    WAVES_CONFIG,
    get_wave_config,
    load_wave_holdings_and_stats,
)


# ---------------------------------------------------------------------------
#  Page config / theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence – Portfolio Wave Console",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
#  Sidebar – wave selector & controls
# ---------------------------------------------------------------------------

st.sidebar.title("🌊 WAVES Console")

# Build selectbox options like: "SPX – S&P 500 Core Equity Wave"
options = [f"{cfg.wave_id} – {cfg.name}" for cfg in WAVES_CONFIG]
default_index = 0  # SPX at top

selected_label = st.sidebar.selectbox("Select Wave", options, index=default_index)
selected_wave_id = selected_label.split("–")[0].strip()

cfg = get_wave_config(selected_wave_id)

st.sidebar.markdown("---")
st.sidebar.caption("Mode: **Standard**  ·  Type: **AI-Managed Wave**")
st.sidebar.caption("Console is read-only – no live trades are placed.")


# ---------------------------------------------------------------------------
#  Main header
# ---------------------------------------------------------------------------

st.markdown(
    "<h2 style='color:#9AE6FF; margin-bottom:0;'>"
    "WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE"
    "</h2>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<h1 style='color:#4FD1FF; margin-top:0.25rem;'>"
    f"{cfg.name} (LIVE Demo)"
    "</h1>",
    unsafe_allow_html=True,
)

subheader = (
    f"Mode: **Standard**  ·  Benchmark: **{cfg.benchmark}**  ·  "
    f"Style: **{cfg.style}**  ·  Type: **{cfg.wave_type}**"
)
st.markdown(subheader)

st.caption(
    "Below is a live rendering of the selected Wave’s holdings, concentration, "
    "and basic risk profile – using the same data Franklin uses, but managed by "
    "a small AI-augmented team instead of hundreds of humans."
)


# ---------------------------------------------------------------------------
#  Load holdings / stats
# ---------------------------------------------------------------------------

try:
    holdings, stats = load_wave_holdings_and_stats(selected_wave_id)
    load_error = None
except Exception as exc:  # noqa: BLE001
    holdings = pd.DataFrame()
    stats = {"num_holdings": 0, "largest_weight": None, "top10_weight": None}
    load_error = str(exc)

if load_error:
    st.error(
        "Wave engine import failed – see details below.\n\n"
        f"```text\n{load_error}\n```"
    )
    st.stop()

if holdings.empty:
    st.warning(
        f"Holdings for {cfg.wave_id} – {cfg.name} are currently empty. "
        "This can happen if the Wave filters select no names from the master universe."
    )
    st.stop()


# ---------------------------------------------------------------------------
#  Top section – Top 10 + Snapshot
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([2.2, 1.1])

with col_left:
    st.subheader("Top 10 holdings")

    top10 = holdings.sort_values("weight", ascending=False).head(10).copy()
    top10["Weight %"] = (top10["weight"] * 100).round(2)

    display_cols = ["ticker", "name", "Weight %"]
    display_df = top10[display_cols].rename(
        columns={"ticker": "Ticker", "name": "Name"}
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

with col_right:
    st.subheader("Wave snapshot")

    num_holdings = stats.get("num_holdings") or 0
    largest_weight = stats.get("largest_weight") or 0.0
    top10_weight = stats.get("top10_weight") or 0.0

    st.markdown(
        f"""
        **{cfg.wave_id} – {cfg.name}**

        • **Total holdings:** {num_holdings:,}  
        • **Largest position:** {largest_weight*100:0.2f}%  
        • **Top-10 weight:** {top10_weight*100:0.2f}%  
        • **Equity vs Cash:** 100% / 0% (demo)  
        """
    )


# ---------------------------------------------------------------------------
#  Second row – charts
# ---------------------------------------------------------------------------

c1, c2, c3 = st.columns(3)

# Ensure we have numeric weights
weights = pd.to_numeric(holdings["weight"], errors="coerce").fillna(0)
sorted_holdings = holdings.assign(weight=weights).sort_values("weight", ascending=False)

with c1:
    st.markdown("#### Top-10 profile – Wave weight distribution")

    chart_df = sorted_holdings.head(10)[["ticker", "weight"]].copy()
    chart_df["Weight %"] = chart_df["weight"] * 100

    st.bar_chart(
        chart_df.set_index("ticker")["Weight %"],
        use_container_width=True,
    )

with c2:
    st.markdown("#### Sector allocation")

    if "sector" in holdings.columns:
        sector_series = (
            holdings["sector"]
            .astype(str)
            .replace({"": "Unclassified", "nan": "Unclassified"})
        )
        sector_weights = (
            pd.DataFrame({"sector": sector_series, "weight": weights})
            .groupby("sector")["weight"]
            .sum()
            .sort_values(ascending=False)
        )

        if not sector_weights.empty:
            sector_chart = (sector_weights * 100).to_frame("Weight %")
            st.bar_chart(sector_chart, use_container_width=True)
        else:
            st.info("No sector data populated yet in the master sheet.")
    else:
        st.info("No sector column found in the master sheet – add one to unlock this view.")

with c3:
    st.markdown("#### Holding rank curve")

    rank_df = sorted_holdings.reset_index(drop=True)
    rank_df["Rank"] = rank_df.index + 1
    rank_df["Weight %"] = rank_df["weight"] * 100

    st.line_chart(
        rank_df.set_index("Rank")["Weight %"],
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
#  Bottom – Mode overview narrative
# ---------------------------------------------------------------------------

st.markdown("---")

overview = textwrap.dedent(
    f"""
    **Mode overview**

    • **Standard mode** shows how the {cfg.name} Wave would be managed in production:  
      benchmark-aware, risk-controlled, and designed to express the Wave’s mandate using
      the same holdings data Franklin relies on today.

    • The **AI-Managed Wave** design lets a small, augmented team oversee thousands of
      securities and dozens of Waves simultaneously – instead of hundreds of siloed
      analysts and PMs – while still operating inside institutional guardrails.

    • Each Wave is carved from a single **Master_Stock_Sheet** universe, which can be
      swapped to Franklin’s internal data feeds without changing the console.
    """
).strip()

st.markdown(overview)