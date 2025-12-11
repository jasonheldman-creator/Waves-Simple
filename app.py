"""
app.py — WAVES Intelligence™ Simple Console (Engine-aligned)

UI for the lightweight waves_engine.py module.

- Discovers Waves from wave_weights.csv via get_available_waves()
- Shows portfolio-level overview (NAV + 365D return + status)
- Shows Wave-level detail:
    • positions snapshot
    • performance history chart + table

Engine contracts (from waves_engine.py):
----------------------------------------
get_available_waves() -> List[str]

get_wave_snapshot(wave_name: str, mode: str = "standard", as_of: date | None = None, **kwargs)
    Returns:
        {
            "wave": <name>,
            "mode": <mode>,
            "as_of": <date>,
            "positions": DataFrame[
                ticker, weight, price, dollar_weight
            ]
        }

get_wave_history(wave_name: str, mode: str = "standard",
                 lookback_days: int = 365, **kwargs) -> DataFrame

    Returns a DataFrame with at least:
        date
        wave_nav
        wave_return
        cum_wave_return
        bench_nav
        bench_return
        cum_bench_return
        daily_alpha
        cum_alpha
"""

from __future__ import annotations

import math
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

from waves_engine import (
    get_available_waves,
    get_wave_snapshot,
    get_wave_history,
)

# ---------------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached wrappers around engine functions
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_wave_list() -> List[str]:
    return get_available_waves()


@st.cache_data(show_spinner=False)
def _cached_snapshot(wave_name: str, mode: str = "standard") -> Dict[str, Any]:
    # In this simple engine, mode is ignored, but we pass it for future-proofing
    return get_wave_snapshot(wave_name=wave_name, mode=mode)


@st.cache_data(show_spinner=False)
def _cached_history(
    wave_name: str, lookback_days: int, mode: str = "standard"
) -> pd.DataFrame:
    return get_wave_history(
        wave_name=wave_name,
        mode=mode,
        lookback_days=lookback_days,
    )


# ---------------------------------------------------------------------------
# Helper formatting functions
# ---------------------------------------------------------------------------


def _fmt_nav(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        return f"{x:0.3f}"
    except Exception:
        return "—"


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        return f"{x*100:0.1f}%"  # x is in decimal form (0.1234 = 12.34%)
    except Exception:
        return "—"


def _fmt_pct_direct(x: float | None) -> str:
    """x is already a percent value (e.g., 12.34)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    try:
        return f"{x:0.1f}%"
    except Exception:
        return "—"


# ---------------------------------------------------------------------------
# Top-level layout
# ---------------------------------------------------------------------------

st.markdown(
    """
# WAVES Intelligence™ Console

Live engine: `wave_weights.csv` + yfinance-driven history
"""
)

tab_overview, tab_detail = st.tabs(["📊 Overview", "📈 Wave Detail"])

waves = _cached_wave_list()

# ---------------------------------------------------------------------------
# TAB 1 — Portfolio-level Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Portfolio-Level Overview")

    rows = []
    for wave in waves:
        nav_last = None
        ret_365 = None
        status = "ok"

        try:
            hist = _cached_history(wave, lookback_days=365)
            if hist is None or hist.empty:
                status = "no data"
            else:
                # Expect decimal returns; if already % adjust accordingly
                nav_last = float(hist["wave_nav"].iloc[-1])

                # Prefer cum_wave_return if present
                if "cum_wave_return" in hist.columns:
                    cum_ret = float(hist["cum_wave_return"].iloc[-1])
                    ret_365 = cum_ret * 100.0
                elif "wave_return" in hist.columns:
                    # Fallback: convert daily returns to cumulative
                    cum_ret = (1.0 + hist["wave_return"]).prod() - 1.0
                    ret_365 = cum_ret * 100.0
                else:
                    status = "missing return cols"

        except Exception as e:
            status = f"error: {e.__class__.__name__}"

        rows.append(
            {
                "Wave": wave,
                "NAV (last)": nav_last,
                "365D Return %": ret_365,
                "Status": status,
            }
        )

    overview_df = pd.DataFrame(rows)

    # Nicely formatted table
    if not overview_df.empty:
        display_df = overview_df.copy()
        display_df["NAV (last)"] = display_df["NAV (last)"].map(_fmt_nav)
        display_df["365D Return %"] = display_df["365D Return %"].map(
            _fmt_pct_direct
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No Waves discovered. Check `wave_weights.csv` in the repo root.")

    st.caption(
        "NAV is normalized to 1.0 at the start of the lookback window. "
        "Returns are cumulative over 365 calendar days using daily NAV changes."
    )

# ---------------------------------------------------------------------------
# TAB 2 — Wave detail
# ---------------------------------------------------------------------------
with tab_detail:
    st.subheader("Wave Detail")

    if not waves:
        st.warning("No Waves available. Check `wave_weights.csv`.")
    else:
        col_sel1, col_sel2 = st.columns([2, 1])

        with col_sel1:
            selected_wave = st.selectbox("Select Wave", waves, index=0)
        with col_sel2:
            mode = st.selectbox(
                "Mode (engine currently treats these the same)",
                options=["standard", "alpha_minus_beta", "private_logic"],
                index=0,
            )

        # History window
        window_choice = st.radio(
            "History window",
            options=[30, 90, 365],
            index=2,
            format_func=lambda d: f"{d} days",
            horizontal=True,
        )

        # -------------------------------------------------------------------
        # Positions Snapshot
        # -------------------------------------------------------------------
        st.markdown("### Positions Snapshot")

        try:
            snap = _cached_snapshot(selected_wave, mode=mode)
            positions = snap.get("positions", None)

            if positions is None or positions.empty:
                st.warning("No positions available for this Wave.")
            else:
                # Ensure clean column order / casing
                cols = []
                for c in ["ticker", "weight", "price", "dollar_weight"]:
                    if c in positions.columns:
                        cols.append(c)

                pos_df = positions[cols].copy()
                # Formatting
                if "weight" in pos_df.columns:
                    pos_df["weight"] = pos_df["weight"].map(_fmt_nav)
                if "price" in pos_df.columns:
                    pos_df["price"] = pos_df["price"].map(
                        lambda x: f"${x:0.2f}" if pd.notna(x) else "—"
                    )
                if "dollar_weight" in pos_df.columns:
                    pos_df["dollar_weight"] = pos_df["dollar_weight"].map(_fmt_nav)

                st.dataframe(
                    pos_df,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(f"Error loading positions for '{selected_wave}': {e}")

        # -------------------------------------------------------------------
        # Performance history
        # -------------------------------------------------------------------
        st.markdown("### Performance History")

        try:
            hist = _cached_history(
                selected_wave,
                lookback_days=int(window_choice),
                mode=mode,
            )

            if hist is None or hist.empty:
                st.warning(
                    f"No history available for '{selected_wave}' "
                    f"over the last {window_choice} days."
                )
            else:
                # Ensure we only keep the expected columns if present
                cols_for_table = []
                for c in [
                    "date",
                    "wave_nav",
                    "wave_return",
                    "cum_wave_return",
                    "bench_nav",
                    "bench_return",
                    "cum_bench_return",
                    "daily_alpha",
                    "cum_alpha",
                ]:
                    if c in hist.columns:
                        cols_for_table.append(c)

                hist_view = hist[cols_for_table].copy()

                # Basic formatting
                if "date" in hist_view.columns:
                    hist_view["date"] = pd.to_datetime(
                        hist_view["date"]
                    ).dt.date

                for col in [
                    "wave_return",
                    "cum_wave_return",
                    "bench_return",
                    "cum_bench_return",
                    "daily_alpha",
                    "cum_alpha",
                ]:
                    if col in hist_view.columns:
                        hist_view[col] = hist_view[col].map(_fmt_pct)

                # Show chart on NAV if present
                if {"date", "wave_nav"}.issubset(hist.columns):
                    chart_df = hist[["date", "wave_nav"]].copy()
                    chart_df["date"] = pd.to_datetime(chart_df["date"])
                    chart_df = chart_df.set_index("date")
                    st.line_chart(chart_df)

                st.dataframe(
                    hist_view,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(
                f"Error loading history for '{selected_wave}': {e}"
            )
            st.caption(
                "If this persists for a specific Wave, double-check tickers and "
                "weights in `wave_weights.csv` (especially for delisted symbols "
                "or exotic tickers)."
            )