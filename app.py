"""
app.py — WAVES Intelligence™ Simple Console v2

UI for the "waves-simple" Streamlit app.

Relies on waves_engine.py for:
    - get_available_waves
    - get_wave_snapshot
    - get_wave_history
    - compute_wave_metrics
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from waves_engine import (
    get_available_waves,
    get_wave_snapshot,
    get_wave_history,
    compute_wave_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_nav(x) -> str:
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return "—"


def _fmt_return(x) -> str:
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "—"


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="WAVES Intelligence™ Console",
        layout="wide",
    )

    st.title("WAVES Intelligence™ Console")
    st.caption(
        "Live engine: **wave_weights.csv** + yfinance-driven history "
        "(optionally enhanced by **Full_Wave_History.csv** when available)."
    )

    tab_overview, tab_detail = st.tabs(["📊 Overview", "📈 Wave Detail"])

    # -------------------------------------------------------------------
    # Overview tab
    # -------------------------------------------------------------------
    with tab_overview:
        st.header("Portfolio-Level Overview")

        try:
            metrics = compute_wave_metrics(lookback_days=365)
        except Exception as e:
            st.error(f"Error computing overview metrics: {e}")
            st.stop()

        if not metrics:
            st.warning("No Waves found in wave_weights.csv.")
        else:
            df = pd.DataFrame(metrics)

            display_df = pd.DataFrame(
                {
                    "Wave": df["Wave"],
                    "NAV (last)": df["nav_last"].apply(_fmt_nav),
                    "365D Return": df["cum_return"].apply(_fmt_return),
                    "Status": df["status"],
                }
            )

            st.dataframe(display_df, use_container_width=True)

            st.caption(
                "NAV is normalized to 1.0 at the start of the 365D window. "
                "Returns are cumulative over the selected window."
            )

    # -------------------------------------------------------------------
    # Wave Detail tab
    # -------------------------------------------------------------------
    with tab_detail:
        st.header("Wave Detail")

        try:
            waves = get_available_waves()
        except Exception as e:
            st.error(f"Error loading Waves from wave_weights.csv: {e}")
            return

        if not waves:
            st.warning("No Waves found in wave_weights.csv.")
            return

        col1, col2 = st.columns(2)

        with col1:
            selected_wave = st.selectbox("Select Wave", waves)

        with col2:
            mode = st.selectbox(
                "Mode (engine currently treats these the same)",
                ["standard", "alpha_minus_beta", "private_logic"],
            )

        st.markdown("---")

        st.subheader("Positions Snapshot")

        # History window selector
        window_label = st.radio(
            "History window",
            options=["30 days", "90 days", "365 days"],
            index=2,
            horizontal=True,
        )

        window_map = {
            "30 days": 30,
            "90 days": 90,
            "365 days": 365,
        }
        lookback_days = window_map[window_label]

        # --- Positions ---
        try:
            snap = get_wave_snapshot(selected_wave, mode=mode)
            positions = snap.get("positions", None)
        except Exception as e:
            st.error(f"Error loading positions for '{selected_wave}': {e}")
            positions = None

        if isinstance(positions, pd.DataFrame) and not positions.empty:
            st.dataframe(positions, use_container_width=True)

            # Optional: simple list of Google Finance links
            try:
                tickers = positions["ticker"].dropna().astype(str).tolist()
                if tickers:
                    links = [
                        f"[{t}](https://www.google.com/finance/quote/{t})"
                        for t in tickers
                    ]
                    st.markdown(
                        "Google Finance links: " + " · ".join(links),
                        help="Quick links to quotes for all holdings in this Wave.",
                    )
            except Exception:
                pass
        else:
            st.info(f"No positions data available for Wave '{selected_wave}'.")

        st.markdown("---")
        st.subheader("Performance History")

        # --- History ---
        try:
            hist = get_wave_history(
                selected_wave,
                mode=mode,
                lookback_days=lookback_days,
            )
        except Exception as e:
            st.error(
                f"Error loading history for '{selected_wave}': {e}"
            )
            hist = None

        if isinstance(hist, pd.DataFrame) and not hist.empty:
            # Ensure we have a proper date index
            if "date" in hist.columns:
                try:
                    chart_df = hist.copy()
                    chart_df["date"] = pd.to_datetime(chart_df["date"])
                    chart_df = chart_df.set_index("date")
                    if "wave_nav" in chart_df.columns:
                        st.line_chart(
                            chart_df[["wave_nav"]],
                            use_container_width=True,
                        )
                except Exception:
                    st.warning(
                        "History loaded but could not build chart (date/index issue)."
                    )

            # Show some summary numbers
            last_nav = (
                hist["wave_nav"].iloc[-1]
                if "wave_nav" in hist.columns
                else float("nan")
            )
            last_cum = (
                hist["cum_wave_return"].iloc[-1]
                if "cum_wave_return" in hist.columns
                else float("nan")
            )

            col_nav, col_ret = st.columns(2)
            with col_nav:
                st.metric("NAV (last)", _fmt_nav(last_nav))
            with col_ret:
                st.metric(f"Cumulative Return ({window_label})", _fmt_return(last_cum))

        else:
            st.info(
                f"No history data available for Wave '{selected_wave}' "
                f"over the last {lookback_days} days."
            )


if __name__ == "__main__":
    main()