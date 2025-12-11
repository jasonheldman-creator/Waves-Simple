"""
app.py — WAVES Intelligence™ Console (Stable v1.3)

Front-end Streamlit app for the simple engine in waves_engine.py

• Overview tab       → Portfolio-level table across all Waves
• Wave Detail tab    → Positions snapshot + performance chart

Relies on the public API from waves_engine.py:

    get_available_waves()
    get_wave_snapshot(...)
    get_wave_history(...)

"""

import datetime as dt

import pandas as pd
import streamlit as st

from waves_engine import (
    get_available_waves,
    get_wave_snapshot,
    get_wave_history,
)


# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)


# ---------------------------------------------------------
# Cached wrappers (so we don’t hammer yfinance)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_wave_list():
    return get_available_waves()


@st.cache_data(show_spinner=False)
def _cached_snapshot(wave: str, mode: str):
    # mode currently ignored by engine but kept for API stability
    return get_wave_snapshot(wave_name=wave, mode=mode)


@st.cache_data(show_spinner=False)
def _cached_history(wave: str, mode: str, lookback_days: int):
    # mode currently ignored by engine but kept for API stability
    return get_wave_history(wave_name=wave, mode=mode, lookback_days=lookback_days)


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------
def _format_pct(x):
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "—"


def _build_overview_table(waves, lookback_days: int = 365) -> pd.DataFrame:
    rows = []
    for w in waves:
        try:
            hist = _cached_history(wave=w, mode="standard", lookback_days=lookback_days)
            if hist is None or hist.empty:
                rows.append(
                    {
                        "Wave": w,
                        "NAV (last)": "—",
                        f"{lookback_days}D Return": "—",
                        "Status": "no history",
                    }
                )
                continue

            hist = hist.sort_values("date")
            last_nav = float(hist["wave_nav"].iloc[-1])
            cum_return = float(hist["cum_wave_return"].iloc[-1])

            rows.append(
                {
                    "Wave": w,
                    "NAV (last)": f"{last_nav:.4f}",
                    f"{lookback_days}D Return": _format_pct(cum_return),
                    "Status": "ok",
                }
            )
        except Exception as exc:  # keep UI alive if one Wave fails
            rows.append(
                {
                    "Wave": w,
                    "NAV (last)": "—",
                    f"{lookback_days}D Return": "—",
                    "Status": f"error: {exc.__class__.__name__}",
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values("Wave")
    return df


# ---------------------------------------------------------
# Title & tabs
# ---------------------------------------------------------
st.title("WAVES Intelligence™ Console")
st.caption("Live engine: wave_weights.csv + yfinance-driven history")

tab_overview, tab_detail = st.tabs(["📊 Overview", "📈 Wave Detail"])


# ---------------------------------------------------------
# Overview Tab
# ---------------------------------------------------------
with tab_overview:
    st.subheader("Portfolio-Level Overview")

    waves = _cached_wave_list()
    if not waves:
        st.error("No Waves discovered in wave_weights.csv.")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Lookback window**")
            horizon_label = st.radio(
                "",
                options=["1 year (365 days)", "90 days", "30 days"],
                index=0,
            )

        if horizon_label.startswith("1 year"):
            lookback = 365
        elif horizon_label.startswith("90"):
            lookback = 90
        else:
            lookback = 30

        with st.spinner("Loading portfolio overview…"):
            df_overview = _build_overview_table(waves, lookback_days=lookback)

        st.dataframe(
            df_overview,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "NAV is normalized to 1.0 at the start of the lookback window. "
            "Returns are cumulative over the selected window."
        )


# ---------------------------------------------------------
# Wave Detail Tab
# ---------------------------------------------------------
with tab_detail:
    st.subheader("Wave Detail")

    waves = _cached_wave_list()
    if not waves:
        st.error("No Waves discovered in wave_weights.csv.")
    else:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            wave_name = st.selectbox("Select Wave", options=waves, index=0)

        with col_right:
            mode = st.selectbox(
                "Mode (engine currently treats these the same)",
                options=["standard", "alpha_minus_beta", "private_logic"],
                index=0,
            )

        # History window selector
        st.markdown("**History window**")
        win_label = st.radio(
            "",
            options=["30 days", "90 days", "365 days"],
            index=2,
            horizontal=True,
        )
        if win_label.startswith("30"):
            lookback_days = 30
        elif win_label.startswith("90"):
            lookback_days = 90
        else:
            lookback_days = 365

        # --- Positions Snapshot ---
        st.markdown("### Positions Snapshot")
        try:
            snap = _cached_snapshot(wave=wave_name, mode=mode)
            positions = snap.get("positions", pd.DataFrame())
            as_of = snap.get("as_of", dt.date.today())

            st.caption(f"As of: **{as_of}**  •  Mode: **{mode}**")
            if isinstance(positions, pd.DataFrame) and not positions.empty:
                st.dataframe(
                    positions.reset_index(drop=True),
                    use_container_width=True,
                )
            else:
                st.info("No positions available for this Wave.")
        except Exception as exc:
            st.error(f"Error loading snapshot for '{wave_name}': {exc}")

        # --- Performance History ---
        st.markdown("### Performance History")
        try:
            hist = _cached_history(wave=wave_name, mode=mode, lookback_days=lookback_days)

            if hist is None or hist.empty:
                st.info("No price history available for this Wave / horizon.")
            else:
                # Make sure date is datetime & set as index for chart
                hist = hist.copy()
                hist["date"] = pd.to_datetime(hist["date"])
                hist = hist.sort_values("date").set_index("date")

                # Simple NAV chart
                st.line_chart(
                    hist[["wave_nav"]],
                    use_container_width=True,
                )

                # Quick metrics row
                last_nav = float(hist["wave_nav"].iloc[-1])
                cum_ret = float(hist["cum_wave_return"].iloc[-1])

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("NAV (last)", f"{last_nav:.4f}")
                with col_b:
                    st.metric(f"Cumulative return ({lookback_days}d)", _format_pct(cum_ret))

                with st.expander("Show raw history data"):
                    st.dataframe(hist.reset_index(), use_container_width=True)

        except Exception as exc:
            st.error(f"Error loading history for '{wave_name}': {exc}")
            st.caption(
                "If this persists for a specific Wave, check tickers and weights in "
                "`wave_weights.csv` (especially for delisted symbols)."
            )