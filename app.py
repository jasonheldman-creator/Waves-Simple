"""
app.py — WAVES Intelligence™ Simple Console (Engine v2.8+ compatible)

Uses the light-weight waves_engine.py:

    get_available_waves()  -> list of Wave names
    get_wave_snapshot(...) -> latest positions for a Wave
    get_wave_history(...)  -> daily NAV & returns for a Wave

This app avoids any old 'metrics' / 'get_wave_history_v2' references and
only calls the functions that currently exist in waves_engine.py.
"""

import datetime as dt
from typing import Dict, Any

import pandas as pd
import streamlit as st

from waves_engine import (
    get_available_waves,
    get_wave_snapshot,
    get_wave_history,
)

# -------------------------------------------------------------------
# Streamlit config
# -------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)


# -------------------------------------------------------------------
# Cached wrappers (so Streamlit doesn't hammer yfinance)
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_wave_list():
    return get_available_waves()


@st.cache_data(show_spinner=False)
def _cached_snapshot(wave: str, mode: str = "standard") -> Dict[str, Any]:
    return get_wave_snapshot(wave_name=wave, mode=mode)


@st.cache_data(show_spinner=False)
def _cached_history(
    wave: str,
    mode: str = "standard",
    lookback_days: int = 365,
) -> pd.DataFrame:
    return get_wave_history(
        wave_name=wave,
        mode=mode,
        lookback_days=lookback_days,
    )


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _safe_history_row(wave: str) -> Dict[str, Any]:
    """
    Build a single summary row for the Overview table using history.
    If anything fails, we just mark the row as N/A instead of crashing.
    """
    try:
        hist = _cached_history(wave, "standard", 365)
        if hist is None or hist.empty:
            return {
                "Wave": wave,
                "NAV (last)": float("nan"),
                "1Y Return %": float("nan"),
                "1Y Alpha %": float("nan"),
                "Status": "no history",
            }

        last = hist.iloc[-1]

        nav = float(last.get("wave_nav", float("nan")))
        cum_ret = float(last.get("cum_wave_return", float("nan")))
        cum_alpha = float(last.get("cum_alpha", float("nan")))

        return {
            "Wave": wave,
            "NAV (last)": nav,
            "1Y Return %": cum_ret * 100.0 if pd.notna(cum_ret) else float("nan"),
            "1Y Alpha %": cum_alpha * 100.0 if pd.notna(cum_alpha) else float("nan"),
            "Status": "ok",
        }
    except Exception as e:
        return {
            "Wave": wave,
            "NAV (last)": float("nan"),
            "1Y Return %": float("nan"),
            "1Y Alpha %": float("nan"),
            "Status": f"error: {type(e).__name__}",
        }


def _format_pct(x):
    if pd.isna(x):
        return "—"
    return f"{x:0.2f}%"


# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------

st.title("WAVES Intelligence™ Console")
st.caption("Live engine: wave_weights.csv + yfinance-driven history")

tabs = st.tabs(["📊 Overview", "📈 Wave Detail"])

# -------------------------------------------------------------------
# TAB 1 — Overview
# -------------------------------------------------------------------
with tabs[0]:
    st.subheader("Portfolio-Level Overview")

    with st.spinner("Loading wave list…"):
        waves = _cached_wave_list()

    if not waves:
        st.error("No Waves found. Check wave_weights.csv in the repo root.")
    else:
        with st.spinner("Building summary…"):
            rows = [_safe_history_row(w) for w in waves]
            df_overview = pd.DataFrame(rows)

        # Nice formatting
        df_show = df_overview.copy()
        df_show["1Y Return %"] = df_show["1Y Return %"].apply(_format_pct)
        df_show["1Y Alpha %"] = df_show["1Y Alpha %"].apply(_format_pct)
        df_show["NAV (last)"] = df_show["NAV (last)"].map(
            lambda x: "—" if pd.isna(x) else f"{x:0.3f}"
        )

        st.dataframe(
            df_show.set_index("Wave"),
            use_container_width=True,
        )

        st.caption(
            "NAV and returns are built from daily closes via yfinance. "
            "Benchmarks are currently stubbed to 0; alpha = Wave return vs flat benchmark."
        )

# -------------------------------------------------------------------
# TAB 2 — Wave Detail
# -------------------------------------------------------------------
with tabs[1]:
    st.subheader("Wave Detail")

    with st.spinner("Loading wave list…"):
        waves = _cached_wave_list()

    if not waves:
        st.error("No Waves found. Check wave_weights.csv in the repo root.")
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

        # Lookback selection
        lookback_label = st.radio(
            "History window",
            options=["30 days", "90 days", "365 days"],
            index=2,
            horizontal=True,
        )
        lookback_map = {
            "30 days": 30,
            "90 days": 90,
            "365 days": 365,
        }
        lookback_days = lookback_map[lookback_label]

        # Snapshot + positions
        st.markdown("### Positions Snapshot")

        try:
            snap = _cached_snapshot(wave_name, mode)
            positions = snap.get("positions", None)

            if positions is None or positions.empty:
                st.warning("No positions found for this Wave.")
            else:
                # Ensure we have nice columns
                pos = positions.copy()
                # Best-effort float formatting
                for col in ["weight", "price", "dollar_weight"]:
                    if col in pos.columns:
                        pos[col] = pd.to_numeric(pos[col], errors="coerce")

                st.dataframe(pos, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading snapshot for '{wave_name}': {e}")

        # History + charts
        st.markdown("### Performance History")

        try:
            hist = _cached_history(wave_name, mode, lookback_days)

            if hist is None or hist.empty:
                st.warning("No history available for this Wave / window.")
            else:
                hist = hist.copy()
                # Make sure 'date' is datetime for charting
                if "date" in hist.columns:
                    hist["date"] = pd.to_datetime(hist["date"])

                # Key series
                nav_col = "wave_nav" if "wave_nav" in hist.columns else None
                alpha_col = "cum_alpha" if "cum_alpha" in hist.columns else None

                if nav_col:
                    st.line_chart(
                        hist.set_index("date")[nav_col],
                        height=300,
                    )
                    st.caption("NAV history")

                if alpha_col:
                    # Convert to percent for readability
                    alpha_series = hist.set_index("date")[alpha_col] * 100.0
                    st.line_chart(alpha_series, height=300)
                    st.caption("Cumulative alpha (%) vs stub benchmark")

                # Summary stats
                last = hist.iloc[-1]
                nav = float(last.get("wave_nav", float("nan")))
                cum_ret = float(last.get("cum_wave_return", float("nan")))
                cum_alpha = float(last.get("cum_alpha", float("nan")))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "NAV (last)",
                        "—" if pd.isna(nav) else f"{nav:0.3f}",
                    )
                with col2:
                    st.metric(
                        "Total Return",
                        _format_pct(cum_ret * 100.0 if not pd.isna(cum_ret) else float("nan")),
                    )
                with col3:
                    st.metric(
                        "Total Alpha",
                        _format_pct(
                            cum_alpha * 100.0 if not pd.isna(cum_alpha) else float("nan")
                        ),
                    )

        except Exception as e:
            st.error(f"Error loading history for '{wave_name}': {e}")