# app.py — WAVES Intelligence™ Simple Console (History-Driven)

import datetime as dt

import pandas as pd
import streamlit as st

from waves_engine import (
    get_available_waves,
    get_wave_snapshot,
    get_wave_snapshots,  # dict of {wave_name: {"positions": df, "history": df}}
)

# ------------------------------------------------------------
# Streamlit page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("WAVES Intelligence™ Institutional Console")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _safe_window(history: pd.DataFrame, days: int) -> pd.DataFrame | None:
    """Return the last `days` rows (or all if shorter)."""
    if history is None or history.empty:
        return None

    h = history.copy()
    if "date" in h.columns:
        h["date"] = pd.to_datetime(h["date"])
        h = h.sort_values("date")
    else:
        h = h.sort_index()

    if len(h) <= days:
        return h
    return h.iloc[-days:]


def _window_returns(history: pd.DataFrame, days: int) -> tuple[float | None, float | None]:
    """
    Compute wave and benchmark returns over the last `days` rows
    using daily return columns if available.
    """
    if history is None or history.empty:
        return None, None

    h = _safe_window(history, days)
    if h is None or h.empty:
        return None, None

    # Prefer daily return columns if present; otherwise fall back to navs
    wave_ret_col = None
    bench_ret_col = None

    for candidate in ["wave_return", "wave_ret", "ret_wave"]:
        if candidate in h.columns:
            wave_ret_col = candidate
            break

    for candidate in ["bench_return", "bench_ret", "ret_bench", "benchmark_return"]:
        if candidate in h.columns:
            bench_ret_col = candidate
            break

    if wave_ret_col and bench_ret_col:
        # Daily returns assumed in decimal form (0.01 = 1%)
        w = (1.0 + h[wave_ret_col].astype(float)).prod() - 1.0
        b = (1.0 + h[bench_ret_col].astype(float)).prod() - 1.0
        return float(w), float(b)

    # Fallback: use nav columns
    wave_nav_col = None
    bench_nav_col = None

    for candidate in ["wave_nav", "nav_wave", "portfolio_nav"]:
        if candidate in h.columns:
            wave_nav_col = candidate
            break

    for candidate in ["bench_nav", "nav_bench", "benchmark_nav"]:
        if candidate in h.columns:
            bench_nav_col = candidate
            break

    if wave_nav_col and bench_nav_col:
        w_start = float(h[wave_nav_col].iloc[0])
        w_end = float(h[wave_nav_col].iloc[-1])
        b_start = float(h[bench_nav_col].iloc[0])
        b_end = float(h[bench_nav_col].iloc[-1])

        if w_start > 0 and b_start > 0:
            w = w_end / w_start - 1.0
            b = b_end / b_start - 1.0
            return w, b

    return None, None


def compute_metrics(history: pd.DataFrame) -> dict:
    """
    Given a Wave's full history DataFrame with at least wave/benchmark
    returns or navs, compute key windows.
    All outputs are in PERCENT form (e.g., 5.23 = 5.23%).
    """
    metrics = {
        "60D Return %": None,
        "60D Alpha %": None,
        "1Y Return %": None,
        "1Y Alpha %": None,
        "SI Return %": None,
        "SI Alpha %": None,
    }

    if history is None or history.empty:
        return metrics

    # 60-day window
    w60, b60 = _window_returns(history, 60)
    if w60 is not None and b60 is not None:
        metrics["60D Return %"] = round(w60 * 100.0, 2)
        metrics["60D Alpha %"] = round((w60 - b60) * 100.0, 2)

    # 1-year window (~252 trading days)
    w1y, b1y = _window_returns(history, 252)
    if w1y is not None and b1y is not None:
        metrics["1Y Return %"] = round(w1y * 100.0, 2)
        metrics["1Y Alpha %"] = round((w1y - b1y) * 100.0, 2)

    # Since inception: just use entire history
    w_all, b_all = _window_returns(history, len(history))
    if w_all is not None and b_all is not None:
        metrics["SI Return %"] = round(w_all * 100.0, 2)
        metrics["SI Alpha %"] = round((w_all - b_all) * 100.0, 2)

    return metrics


# ------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------
@st.cache_data(ttl=120)
def load_all_snapshots(mode_key: str = "standard") -> dict:
    """
    mode_key is passed through to the engine; the engine can choose to
    use or ignore it.
    Returns: {wave_name: {"positions": df, "history": df}}
    """
    return get_wave_snapshots(mode=mode_key)


@st.cache_data(ttl=120)
def load_wave_names() -> list[str]:
    return sorted(get_available_waves())


# ------------------------------------------------------------
# Mode selection + data load
# ------------------------------------------------------------
mode = st.radio(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    horizontal=True,
)

mode_key = "standard"
if "alpha" in mode.lower():
    mode_key = "alpha_minus_beta"
elif "private" in mode.lower():
    mode_key = "private_logic"

snapshots = load_all_snapshots(mode_key=mode_key)
wave_names = sorted(list(snapshots.keys())) or load_wave_names()

tab_overview, tab_detail = st.tabs(["📊 Overview", "📈 Wave Detail"])

# ------------------------------------------------------------
# OVERVIEW TAB
# ------------------------------------------------------------
with tab_overview:
    st.subheader("Portfolio-Level Overview")

    sort_options = [
        "1Y Alpha %",
        "1Y Return %",
        "60D Alpha %",
        "60D Return %",
        "SI Alpha %",
        "SI Return %",
    ]
    sort_by = st.selectbox("Sort by", sort_options, index=0)

    direction = st.radio(
        "Direction",
        ["High → Low", "Low → High"],
        horizontal=True,
    )
    ascending = direction.startswith("Low")

    rows = []
    for w in wave_names:
        snap = snapshots.get(w, {})
        positions = snap.get("positions")
        history = snap.get("history")

        holdings_count = len(positions) if positions is not None else 0
        total_weight = (
            float(positions["weight"].sum()) if positions is not None and "weight" in positions.columns else None
        )

        metrics = compute_metrics(history)

        row = {
            "Wave": w,
            "Holdings": holdings_count,
            "Total Weight": round(total_weight, 4) if total_weight is not None else None,
        }
        row.update(metrics)
        rows.append(row)

    overview_df = pd.DataFrame(rows)

    # If the sort column is entirely NaN, fallback to Wave name
    if sort_by in overview_df.columns and overview_df[sort_by].notna().any():
        overview_df = overview_df.sort_values(sort_by, ascending=ascending)
    else:
        overview_df = overview_df.sort_values("Wave")

    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True,
    )

    # Download button
    csv_bytes = overview_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Overview as CSV",
        data=csv_bytes,
        file_name=f"waves_overview_{dt.date.today().isoformat()}.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------
# WAVE DETAIL TAB
# ------------------------------------------------------------
with tab_detail:
    st.subheader("Wave Detail")

    if not wave_names:
        st.info("No Waves found. Check `wave_weights.csv` and the engine configuration.")
    else:
        wave_choice = st.selectbox("Select a Wave", wave_names)

        snap = snapshots.get(wave_choice)
        if snap is None:
            st.warning(f"No snapshot found for **{wave_choice}**.")
        else:
            positions = snap.get("positions")
            history = snap.get("history")

            # --- Summary metrics for this wave ---
            metrics = compute_metrics(history)
            cols = st.columns(3)
            cols[0].metric("1Y Return", f"{metrics['1Y Return %'] or 0:.2f}%")
            cols[1].metric("1Y Alpha", f"{metrics['1Y Alpha %'] or 0:.2f}%")
            cols[2].metric("SI Alpha", f"{metrics['SI Alpha %'] or 0:.2f}%")

            st.markdown("### Current Positions")
            if positions is None or positions.empty:
                st.info("No positions data available for this Wave.")
            else:
                st.dataframe(
                    positions,
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### Performance History")
            if history is None or history.empty:
                st.info("No history data available for this Wave.")
            else:
                h = history.copy()
                if "date" in h.columns:
                    h["date"] = pd.to_datetime(h["date"])
                    h = h.sort_values("date")
                    h = h.set_index("date")

                # Try to find nav columns for chart
                nav_cols = []
                for candidate in ["wave_nav", "portfolio_nav"]:
                    if candidate in h.columns:
                        nav_cols.append(candidate)
                        break
                for candidate in ["bench_nav", "benchmark_nav"]:
                    if candidate in h.columns:
                        nav_cols.append(candidate)
                        break

                if nav_cols:
                    st.line_chart(h[nav_cols])
                else:
                    st.line_chart(h)