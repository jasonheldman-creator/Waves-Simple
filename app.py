# app.py — WAVES Intelligence™ Institutional Console
# Mobile-friendly Streamlit UI
#
# Sections:
#   • Portfolio-Level Overview (All Waves)
#   • Multi-Window Alpha Capture (All Waves)
#   • Risk & WaveScore Ingredients (All Waves, over selected window)
#   • Benchmark ETF Mix
#   • Wave Detail (NAV chart, performance vs benchmark, mode comparison, top-10 holdings)

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from waves_engine import (
    USE_FULL_WAVE_HISTORY,
    get_all_waves,
    get_modes,
    compute_history_nav,
    get_benchmark_mix_table,
    get_wave_holdings,
)

# ------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)

# ------------------------------------------------------------
# Helper functions (cached history per Wave/mode)
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_history_cached(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    """
    Cached wrapper around compute_history_nav from waves_engine.py.

    Returns a DataFrame indexed by Date with columns:
        ['wave_nav', 'bm_nav', 'wave_ret', 'bm_ret']
    """
    return compute_history_nav(wave_name, mode, days)


def compute_window_stats(hist: pd.DataFrame, window: int) -> tuple[float, float]:
    """
    Compute cumulative total returns for Wave and Benchmark over the last `window` days.

    Returns:
        (wave_return, benchmark_return) as decimal returns (e.g. 0.12 = +12%)
        If insufficient data, returns (nan, nan).
    """
    if hist is None or hist.empty:
        return (np.nan, np.nan)

    if len(hist) < max(2, window):
        return (np.nan, np.nan)

    window_hist = hist.iloc[-window:]
    w_ret = (1.0 + window_hist["wave_ret"]).prod() - 1.0
    b_ret = (1.0 + window_hist["bm_ret"]).prod() - 1.0
    return (w_ret, b_ret)


def format_pct(x: float, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def compute_risk_metrics(hist: pd.DataFrame) -> dict:
    """
    Compute risk stats from daily returns and NAV series over the
    *currently selected window* (whatever history length we pass in).

    Assumes hist has columns: ['wave_nav', 'bm_nav', 'wave_ret', 'bm_ret'].

    Metrics:
        • Wave Vol (annualized)
        • Benchmark Vol (annualized)
        • Max Drawdown (Wave)
        • Max Drawdown (Benchmark)
        • Tracking Error (annualized)
        • Information Ratio (annualized excess / TE)
    """
    out = {
        "wave_vol": np.nan,
        "bm_vol": np.nan,
        "wave_maxdd": np.nan,
        "bm_maxdd": np.nan,
        "tracking_error": np.nan,
        "information_ratio": np.nan,
    }

    if hist is None or hist.empty or len(hist) < 10:
        return out

    wave_ret = hist["wave_ret"].dropna()
    bm_ret = hist["bm_ret"].dropna()
    common_index = wave_ret.index.intersection(bm_ret.index)
    if len(common_index) < 10:
        return out

    wave_ret = wave_ret.loc[common_index]
    bm_ret = bm_ret.loc[common_index]

    # Annualized vol
    wave_vol = wave_ret.std() * np.sqrt(252)
    bm_vol = bm_ret.std() * np.sqrt(252)

    # Max drawdown helper
    def max_drawdown(nav: pd.Series) -> float:
        if nav is None or nav.empty:
            return np.nan
        running_max = nav.cummax()
        drawdowns = (nav / running_max) - 1.0
        return drawdowns.min()  # will be ≤ 0

    wave_nav = hist["wave_nav"].loc[common_index]
    bm_nav = hist["bm_nav"].loc[common_index]

    wave_maxdd = max_drawdown(wave_nav)
    bm_maxdd = max_drawdown(bm_nav)

    # Tracking error & Information Ratio
    excess = wave_ret - bm_ret
    te = excess.std() * np.sqrt(252)

    if te > 0:
        ann_excess_return = excess.mean() * 252.0
        ir = ann_excess_return / te
    else:
        ir = np.nan

    out["wave_vol"] = wave_vol
    out["bm_vol"] = bm_vol
    out["wave_maxdd"] = wave_maxdd
    out["bm_maxdd"] = bm_maxdd
    out["tracking_error"] = te
    out["information_ratio"] = ir

    return out


# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------

st.sidebar.title("WAVES Intelligence™")
st.sidebar.caption("Vector OS • Institutional Console")

waves = get_all_waves()
modes = get_modes()
default_mode = "Standard" if "Standard" in modes else modes[0]

selected_mode = st.sidebar.radio(
    "Select Mode",
    options=modes,
    index=modes.index(default_mode) if default_mode in modes else 0,
    help="Applies to Wave returns; benchmarks are always Standard.",
)

# History window selector
history_options = {
    "1Y (365D)": 365,
    "3Y (756D)": 756,
    "5Y (1260D)": 1260,
}
history_label = st.sidebar.selectbox(
    "History Window",
    options=list(history_options.keys()),
    index=0,
)
lookback_days = history_options[history_label]


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("WAVES Intelligence™ Console")
st.caption(
    f"Live Alpha Capture • Composite Benchmarks • Mode-aware Dynamic NAV • "
    f"Risk & WaveScore Ingredients ({history_label})"
)

if USE_FULL_WAVE_HISTORY:
    st.warning(
        "USE_FULL_WAVE_HISTORY is True in waves_engine.py — this UI is configured "
        "for rolling window analytics based on the selected History Window.",
        icon="⚠️",
    )

# ------------------------------------------------------------
# SECTION 1 — Portfolio-Level Overview (All Waves)
# ------------------------------------------------------------

st.subheader("Portfolio-Level Overview (All Waves)")

overview_rows = []

for wave in waves:
    hist = get_history_cached(wave, selected_mode, days=lookback_days)
    # 30D / 365D returns & alpha vs benchmark
    w30, b30 = compute_window_stats(hist, 30)
    w365, b365 = compute_window_stats(hist, 365)

    overview_rows.append(
        {
            "Wave": wave,
            "30D Return": w30,
            "30D Alpha": (w30 - b30) if not (math.isnan(w30) or math.isnan(b30)) else np.nan,
            "365D Return": w365,
            "365D Alpha": (w365 - b365) if not (math.isnan(w365) or math.isnan(b365)) else np.nan,
        }
    )

overview_df = pd.DataFrame(overview_rows)

if not overview_df.empty:
    display_df = overview_df.copy()
    for col in ["30D Return", "30D Alpha", "365D Return", "365D Alpha"]:
        display_df[col] = display_df[col].apply(lambda x: format_pct(x, 2))
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No overview data available yet for the configured Waves.", icon="ℹ️")


# ------------------------------------------------------------
# SECTION 2 — Multi-Window Alpha Capture (All Waves)
# ------------------------------------------------------------

st.subheader("Multi-Window Alpha Capture (All Waves)")

alpha_rows = []
windows = [1, 30, 60, 365]

for wave in waves:
    hist = get_history_cached(wave, selected_mode, days=lookback_days)
    row = {"Wave": wave}
    for w in windows:
        w_ret, b_ret = compute_window_stats(hist, w)
        row[f"{w}D Return"] = w_ret
        row[f"{w}D Alpha"] = (
            (w_ret - b_ret) if not (math.isnan(w_ret) or math.isnan(b_ret)) else np.nan
        )
    alpha_rows.append(row)

alpha_df = pd.DataFrame(alpha_rows)

if not alpha_df.empty:
    display_alpha = alpha_df.copy()
    for w in windows:
        display_alpha[f"{w}D Return"] = display_alpha[f"{w}D Return"].apply(
            lambda x: format_pct(x, 2)
        )
        display_alpha[f"{w}D Alpha"] = display_alpha[f"{w}D Alpha"].apply(
            lambda x: format_pct(x, 2)
        )
    st.dataframe(display_alpha, use_container_width=True, hide_index=True)
else:
    st.info("No alpha capture data available yet for the configured Waves.", icon="ℹ️")


# ------------------------------------------------------------
# SECTION 3 — Risk & WaveScore Ingredients (All Waves)
# ------------------------------------------------------------

st.subheader(f"Risk & WaveScore Ingredients (All Waves, {history_label})")

risk_rows = []

for wave in waves:
    hist = get_history_cached(wave, selected_mode, days=lookback_days)
    metrics = compute_risk_metrics(hist)

    risk_rows.append(
        {
            "Wave": wave,
            "Wave Vol (Ann.)": metrics["wave_vol"],
            "Benchmark Vol (Ann.)": metrics["bm_vol"],
            "Max Drawdown (Wave)": metrics["wave_maxdd"],
            "Max Drawdown (Benchmark)": metrics["bm_maxdd"],
            "Tracking Error (Ann.)": metrics["tracking_error"],
            "Information Ratio": metrics["information_ratio"],
        }
    )

risk_df = pd.DataFrame(risk_rows)

if not risk_df.empty:
    display_risk = risk_df.copy()

    for col in [
        "Wave Vol (Ann.)",
        "Benchmark Vol (Ann.)",
        "Max Drawdown (Wave)",
        "Max Drawdown (Benchmark)",
        "Tracking Error (Ann.)",
    ]:
        display_risk[col] = display_risk[col].apply(lambda x: format_pct(x, 2))

    display_risk["Information Ratio"] = display_risk["Information Ratio"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"
    )

    st.dataframe(display_risk, use_container_width=True, hide_index=True)
else:
    st.info("No risk analytics available yet for the configured Waves.", icon="ℹ️")


# ------------------------------------------------------------
# SECTION 4 — Benchmark ETF Mix (Composite Benchmarks)
# ------------------------------------------------------------

st.subheader("Benchmark ETF Mix (Composite Benchmarks)")

bmix_df = get_benchmark_mix_table()
if bmix_df is not None and not bmix_df.empty:
    st.dataframe(bmix_df, use_container_width=True, hide_index=True)
else:
    st.info("Benchmark mix table not available.", icon="ℹ️")


# ------------------------------------------------------------
# SECTION 5 — Wave Detail View
# ------------------------------------------------------------

st.subheader("Wave Detail View")

col_sel, col_spacer = st.columns([2, 1])
with col_sel:
    selected_wave = st.selectbox("Select Wave", options=waves, index=0)

detail_hist = get_history_cached(selected_wave, selected_mode, days=lookback_days)

if detail_hist is None or detail_hist.empty:
    st.warning(f"No history available for {selected_wave}.", icon="⚠️")
else:
    # NAV chart (Wave vs Benchmark)
    st.markdown(f"**NAV (Wave vs Benchmark, {history_label})**")

    nav_df = detail_hist[["wave_nav", "bm_nav"]].copy()
    nav_df.columns = ["Wave NAV", "Benchmark NAV"]
    st.line_chart(nav_df, use_container_width=True)

    # Performance vs benchmark (30D / 365D)
    st.markdown("**Performance vs Benchmark (30D / 365D)**")

    w30, b30 = compute_window_stats(detail_hist, 30)
    w365, b365 = compute_window_stats(detail_hist, 365)

    perf_df = pd.DataFrame(
        {
            "Window": ["30D", "365D"],
            "Wave Return": [format_pct(w30), format_pct(w365)],
            "Benchmark Return": [format_pct(b30), format_pct(b365)],
            "Alpha": [
                format_pct((w30 - b30) if not (math.isnan(w30) or math.isnan(b30)) else np.nan),
                format_pct(
                    (w365 - b365) if not (math.isnan(w365) or math.isnan(b365)) else np.nan
                ),
            ],
        }
    )

    st.table(perf_df)

    # Mode comparison (Standard vs AMB vs Private Logic) at 365D
    st.markdown("**Mode Comparison (365D)**")

    mode_rows = []
    for m in modes:
        h = get_history_cached(selected_wave, m, days=lookback_days)
        if h is None or h.empty:
            mode_rows.append(
                {"Mode": m, "365D Return": np.nan, "365D Alpha": np.nan}
            )
            continue
        mw, mb = compute_window_stats(h, 365)
        mode_rows.append(
            {
                "Mode": m,
                "365D Return": mw,
                "365D Alpha": (mw - mb) if not (math.isnan(mw) or math.isnan(mb)) else np.nan,
            }
        )

    mode_df = pd.DataFrame(mode_rows)
    if not mode_df.empty:
        display_mode = mode_df.copy()
        display_mode["365D Return"] = display_mode["365D Return"].apply(
            lambda x: format_pct(x, 2)
        )
        display_mode["365D Alpha"] = display_mode["365D Alpha"].apply(
            lambda x: format_pct(x, 2)
        )
        st.table(display_mode)

    # Top-10 holdings with Google Finance links
    st.markdown("**Top-10 Holdings (Click for Google Finance)**")

    holdings_df = get_wave_holdings(selected_wave)
    if holdings_df is None or holdings_df.empty:
        st.info("No holdings data available for this Wave.", icon="ℹ️")
    else:
        def google_link(ticker: str) -> str:
            if not isinstance(ticker, str) or ticker.strip() == "":
                return ""
            base = "https://www.google.com/finance/quote/"
            return f"{base}{ticker}"

        holdings_display = holdings_df.copy()
        if "Ticker" in holdings_display.columns:
            holdings_display["Google Finance"] = holdings_display["Ticker"].apply(
                lambda t: google_link(t)
            )

        st.dataframe(holdings_display, use_container_width=True, hide_index=True)