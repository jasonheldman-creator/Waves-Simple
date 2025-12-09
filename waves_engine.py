"""
app.py

WAVES Intelligence™ Institutional Console — Vector 2.0 Max (Link-Enabled)

Tabs:
  1) Vector 2.0 Dashboard     – summary of all Waves + selected Wave alpha boxes
  2) Wave Explorer            – detail view for selected Wave + Trading Signal
  3) Alpha Matrix (All Waves) – intraday/30D/60D/1Y alpha + 1Y returns
  4) Alpha Lab (Selected Wave)– IR, hit-rate, max DD, best/worst alpha day
  5) History & Logs           – performance & positions logs
  6) Engine Diagnostics & VIX – raw engine exposure/beta across modes
  7) Human Override           – exposure overrides (display-only)
  8) SmartSafe / Cash         – SmartSafe metrics (if defined)

Features:
  - Mode-aware + VIX-gated metrics: Standard / Alpha-Minus-Beta / Private Logic
  - Intraday / 30D / 60D / 1Y Alpha Captured boxes
  - Human override exposure slider per Wave+Mode
  - Clickable Google Finance links for Top 10 holdings (ticker text is the link)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from waves_engine import WavesEngine


# ----------------------------------------------------------------------
# Formatting & analytics helpers
# ----------------------------------------------------------------------
def _fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:0.2f}%"


def _fmt_pct_diff(wave, bm):
    if wave is None or bm is None or pd.isna(wave) or pd.isna(bm):
        return "—"
    diff = (wave - bm) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:0.2f} pts vs BM"


def _trading_signal(alpha_30d, alpha_1y, beta, exposure):
    """Simple qualitative trading-style signal based on alpha profile."""
    alpha_30d = float(alpha_30d) if alpha_30d is not None and not pd.isna(alpha_30d) else 0.0
    alpha_1y = float(alpha_1y) if alpha_1y is not None and not pd.isna(alpha_1y) else 0.0
    beta = float(beta) if beta is not None and not pd.isna(beta) else 1.0
    exposure = float(exposure) if exposure is not None and not pd.isna(exposure) else 1.0

    small = 0.01   # 1%
    medium = 0.03  # 3%
    large = 0.06   # 6%

    if alpha_1y > large and alpha_30d > medium:
        return (
            "📈 **Strong Uptrend / Core Hold** — 1-Year alpha is strong and the last "
            "30 days are also beating the benchmark. "
            f"β≈{beta:0.2f}, exposure≈{exposure:0.2f}× look justified."
        )
    if alpha_1y > medium and alpha_30d < 0:
        return (
            "🔁 **Rebalance / Buy-the-Dip Candidate** — 1-Year alpha is positive but "
            "recent 30-Day alpha is soft. Likely a pullback within an uptrend."
        )
    if alpha_1y < -small and alpha_30d < -small:
        return (
            "⚠️ **Underperformer / Watchlist** — both 1-Year and 30-Day alpha are "
            "negative. Candidate for risk reduction or strategy review."
        )
    if abs(alpha_1y) < small and abs(alpha_30d) < small:
        return (
            "➖ **Benchmark-Like Behavior** — little alpha either way over 30-Day and "
            "1-Year windows. Acting like its benchmark."
        )

    return (
        "ℹ️ **Mixed Signal** — alpha profile is in between clear categories. Use "
        "Diagnostics + Alpha Lab to decide whether to lean risk-on or risk-off."
    )


def _alpha_lab_stats(history_30d: pd.DataFrame) -> dict:
    """IR / hit-rate / best-worst alpha day / max drawdown from 30-day alpha series."""
    if history_30d is None or history_30d.empty:
        return {}

    df = history_30d.copy()
    if "alpha_captured" not in df.columns or "wave_value" not in df.columns:
        return {}

    alpha = df["alpha_captured"].dropna()
    if alpha.empty:
        return {}

    mean = alpha.mean()
    vol = alpha.std(ddof=1)
    ir = None
    if vol and vol > 0:
        ir = (mean / vol) * np.sqrt(252)

    hit_rate = (alpha > 0).mean()
    best = alpha.max()
    worst = alpha.min()

    curve = df["wave_value"].dropna()
    if curve.empty:
        max_dd = None
    else:
        running_max = curve.cummax()
        drawdown = curve / running_max - 1.0
        max_dd = drawdown.min()

    return {
        "mean_daily_alpha": mean,
        "vol_daily_alpha": vol,
        "ir_annualized": ir,
        "hit_rate": hit_rate,
        "best_alpha": best,
        "worst_alpha": worst,
        "max_drawdown": max_dd,
    }


# ----------------------------------------------------------------------
# Human override helpers (display-only)
# ----------------------------------------------------------------------
def _override_key(wave: str, mode_key: str) -> str:
    return f"override_exposure__{wave}__{mode_key}"


def _get_override_multiplier(wave: str, mode_key: str) -> float:
    key = _override_key(wave, mode_key)
    return float(st.session_state.get(key, 1.0))


def _set_override_multiplier(wave: str, mode_key: str, value: float) -> None:
    key = _override_key(wave, mode_key)
    st.session_state[key] = float(value)


def _apply_override_to_perf(perf: dict, wave: str, mode_key: str) -> dict:
    """
    Scale alpha/returns/exposure by a user override multiplier.

    Affects ONLY what the console displays (Dashboard, Wave Explorer, Alpha Matrix,
    Alpha Lab). The underlying engine & logs remain raw.
    """
    if perf is None:
        return perf

    mult = _get_override_multiplier(wave, mode_key)
    if mult == 1.0:
        return perf

    new = dict(perf)

    scale_fields = [
        "intraday_alpha_captured",
        "alpha_30d",
        "alpha_60d",
        "alpha_1y",
        "return_30d_wave",
        "return_60d_wave",
        "return_1y_wave",
    ]
    for fld in scale_fields:
        if fld in new and new[fld] is not None and not pd.isna(new[fld]):
            new[fld] = new[fld] * mult

    # Scale daily series
    hist = new.get("history_30d")
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        hist = hist.copy()
        if "wave_return" in hist.columns:
            hist["wave_return"] = hist["wave_return"] * mult
        if "alpha_captured" in hist.columns:
            hist["alpha_captured"] = hist["alpha_captured"] * mult
        if "wave_return" in hist.columns:
            hist["wave_value"] = (1.0 + hist["wave_return"]).cumprod()
        new["history_30d"] = hist

    # Approximate exposure / beta scaling
    if "exposure_final" in new and new["exposure_final"] is not None:
        new["exposure_final"] = new["exposure_final"] * mult
    if "beta_realized" in new and new["beta_realized"] is not None:
        new["beta_realized"] = new["beta_realized"] * mult

    return new


# ----------------------------------------------------------------------
# Cache reset on app start
# ----------------------------------------------------------------------
def clear_streamlit_cache_once():
    if "cache_cleared" in st.session_state:
        return
    try:
        if hasattr(st, "cache_data"):
            st.cache_data.clear()
        if hasattr(st, "cache_resource"):
            st.cache_resource.clear()
    except Exception:
        pass
    st.session_state["cache_cleared"] = True


clear_streamlit_cache_once()

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ — Vector 2.0 Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console — Vector 2.0")
st.caption(
    "Live Wave Engine • Mode-Aware + VIX-Gated • Beta-Adjusted Alpha Captured • "
    "Intraday + 30/60/1-Year • Multi-Wave Dashboard"
)

# ----------------------------------------------------------------------
# Engine init
# ----------------------------------------------------------------------
try:
    engine = WavesEngine(list_path="list.csv", weights_path="wave_weights.csv")
except Exception as e:
    st.error(f"Engine failed to initialize: {e}")
    st.stop()

# Exclude crypto income variants for this console
EXCLUDED_WAVES = {
    "Crypto Income Wave",
    "Crypto Income",
    "Crypto Wave",
    "Crypto Income APW",
    "Crypto Income AIW",
}

all_waves = engine.get_wave_names()
waves = [w for w in all_waves if w not in EXCLUDED_WAVES]

if not waves:
    st.error("No active Waves detected (after exclusions).")
    st.stop()

# Optional SmartSafe wave
SMARTSAFE_NAMES = {
    "SmartSafe Wave",
    "SmartSafe",
    "SmartSafe™",
    "SmartSafe Cash Wave",
}
smartsafe_wave = next((w for w in waves if w in SMARTSAFE_NAMES), None)

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Wave & Mode")

selected_wave = st.sidebar.selectbox("Selected Wave", waves, index=0)

mode_label = st.sidebar.selectbox(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic"],
    index=0,
)

mode_map = {
    "Standard": "standard",
    "Alpha-Minus-Beta": "alpha-minus-beta",
    "Private Logic": "private_logic",
}
selected_mode_key = mode_map[mode_label]

st.sidebar.markdown("---")
st.sidebar.markdown("**Files in use:**")
st.sidebar.code("list.csv\nwave_weights.csv", language="text")

st.sidebar.markdown("**Log Directory:**")
st.sidebar.code("logs/positions\nlogs/performance", language="text")


# ----------------------------------------------------------------------
# Helper: get performance for app (with override)
# ----------------------------------------------------------------------
def get_perf_for_app(wave: str, mode_key: str, days: int = 30, use_override: bool = True):
    perf = engine.get_wave_performance(wave, mode=mode_key, days=days, log=False)
    if use_override:
        perf = _apply_override_to_perf(perf, wave, mode_key)
    return perf


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
(
    tab_dashboard,
    tab_wave,
    tab_alpha_matrix,
    tab_alpha_lab,
    tab_history,
    tab_diagnostics,
    tab_human,
    tab_smartsafe,
) = st.tabs(
    [
        "Vector 2.0 Dashboard",
        "Wave Explorer",
        "Alpha Matrix (All Waves)",
        "Alpha Lab (Selected Wave)",
        "History & Logs",
        "Engine Diagnostics & VIX",
        "Human Override",
        "SmartSafe / Cash",
    ]
)


# ----------------------------------------------------------------------
# Helper: performance for all waves
# ----------------------------------------------------------------------
def get_all_wave_performance(mode_key: str) -> pd.DataFrame:
    rows = []
    for w in waves:
        try:
            perf = get_perf_for_app(w, mode_key, days=30, use_override=True)
        except Exception:
            perf = None
        if perf is not None:
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": perf["benchmark"],
                    "Realized Beta (≈60d)": perf["beta_realized"],
                    "Exposure (Net)": perf.get("exposure_final", None),
                    "Intraday Alpha Captured": perf["intraday_alpha_captured"],
                    "Alpha 30D": perf["alpha_30d"],
                    "Alpha 60D": perf["alpha_60d"],
                    "Alpha 1Y": perf["alpha_1y"],
                    "Return 1Y Wave": perf["return_1y_wave"],
                    "Return 1Y BM": perf["return_1y_benchmark"],
                }
            )
        else:
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": "—",
                    "Realized Beta (≈60d)": None,
                    "Exposure (Net)": None,
                    "Intraday Alpha Captured": None,
                    "Alpha 30D": None,
                    "Alpha 60D": None,
                    "Alpha 1Y": None,
                    "Return 1Y Wave": None,
                    "Return 1Y BM": None,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# TAB 1 — Dashboard
# ----------------------------------------------------------------------
with tab_dashboard:
    st.subheader(f"Mode Snapshot — {mode_label}")

    perf_df = get_all_wave_performance(selected_mode_key)

    # High-level stats
    numeric_cols = [
        "Intraday Alpha Captured",
        "Alpha 30D",
        "Alpha 60D",
        "Alpha 1Y",
    ]
    stats_df = perf_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    avg_intraday = stats_df["Intraday Alpha Captured"].mean()
    avg_30 = stats_df["Alpha 30D"].mean()
    avg_60 = stats_df["Alpha 60D"].mean()
    avg_1y = stats_df["Alpha 1Y"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Intraday Alpha (All Waves)", _fmt_pct(avg_intraday))
    c2.metric("Avg 30-Day Alpha (All Waves)", _fmt_pct(avg_30))
    c3.metric("Avg 60-Day Alpha (All Waves)", _fmt_pct(avg_60))
    c4.metric("Avg 1-Year Alpha (All Waves)", _fmt_pct(avg_1y))

    # Selected wave alpha boxes
    st.markdown(f"##### Selected Wave Alpha — {selected_wave}")
    try:
        perf_sel = get_perf_for_app(selected_wave, selected_mode_key, days=30, use_override=True)
    except Exception:
        perf_sel = None

    if perf_sel is not None:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Intraday Alpha Captured", _fmt_pct(perf_sel["intraday_alpha_captured"]))
        sc2.metric("30-Day Alpha Captured", _fmt_pct(perf_sel["alpha_30d"]))
        sc3.metric("60-Day Alpha Captured", _fmt_pct(perf_sel["alpha_60d"]))
        sc4.metric("1-Year Alpha Captured", _fmt_pct(perf_sel["alpha_1y"]))

    # Top / bottom (30D alpha)
    perf_df["Alpha 30D_safe"] = pd.to_numeric(perf_df["Alpha 30D"], errors="coerce").fillna(-9999)
    top_row = perf_df.sort_values("Alpha 30D_safe", ascending=False).head(1)
    bot_row = perf_df.sort_values("Alpha 30D_safe", ascending=True).head(1)

    tc, bc = st.columns(2)
    if not top_row.empty:
        r = top_row.iloc[0]
        tc.markdown("##### Top Wave (30-Day Alpha)")
        tc.metric(
            r["Wave"],
            _fmt_pct(r["Alpha 30D"]),
            f"β≈{r['Realized Beta (≈60d)']:.2f}" if pd.notna(r["Realized Beta (≈60d)"]) else "β NA",
        )
    if not bot_row.empty:
        r = bot_row.iloc[0]
        bc.markdown("##### Bottom Wave (30-Day Alpha)")
        bc.metric(
            r["Wave"],
            _fmt_pct(r["Alpha 30D"]),
            f"β≈{r['Realized Beta (≈60d)']:.2f}" if pd.notna(r["Realized Beta (≈60d)"]) else "β NA",
        )

    st.markdown("#### All Waves — Quick View")
    display_df = perf_df[
        [
            "Wave",
            "Benchmark",
            "Realized Beta (≈60d)",
            "Exposure (Net)",
            "Intraday Alpha Captured",
            "Alpha 30D",
            "Alpha 60D",
            "Alpha 1Y",
        ]
    ].copy()

    for col in ["Intraday Alpha Captured", "Alpha 30D", "Alpha 60D", "Alpha 1Y"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
            display_df[col] = (display_df[col] * 100).round(2)

    for col in ["Realized Beta (≈60d)", "Exposure (Net)"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)

    display_df = display_df.rename(
        columns={
            "Intraday Alpha Captured": "Intraday Alpha (%)",
            "Alpha 30D": "Alpha 30D (%)",
            "Alpha 60D": "Alpha 60D (%)",
            "Alpha 1Y": "Alpha 1Y (%)",
        }
    )

    st.dataframe(display_df, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------
# TAB 2 — Wave Explorer
# ----------------------------------------------------------------------
with tab_wave:
    st.subheader(f"Wave Explorer — {selected_wave} ({mode_label})")

    top_row = st.columns([2.2, 1.2])
    perf_col, snapshot_col = top_row

    bottom_row = st.columns([2.0, 1.4])
    chart_col, holdings_col = bottom_row

    # Snapshot
    with snapshot_col:
        st.markdown("##### Snapshot")
        try:
            holdings_df = engine.get_wave_holdings(selected_wave)
            num_holdings = len(holdings_df)
            benchmark = engine.get_benchmark(selected_wave)
        except Exception as e:
            st.error(f"Could not load snapshot for {selected_wave}: {e}")
            holdings_df = None
            num_holdings = 0
            benchmark = "SPY"

        m1, m2 = st.columns(2)
        m1.metric("Wave", selected_wave)
        m2.metric("Benchmark", benchmark)

        st.write(f"**Mode:** {mode_label}")
        st.write(f"**Holdings:** {num_holdings:,}")

        if holdings_df is not None and "sector" in holdings_df.columns:
            sector_weights = (
                holdings_df.groupby("sector")["weight"].sum().sort_values(ascending=False)
            )
            if not sector_weights.empty:
                top_sector = sector_weights.index[0]
                top_sector_weight = float(sector_weights.iloc[0])
                st.write(f"**Top Sector:** {top_sector} ({top_sector_weight:.1%})")

    # Performance + signal
    with perf_col:
        st.markdown("##### Performance & Alpha")
        try:
            perf = get_perf_for_app(selected_wave, selected_mode_key, days=30, use_override=True)
        except Exception as e:
            st.error(f"Could not compute performance for {selected_wave}: {e}")
            perf = None

        if perf is not None:
            beta = perf["beta_realized"]
            exposure = perf.get("exposure_final", None)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Intraday Alpha Captured", _fmt_pct(perf["intraday_alpha_captured"]))
            c2.metric("30-Day Alpha Captured", _fmt_pct(perf["alpha_30d"]))
            c3.metric("60-Day Alpha Captured", _fmt_pct(perf["alpha_60d"]))
            c4.metric("1-Year Alpha Captured", _fmt_pct(perf["alpha_1y"]))

            st.markdown(f"**Realized Beta (≈60d):** {beta:0.2f}")
            if exposure is not None:
                st.markdown(f"**Net Exposure (Mode + VIX + Human):** {exposure:0.2f}×")
            st.markdown(f"**Benchmark:** {perf['benchmark']}")

            r1, r2, r3 = st.columns(3)
            r1.metric(
                "30-Day Wave Return",
                _fmt_pct(perf["return_30d_wave"]),
                delta=_fmt_pct_diff(perf["return_30d_wave"], perf["return_30d_benchmark"]),
            )
            r2.metric(
                "60-Day Wave Return",
                _fmt_pct(perf["return_60d_wave"]),
                delta=_fmt_pct_diff(perf["return_60d_wave"], perf["return_60d_benchmark"]),
            )
            r3.metric(
                "1-Year Wave Return",
                _fmt_pct(perf["return_1y_wave"]),
                delta=_fmt_pct_diff(perf["return_1y_wave"], perf["return_1y_benchmark"]),
            )

            st.markdown("##### Trading Signal (Non-binding)")
            st.markdown(
                _trading_signal(
                    perf["alpha_30d"],
                    perf["alpha_1y"],
                    perf["beta_realized"],
                    perf.get("exposure_final", None),
                )
            )

    # Chart + recent alpha table
    with chart_col:
        if perf is not None:
            st.markdown(f"##### 30-Day Curve — {selected_wave} vs {perf['benchmark']}")
            history = perf["history_30d"]
            chart_data = history[["wave_value", "benchmark_value"]]
            st.line_chart(chart_data)

            hist_df = history.reset_index()
            date_col = hist_df.columns[0]
            hist_df = hist_df.rename(columns={date_col: "date"})
            hist_df["date"] = pd.to_datetime(hist_df["date"]).dt.date
            hist_df["wave_return_pct"] = hist_df["wave_return"] * 100
            hist_df["benchmark_return_pct"] = hist_df["benchmark_return"] * 100
            hist_df["alpha_captured_pct"] = hist_df["alpha_captured"] * 100

            display_cols = ["date", "wave_return_pct", "benchmark_return_pct", "alpha_captured_pct"]
            hist_display = hist_df[display_cols].tail(15).iloc[::-1]
            hist_display = hist_display.rename(
                columns={
                    "date": "Date",
                    "wave_return_pct": "Wave Return (%)",
                    "benchmark_return_pct": "Benchmark Return (%)",
                    "alpha_captured_pct": "Alpha Captured (%)",
                }
            ).round(3)

            st.markdown("###### Recent Daily Returns & Alpha (Last 15 Days)")
            st.dataframe(hist_display, hide_index=True, use_container_width=True)

    # Top 10 holdings (ticker text is clickable)
    with holdings_col:
        st.markdown("##### Top 10 Holdings (Click Ticker for Google Finance)")
        try:
            top10 = engine.get_top_holdings(selected_wave, n=10)
        except Exception as e:
            st.error(f"Could not load holdings for {selected_wave}: {e}")
            top10 = None

        if top10 is not None and not top10.empty:

            def google_finance_url(ticker: str) -> str:
                # generic (Google will figure out the exchange)
                return f"https://www.google.com/finance/quote/{ticker}"

            st.markdown(
                "| Ticker | Company | Weight |\n|:------:|:--------|-------:|",
                unsafe_allow_html=True,
            )
            for _, row in top10.iterrows():
                tkr = str(row["ticker"])
                company = str(row.get("company", "") or "")
                weight = float(row.get("weight", 0.0))
                url = google_finance_url(tkr)
                st.markdown(
                    f"| [{tkr}]({url}) | {company} | {weight:.2%} |",
                    unsafe_allow_html=True,
                )
        else:
            st.write("No holdings found for this Wave.")


# ----------------------------------------------------------------------
# TAB 3 — Alpha Matrix
# ----------------------------------------------------------------------
with tab_alpha_matrix:
    st.subheader(f"Alpha Matrix — All Waves ({mode_label} Mode)")

    perf_df = get_all_wave_performance(selected_mode_key)

    matrix_df = perf_df[
        [
            "Wave",
            "Benchmark",
            "Realized Beta (≈60d)",
            "Exposure (Net)",
            "Intraday Alpha Captured",
            "Alpha 30D",
            "Alpha 60D",
            "Alpha 1Y",
            "Return 1Y Wave",
            "Return 1Y BM",
        ]
    ].copy()

    for col in [
        "Intraday Alpha Captured",
        "Alpha 30D",
        "Alpha 60D",
        "Alpha 1Y",
        "Return 1Y Wave",
        "Return 1Y BM",
    ]:
        if col in matrix_df.columns:
            matrix_df[col] = pd.to_numeric(matrix_df[col], errors="coerce")
            matrix_df[col] = (matrix_df[col] * 100).round(2)

    for col in ["Realized Beta (≈60d)", "Exposure (Net)"]:
        if col in matrix_df.columns:
            matrix_df[col] = pd.to_numeric(matrix_df[col], errors="coerce").round(2)

    matrix_df = matrix_df.rename(
        columns={
            "Intraday Alpha Captured": "Intraday Alpha (%)",
            "Alpha 30D": "Alpha 30D (%)",
            "Alpha 60D": "Alpha 60D (%)",
            "Alpha 1Y": "Alpha 1Y (%)",
            "Return 1Y Wave": "1Y Wave Return (%)",
            "Return 1Y BM": "1Y Benchmark Return (%)",
        }
    )

    sort_options = {
        "Intraday Alpha (%)": "Intraday Alpha (%)",
        "Alpha 30D (%)": "Alpha 30D (%)",
        "Alpha 60D (%)": "Alpha 60D (%)",
        "Alpha 1Y (%)": "Alpha 1Y (%)",
        "1Y Wave Return (%)": "1Y Wave Return (%)",
        "1Y Benchmark Return (%)": "1Y Benchmark Return (%)",
    }
    sort_label = st.selectbox("Sort Waves by", list(sort_options.keys()), index=1)
    sort_col = sort_options[sort_label]

    if sort_col in matrix_df.columns:
        matrix_df = matrix_df.sort_values(by=sort_col, ascending=False, na_position="last")

    st.dataframe(matrix_df, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------
# TAB 4 — Alpha Lab
# ----------------------------------------------------------------------
with tab_alpha_lab:
    st.subheader(f"Alpha Lab — {selected_wave} ({mode_label})")

    try:
        perf_lab = get_perf_for_app(selected_wave, selected_mode_key, days=30, use_override=True)
    except Exception as e:
        st.error(f"Could not compute performance for Alpha Lab: {e}")
        perf_lab = None

    if perf_lab is not None:
        hist = perf_lab["history_30d"]
        stats = _alpha_lab_stats(hist)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Daily Alpha (30D)", _fmt_pct(stats.get("mean_daily_alpha")))
        c2.metric("Daily Alpha Volatility", _fmt_pct(stats.get("vol_daily_alpha")))
        c3.metric(
            "IR (Annualized)",
            f"{stats.get('ir_annualized', 0):0.2f}"
            if stats.get("ir_annualized") is not None
            else "—",
        )
        c4.metric("Hit Rate (Alpha > 0)", _fmt_pct(stats.get("hit_rate")))

        c5, c6, c7 = st.columns(3)
        c5.metric("Best Alpha Day", _fmt_pct(stats.get("best_alpha")))
        c6.metric("Worst Alpha Day", _fmt_pct(stats.get("worst_alpha")))
        c7.metric("Max Drawdown (30D Curve)", _fmt_pct(stats.get("max_drawdown")))

        st.markdown("###### Daily Alpha Series (Last 30 Trading Days)")
        alpha_df = hist[["alpha_captured"]].copy()
        alpha_df["Alpha Captured (%)"] = alpha_df["alpha_captured"] * 100
        alpha_df = alpha_df.drop(columns=["alpha_captured"])
        st.dataframe(alpha_df.tail(30), use_container_width=True)


# ----------------------------------------------------------------------
# TAB 5 — History & Logs
# ----------------------------------------------------------------------
with tab_history:
    st.subheader(f"History & Logs — {selected_wave}")

    logs_perf_root = Path("logs") / "performance"
    logs_pos_root = Path("logs") / "positions"

    perf_fname = logs_perf_root / f"{selected_wave.replace(' ', '_')}_performance_daily.csv"
    pos_prefix = f"{selected_wave.replace(' ', '_')}_positions_"

    if perf_fname.exists():
        df_log = pd.read_csv(perf_fname)
        df_show = df_log.iloc[::-1].head(50)
        st.markdown("##### Last 50 Performance Log Entries")
        st.dataframe(df_show, hide_index=True, use_container_width=True)
    else:
        st.info("No performance log found yet for this Wave.")

    st.markdown("---")
    st.markdown("##### Recent Positions Snapshots")
    if logs_pos_root.exists():
        pos_files = sorted(p for p in logs_pos_root.iterdir() if p.name.startswith(pos_prefix))
        if pos_files:
            for p in pos_files[-5:][::-1]:
                st.write(f"- {p.name}")
        else:
            st.write("No positions snapshots found yet for this Wave.")
    else:
        st.write("Positions log directory not found.")


# ----------------------------------------------------------------------
# TAB 6 — Engine Diagnostics (raw engine, no override)
# ----------------------------------------------------------------------
with tab_diagnostics:
    st.subheader("Engine Diagnostics & VIX (Raw Engine Numbers)")

    rows = []
    for label, key in mode_map.items():
        try:
            perf = engine.get_wave_performance(selected_wave, mode=key, days=30, log=False)
        except Exception:
            perf = None
        if perf is not None:
            rows.append(
                {
                    "Mode": label,
                    "Benchmark": perf["benchmark"],
                    "Realized Beta (≈60d)": perf["beta_realized"],
                    "Exposure (Net, Engine)": perf.get("exposure_final", None),
                    "Alpha 30D": perf["alpha_30d"],
                    "Alpha 60D": perf["alpha_60d"],
                    "Alpha 1Y": perf["alpha_1y"],
                }
            )

    diag_df = pd.DataFrame(rows)
    if not diag_df.empty:
        for col in ["Alpha 30D", "Alpha 60D", "Alpha 1Y"]:
            diag_df[col] = pd.to_numeric(diag_df[col], errors="coerce")
            diag_df[col] = (diag_df[col] * 100).round(2)
        for col in ["Realized Beta (≈60d)", "Exposure (Net, Engine)"]:
            diag_df[col] = pd.to_numeric(diag_df[col], errors="coerce").round(2)

        diag_df = diag_df.rename(
            columns={
                "Alpha 30D": "Alpha 30D (%)",
                "Alpha 60D": "Alpha 60D (%)",
                "Alpha 1Y": "Alpha 1Y (%)",
            }
        )
        st.markdown(f"##### Mode Comparison — {selected_wave}")
        st.dataframe(diag_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "VIX overlay lives inside the engine. Higher VIX levels automatically throttle "
        "exposure (especially in Alpha-Minus-Beta), while Private Logic allows modest "
        "risk-on bias when volatility is cheap. This tab shows **raw engine** numbers "
        "without any human override."
    )


# ----------------------------------------------------------------------
# TAB 7 — Human Override
# ----------------------------------------------------------------------
with tab_human:
    st.subheader("Human Override — Exposure Controls (Display-Only)")

    base_perf = engine.get_wave_performance(selected_wave, mode=selected_mode_key, days=30, log=False)
    base_exposure = base_perf.get("exposure_final", 1.0)

    current_mult = _get_override_multiplier(selected_wave, selected_mode_key)
    slider_val = st.slider(
        "Exposure Override Multiplier (relative to engine's exposure)",
        min_value=0.50,
        max_value=1.50,
        value=float(current_mult),
        step=0.05,
        help=(
            "1.00 = no change. <1 = more defensive; >1 = more aggressive. "
            "Affects ONLY display metrics (Dashboard, Explorer, Matrix, Alpha Lab)."
        ),
    )
    _set_override_multiplier(selected_wave, selected_mode_key, slider_val)

    eff_exposure = base_exposure * slider_val
    c1, c2 = st.columns(2)
    c1.metric("Engine Exposure (Net)", f"{base_exposure:0.2f}×")
    c2.metric("Effective Exposure (with Override)", f"{eff_exposure:0.2f}×")

    st.markdown(
        "> Use this tab as a *human-in-the-loop* risk dial. The underlying engine "
        "and logs are unchanged — this only rescales what the console displays."
    )


# ----------------------------------------------------------------------
# TAB 8 — SmartSafe / Cash
# ----------------------------------------------------------------------
with tab_smartsafe:
    st.subheader("SmartSafe™ / Cash Engine")

    if smartsafe_wave is None:
        st.info(
            "No SmartSafe™ Wave detected in wave_weights.csv. "
            "To activate this tab, add a wave named one of: "
            f"{', '.join(SMARTSAFE_NAMES)}."
        )
    else:
        st.markdown(f"##### SmartSafe Wave: {smartsafe_wave} ({mode_label} Mode)")

        try:
            perf = get_perf_for_app(smartsafe_wave, selected_mode_key, days=30, use_override=True)
        except Exception as e:
            st.error(f"Could not compute performance for SmartSafe Wave: {e}")
            perf = None

        if perf is not None:
            beta = perf["beta_realized"]
            exposure = perf.get("exposure_final", None)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Intraday Alpha Captured", _fmt_pct(perf["intraday_alpha_captured"]))
            c2.metric("30-Day Alpha Captured", _fmt_pct(perf["alpha_30d"]))
            c3.metric("60-Day Alpha Captured", _fmt_pct(perf["alpha_60d"]))
            c4.metric("1-Year Alpha Captured", _fmt_pct(perf["alpha_1y"]))

            st.markdown(f"**Realized Beta (≈60d):** {beta:0.2f}")
            if exposure is not None:
                st.markdown(f"**Net Exposure (Mode + VIX + Human):** {exposure:0.2f}×")
            st.markdown(f"**Benchmark:** {perf['benchmark']}")

            st.markdown(
                "> SmartSafe is designed to run at very low beta, using the VIX ladder "
                "and mode rules to throttle risk while still capturing enhanced yield "
                "relative to pure cash."
            )


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Engine: WAVES Intelligence™ • list.csv = total market universe • "
    "wave_weights.csv = Wave definitions • Alpha = Mode-Aware + VIX-Gated, "
    "Beta-Adjusted Alpha Captured • Human Override = display-only risk dials • "
    "Modes: Standard / Alpha-Minus-Beta / Private Logic • Console: Vector 2.0 Max."
)