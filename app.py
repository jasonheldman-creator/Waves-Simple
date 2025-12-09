"""
app.py

WAVES Intelligence™ Institutional Console — Vector 2.0 (Enhanced)

- Multi-tab institutional console on top of the VIX-gated, mode-aware WavesEngine.
- Shows Intraday / 30D / 60D / 1Y Alpha Captured in metric boxes for each Wave.
- Adds trading-style "Signal" interpretation based on alpha profile + beta/exposure.
- Fixes numeric TypeErrors by coercing numeric columns before scaling.

Tabs:
  1) Vector 2.0 Dashboard     – summary of all Waves, top/bottom alpha + selected Wave alpha boxes
  2) Wave Explorer            – full detail view for selected Wave + Trading Signal
  3) Alpha Matrix (All Waves) – Intraday/30D/60D/1Y Alpha + 1Y returns for ALL Waves
  4) History & Logs           – recent performance logs for selected Wave
  5) Engine Diagnostics & VIX – per-mode beta & exposure comparison
  6) SmartSafe / Cash         – SmartSafe wave view (if defined)

Modes:
  - Standard
  - Alpha-Minus-Beta
  - Private Logic
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from waves_engine import WavesEngine


# ----------------------------------------------------------------------
# Formatting helpers
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
    """
    Simple qualitative trading-style signal based on alpha profile.

    This does NOT place trades — it's an interpretation layer:
    - Strong Uptrend / Core Hold
    - Short-Term Soft Patch / Consider Trim
    - Rebound Setup
    - Underperformer / Watchlist
    """
    alpha_30d = float(alpha_30d) if alpha_30d is not None and not pd.isna(alpha_30d) else 0.0
    alpha_1y = float(alpha_1y) if alpha_1y is not None and not pd.isna(alpha_1y) else 0.0
    beta = float(beta) if beta is not None and not pd.isna(beta) else 1.0
    exposure = float(exposure) if exposure is not None and not pd.isna(exposure) else 1.0

    # Thresholds (in decimal alpha)
    small = 0.01   # 1%
    medium = 0.03  # 3%
    large = 0.06   # 6%

    if alpha_1y > large and alpha_30d > medium:
        return (
            "📈 **Strong Uptrend / Core Hold** — 1-Year alpha is strong and the last "
            "30 days are also beating the benchmark. Current beta and exposure "
            f"({beta:0.2f}, {exposure:0.2f}×) look justified."
        )
    if alpha_1y > medium and alpha_30d < 0:
        return (
            "🔁 **Rebalance / Buy-the-Dip Candidate** — 1-Year alpha is positive but "
            "the most recent 30 days are soft. Consider whether this is a healthy "
            "pullback within an uptrend."
        )
    if alpha_1y < -small and alpha_30d < -small:
        return (
            "⚠️ **Underperformer / Watchlist** — both 1-Year and 30-Day alpha are "
            "negative. Treat as a candidate for risk reduction or strategy review."
        )
    if abs(alpha_1y) < small and abs(alpha_30d) < small:
        return (
            "➖ **Benchmark-Like Behavior** — little alpha either way over both 30-Day "
            "and 1-Year windows. Acting like its benchmark; no strong signal."
        )

    return (
        "ℹ️ **Mixed Signal** — alpha profile is in between clear categories. Use the "
        "metrics and diagnostics to decide whether to lean risk-on or risk-off."
    )


# ----------------------------------------------------------------------
# Hard cache reset on app start
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

# Exclude Crypto Income / Crypto Wave variants from this console
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

# Optional: SmartSafe wave name if present
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
# Tabs
# ----------------------------------------------------------------------
(
    tab_dashboard,
    tab_wave,
    tab_alpha_matrix,
    tab_history,
    tab_diagnostics,
    tab_smartsafe,
) = st.tabs(
    [
        "Vector 2.0 Dashboard",
        "Wave Explorer",
        "Alpha Matrix (All Waves)",
        "History & Logs",
        "Engine Diagnostics & VIX",
        "SmartSafe / Cash",
    ]
)

# ----------------------------------------------------------------------
# Helper to compute performance for all waves for current mode
# ----------------------------------------------------------------------
def get_all_wave_performance(mode_key: str) -> pd.DataFrame:
    rows = []
    for wave_name in waves:
        try:
            perf = engine.get_wave_performance(
                wave_name, mode=mode_key, days=30, log=False
            )
        except Exception:
            perf = None

        if perf is not None:
            rows.append(
                {
                    "Wave": wave_name,
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
                    "Wave": wave_name,
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

    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------
# TAB 1 — Vector 2.0 Dashboard
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

    # Selected Wave alpha boxes (including 1-Year)
    st.markdown(f"##### Selected Wave Alpha — {selected_wave}")
    try:
        perf_sel = engine.get_wave_performance(
            selected_wave, mode=selected_mode_key, days=30, log=False
        )
    except Exception:
        perf_sel = None

    if perf_sel is not None:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Intraday Alpha Captured", _fmt_pct(perf_sel["intraday_alpha_captured"]))
        sc2.metric("30-Day Alpha Captured", _fmt_pct(perf_sel["alpha_30d"]))
        sc3.metric("60-Day Alpha Captured", _fmt_pct(perf_sel["alpha_60d"]))
        sc4.metric("1-Year Alpha Captured", _fmt_pct(perf_sel["alpha_1y"]))

    # Top / bottom alpha waves (30D as primary)
    perf_df["Alpha 30D_safe"] = pd.to_numeric(
        perf_df["Alpha 30D"], errors="coerce"
    ).fillna(-9999)
    top_wave_row = perf_df.sort_values("Alpha 30D_safe", ascending=False).head(1)
    bottom_wave_row = perf_df.sort_values("Alpha 30D_safe", ascending=True).head(1)

    tcol, bcol = st.columns(2)
    if not top_wave_row.empty:
        row = top_wave_row.iloc[0]
        tcol.markdown("##### Top Wave (30-Day Alpha)")
        tcol.metric(
            f"{row['Wave']}",
            _fmt_pct(row["Alpha 30D"]),
            f"β≈{row['Realized Beta (≈60d)']:.2f}"
            if pd.notna(row["Realized Beta (≈60d)"])
            else "β NA",
        )

    if not bottom_wave_row.empty:
        row = bottom_wave_row.iloc[0]
        bcol.markdown("##### Bottom Wave (30-Day Alpha)")
        bcol.metric(
            f"{row['Wave']}",
            _fmt_pct(row["Alpha 30D"]),
            f"β≈{row['Realized Beta (≈60d)']:.2f}"
            if pd.notna(row["Realized Beta (≈60d)"])
            else "β NA",
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

    # Ensure numeric and scale
    for col in [
        "Intraday Alpha Captured",
        "Alpha 30D",
        "Alpha 60D",
        "Alpha 1Y",
    ]:
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
# TAB 2 — Wave Explorer (Selected Wave Detail)
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

        st.write("")
        st.write(f"**Mode:** {mode_label}")
        st.write(f"**Holdings:** {num_holdings:,}")

        if holdings_df is not None and "sector" in holdings_df.columns:
            sector_weights = (
                holdings_df.groupby("sector")["weight"]
                .sum()
                .sort_values(ascending=False)
            )
            if not sector_weights.empty:
                top_sector = sector_weights.index[0]
                top_sector_weight = float(sector_weights.iloc[0])
                st.write(f"**Top Sector:** {top_sector} ({top_sector_weight:.1%})")

    # Performance metrics + trading signal
    with perf_col:
        st.markdown("##### Performance & Alpha")

        try:
            perf = engine.get_wave_performance(
                selected_wave, mode=selected_mode_key, days=30, log=True
            )
        except Exception as e:
            st.error(f"Could not compute performance for {selected_wave}: {e}")
            perf = None

        if perf is not None:
            beta = perf["beta_realized"]
            exposure = perf.get("exposure_final", None)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Intraday Alpha Captured",
                _fmt_pct(perf["intraday_alpha_captured"]),
            )
            c2.metric("30-Day Alpha Captured", _fmt_pct(perf["alpha_30d"]))
            c3.metric("60-Day Alpha Captured", _fmt_pct(perf["alpha_60d"]))
            c4.metric("1-Year Alpha Captured", _fmt_pct(perf["alpha_1y"]))

            st.markdown(f"**Realized Beta (≈60d):** {beta:0.2f}")
            if exposure is not None:
                st.markdown(f"**Net Exposure (Mode + VIX):** {exposure:0.2f}×")
            st.markdown(f"**Benchmark:** {perf['benchmark']}")

            st.write("")
            r1, r2, r3 = st.columns(3)
            r1.metric(
                "30-Day Wave Return",
                _fmt_pct(perf["return_30d_wave"]),
                delta=_fmt_pct_diff(
                    perf["return_30d_wave"], perf["return_30d_benchmark"]
                ),
            )
            r2.metric(
                "60-Day Wave Return",
                _fmt_pct(perf["return_60d_wave"]),
                delta=_fmt_pct_diff(
                    perf["return_60d_wave"], perf["return_60d_benchmark"]
                ),
            )
            r3.metric(
                "1-Year Wave Return",
                _fmt_pct(perf["return_1y_wave"]),
                delta=_fmt_pct_diff(
                    perf["return_1y_wave"], perf["return_1y_benchmark"]
                ),
            )

            # Trading-style signal box
            st.markdown("##### Trading Signal (Non-binding)")
            signal_text = _trading_signal(
                perf["alpha_30d"],
                perf["alpha_1y"],
                perf["beta_realized"],
                perf.get("exposure_final", None),
            )
            st.markdown(signal_text)

    # 30-day chart + alpha table
    with chart_col:
        if perf is not None:
            st.markdown(
                f"##### 30-Day Curve — {selected_wave} vs {perf['benchmark']}"
            )

            history = perf["history_30d"]
            chart_data = history[["wave_value", "benchmark_value"]]
            st.line_chart(chart_data)

            hist_df = history.copy()
            hist_df = hist_df.reset_index()
            date_col = hist_df.columns[0]
            hist_df = hist_df.rename(columns={date_col: "date"})
            hist_df["date"] = pd.to_datetime(hist_df["date"]).dt.date

            hist_df["wave_return_pct"] = hist_df["wave_return"] * 100
            hist_df["benchmark_return_pct"] = hist_df["benchmark_return"] * 100
            hist_df["alpha_captured_pct"] = hist_df["alpha_captured"] * 100

            display_cols = [
                "date",
                "wave_return_pct",
                "benchmark_return_pct",
                "alpha_captured_pct",
            ]
            hist_display = hist_df[display_cols].tail(15).iloc[::-1]
            hist_display = hist_display.rename(
                columns={
                    "date": "Date",
                    "wave_return_pct": "Wave Return (%)",
                    "benchmark_return_pct": "Benchmark Return (%)",
                    "alpha_captured_pct": "Alpha Captured (%)",
                }
            )
            hist_display = hist_display.round(3)

            st.markdown("###### Recent Daily Returns & Alpha (Last 15 Days)")
            st.dataframe(hist_display, hide_index=True, use_container_width=True)

    # Top 10 holdings
    with holdings_col:
        st.markdown("##### Top 10 Holdings")

        try:
            top10 = engine.get_top_holdings(selected_wave, n=10)
        except Exception as e:
            st.error(f"Could not load holdings for {selected_wave}: {e}")
            top10 = None

        if top10 is not None and not top10.empty:

            def google_finance_url(ticker: str) -> str:
                return f"https://www.google.com/finance/quote/{ticker}:NASDAQ"

            display_df = top10.copy()

            if "company" not in display_df.columns:
                display_df["company"] = ""

            display_df = display_df[["ticker", "company", "weight"]].copy()
            display_df["weight"] = display_df["weight"].round(4)
            display_df["Google Finance"] = display_df["ticker"].apply(
                google_finance_url
            )

            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.write("No holdings found for this Wave.")

# ----------------------------------------------------------------------
# TAB 3 — Alpha Matrix (All Waves)
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

    # Ensure numeric then scale to %
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

    # Sort control
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
        matrix_df = matrix_df.sort_values(
            by=sort_col, ascending=False, na_position="last"
        )

    st.dataframe(matrix_df, hide_index=True, use_container_width=True)

# ----------------------------------------------------------------------
# TAB 4 — History & Logs
# ----------------------------------------------------------------------
with tab_history:
    st.subheader(f"History & Logs — {selected_wave}")

    logs_root = Path("logs") / "performance"
    fname = logs_root / f"{selected_wave.replace(' ', '_')}_performance_daily.csv"

    if fname.exists():
        df_log = pd.read_csv(fname)
        df_log_display = df_log.copy()
        df_log_display = df_log_display.iloc[::-1].head(50)
        st.markdown("##### Last 50 Performance Log Entries")
        st.dataframe(df_log_display, hide_index=True, use_container_width=True)
    else:
        st.info("No performance log found yet for this Wave (file not created).")

# ----------------------------------------------------------------------
# TAB 5 — Engine Diagnostics & VIX
# ----------------------------------------------------------------------
with tab_diagnostics:
    st.subheader("Engine Diagnostics & VIX")

    diagnostics_rows = []
    for label, key in mode_map.items():
        try:
            perf = engine.get_wave_performance(
                selected_wave, mode=key, days=30, log=False
            )
        except Exception:
            perf = None

        if perf is not None:
            diagnostics_rows.append(
                {
                    "Mode": label,
                    "Benchmark": perf["benchmark"],
                    "Realized Beta (≈60d)": perf["beta_realized"],
                    "Exposure (Net)": perf.get("exposure_final", None),
                    "Alpha 30D": perf["alpha_30d"],
                    "Alpha 60D": perf["alpha_60d"],
                    "Alpha 1Y": perf["alpha_1y"],
                }
            )

    diag_df = pd.DataFrame(diagnostics_rows)
    if not diag_df.empty:
        for col in ["Alpha 30D", "Alpha 60D", "Alpha 1Y"]:
            diag_df[col] = pd.to_numeric(diag_df[col], errors="coerce")
            diag_df[col] = (diag_df[col] * 100).round(2)

        for col in ["Realized Beta (≈60d)", "Exposure (Net)"]:
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
        "risk-on bias when volatility is cheap."
    )
    st.caption(
        "This tab summarizes the resulting exposures and betas per mode for the selected Wave."
    )

# ----------------------------------------------------------------------
# TAB 6 — SmartSafe / Cash
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
            perf = engine.get_wave_performance(
                smartsafe_wave, mode=selected_mode_key, days=30, log=True
            )
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
                st.markdown(f"**Net Exposure (Mode + VIX):** {exposure:0.2f}×")
            st.markdown(f"**Benchmark:** {perf['benchmark']}")

            st.markdown(
                "> SmartSafe is designed to run at very low beta, using the VIX ladder and "
                "mode rules to throttle risk while still capturing enhanced yield relative "
                "to pure cash."
            )

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Engine: WAVES Intelligence™ • list.csv = total market universe • "
    "wave_weights.csv = Wave definitions • Alpha = Mode-Aware + VIX-Gated, "
    "Beta-Adjusted Alpha Captured • Modes: Standard / Alpha-Minus-Beta / "
    "Private Logic • Console: Vector 2.0 multi-tab view."
)