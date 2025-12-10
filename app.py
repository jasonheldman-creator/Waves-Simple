import numpy as np
import pandas as pd
import streamlit as st

from waves_engine import WavesEngine

# ---------------------------------------------------------
# Global setup & cache reset
# ---------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

# Clear caches on each rerun to avoid stale logic
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass

# ---------------------------------------------------------
# Initialize engine (singleton)
# ---------------------------------------------------------
@st.cache_resource
def get_engine() -> WavesEngine:
    return WavesEngine(list_path="list.csv", weights_path="wave_weights.csv", logs_root="logs")


engine = get_engine()

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:0.2f}%"


def fmt_beta(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:0.2f}"


def describe_benchmark(bm):
    if isinstance(bm, dict):
        parts = [f"{int(w * 100)}% {t}" for t, w in bm.items()]
        return " + ".join(parts)
    return str(bm)


@st.cache_data(show_spinner=False)
def get_wave_metrics(wave: str, mode: str):
    return engine.get_wave_performance(wave, mode=mode, log=False)


@st.cache_data(show_spinner=False)
def get_wave_top_holdings_dynamic(wave: str, mode: str, n: int = 10) -> pd.DataFrame:
    """
    Build a top-holdings table using the engine's dynamic current_weights,
    merged with static metadata from the universe.
    """
    # Base holdings (static definition)
    base = engine.get_wave_holdings(wave)
    if base is None or base.empty:
        return base

    metrics = get_wave_metrics(wave, mode)
    current_weights = metrics.get("current_weights")

    if current_weights is None or current_weights.empty:
        # fallback to static weights
        base = base.copy()
        base = base.sort_by("weight", ascending=False).head(n)
        return base

    base = base.copy()
    # current_weights index is ticker; map into base
    base["dynamic_weight"] = base["ticker"].map(current_weights).astype(float)
    base["dynamic_weight"] = base["dynamic_weight"].fillna(0.0)

    # Filter out zero dynamic weights
    base = base[base["dynamic_weight"] > 0.0]

    if base.empty:
        return base

    base = base.sort_values("dynamic_weight", ascending=False).head(n).reset_index(drop=True)
    return base


# ---------------------------------------------------------
# UI — Header
# ---------------------------------------------------------
st.markdown(
    "<h1 style='font-size: 2.6rem;'>WAVES Intelligence™ Institutional Console</h1>",
    unsafe_allow_html=True,
)
st.caption("Live Wave Engine • Dynamic Weights • Alpha Capture • Benchmark-Relative Performance")

# Mode selector
mode = st.sidebar.radio(
    "Mode",
    options=["standard", "alpha-minus-beta", "private_logic"],
    format_func=lambda m: {
        "standard": "Standard",
        "alpha-minus-beta": "Alpha-Minus-Beta",
        "private_logic": "Private Logic™",
    }[m],
)

# Optional manual cache reset
if st.sidebar.button("Force Reload Engine & Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

waves = engine.get_wave_names()

# ---------------------------------------------------------
# Layout — Tabs
# ---------------------------------------------------------
tab_dashboard, tab_explorer, tab_alpha, tab_history, tab_about = st.tabs(
    ["Dashboard", "Wave Explorer", "Alpha Matrix", "History (30-Day)", "About / Diagnostics"]
)

# ---------------------------------------------------------
# DASHBOARD TAB
# ---------------------------------------------------------
with tab_dashboard:
    st.subheader(f"Dashboard — Mode: {mode}")

    rows = []
    for w in waves:
        try:
            m = get_wave_metrics(w, mode)
            bm = describe_benchmark(m["benchmark"])
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": bm,
                    "Intraday Alpha": m["intraday_alpha_captured"],
                    "Alpha 30D": m["alpha_30d"],
                    "Alpha 60D": m["alpha_60d"],
                    "Alpha 1Y": m["alpha_1y"],
                    "Return 30D (Wave)": m["return_30d_wave"],
                    "Return 30D (BM)": m["return_30d_benchmark"],
                    "Return 60D (Wave)": m["return_60d_wave"],
                    "Return 60D (BM)": m["return_60d_benchmark"],
                    "Return 1Y (Wave)": m["return_1y_wave"],
                    "Return 1Y (BM)": m["return_1y_benchmark"],
                    "Beta (≈60D)": m["beta_realized"],
                    "Exposure": m["exposure_final"],
                }
            )
        except Exception:
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": "ERROR",
                    "Intraday Alpha": np.nan,
                    "Alpha 30D": np.nan,
                    "Alpha 60D": np.nan,
                    "Alpha 1Y": np.nan,
                    "Return 30D (Wave)": np.nan,
                    "Return 30D (BM)": np.nan,
                    "Return 60D (Wave)": np.nan,
                    "Return 60D (BM)": np.nan,
                    "Return 1Y (Wave)": np.nan,
                    "Return 1Y (BM)": np.nan,
                    "Beta (≈60D)": np.nan,
                    "Exposure": np.nan,
                }
            )

    dash_df = pd.DataFrame(rows)

    # Summary metrics (averages across waves)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Avg 30-Day Alpha",
            fmt_pct(dash_df["Alpha 30D"].mean(skipna=True)),
        )
    with c2:
        st.metric(
            "Avg 60-Day Alpha",
            fmt_pct(dash_df["Alpha 60D"].mean(skipna=True)),
        )
    with c3:
        st.metric(
            "Avg 1-Year Alpha",
            fmt_pct(dash_df["Alpha 1Y"].mean(skipna=True)),
        )

    # All Waves snapshot
    display_df = dash_df.copy()
    pct_cols = [
        "Intraday Alpha",
        "Alpha 30D",
        "Alpha 60D",
        "Alpha 1Y",
        "Return 30D (Wave)",
        "Return 30D (BM)",
        "Return 60D (Wave)",
        "Return 60D (BM)",
        "Return 1Y (Wave)",
        "Return 1Y (BM)",
    ]

    for col in pct_cols:
        display_df[col] = display_df[col].apply(fmt_pct)

    display_df["Beta (≈60D)"] = display_df["Beta (≈60D)"].apply(fmt_beta)
    display_df["Exposure"] = display_df["Exposure"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:0.2f}"
    )

    st.markdown("### All Waves Snapshot")
    st.dataframe(
        display_df.set_index("Wave"),
        use_container_width=True,
    )

# ---------------------------------------------------------
# WAVE EXPLORER TAB
# ---------------------------------------------------------
with tab_explorer:
    st.subheader("Wave Explorer")

    col_sel, col_mode = st.columns([2, 1])
    with col_sel:
        wave_sel = st.selectbox("Select Wave", options=waves)
    with col_mode:
        st.write("")
        st.write(f"Mode: **{mode}**")

    try:
        metrics = get_wave_metrics(wave_sel, mode)
        dyn_holdings = get_wave_top_holdings_dynamic(wave_sel, mode, n=10)
    except Exception as e:
        st.error(f"Unable to compute performance for {wave_sel}: {e}")
        metrics, dyn_holdings = None, None

    if metrics is not None:
        bm_desc = describe_benchmark(metrics["benchmark"])

        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Intraday Alpha", fmt_pct(metrics["intraday_alpha_captured"]))
        with m2:
            st.metric("30-Day Alpha", fmt_pct(metrics["alpha_30d"]))
        with m3:
            st.metric("60-Day Alpha", fmt_pct(metrics["alpha_60d"]))
        with m4:
            st.metric("1-Year Alpha", fmt_pct(metrics["alpha_1y"]))

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("30D Return (Wave)", fmt_pct(metrics["return_30d_wave"]))
        with r2:
            st.metric("30D Return (Benchmark)", fmt_pct(metrics["return_30d_benchmark"]))
        with r3:
            st.metric("1Y Return (Wave)", fmt_pct(metrics["return_1y_wave"]))
        with r4:
            st.metric("1Y Return (Benchmark)", fmt_pct(metrics["return_1y_benchmark"]))

        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Beta (≈60D)", fmt_beta(metrics["beta_realized"]))
        with b2:
            st.metric(
                "Current Exposure",
                "—" if metrics["exposure_final"] is None else f"{metrics['exposure_final']:0.2f}",
            )
        with b3:
            vix_val = metrics.get("vix_last", None)
            regime = metrics.get("vol_regime", "unknown")
            if vix_val is None or (isinstance(vix_val, float) and np.isnan(vix_val)):
                st.metric("VIX / Regime", "—")
            else:
                st.metric("VIX / Regime", f"{vix_val:0.2f} ({regime})")

        st.caption(f"Benchmark: **{bm_desc}**")

        # 30-day history chart
        hist = metrics["history_30d"].copy()
        hist.index = pd.to_datetime(hist.index)
        st.markdown("#### 30-Day Value Curve")
        st.line_chart(
            hist[["wave_value", "benchmark_value"]],
            use_container_width=True,
        )

    # Top holdings with dynamic weights + Google Finance links
    st.markdown("#### Top 10 Dynamic Holdings")
    if dyn_holdings is not None and not dyn_holdings.empty:
        links = []
        for _, row in dyn_holdings.iterrows():
            ticker = row["ticker"]
            url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
            links.append(f"[{ticker}]({url})")

        holdings_display = dyn_holdings.copy()
        holdings_display["Ticker"] = links
        # Use dynamic_weight for display
        holdings_display["Dynamic Weight"] = holdings_display["dynamic_weight"].apply(fmt_pct)
        holdings_display.rename(
            columns={"company": "Company", "sector": "Sector"},
            inplace=True,
        )
        holdings_display = holdings_display[
            ["Ticker", "Company", "Sector", "Dynamic Weight"]
        ]
        st.dataframe(holdings_display, use_container_width=True)
    else:
        st.info("No dynamic holdings available for this Wave.")

# ---------------------------------------------------------
# ALPHA MATRIX TAB
# ---------------------------------------------------------
with tab_alpha:
    st.subheader("Alpha Matrix (All Waves)")

    rows = []
    for w in waves:
        try:
            m = get_wave_metrics(w, mode)
            rows.append(
                {
                    "Wave": w,
                    "Alpha 30D": m["alpha_30d"],
                    "Alpha 60D": m["alpha_60d"],
                    "Alpha 1Y": m["alpha_1y"],
                    "Wave 1Y Return": m["return_1y_wave"],
                    "Benchmark 1Y Return": m["return_1y_benchmark"],
                }
            )
        except Exception:
            rows.append(
                {
                    "Wave": w,
                    "Alpha 30D": np.nan,
                    "Alpha 60D": np.nan,
                    "Alpha 1Y": np.nan,
                    "Wave 1Y Return": np.nan,
                    "Benchmark 1Y Return": np.nan,
                }
            )

    alpha_df = pd.DataFrame(rows)

    # Sorting control
    sort_choice = st.selectbox(
        "Sort Waves by",
        options=["Alpha 30D", "Alpha 60D", "Alpha 1Y", "Wave 1Y Return"],
    )
    alpha_df = alpha_df.sort_values(sort_choice, ascending=False)

    display = alpha_df.copy()
    for col in ["Alpha 30D", "Alpha 60D", "Alpha 1Y", "Wave 1Y Return", "Benchmark 1Y Return"]:
        display[col] = display[col].apply(fmt_pct)

    st.dataframe(display.set_index("Wave"), use_container_width=True)

# ---------------------------------------------------------
# HISTORY TAB (30-Day)
# ---------------------------------------------------------
with tab_history:
    st.subheader("30-Day History — Wave & Benchmark")

    wave_hist = st.selectbox("Select Wave", options=waves, key="history_wave")

    try:
        metrics = get_wave_metrics(wave_hist, mode)
        hist = metrics["history_30d"].copy()
        hist.index = pd.to_datetime(hist.index)
    except Exception as e:
        st.error(f"Unable to load 30-day history for {wave_hist}: {e}")
        hist = None

    if hist is not None and not hist.empty:
        st.line_chart(hist[["wave_value", "benchmark_value"]], use_container_width=True)
    else:
        st.info("No 30-day history available for this Wave yet.")

# ---------------------------------------------------------
# ABOUT / DIAGNOSTICS TAB
# ---------------------------------------------------------
with tab_about:
    st.subheader("About / Diagnostics")

    st.markdown(
        """
        **Engine:** WAVES Intelligence™ Vector 2.6  
        **Weighting:** Dynamic risk-parity + signal tilt + VIX regime + mode overlay  
        **Benchmark Map:** v1.1 (style-correct ETF & blended benchmarks)  

        - Alpha is computed as: **Wave Return − Benchmark Return**  
        - Benchmarks are chosen to match sector, style, and cap profile for each Wave.  
        - Dynamic weights adjust per day based on volatility, momentum, VIX regime, and mode.  
        - Exposure overlay further scales risk up/down based on VIX.  
        """
    )

    diag_rows = []
    for w in waves:
        try:
            m = get_wave_metrics(w, mode)
            diag_rows.append(
                {
                    "Wave": w,
                    "Benchmark": describe_benchmark(m["benchmark"]),
                    "Beta (≈60D)": m["beta_realized"],
                    "Exposure": m["exposure_final"],
                    "VIX (Last)": m.get("vix_last", np.nan),
                    "Regime": m.get("vol_regime", "unknown"),
                }
            )
        except Exception as e:
            diag_rows.append(
                {
                    "Wave": w,
                    "Benchmark": f"ERROR: {e}",
                    "Beta (≈60D)": np.nan,
                    "Exposure": np.nan,
                    "VIX (Last)": np.nan,
                    "Regime": "error",
                }
            )

    diag_df = pd.DataFrame(diag_rows)
    diag_df["Beta (≈60D)"] = diag_df["Beta (≈60D)"].apply(fmt_beta)
    diag_df["Exposure"] = diag_df["Exposure"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:0.2f}"
    )
    diag_df["VIX (Last)"] = diag_df["VIX (Last)"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:0.2f}"
    )

    st.markdown("### Benchmark, Risk & Regime Diagnostics")
    st.dataframe(diag_df.set_index("Wave"), use_container_width=True)

    st.caption("Engine logs are written under ./logs/performance for reproducibility.")