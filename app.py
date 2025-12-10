import numpy as np
import pandas as pd
import streamlit as st

from waves_engine import WavesEngine, SmartSafeSweepEngine

# ---------------------------------------------------------
# Hard reset caches each run so we never see stale logic
# ---------------------------------------------------------
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.markdown(
    "<h1 style='font-size: 2.6rem;'>WAVES Intelligence™ Institutional Console</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "Live Wave Engine • Dynamic Weights • VIX-Aware Alpha Capture • "
    "SmartSafe™ Sweep • Benchmark-Relative Performance"
)

# ---------------------------------------------------------
# Helper formatting
# ---------------------------------------------------------
SMARTSAFE_EST_YIELD = 0.0425  # 4.25% estimated annual yield


def fmt_pct(x):
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "—"
    except Exception:
        pass
    return f"{x * 100:0.2f}%"


def fmt_beta(x):
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "—"
    except Exception:
        pass
    return f"{x:0.2f}"


def describe_benchmark(bm):
    if isinstance(bm, dict):
        parts = [f"{int(w * 100)}% {t}" for t, w in bm.items()]
        return " + ".join(parts)
    return str(bm)


# ---------------------------------------------------------
# Engine initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_engine() -> WavesEngine:
    return WavesEngine(list_path="list.csv", weights_path="wave_weights.csv", logs_root="logs")


engine = get_engine()
sweep_engine = SmartSafeSweepEngine(engine)

# ---------------------------------------------------------
# Cached metric helpers
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_wave_metrics(wave: str, mode: str):
    return engine.get_wave_performance(wave, mode=mode, log=False)


@st.cache_data(show_spinner=False)
def get_wave_top_holdings_dynamic(wave: str, mode: str, n: int = 10) -> pd.DataFrame:
    """Top holdings using dynamic current_weights merged with static metadata."""
    base = engine.get_wave_holdings(wave)
    if base is None or base.empty:
        return base

    metrics = get_wave_metrics(wave, mode)
    current_weights = metrics.get("current_weights")

    if current_weights is None or current_weights.empty:
        return pd.DataFrame()

    base = base.copy()
    base["dynamic_weight"] = base["ticker"].map(current_weights).astype(float)
    base["dynamic_weight"] = base["dynamic_weight"].fillna(0.0)
    base = base[base["dynamic_weight"] > 0.0]

    if base.empty:
        return base

    base = base.sort_values("dynamic_weight", ascending=False).head(n).reset_index(drop=True)
    return base


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Engine Controls")

mode = st.sidebar.radio(
    "Mode",
    options=["standard", "alpha-minus-beta", "private_logic"],
    format_func=lambda m: {
        "standard": "Standard",
        "alpha-minus-beta": "Alpha-Minus-Beta",
        "private_logic": "Private Logic™",
    }[m],
)

risk_level = st.sidebar.selectbox(
    "SmartSafe™ Household Risk Level (for Sweep Engine)",
    options=["Conservative", "Moderate", "Aggressive"],
    index=1,
)

if st.sidebar.button("Force Reload Engine & Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

waves = engine.get_wave_names()

# ---------------------------------------------------------
# Build high-level metrics table for dashboard & alpha matrix
# ---------------------------------------------------------
rows = []
for w in waves:
    try:
        m = get_wave_metrics(w, mode)
        rows.append(
            {
                "Wave": w,
                "Benchmark": describe_benchmark(m["benchmark"]),
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
                "VIX Last": m["vix_last"],
                "Regime": m["vol_regime"],
                "Turnover Annual": m["turnover_annual"],
                "Slippage Drag Annual": m["slippage_annual_drag"],
                "TLH Candidates": m["tlh_candidate_count"],
                "TLH Weight": m["tlh_candidate_weight"],
                "UAPV Unit Price": m["uapv_unit_price"],
            }
        )
    except Exception as e:
        rows.append(
            {
                "Wave": w,
                "Benchmark": f"ERROR: {e}",
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
                "VIX Last": np.nan,
                "Regime": "error",
                "Turnover Annual": np.nan,
                "Slippage Drag Annual": np.nan,
                "TLH Candidates": np.nan,
                "TLH Weight": np.nan,
                "UAPV Unit Price": np.nan,
            }
        )

metrics_df = pd.DataFrame(rows)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_dashboard, tab_explorer, tab_alpha, tab_sweep, tab_history, tab_about = st.tabs(
    [
        "Dashboard",
        "Wave Explorer",
        "Alpha Matrix",
        "SmartSafe™ / Sweep",
        "History (30-Day)",
        "About / Diagnostics",
    ]
)

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
with tab_dashboard:
    st.subheader(f"Dashboard — Mode: {mode}")

    valid = metrics_df.replace([np.inf, -np.inf], np.nan)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        avg_30d_alpha = valid["Alpha 30D"].mean(skipna=True)
        st.metric("Avg 30-Day Alpha", fmt_pct(avg_30d_alpha))
    with c2:
        avg_60d_alpha = valid["Alpha 60D"].mean(skipna=True)
        st.metric("Avg 60-Day Alpha", fmt_pct(avg_60d_alpha))
    with c3:
        avg_1y_alpha = valid["Alpha 1Y"].mean(skipna=True)
        st.metric("Avg 1-Year Alpha", fmt_pct(avg_1y_alpha))
    with c4:
        st.metric("SmartSafe™ Estimated Yield", fmt_pct(SMARTSAFE_EST_YIELD))

    display_df = metrics_df.copy()
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
    display_df["VIX Last"] = display_df["VIX Last"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:0.2f}"
    )
    display_df["UAPV Unit Price"] = display_df["UAPV Unit Price"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:0.4f}"
    )

    st.markdown("### All Waves Snapshot")
    st.dataframe(
        display_df.set_index("Wave")[
            [
                "Benchmark",
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
                "Beta (≈60D)",
                "Exposure",
                "VIX Last",
                "Regime",
                "UAPV Unit Price",
            ]
        ],
        use_container_width=True,
    )

# ---------------------------------------------------------
# WAVE EXPLORER
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
        m = get_wave_metrics(wave_sel, mode)
        dyn_holdings = get_wave_top_holdings_dynamic(wave_sel, mode, n=10)
    except Exception as e:
        st.error(f"Unable to compute performance for {wave_sel}: {e}")
        m, dyn_holdings = None, None

    if m is not None:
        bm_desc = describe_benchmark(m["benchmark"])

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Intraday Alpha", fmt_pct(m["intraday_alpha_captured"]))
        with m2:
            st.metric("30-Day Alpha", fmt_pct(m["alpha_30d"]))
        with m3:
            st.metric("60-Day Alpha", fmt_pct(m["alpha_60d"]))
        with m4:
            st.metric("1-Year Alpha", fmt_pct(m["alpha_1y"]))

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("30D Return (Wave)", fmt_pct(m["return_30d_wave"]))
        with r2:
            st.metric("30D Return (Benchmark)", fmt_pct(m["return_30d_benchmark"]))
        with r3:
            st.metric("1Y Return (Wave)", fmt_pct(m["return_1y_wave"]))
        with r4:
            st.metric("1Y Return (Benchmark)", fmt_pct(m["return_1y_benchmark"]))

        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Beta (≈60D)", fmt_beta(m["beta_realized"]))
        with b2:
            st.metric(
                "Current Exposure",
                "—" if m["exposure_final"] is None else f"{m['exposure_final']:0.2f}",
            )
        with b3:
            vix_val = m.get("vix_last", None)
            regime = m.get("vol_regime", "unknown")
            if vix_val is None or (isinstance(vix_val, float) and np.isnan(vix_val)):
                st.metric("VIX / Regime", "—")
            else:
                st.metric("VIX / Regime", f"{vix_val:0.2f} ({regime})")

        st.caption(f"Benchmark: **{bm_desc}**")

        hist = m["history_30d"].copy()
        hist.index = pd.to_datetime(hist.index)

        st.markdown("#### 30-Day Value Curve")
        st.line_chart(
            hist[["wave_value", "benchmark_value"]],
            use_container_width=True,
        )

        st.markdown("#### Top 10 Dynamic Holdings")
        if dyn_holdings is not None and not dyn_holdings.empty:
            links = []
            for _, row in dyn_holdings.iterrows():
                ticker = row["ticker"]
                # Default to NASDAQ suffix; still works for many large caps/ETFs
                url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
                links.append(f"[{ticker}]({url})")

            holdings_display = dyn_holdings.copy()
            holdings_display["Ticker"] = links
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
# ALPHA MATRIX
# ---------------------------------------------------------
with tab_alpha:
    st.subheader("Alpha Matrix (All Waves)")

    sort_choice = st.selectbox(
        "Sort Waves by",
        options=["Alpha 30D", "Alpha 60D", "Alpha 1Y", "Wave 1Y Return"],
        index=0,
    )

    alpha_df = metrics_df.copy()
    sort_map = {
        "Alpha 30D": "Alpha 30D",
        "Alpha 60D": "Alpha 60D",
        "Alpha 1Y": "Alpha 1Y",
        "Wave 1Y Return": "Return 1Y (Wave)",
    }
    alpha_df = alpha_df.sort_values(sort_map[sort_choice], ascending=False)

    disp = alpha_df[
        [
            "Wave",
            "Benchmark",
            "Alpha 30D",
            "Alpha 60D",
            "Alpha 1Y",
            "Return 1Y (Wave)",
            "Return 1Y (BM)",
        ]
    ].copy()

    for col in ["Alpha 30D", "Alpha 60D", "Alpha 1Y", "Return 1Y (Wave)", "Return 1Y (BM)"]:
        disp[col] = disp[col].apply(fmt_pct)

    st.dataframe(disp.set_index("Wave"), use_container_width=True)

# ---------------------------------------------------------
# SMARTSAFE / SWEEP TAB
# ---------------------------------------------------------
with tab_sweep:
    st.subheader("SmartSafe™ Sweep Engine — Household Allocation")

    st.markdown(
        """
        This view uses the **SmartSafeSweepEngine** on top of all active Waves to suggest a 
        household-level allocation between **risk Waves** and the **SmartSafe Wave**, based on:
        - Selected **Mode** (Standard / Alpha-Minus-Beta / Private Logic™)
        - Sidebar **Risk Level** (Conservative / Moderate / Aggressive)
        - Current **volatility regime** inferred from the Wave engine

        SmartSafe™ is modeled as a capital-preservation anchor with an estimated annual yield 
        of approximately **4.25%**, and risk Waves are blended to optimize risk-adjusted exposure.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Selected Mode", mode)
    with c2:
        st.metric("Household Risk Level", risk_level)

    st.metric("SmartSafe™ Estimated Yield", fmt_pct(SMARTSAFE_EST_YIELD))

    if st.button("Run SmartSafe™ Sweep Allocation"):
        try:
            alloc = sweep_engine.recommend_allocation(mode=mode, risk_level=risk_level)
            eval_result = sweep_engine.evaluate_portfolio(allocations=alloc, mode=mode)

            st.markdown("### Recommended Allocation by Wave")
            alloc_df = pd.DataFrame(
                [
                    {"Wave": w, "Allocation Weight": float(a)}
                    for w, a in alloc.items()
                ]
            ).sort_values("Allocation Weight", ascending=False)
            alloc_df["Allocation Weight"] = alloc_df["Allocation Weight"].apply(fmt_pct)
            st.dataframe(alloc_df.set_index("Wave"), use_container_width=True)

            st.markdown("### Blended Portfolio Alpha & Return (Approximate)")
            if eval_result:
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Blended 30D Alpha", fmt_pct(eval_result.get("alpha_30d_blended")))
                with m2:
                    st.metric("Blended 60D Alpha", fmt_pct(eval_result.get("alpha_60d_blended")))
                with m3:
                    st.metric("Blended 1Y Alpha", fmt_pct(eval_result.get("alpha_1y_blended")))

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric(
                        "Blended 30D Return (Wave)",
                        fmt_pct(eval_result.get("return_30d_wave_blended")),
                    )
                with r2:
                    st.metric(
                        "Blended 60D Return (Wave)",
                        fmt_pct(eval_result.get("return_60d_wave_blended")),
                    )
                with r3:
                    st.metric(
                        "Blended 1Y Return (Wave)",
                        fmt_pct(eval_result.get("return_1y_wave_blended")),
                    )
            else:
                st.info("Unable to compute blended portfolio metrics for the current configuration.")
        except Exception as e:
            st.error(f"Error running SmartSafe™ Sweep Engine: {e}")

# ---------------------------------------------------------
# HISTORY TAB
# ---------------------------------------------------------
with tab_history:
    st.subheader("30-Day History — Wave vs Benchmark")

    wave_hist = st.selectbox("Select Wave", options=waves, key="history_wave")

    try:
        m_hist = get_wave_metrics(wave_hist, mode)
        hist_df = m_hist["history_30d"].copy()
        hist_df.index = pd.to_datetime(hist_df.index)
    except Exception as e:
        st.error(f"Unable to load 30-day history for {wave_hist}: {e}")
        hist_df = None

    if hist_df is not None and not hist_df.empty:
        st.line_chart(hist_df[["wave_value", "benchmark_value"]], use_container_width=True)
    else:
        st.info("No 30-day history available for this Wave yet.")

# ---------------------------------------------------------
# ABOUT / DIAGNOSTICS
# ---------------------------------------------------------
with tab_about:
    st.subheader("About / Diagnostics")

    st.markdown(
        """
        **Engine:** WAVES Intelligence™ Vector 2.8+  
        **Weighting:** Dynamic risk-parity + signal tilt + VIX regime + mode overlay  
        **Benchmarks:** Custom blended ETF & index mappings for each Wave  
        **Alpha:** Wave return − Benchmark return (with VIX-gated exposure and slippage)  

        **SmartSafe™:**  
        - Modeled as a capital-preservation Wave with an estimated yield of ~4.25%  
        - Acts as a stable anchor in the SmartSafe™ Sweep Engine  
        - Designed as the cash-equivalent Wave for multi-Wave household allocations  

        - TLH signals show how many holdings are >10% below their 60-day high and how
          much of the Wave's dynamic weight they represent.  
        - Turnover and slippage drag are annualized approximations based on dynamic
          weights and a 5 bps slippage assumption.  
        - UAPV Unit Price is the current Wave value starting from 1.0, suitable as a
          live token/unit price for a UAPV-style ledger.  
        """
    )

    diag = metrics_df.copy()
    diag["Beta (≈60D)"] = diag["Beta (≈60D)"].apply(fmt_beta)
    diag["Exposure"] = diag["Exposure"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:0.2f}"
    )
    diag["VIX Last"] = diag["VIX Last"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:0.2f}"
    )
    diag["Turnover Annual"] = diag["Turnover Annual"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:0.2f}"
    )
    diag["Slippage Drag Annual"] = diag["Slippage Drag Annual"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:0.4f}"
    )
    diag["TLH Weight"] = diag["TLH Weight"].apply(fmt_pct)
    diag["UAPV Unit Price"] = diag["UAPV Unit Price"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:0.4f}"
    )

    st.markdown("### Engine & Risk Diagnostics")
    st.dataframe(
        diag.set_index("Wave")[
            [
                "Benchmark",
                "Beta (≈60D)",
                "Exposure",
                "VIX Last",
                "Regime",
                "Turnover Annual",
                "Slippage Drag Annual",
                "TLH Candidates",
                "TLH Weight",
                "UAPV Unit Price",
            ]
        ],
        use_container_width=True,
    )

    if st.checkbox("Show raw metrics dataframe"):
        st.dataframe(metrics_df, use_container_width=True)