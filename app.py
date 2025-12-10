# app.py
# WAVES Intelligence™ Institutional Console
# Uses waves_engine.WavesEngine v2.1

import streamlit as st
import pandas as pd

from waves_engine import (
    build_engine,
    RISK_WAVES,
    SMARTSAFE_WAVE,
)


# --------- STREAMLIT CONFIG & CACHE RESET ---------

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

# Force a fresh run every deploy/reboot (ignore stale cache)
try:
    st.cache_data.clear()
except Exception:
    pass


@st.cache_data(show_spinner=True)
def load_engine():
    eng = build_engine()
    return eng


# --------- LOAD ENGINE ---------

eng = load_engine()
metrics = eng.metrics_by_wave
wave_nav = eng.wave_nav
bm_nav = eng.benchmark_nav

risk_waves = [w for w in RISK_WAVES if w in metrics]


# --------- UI LAYOUT ---------

st.title("WAVES Intelligence™ Institutional Console")
st.caption(
    "Live Wave Engine • Alpha Capture • Benchmark-Relative Performance"
)

tabs = st.tabs(
    [
        "Dashboard",
        "Wave Explorer",
        "Alpha Matrix",
        "History (30-Day)",
        "SmartSafe",
    ]
)

# ---------- DASHBOARD TAB ----------

with tabs[0]:
    st.subheader("Dashboard — Mode: standard")

    col1, col2, col3 = st.columns(3)

    # Aggregate alpha across risk Waves
    alpha_30_vals = [
        m.alpha_30d for w, m in metrics.items()
        if w in risk_waves and m.alpha_30d is not None
    ]
    alpha_60_vals = [
        m.alpha_60d for w, m in metrics.items()
        if w in risk_waves and m.alpha_60d is not None
    ]
    alpha_1y_vals = [
        m.alpha_1y for w, m in metrics.items()
        if w in risk_waves and m.alpha_1y is not None
    ]

    avg30 = sum(alpha_30_vals) / len(alpha_30_vals) if alpha_30_vals else 0.0
    avg60 = sum(alpha_60_vals) / len(alpha_60_vals) if alpha_60_vals else 0.0
    avg1y = sum(alpha_1y_vals) / len(alpha_1y_vals) if alpha_1y_vals else 0.0

    col1.metric("Avg 30-Day Alpha (All Risk Waves)", f"{avg30:0.2f}%")
    col2.metric("Avg 60-Day Alpha (All Risk Waves)", f"{avg60:0.2f}%")
    col3.metric("Avg 1-Year Alpha (All Risk Waves)", f"{avg1y:0.2f}%")

    st.markdown("---")
    st.markdown("### All Waves Snapshot")

    rows = []
    for wave in risk_waves:
        m = metrics[wave]
        rows.append(
            {
                "Wave": wave,
                "Alpha 30D (%)": m.alpha_30d,
                "Alpha 60D (%)": m.alpha_60d,
                "Alpha 1Y (%)": m.alpha_1y,
                "1Y Wave Return (%)": m.wave_1y,
                "1Y Benchmark Return (%)": m.benchmark_1y,
            }
        )

    df = pd.DataFrame(rows).set_index("Wave")
    st.dataframe(df, use_container_width=True)


# ---------- WAVE EXPLORER TAB ----------

with tabs[1]:
    st.subheader("Wave Explorer")

    selected_wave = st.selectbox("Select Wave", risk_waves)

    m = metrics[selected_wave]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha 30D", f"{(m.alpha_30d or 0.0):0.2f}%")
    c2.metric("Alpha 60D", f"{(m.alpha_60d or 0.0):0.2f}%")
    c3.metric("Alpha 1Y", f"{(m.alpha_1y or 0.0):0.2f}%")
    c4.metric("1Y Wave Return", f"{(m.wave_1y or 0.0):0.2f}%")

    st.markdown("#### Wave vs Benchmark — 1Y Value")

    nav = wave_nav.get(selected_wave)
    bm = bm_nav.get(selected_wave)

    if nav is not None and not nav.empty:
        chart_df = pd.DataFrame({"Wave": nav})
        if bm is not None and not bm.empty:
            # Align indices
            chart_df = chart_df.join(bm.rename("Benchmark"), how="inner")
        st.line_chart(chart_df)
    else:
        st.info("No NAV data available for this Wave.")

    # Top 10 holdings with Google Finance links
    st.markdown("#### Top 10 Holdings")

    if selected_wave in eng.wave_weights:
        s = eng.wave_weights[selected_wave].sort_values(ascending=False).head(10)
        holdings_rows = []
        for ticker, w in s.items():
            url = f"https://www.google.com/finance?q={ticker}"
            link = f"[{ticker}]({url})"
            holdings_rows.append(
                {"Ticker": link, "Weight (%)": float(w * 100.0)}
            )
        hold_df = pd.DataFrame(holdings_rows)
        st.markdown(
            hold_df.to_markdown(index=False),
            unsafe_allow_html=True,
        )
    else:
        st.write("No holdings data for this Wave.")


# ---------- ALPHA MATRIX TAB ----------

with tabs[2]:
    st.subheader("Alpha Matrix (All Risk Waves)")

    rows = []
    for wave in risk_waves:
        m = metrics[wave]
        rows.append(
            {
                "Wave": wave,
                "Alpha 30D (%)": m.alpha_30d,
                "Alpha 60D (%)": m.alpha_60d,
                "Alpha 1Y (%)": m.alpha_1y,
                "1Y Wave Return (%)": m.wave_1y,
                "1Y Benchmark Return (%)": m.benchmark_1y,
            }
        )

    df = pd.DataFrame(rows)

    sort_field = st.selectbox(
        "Sort Waves by",
        ["Alpha 30D (%)", "Alpha 60D (%)", "Alpha 1Y (%)", "1Y Wave Return (%)"],
        index=0,
    )

    df_sorted = df.sort_values(sort_field, ascending=False).set_index("Wave")
    st.dataframe(df_sorted, use_container_width=True)


# ---------- HISTORY (30-DAY) TAB ----------

with tabs[3]:
    st.subheader("History (Last 30 Trading Days)")

    for wave in risk_waves:
        nav = wave_nav.get(wave)
        bm = bm_nav.get(wave)
        if nav is None or nav.empty:
            continue

        sub_nav = nav.iloc[-WINDOW_30D:]
        chart_df = pd.DataFrame({"Wave": sub_nav})
        if bm is not None and not bm.empty:
            sub_bm = bm.reindex(sub_nav.index).dropna()
            if not sub_bm.empty:
                chart_df = chart_df.join(sub_bm.rename("Benchmark"), how="inner")

        st.markdown(f"#### {wave}")
        st.line_chart(chart_df)


# ---------- SMARTSAFE TAB ----------

with tabs[4]:
    st.subheader("SmartSafe Wave — Cash & Safety Engine")

    m = metrics.get(SMARTSAFE_WAVE)
    nav = wave_nav.get(SMARTSAFE_WAVE)

    col1, col2 = st.columns(2)
    if m and m.wave_1y is not None:
        col1.metric("Estimated 1-Year Yield (SmartSafe Wave)", f"{m.wave_1y:0.2f}%")
    else:
        col1.metric("Estimated 1-Year Yield (SmartSafe Wave)", "—")

    if nav is not None and not nav.empty:
        col2.metric("History Length (trading days)", f"{len(nav)}")
    else:
        col2.metric("History Length (trading days)", "—")

    st.markdown("---")
    st.markdown("#### SmartSafe NAV History")
    if nav is not None and not nav.empty:
        st.line_chart(nav)
    else:
        st.info("No SmartSafe data available yet.")

    st.markdown(
        """
**SmartSafe Mechanics (Summary)**  

- Short-duration Treasury ETFs (SGOV, BIL, SHV) blended for yield + safety  
- Dynamic allocation within that ladder based on relative yields  
- VIX-driven risk-off overlay at the *household* level (implemented in higher OS layer)  
- Tokenization-ready via UAPV™ units and daily compounding NAV  
        """
    )