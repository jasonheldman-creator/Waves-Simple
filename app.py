# app.py
# WAVES Intelligence™ — Institutional Console (Full Feature)
#
# Requirements:
# - Streamlit
# - pandas, numpy
# - yfinance
# - matplotlib
#
# Deploy:
# - Place app.py and waves_engine.py in repo root.
# - Streamlit Cloud: set main file to app.py

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import waves_engine as we


# -----------------------------
# Streamlit setup
# -----------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
    initial_sidebar_state="collapsed",  # mobile friendly
)

# Lightweight mobile styling
st.markdown(
    """
    <style>
      .block-container { padding-top: 0.8rem; padding-bottom: 1.2rem; }
      [data-testid="stMetricValue"] { font-size: 1.25rem; }
      [data-testid="stMetricLabel"] { font-size: 0.85rem; }
      .small-note { font-size: 0.85rem; opacity: 0.85; }
      .tight hr { margin: 0.6rem 0; }
      .stDataFrame { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("WAVES Intelligence™ — Institutional Console")
st.caption("Vector Engine • SmartSafe™ • Mode-separated outcomes • Full transparency")


# -----------------------------
# Utility
# -----------------------------

def fmt_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x*100:.2f}%"

def fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.2f}"

def google_finance_link(ticker: str) -> str:
    t = ticker.replace("-", ".")
    url = f"https://www.google.com/finance/quote/{t}:NASDAQ"
    # Many tickers are NYSE; Google still resolves often. Keep it simple + universal:
    url2 = f"https://www.google.com/finance/quote/{t}"
    return url2

def letter_grade(score: float) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    return "D"


@st.cache_data(ttl=180, show_spinner=False)
def cached_nav(wave: str, mode: str, days: int = 420):
    end = datetime.utcnow() + timedelta(days=1)
    start = datetime.utcnow() - timedelta(days=days)
    navres = we.compute_history_nav(wave, mode, start=start, end=end)
    ws = we.compute_wavescore(wave, mode, navres)
    return navres, ws

@st.cache_data(ttl=180, show_spinner=False)
def cached_overview(mode: str):
    waves = we.get_all_waves()
    rows = []

    for w in waves:
        navres, ws = cached_nav(w, mode, days=420)

        nav = navres.nav
        bnav = navres.bench_nav

        def period_return(series: pd.Series, d: int) -> float:
            if series is None or series.empty:
                return 0.0
            if len(series) < 3:
                return 0.0
            idx = max(0, len(series) - d - 1)
            base = series.iloc[idx]
            last = series.iloc[-1]
            if base == 0:
                return 0.0
            return float(last / base - 1.0)

        r1 = period_return(nav, 1)
        r30 = period_return(nav, 30)
        r60 = period_return(nav, 60)
        r365 = period_return(nav, 252)  # trading days approx

        br1 = period_return(bnav, 1)
        br30 = period_return(bnav, 30)
        br60 = period_return(bnav, 60)
        br365 = period_return(bnav, 252)

        rows.append({
            "Wave": w,
            "Mode": mode,
            "1D Return": r1,
            "30D Return": r30,
            "60D Return": r60,
            "365D Return": r365,
            "1D Alpha": r1 - br1,
            "30D Alpha": r30 - br30,
            "60D Alpha": r60 - br60,
            "365D Alpha": r365 - br365,
            "Vol (Ann)": navres.meta.get("ann_vol", 0.0),
            "MaxDD": navres.meta.get("max_drawdown", 0.0),
            "TE": navres.meta.get("tracking_error", 0.0),
            "IR": navres.meta.get("information_ratio", 0.0),
            "Beta": navres.meta.get("beta", 1.0),
            "WaveScore": ws.get("WaveScore", 0.0),
            "Grade": letter_grade(ws.get("WaveScore", 0.0)),
        })

    df = pd.DataFrame(rows)
    # Sort: best WaveScore first
    df = df.sort_values(["WaveScore", "365D Alpha", "30D Alpha"], ascending=False).reset_index(drop=True)
    return df


def plot_nav(nav: pd.Series, bnav: pd.Series, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if nav is not None and not nav.empty:
        ax.plot(nav.index, nav.values, label="Wave NAV")
    if bnav is not None and not bnav.empty:
        ax.plot(bnav.index, bnav.values, label="Benchmark NAV")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)


def correlation_heatmap(df_rets: pd.DataFrame, title: str):
    corr = df_rets.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


def rules_narrative(mode: str, overview_row: pd.Series) -> str:
    # Rules-based narrative panel (Vector OS Insight Layer)
    w = overview_row["Wave"]
    score = overview_row["WaveScore"]
    alpha30 = overview_row["30D Alpha"]
    alpha365 = overview_row["365D Alpha"]
    vol = overview_row["Vol (Ann)"]
    mdd = overview_row["MaxDD"]
    beta = overview_row["Beta"]

    risk_phrase = "contained" if vol < 0.18 and abs(mdd) < 0.18 else "elevated"
    alpha_phrase = "strong" if alpha30 > 0.01 else ("weak" if alpha30 < -0.01 else "flat")
    trend_phrase = "compounding" if alpha365 > 0.03 else ("lagging" if alpha365 < -0.03 else "neutral")

    return (
        f"**Vector OS Insight — {w} ({mode})**\n\n"
        f"- WaveScore: **{score:.1f} ({letter_grade(score)})**\n"
        f"- 30D alpha is **{alpha_phrase}** ({alpha30*100:.2f}%), with risk conditions **{risk_phrase}**.\n"
        f"- 365D alpha trend is **{trend_phrase}** ({alpha365*100:.2f}%).\n"
        f"- Vol: {vol*100:.1f}%, MaxDD: {mdd*100:.1f}%, Beta: {beta:.2f}\n\n"
        f"**Rule read:** If WaveScore stays above 80 and 30D alpha remains positive, maintain allocation. "
        f"If 30D alpha turns negative while drawdown widens, SmartSafe™ rules will increasingly sweep to cash proxies."
    )


# -----------------------------
# Controls (top)
# -----------------------------

c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
with c1:
    mode = st.selectbox("Mode", we.get_modes(), index=0)
with c2:
    wave = st.selectbox("Wave", we.get_all_waves(), index=0)
with c3:
    st.markdown(
        "<div class='small-note'>Tip: On iPhone, keep sidebar collapsed. Use the tabs below to navigate the full console.</div>",
        unsafe_allow_html=True
    )

tabs = st.tabs([
    "Overview",
    "Wave Detail",
    "Risk & WaveScore",
    "Benchmark Transparency",
    "Market Intel",
    "Factor Decomposition",
    "Vector OS Insight",
])


# -----------------------------
# OVERVIEW TAB
# -----------------------------
with tabs[0]:
    st.subheader("All Waves — Returns + Alpha Capture (Mode-separated)")
    with st.spinner("Computing overview (cached)…"):
        ov = cached_overview(mode)

    # show compact metrics row
    top = ov.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top WaveScore", f"{top['WaveScore']:.1f}", top["Wave"])
    m2.metric("Top 30D Alpha", fmt_pct(float(ov["30D Alpha"].max())), "best wave")
    m3.metric("Top 365D Alpha", fmt_pct(float(ov["365D Alpha"].max())), "best wave")
    m4.metric("Median Vol", fmt_pct(float(ov["Vol (Ann)"].median())), "annualized")

    show_cols = [
        "Wave",
        "1D Return", "1D Alpha",
        "30D Return", "30D Alpha",
        "60D Return", "60D Alpha",
        "365D Return", "365D Alpha",
        "Vol (Ann)", "MaxDD", "TE", "IR", "Beta",
        "WaveScore", "Grade",
    ]
    ov_show = ov[show_cols].copy()

    # format
    for c in ["1D Return","30D Return","60D Return","365D Return","1D Alpha","30D Alpha","60D Alpha","365D Alpha","Vol (Ann)","MaxDD","TE"]:
        ov_show[c] = ov_show[c].astype(float)
    for c in ["IR","Beta","WaveScore"]:
        ov_show[c] = ov_show[c].astype(float)

    st.dataframe(
        ov_show.style.format({
            "1D Return": "{:.2%}", "30D Return": "{:.2%}", "60D Return": "{:.2%}", "365D Return": "{:.2%}",
            "1D Alpha": "{:.2%}", "30D Alpha": "{:.2%}", "60D Alpha": "{:.2%}", "365D Alpha": "{:.2%}",
            "Vol (Ann)": "{:.2%}", "MaxDD": "{:.2%}", "TE": "{:.2%}",
            "IR": "{:.2f}", "Beta": "{:.2f}", "WaveScore": "{:.1f}",
        }),
        use_container_width=True,
        height=520
    )


# -----------------------------
# WAVE DETAIL TAB
# -----------------------------
with tabs[1]:
    st.subheader("Wave Detail — NAV vs Benchmark + Key Stats")
    with st.spinner("Computing NAV…"):
        navres, ws = cached_nav(wave, mode, days=420)

    cA, cB, cC, cD = st.columns(4)
    cA.metric("WaveScore", f"{ws.get('WaveScore',0.0):.1f}", letter_grade(ws.get("WaveScore",0.0)))
    cB.metric("Ann Vol", fmt_pct(navres.meta.get("ann_vol", 0.0)))
    cC.metric("Max Drawdown", fmt_pct(navres.meta.get("max_drawdown", 0.0)))
    cD.metric("Information Ratio", fmt_num(navres.meta.get("information_ratio", 0.0)))

    plot_nav(navres.nav, navres.bench_nav, f"{wave} ({mode}) — NAV vs Benchmark")

    st.markdown("---")

    st.subheader("Top Holdings (Top-10) — with Google Finance links")
    h = we.get_wave_holdings(wave, mode).copy()
    if h.empty:
        st.info("No holdings returned.")
    else:
        h["Google"] = h["ticker"].apply(lambda t: google_finance_link(t))
        top10 = h.head(10)[["ticker", "weight", "weight_effective", "Google"]].copy()
        st.dataframe(
            top10.style.format({"weight": "{:.2%}", "weight_effective": "{:.2%}"}),
            use_container_width=True,
            height=380
        )
        st.caption("Tip: Tap-and-hold the Google link to open in a new tab on iPhone.")


# -----------------------------
# RISK & WAVESCORE TAB
# -----------------------------
with tabs[2]:
    st.subheader("Risk Analytics + WaveScore™ Leaderboard")
    with st.spinner("Building leaderboard…"):
        ov = cached_overview(mode)

    st.markdown("### WaveScore™ Leaderboard")
    board = ov[["Wave", "WaveScore", "Grade", "IR", "Vol (Ann)", "MaxDD", "TE", "Beta", "30D Alpha", "365D Alpha"]].copy()
    st.dataframe(
        board.style.format({
            "WaveScore": "{:.1f}", "IR": "{:.2f}", "Vol (Ann)": "{:.2%}", "MaxDD": "{:.2%}", "TE": "{:.2%}",
            "Beta": "{:.2f}", "30D Alpha": "{:.2%}", "365D Alpha": "{:.2%}",
        }),
        use_container_width=True,
        height=520
    )

    st.markdown("---")
    st.markdown("### Selected Wave — Risk Ingredients")
    navres, ws = cached_nav(wave, mode, days=420)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Tracking Error", fmt_pct(navres.meta.get("tracking_error", 0.0)))
    r2.metric("Beta", fmt_num(navres.meta.get("beta", 1.0)), f"target {ws.get('BetaTarget', 1.0):.2f}")
    r3.metric("Gross Exposure", fmt_num(navres.meta.get("gross_exposure", 1.0)))
    r4.metric("WaveScore Grade", letter_grade(ws.get("WaveScore", 0.0)))

    subs = {
        "ReturnQuality": ws.get("ReturnQuality", 0.0),
        "RiskControl": ws.get("RiskControl", 0.0),
        "Consistency": ws.get("Consistency", 0.0),
        "Resilience": ws.get("Resilience", 0.0),
        "Efficiency": ws.get("Efficiency", 0.0),
        "TransparencyGov": ws.get("TransparencyGov", 0.0),
    }
    subs_df = pd.DataFrame([{"SubScore": k, "Score": float(v)} for k, v in subs.items()]).sort_values("Score", ascending=False)
    st.dataframe(subs_df.style.format({"Score": "{:.1f}"}), use_container_width=True, height=260)


# -----------------------------
# BENCHMARK TRANSPARENCY TAB
# -----------------------------
with tabs[3]:
    st.subheader("Benchmark Transparency — Composite + Static Fallbacks")
    bmix = we.get_benchmark_mix_table(wave).copy()
    st.dataframe(
        bmix.style.format({"weight": "{:.2%}"}),
        use_container_width=True,
        height=460
    )
    st.caption("Auto composite is the primary benchmark used for alpha unless its data is unavailable, in which case fallback_static is used.")


# -----------------------------
# MARKET INTEL TAB
# -----------------------------
with tabs[4]:
    st.subheader("Market Intel — Multi-Asset Dashboard + WAVES Reaction Snapshot")

    st.markdown("### Multi-Asset Proxies (simple read)")
    proxies = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "DBC", "^VIX"]
    try:
        import yfinance as yf
        end = datetime.utcnow() + timedelta(days=1)
        start = datetime.utcnow() - timedelta(days=180)
        px = yf.download(proxies, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)["Close"]
        px = px.dropna(how="all").ffill()
        rets = px.pct_change().fillna(0.0)

        last = px.iloc[-1]
        r5 = (px.iloc[-1] / px.iloc[-6] - 1.0) if len(px) > 6 else pd.Series({c: 0.0 for c in px.columns})
        r30 = (px.iloc[-1] / px.iloc[-31] - 1.0) if len(px) > 31 else pd.Series({c: 0.0 for c in px.columns})

        out = pd.DataFrame({
            "Last": last,
            "5D": r5,
            "30D": r30,
            "Vol(60D ann)": rets.rolling(60).std(ddof=0).iloc[-1] * np.sqrt(252.0) if len(rets) > 60 else 0.0
        })
        st.dataframe(out.style.format({"Last": "{:.2f}", "5D": "{:.2%}", "30D": "{:.2%}", "Vol(60D ann)": "{:.2%}"}), use_container_width=True, height=340)
    except Exception as e:
        st.warning(f"Market intel data not available right now: {e}")

    st.markdown("---")
    st.markdown("### WAVES Reaction Snapshot (how Waves behaved vs benchmark)")
    with st.spinner("Computing reaction snapshot…"):
        ov = cached_overview(mode)
    snap = ov[["Wave", "1D Alpha", "30D Alpha", "60D Alpha", "IR", "Vol (Ann)", "MaxDD", "WaveScore"]].copy()
    st.dataframe(
        snap.style.format({"1D Alpha": "{:.2%}", "30D Alpha": "{:.2%}", "60D Alpha": "{:.2%}", "IR": "{:.2f}", "Vol (Ann)": "{:.2%}", "MaxDD": "{:.2%}", "WaveScore": "{:.1f}"}),
        use_container_width=True,
        height=480
    )


# -----------------------------
# FACTOR DECOMPOSITION TAB
# -----------------------------
with tabs[5]:
    st.subheader("Factor Decomposition — Betas + Correlation Matrix (heatmap)")

    st.markdown("### Selected Wave — Simple Factor Betas (proxy)")
    # Proxy factors: SPY (market), QQQ (growth/tech), IWM (size), TLT (rates), GLD (gold/defensive), DBC (commodities)
    factors = ["SPY", "QQQ", "IWM", "TLT", "GLD", "DBC"]
    try:
        import yfinance as yf
        navres, ws = cached_nav(wave, mode, days=420)

        end = datetime.utcnow() + timedelta(days=1)
        start = datetime.utcnow() - timedelta(days=420)
        px = yf.download(factors, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)["Close"]
        px = px.dropna(how="all").ffill()
        frets = px.pct_change().fillna(0.0)

        # align with portfolio returns
        pr = navres.port_rets.copy()
        idx = pr.index.intersection(frets.index)
        pr = pr.loc[idx]
        X = frets.loc[idx, factors].values
        y = pr.values

        # OLS with intercept
        X2 = np.column_stack([np.ones(len(X)), X])
        beta_hat, *_ = np.linalg.lstsq(X2, y, rcond=None)
        # beta_hat[0] intercept, rest factors
        betas = beta_hat[1:]
        bdf = pd.DataFrame({"Factor": factors, "Beta": betas}).sort_values("Beta", ascending=False).reset_index(drop=True)
        st.dataframe(bdf.style.format({"Beta": "{:.3f}"}), use_container_width=True, height=280)

        st.markdown("### Correlation Matrix (factors)")
        correlation_heatmap(frets[factors].tail(120), "Factor Correlations (last ~120 trading days)")
    except Exception as e:
        st.warning(f"Factor decomposition unavailable right now: {e}")


# -----------------------------
# VECTOR OS INSIGHT TAB
# -----------------------------
with tabs[6]:
    st.subheader("Vector OS Insight Layer — Rules-based Narrative Panel")
    with st.spinner("Generating narrative…"):
        ov = cached_overview(mode)
    row = ov[ov["Wave"] == wave].iloc[0]
    st.markdown(rules_narrative(mode, row))

    st.markdown("---")
    st.markdown("### Current Holdings Overlay Notes")
    h = we.get_wave_holdings(wave, mode)
    if h.empty:
        st.info("No holdings returned.")
    else:
        # notes are repeated per row; show once
        note = h["notes"].iloc[0] if "notes" in h.columns and len(h) > 0 else ""
        st.write(note if note else "No overlay notes.")