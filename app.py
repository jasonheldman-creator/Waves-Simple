# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# + Volatility Regime Attribution + Diagnostics + Recommendations + Session Overrides Preview

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import waves_engine as we

try:
    import yfinance as yf
except ImportError:
    yf = None


# ------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Caching
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_spy_vix(days: int = 365) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)

    data = yf.download(
        tickers=["SPY", "^VIX"],
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)
    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    if len(data) > days:
        data = data.iloc[-days:]
    return data


@st.cache_data(show_spinner=False)
def fetch_market_assets(days: int = 365) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 10)
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"]

    data = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)
    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    if len(data) > days:
        data = data.iloc[-days:]
    return data


@st.cache_data(show_spinner=False)
def compute_wave_history(
    wave_name: str,
    mode: str,
    days: int = 365,
    overrides: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    try:
        return we.compute_history_nav(wave_name, mode=mode, days=days, overrides=overrides)
    except Exception:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret", "vix", "vix_regime", "market_regime"])


@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    try:
        return we.get_benchmark_mix_table()
    except Exception:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    try:
        return we.get_wave_holdings(wave_name)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


# ------------------------------------------------------------
# Analytics helpers
# ------------------------------------------------------------

def compute_return_from_nav(nav: pd.Series, window: int) -> float:
    if nav is None or len(nav) < 2:
        return float("nan")
    if len(nav) < window:
        window = len(nav)
    if window < 2:
        return float("nan")
    sub = nav.iloc[-window:]
    start = float(sub.iloc[0])
    end = float(sub.iloc[-1])
    if start <= 0:
        return float("nan")
    return (end / start) - 1.0


def annualized_vol(daily_ret: pd.Series) -> float:
    if daily_ret is None or len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(252))


def max_drawdown(nav: pd.Series) -> float:
    if nav is None or len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    return float(dd.min())


def tracking_error(daily_wave: pd.Series, daily_bm: pd.Series) -> float:
    if daily_wave is None or daily_bm is None:
        return float("nan")
    diff = (daily_wave - daily_bm).dropna()
    if len(diff) < 2:
        return float("nan")
    return float(diff.std() * np.sqrt(252))


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    if nav_wave is None or nav_bm is None or len(nav_wave) < 2:
        return float("nan")
    if te is None or te <= 0:
        return float("nan")
    ret_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
    ret_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
    excess = ret_wave - ret_bm
    return float(excess / te)


def simple_ret(series: pd.Series, window: int) -> float:
    if series is None or len(series) < 2:
        return float("nan")
    if len(series) < window:
        window = len(series)
    if window < 2:
        return float("nan")
    sub = series.iloc[-window:]
    return float(sub.iloc[-1] / sub.iloc[0] - 1.0)


def regress_factors(wave_ret: pd.Series, factor_ret: pd.DataFrame) -> dict:
    df = pd.concat([wave_ret.rename("wave"), factor_ret], axis=1).dropna()
    if df.shape[0] < 60 or df.shape[1] < 2:
        return {col: float("nan") for col in factor_ret.columns}
    y = df["wave"].values
    X = df[factor_ret.columns].values
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    except Exception:
        return {col: float("nan") for col in factor_ret.columns}
    betas = beta[1:]
    return {col: float(b) for col, b in zip(factor_ret.columns, betas)}


# ------------------------------------------------------------
# WaveScore proto (same as your current logic)
# ------------------------------------------------------------

def _grade_from_score(score: float) -> str:
    if math.isnan(score):
        return "N/A"
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    return "D"


def compute_wavescore_for_all_waves(all_waves: list[str], mode: str, days: int = 365, overrides: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    rows = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days, overrides=overrides)
        if hist.empty or len(hist) < 20:
            rows.append({"Wave": wave, "WaveScore": np.nan, "Grade": "N/A",
                        "Return Quality": np.nan, "Risk Control": np.nan, "Consistency": np.nan,
                        "Resilience": np.nan, "Efficiency": np.nan, "Transparency": 10.0,
                        "IR_365D": np.nan, "Alpha_365D": np.nan})
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        vol_wave = annualized_vol(wave_ret)
        vol_bm = annualized_vol(bm_ret)
        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)

        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)

        hit_rate = float((wave_ret >= bm_ret).mean()) if len(wave_ret) > 0 else np.nan

        if len(nav_wave) > 1:
            trough = float(nav_wave.min())
            peak = float(nav_wave.max())
            last = float(nav_wave.iloc[-1])
            if peak > trough and trough > 0:
                recovery_frac = float((last - trough) / (peak - trough))
                recovery_frac = float(np.clip(recovery_frac, 0.0, 1.0))
            else:
                recovery_frac = np.nan
        else:
            recovery_frac = np.nan

        vol_ratio = (vol_wave / vol_bm) if (vol_bm and not math.isnan(vol_bm)) else np.nan

        # Return Quality (0–25)
        rq_ir = 0.0 if math.isnan(ir) else float(np.clip(ir, 0.0, 1.5) / 1.5 * 15.0)
        rq_alpha = 0.0 if math.isnan(alpha_365) else float(np.clip((alpha_365 + 0.05) / 0.15, 0.0, 1.0) * 10.0)
        return_quality = float(np.clip(rq_ir + rq_alpha, 0.0, 25.0))

        # Risk Control (0–25)
        if math.isnan(vol_ratio):
            risk_control = 0.0
        else:
            penalty = max(0.0, abs(vol_ratio - 0.9) - 0.1)
            risk_control = float(np.clip(1.0 - penalty / 0.6, 0.0, 1.0) * 25.0)

        # Consistency (0–15)
        consistency = 0.0 if math.isnan(hit_rate) else float(np.clip(hit_rate, 0.0, 1.0) * 15.0)

        # Resilience (0–10)
        if math.isnan(recovery_frac) or math.isnan(mdd_wave) or math.isnan(mdd_bm):
            resilience = 0.0
        else:
            rec_part = float(np.clip(recovery_frac, 0.0, 1.0) * 6.0)
            dd_edge = (mdd_bm - mdd_wave)
            dd_part = float(np.clip((dd_edge + 0.10) / 0.20, 0.0, 1.0) * 4.0)
            resilience = float(np.clip(rec_part + dd_part, 0.0, 10.0))

        # Efficiency (0–15)
        if math.isnan(te):
            efficiency = 0.0
        else:
            efficiency = float(np.clip((0.25 - te) / 0.20, 0.0, 1.0) * 15.0)

        transparency = 10.0

        total = float(np.clip(return_quality + risk_control + consistency + resilience + efficiency + transparency, 0.0, 100.0))
        grade = _grade_from_score(total)

        rows.append({
            "Wave": wave,
            "WaveScore": total,
            "Grade": grade,
            "Return Quality": return_quality,
            "Risk Control": risk_control,
            "Consistency": consistency,
            "Resilience": resilience,
            "Efficiency": efficiency,
            "Transparency": transparency,
            "IR_365D": ir,
            "Alpha_365D": alpha_365,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("Wave")


# ------------------------------------------------------------
# NEW: Volatility Regime Attribution + Diagnostics + Recos
# ------------------------------------------------------------

def vol_regime_attribution(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Returns table by VIX regime: count, wave return, bm return, alpha
    """
    if hist is None or hist.empty or "vix_regime" not in hist.columns:
        return pd.DataFrame(columns=["Regime", "Days", "Wave Return", "BM Return", "Alpha"])

    df = hist.copy()
    df = df.dropna(subset=["wave_ret", "bm_ret"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["Regime", "Days", "Wave Return", "BM Return", "Alpha"])

    out_rows = []
    for regime, g in df.groupby("vix_regime"):
        if len(g) < 2:
            continue
        wave_nav = (1.0 + g["wave_ret"]).cumprod()
        bm_nav = (1.0 + g["bm_ret"]).cumprod()
        wret = float(wave_nav.iloc[-1] / wave_nav.iloc[0] - 1.0) if len(wave_nav) > 1 else np.nan
        bret = float(bm_nav.iloc[-1] / bm_nav.iloc[0] - 1.0) if len(bm_nav) > 1 else np.nan
        out_rows.append({
            "Regime": str(regime),
            "Days": int(len(g)),
            "Wave Return": wret,
            "BM Return": bret,
            "Alpha": wret - bret if (pd.notna(wret) and pd.notna(bret)) else np.nan,
        })

    if not out_rows:
        return pd.DataFrame(columns=["Regime", "Days", "Wave Return", "BM Return", "Alpha"])

    out = pd.DataFrame(out_rows).sort_values("Days", ascending=False)
    return out


def run_diagnostics(selected_wave: str, mode: str, hist: pd.DataFrame, holdings_df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Returns list of diagnostics {level, title, detail}
    """
    diags = []

    if hist is None or hist.empty or len(hist) < 30:
        diags.append({"level": "FAIL", "title": "Insufficient history", "detail": "Not enough data points to evaluate this Wave reliably."})
        return diags

    # Data integrity
    if hist["wave_ret"].isna().mean() > 0.05 or hist["bm_ret"].isna().mean() > 0.05:
        diags.append({"level": "WARN", "title": "Missing returns detected", "detail": "Some daily returns are missing/NaN. Could indicate price gaps or ticker issues."})
    if "vix" in hist.columns and hist["vix"].isna().mean() > 0.10:
        diags.append({"level": "WARN", "title": "VIX series missing frequently", "detail": "VIX is missing often; volatility regime attribution may be unstable."})

    # Concentration
    if holdings_df is not None and not holdings_df.empty and "Weight" in holdings_df.columns:
        top1 = float(holdings_df["Weight"].max())
        top3 = float(holdings_df.sort_values("Weight", ascending=False)["Weight"].head(3).sum())
        if top1 > 0.20:
            diags.append({"level": "WARN", "title": "High single-name concentration", "detail": f"Top holding is {top1*100:0.1f}% of the Wave."})
        if top3 > 0.45:
            diags.append({"level": "WARN", "title": "High top-3 concentration", "detail": f"Top-3 holdings sum to {top3*100:0.1f}%."})

    # Alpha stability (hit-rate)
    hit = float((hist["wave_ret"] >= hist["bm_ret"]).mean())
    if hit < 0.48:
        diags.append({"level": "WARN", "title": "Low daily hit-rate vs benchmark", "detail": f"Hit-rate is {hit*100:0.1f}% (days beating benchmark). Consider more defensive gating or reducing noise."})

    # Drawdown edge
    nav_wave = hist["wave_nav"]
    nav_bm = hist["bm_nav"]
    mdd_w = max_drawdown(nav_wave)
    mdd_b = max_drawdown(nav_bm)
    if pd.notna(mdd_w) and pd.notna(mdd_b) and (mdd_w < mdd_b - 0.04):
        diags.append({"level": "WARN", "title": "Wave drawdown deeper than benchmark", "detail": f"MaxDD Wave {mdd_w*100:0.1f}% vs BM {mdd_b*100:0.1f}%."})

    return diags


def recommend_overrides(selected_wave: str, mode: str, hist: pd.DataFrame) -> List[Tuple[str, Dict[str, Any], str]]:
    """
    Returns list of (name, overrides_dict, rationale).
    Session-only safe knobs.
    """
    recos: List[Tuple[str, Dict[str, Any], str]] = []
    if hist is None or hist.empty or len(hist) < 60:
        return recos

    # Regime weakness: if Stress regime alpha is negative, suggest stronger safe allocation in stress
    vr = vol_regime_attribution(hist)
    if not vr.empty:
        stress = vr[vr["Regime"].astype(str) == "Stress"]
        if not stress.empty:
            a = float(stress["Alpha"].iloc[0]) if pd.notna(stress["Alpha"].iloc[0]) else np.nan
            if pd.notna(a) and a < 0:
                recos.append((
                    "Increase SmartSafe in Panic/Downtrend (defensive)",
                    {
                        "REGIME_GATING": {
                            mode: {
                                "panic": min(0.90, 1.15 * we.get_engine_default_params()["REGIME_GATING"][mode]["panic"]),
                                "downtrend": min(0.75, 1.15 * we.get_engine_default_params()["REGIME_GATING"][mode]["downtrend"]),
                            }
                        }
                    },
                    "Stress-regime alpha is negative. Increasing baseline SmartSafe gating in risk-off regimes often improves drawdown/consistency."
                ))

    # High vol ratio: if realized vol is much higher than benchmark, suggest tighter exposure caps
    wave_vol = annualized_vol(hist["wave_ret"])
    bm_vol = annualized_vol(hist["bm_ret"])
    if pd.notna(wave_vol) and pd.notna(bm_vol) and bm_vol > 0:
        ratio = wave_vol / bm_vol
        if ratio > 1.30:
            recos.append((
                "Tighten exposure cap (reduce hot risk)",
                {
                    "MODE_EXPOSURE_CAPS": {
                        mode: (we.get_engine_default_params()["MODE_EXPOSURE_CAPS"][mode][0], max(0.95, we.get_engine_default_params()["MODE_EXPOSURE_CAPS"][mode][1] - 0.10))
                    }
                },
                f"Wave volatility is {ratio:0.2f}× benchmark. Reducing max exposure cap can stabilize alpha and reduce drawdowns."
            ))

    # Low hit rate: suggest slightly stronger VIX safe multiplier (especially for AMB)
    hit = float((hist["wave_ret"] >= hist["bm_ret"]).mean())
    if hit < 0.48:
        mult = we.get_engine_default_params()["VIX_SAFE_MODE_MULT"].get(mode, 1.0)
        recos.append((
            "Increase VIX safe allocation sensitivity",
            {"VIX_SAFE_MODE_MULT": {mode: min(2.0, mult + 0.15)}},
            "Daily hit-rate vs benchmark is low. More responsive VIX-to-SmartSafe behavior can reduce noisy underperformance days."
        ))

    return recos


def preview_alpha_impact(wave: str, mode: str, base_days: int, overrides: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Returns preview metrics comparing baseline vs overridden:
    alpha_365, alpha_30, wavescore
    """
    base = compute_wave_history(wave, mode=mode, days=base_days, overrides=None)
    test = compute_wave_history(wave, mode=mode, days=base_days, overrides=overrides)

    def _alpha(df: pd.DataFrame, window: int) -> float:
        if df is None or df.empty or len(df) < 2:
            return np.nan
        nav_w = df["wave_nav"]
        nav_b = df["bm_nav"]
        rw = compute_return_from_nav(nav_w, window=min(window, len(nav_w)))
        rb = compute_return_from_nav(nav_b, window=min(window, len(nav_b)))
        return float(rw - rb) if (pd.notna(rw) and pd.notna(rb)) else np.nan

    out = {
        "Base Alpha 30D": _alpha(base, 30),
        "Test Alpha 30D": _alpha(test, 30),
        "Base Alpha 365D": _alpha(base, 365),
        "Test Alpha 365D": _alpha(test, 365),
    }
    return out


# ------------------------------------------------------------
# Sidebar + session overrides
# ------------------------------------------------------------

all_waves = we.get_all_waves()
all_modes = we.get_modes()

if "session_overrides" not in st.session_state:
    st.session_state["session_overrides"] = None  # type: ignore[assignment]
if "use_overrides" not in st.session_state:
    st.session_state["use_overrides"] = False

with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Mini Bloomberg Console • Vector OS™")

    mode = st.selectbox("Mode", all_modes, index=0)
    selected_wave = st.selectbox("Select Wave", all_waves, index=0)

    st.markdown("---")
    st.markdown("**Display settings**")
    nav_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

    st.markdown("---")
    st.markdown("**Optimization Layer**")
    st.session_state["use_overrides"] = st.toggle(
        "Apply session-only overrides (preview/test)",
        value=st.session_state["use_overrides"],
        help="This does NOT write to disk. Reboot resets to default.",
    )

    if st.button("Reset overrides (session)"):
        st.session_state["session_overrides"] = None
        st.session_state["use_overrides"] = False
        st.success("Session overrides reset.")


def active_overrides() -> Optional[Dict[str, Any]]:
    if st.session_state.get("use_overrides") and st.session_state.get("session_overrides"):
        return st.session_state["session_overrides"]
    return None


# ------------------------------------------------------------
# Header + tabs
# ------------------------------------------------------------

st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders")

tab_console, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console", "Market Intel", "Factor Decomposition", "Vector OS Insight Layer"]
)


# ============================================================
# TAB 1: Console
# ============================================================

with tab_console:
    st.subheader("Market Regime Monitor — SPY vs VIX")

    spy_vix = fetch_spy_vix(days=nav_days)

    if spy_vix.empty or "SPY" not in spy_vix.columns or "^VIX" not in spy_vix.columns:
        st.warning("Unable to load SPY/VIX data at the moment.")
    else:
        spy = spy_vix["SPY"].copy()
        vix = spy_vix["^VIX"].copy()
        spy_norm = (spy / spy.iloc[0] * 100.0) if len(spy) else spy

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spy_vix.index, y=spy_norm, name="SPY (Index = 100)", mode="lines"))
        fig.add_trace(go.Scatter(x=spy_vix.index, y=vix, name="VIX Level", mode="lines", yaxis="y2"))

        fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Date"),
            yaxis=dict(title="SPY (Indexed)", rangemode="tozero"),
            yaxis2=dict(title="VIX", overlaying="y", side="right", rangemode="tozero"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Portfolio-Level Overview (All Waves)
    st.subheader("Portfolio-Level Overview (All Waves)")

    ov_rows = []
    for w in all_waves:
        hist_365 = compute_wave_history(w, mode=mode, days=365, overrides=active_overrides())
        if hist_365.empty or len(hist_365) < 2:
            ov_rows.append({"Wave": w, "365D Return": np.nan, "365D Alpha": np.nan, "30D Return": np.nan, "30D Alpha": np.nan})
            continue
        nav_w = hist_365["wave_nav"]
        nav_b = hist_365["bm_nav"]

        r365w = compute_return_from_nav(nav_w, window=len(nav_w))
        r365b = compute_return_from_nav(nav_b, window=len(nav_b))
        a365 = r365w - r365b

        r30w = compute_return_from_nav(nav_w, window=min(30, len(nav_w)))
        r30b = compute_return_from_nav(nav_b, window=min(30, len(nav_b)))
        a30 = r30w - r30b

        ov_rows.append({"Wave": w, "365D Return": r365w, "365D Alpha": a365, "30D Return": r30w, "30D Alpha": a30})

    overview_df = pd.DataFrame(ov_rows)
    fmt = overview_df.copy()
    for c in ["365D Return", "365D Alpha", "30D Return", "30D Alpha"]:
        fmt[c] = fmt[c].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
    st.dataframe(fmt.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    # Multi-Window Alpha Capture
    st.subheader(f"Multi-Window Alpha Capture (All Waves · Mode = {mode})")

    alpha_rows = []
    for w in all_waves:
        hist_365 = compute_wave_history(w, mode=mode, days=365, overrides=active_overrides())
        if hist_365.empty or len(hist_365) < 2:
            alpha_rows.append({"Wave": w, "1D Ret": np.nan, "1D Alpha": np.nan, "30D Ret": np.nan, "30D Alpha": np.nan,
                               "60D Ret": np.nan, "60D Alpha": np.nan, "365D Ret": np.nan, "365D Alpha": np.nan})
            continue
        nav_w = hist_365["wave_nav"]
        nav_b = hist_365["bm_nav"]

        if len(nav_w) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1w - r1b
        else:
            r1w = r1b = a1 = np.nan

        r30w = compute_return_from_nav(nav_w, window=min(30, len(nav_w)))
        r30b = compute_return_from_nav(nav_b, window=min(30, len(nav_b)))
        a30 = r30w - r30b

        r60w = compute_return_from_nav(nav_w, window=min(60, len(nav_w)))
        r60b = compute_return_from_nav(nav_b, window=min(60, len(nav_b)))
        a60 = r60w - r60b

        r365w = compute_return_from_nav(nav_w, window=len(nav_w))
        r365b = compute_return_from_nav(nav_b, window=len(nav_b))
        a365 = r365w - r365b

        alpha_rows.append({"Wave": w, "1D Ret": r1w, "1D Alpha": a1, "30D Ret": r30w, "30D Alpha": a30,
                           "60D Ret": r60w, "60D Alpha": a60, "365D Ret": r365w, "365D Alpha": a365})

    alpha_df = pd.DataFrame(alpha_rows)
    fmt_a = alpha_df.copy()
    for c in ["1D Ret","1D Alpha","30D Ret","30D Alpha","60D Ret","60D Alpha","365D Ret","365D Alpha"]:
        fmt_a[c] = fmt_a[c].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
    st.dataframe(fmt_a.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    # Risk & WaveScore Ingredients
    st.subheader("Risk & WaveScore Ingredients (All Waves · 365D Window)")

    risk_rows = []
    for w in all_waves:
        hist_365 = compute_wave_history(w, mode=mode, days=365, overrides=active_overrides())
        if hist_365.empty or len(hist_365) < 2:
            risk_rows.append({"Wave": w, "Wave Vol (365D)": np.nan, "Benchmark Vol (365D)": np.nan,
                              "Max Drawdown (Wave)": np.nan, "Max Drawdown (Benchmark)": np.nan,
                              "Tracking Error": np.nan, "Information Ratio": np.nan})
            continue
        wret = hist_365["wave_ret"]
        bret = hist_365["bm_ret"]
        nav_w = hist_365["wave_nav"]
        nav_b = hist_365["bm_nav"]
        vol_w = annualized_vol(wret)
        vol_b = annualized_vol(bret)
        mdd_w = max_drawdown(nav_w)
        mdd_b = max_drawdown(nav_b)
        te = tracking_error(wret, bret)
        ir = information_ratio(nav_w, nav_b, te)
        risk_rows.append({"Wave": w, "Wave Vol (365D)": vol_w, "Benchmark Vol (365D)": vol_b,
                          "Max Drawdown (Wave)": mdd_w, "Max Drawdown (Benchmark)": mdd_b,
                          "Tracking Error": te, "Information Ratio": ir})

    risk_df = pd.DataFrame(risk_rows)
    fmt_r = risk_df.copy()
    for c in ["Wave Vol (365D)", "Benchmark Vol (365D)", "Tracking Error"]:
        fmt_r[c] = fmt_r[c].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
    for c in ["Max Drawdown (Wave)", "Max Drawdown (Benchmark)"]:
        fmt_r[c] = fmt_r[c].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
    fmt_r["Information Ratio"] = fmt_r["Information Ratio"].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")
    st.dataframe(fmt_r.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    # WaveScore Leaderboard
    st.subheader("WaveScore™ Leaderboard (Proto v1.0 · 365D Data)")
    ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365, overrides=active_overrides())
    fmt_ws = ws_df.copy()
    fmt_ws["WaveScore"] = fmt_ws["WaveScore"].apply(lambda x: f"{x:0.1f}" if pd.notna(x) else "—")
    for c in ["Return Quality","Risk Control","Consistency","Resilience","Efficiency"]:
        fmt_ws[c] = fmt_ws[c].apply(lambda x: f"{x:0.1f}" if pd.notna(x) else "—")
    fmt_ws["Alpha_365D"] = fmt_ws["Alpha_365D"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
    fmt_ws["IR_365D"] = fmt_ws["IR_365D"].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")
    st.dataframe(fmt_ws.set_index("Wave"), use_container_width=True)

    st.markdown("---")

    # Benchmark ETF Mix
    st.subheader("Benchmark ETF Mix (Composite Benchmarks)")
    bm_mix = get_benchmark_mix()
    if bm_mix.empty:
        st.info("No benchmark mix data available.")
    else:
        fmt_bm = bm_mix.copy()
        fmt_bm["Weight"] = fmt_bm["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(fmt_bm, use_container_width=True)

    st.markdown("---")

    # ========================================================
    # NEW: Volatility Regime Attribution + Diagnostics + Recos
    # ========================================================

    st.subheader("Volatility Regime Attribution + Diagnostics (Selected Wave)")

    hist_sel = compute_wave_history(selected_wave, mode=mode, days=365, overrides=active_overrides())
    holdings_sel = get_wave_holdings(selected_wave)

    col_vra, col_diag = st.columns([1.2, 1.0])

    with col_vra:
        st.markdown(f"**Volatility Regime Attribution — {selected_wave}**")
        vra = vol_regime_attribution(hist_sel)
        if vra.empty:
            st.info("Not enough data for volatility regime attribution.")
        else:
            fmt_vra = vra.copy()
            for c in ["Wave Return", "BM Return", "Alpha"]:
                fmt_vra[c] = fmt_vra[c].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
            st.dataframe(fmt_vra.set_index("Regime"), use_container_width=True)
            st.caption("Alpha decomposed by VIX regimes: Low / Medium / High / Stress (VIX-based).")

    with col_diag:
        st.markdown("**Diagnostics**")
        diags = run_diagnostics(selected_wave, mode, hist_sel, holdings_sel)
        if not diags:
            st.success("PASS — No issues detected.")
        else:
            for d in diags:
                lvl = d["level"]
                if lvl == "FAIL":
                    st.error(f"FAIL — {d['title']}\n\n{d['detail']}")
                elif lvl == "WARN":
                    st.warning(f"WARN — {d['title']}\n\n{d['detail']}")
                else:
                    st.info(f"{lvl} — {d['title']}\n\n{d['detail']}")

    st.markdown("---")

    st.subheader("Auto Recommendations (Preview-First, Session-Only Apply)")

    recos = recommend_overrides(selected_wave, mode, hist_sel)
    if not recos:
        st.info("No high-confidence recommendations detected (or not enough history).")
    else:
        for name, overrides, rationale in recos:
            with st.expander(name, expanded=False):
                st.write(rationale)

                impact = preview_alpha_impact(selected_wave, mode, 365, overrides)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Base Alpha 30D", f"{impact['Base Alpha 30D']*100:0.2f}%" if pd.notna(impact["Base Alpha 30D"]) else "—")
                c2.metric("Test Alpha 30D", f"{impact['Test Alpha 30D']*100:0.2f}%" if pd.notna(impact["Test Alpha 30D"]) else "—")
                c3.metric("Base Alpha 365D", f"{impact['Base Alpha 365D']*100:0.2f}%" if pd.notna(impact["Base Alpha 365D"]) else "—")
                c4.metric("Test Alpha 365D", f"{impact['Test Alpha 365D']*100:0.2f}%" if pd.notna(impact["Test Alpha 365D"]) else "—")

                st.code(overrides, language="python")

                if st.button(f"Apply this override (session) → {name}"):
                    st.session_state["session_overrides"] = overrides
                    st.session_state["use_overrides"] = True
                    st.success("Applied in this session. (Reboot resets to default.)")

    st.markdown("---")

    # Wave Detail View
    st.subheader(f"Wave Detail — {selected_wave} (Mode: {mode})")

    col_chart, col_stats = st.columns([2.0, 1.0])

    with col_chart:
        hist = compute_wave_history(selected_wave, mode=mode, days=nav_days, overrides=active_overrides())
        if hist.empty or len(hist) < 2:
            st.warning("Not enough data to display NAV chart.")
        else:
            nav_wave = hist["wave_nav"]
            nav_bm = hist["bm_nav"]

            fig_nav = go.Figure()
            fig_nav.add_trace(go.Scatter(x=hist.index, y=nav_wave, name=f"{selected_wave} NAV", mode="lines"))
            fig_nav.add_trace(go.Scatter(x=hist.index, y=nav_bm, name="Benchmark NAV", mode="lines"))
            fig_nav.update_layout(
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis=dict(title="Date"),
                yaxis=dict(title="NAV (Normalized)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=380,
            )
            st.plotly_chart(fig_nav, use_container_width=True)

    with col_stats:
        hist = compute_wave_history(selected_wave, mode=mode, days=nav_days, overrides=active_overrides())
        if hist.empty or len(hist) < 2:
            st.info("No stats available.")
        else:
            nav_wave = hist["wave_nav"]
            nav_bm = hist["bm_nav"]

            ret_30_wave = compute_return_from_nav(nav_wave, window=min(30, len(nav_wave)))
            ret_30_bm = compute_return_from_nav(nav_bm, window=min(30, len(nav_bm)))
            alpha_30 = ret_30_wave - ret_30_bm

            ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
            ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
            alpha_365 = ret_365_wave - ret_365_bm

            st.markdown("**Performance vs Benchmark**")
            st.metric("30D Return", f"{ret_30_wave*100:0.2f}%" if not math.isnan(ret_30_wave) else "—")
            st.metric("30D Alpha", f"{alpha_30*100:0.2f}%" if not math.isnan(alpha_30) else "—")
            st.metric("365D Return", f"{ret_365_wave*100:0.2f}%" if not math.isnan(ret_365_wave) else "—")
            st.metric("365D Alpha", f"{alpha_365*100:0.2f}%" if not math.isnan(alpha_365) else "—")

    st.markdown("#### Mode Comparison (365D)")
    mode_rows = []
    for m in all_modes:
        hist_m = compute_wave_history(selected_wave, mode=m, days=365, overrides=active_overrides())
        if hist_m.empty or len(hist_m) < 2:
            mode_rows.append({"Mode": m, "365D Return": np.nan})
        else:
            r = compute_return_from_nav(hist_m["wave_nav"], window=len(hist_m["wave_nav"]))
            mode_rows.append({"Mode": m, "365D Return": r})
    mode_df = pd.DataFrame(mode_rows)
    fmt_mode = mode_df.copy()
    fmt_mode["365D Return"] = fmt_mode["365D Return"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
    st.dataframe(fmt_mode.set_index("Mode"), use_container_width=True)

    st.markdown("#### Top-10 Holdings")
    holdings_df = get_wave_holdings(selected_wave)
    if holdings_df.empty:
        st.info("No holdings available for this Wave.")
    else:
        def google_link(ticker: str) -> str:
            return f"[{ticker}](https://www.google.com/finance/quote/{ticker})"

        fmt_hold = holdings_df.copy()
        fmt_hold["Weight"] = fmt_hold["Weight"].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        fmt_hold["Google Finance"] = fmt_hold["Ticker"].apply(google_link)

        st.dataframe(fmt_hold[["Ticker", "Name", "Weight", "Google Finance"]], use_container_width=True)


# ============================================================
# TAB 2: Market Intel
# ============================================================

with tab_market:
    st.subheader("Global Market Dashboard")
    market_df = fetch_market_assets(days=nav_days)
    if market_df.empty:
        st.warning("Unable to load multi-asset market data right now.")
    else:
        assets = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ-100",
            "IWM": "US Small Caps",
            "TLT": "US 20+Y Treasuries",
            "GLD": "Gold",
            "BTC-USD": "Bitcoin (USD)",
            "^VIX": "VIX (Implied Vol)",
            "^TNX": "US 10Y Yield",
        }
        rows = []
        for tkr, label in assets.items():
            if tkr not in market_df.columns:
                continue
            s = market_df[tkr]
            rows.append({"Ticker": tkr, "Asset": label, "Last": float(s.iloc[-1]) if len(s) else np.nan,
                         "1D Return": simple_ret(s, 2), "30D Return": simple_ret(s, 30)})
        snap_df = pd.DataFrame(rows)
        fmt_snap = snap_df.copy()
        fmt_snap["Last"] = fmt_snap["Last"].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")
        for c in ["1D Return", "30D Return"]:
            fmt_snap[c] = fmt_snap[c].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
        st.dataframe(fmt_snap.set_index("Ticker"), use_container_width=True)


# ============================================================
# TAB 3: Factor Decomposition
# ============================================================

with tab_factors:
    st.subheader("Factor Decomposition (Institution-Level Analytics)")
    st.caption("Wave daily returns are regressed on SPY, QQQ, IWM, TLT, GLD, BTC-USD.")

    factor_days = min(nav_days, 365)
    factor_prices = fetch_market_assets(days=factor_days)
    needed = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD"]
    missing = [t for t in needed if t not in factor_prices.columns]

    if factor_prices.empty or missing:
        st.warning("Unable to load all factor price series. " + (f"Missing: {', '.join(missing)}" if missing else ""))
    else:
        factor_returns = factor_prices[needed].pct_change().dropna()
        factor_returns = factor_returns.rename(columns={
            "SPY": "MKT_SPY",
            "QQQ": "GROWTH_QQQ",
            "IWM": "SIZE_IWM",
            "TLT": "RATES_TLT",
            "GLD": "GOLD_GLD",
            "BTC-USD": "CRYPTO_BTC",
        })

        rows = []
        for w in all_waves:
            hist = compute_wave_history(w, mode=mode, days=factor_days, overrides=active_overrides())
            if hist.empty or "wave_ret" not in hist.columns:
                rows.append({"Wave": w, "β_SPY": np.nan, "β_QQQ": np.nan, "β_IWM": np.nan, "β_TLT": np.nan, "β_GLD": np.nan, "β_BTC": np.nan})
                continue
            betas = regress_factors(hist["wave_ret"], factor_returns)
            rows.append({
                "Wave": w,
                "β_SPY": betas.get("MKT_SPY", np.nan),
                "β_QQQ": betas.get("GROWTH_QQQ", np.nan),
                "β_IWM": betas.get("SIZE_IWM", np.nan),
                "β_TLT": betas.get("RATES_TLT", np.nan),
                "β_GLD": betas.get("GOLD_GLD", np.nan),
                "β_BTC": betas.get("CRYPTO_BTC", np.nan),
            })

        beta_df = pd.DataFrame(rows)
        fmt_beta = beta_df.copy()
        for c in ["β_SPY","β_QQQ","β_IWM","β_TLT","β_GLD","β_BTC"]:
            fmt_beta[c] = fmt_beta[c].apply(lambda x: f"{x:0.2f}" if pd.notna(x) else "—")
        st.dataframe(fmt_beta.set_index("Wave"), use_container_width=True)

        st.markdown("---")
        st.subheader(f"Correlation Matrix — Waves (Daily Returns · Mode = {mode})")
        corr_days = min(nav_days, 365)
        ret_panel = {}
        for w in all_waves:
            hist = compute_wave_history(w, mode=mode, days=corr_days, overrides=active_overrides())
            if hist.empty or "wave_ret" not in hist.columns:
                continue
            ret_panel[w] = hist["wave_ret"]
        if not ret_panel:
            st.info("No return data available to compute correlations.")
        else:
            ret_df = pd.DataFrame(ret_panel).dropna(how="all")
            if ret_df.empty or ret_df.shape[1] < 2:
                st.info("Not enough overlapping data to compute correlations.")
            else:
                corr = ret_df.corr()
                st.dataframe(corr, use_container_width=True)


# ============================================================
# TAB 4: Vector OS Insight Layer
# ============================================================

with tab_vector:
    st.subheader("Vector OS Insight Layer — AI Chat / Insight Panel")
    st.caption("Rules-based narrative using WaveScore proto, alpha, volatility, and drawdown.")

    ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365, overrides=active_overrides())
    ws_row = ws_df[ws_df["Wave"] == selected_wave]
    hist = compute_wave_history(selected_wave, mode=mode, days=365, overrides=active_overrides())

    if ws_row.empty or hist.empty or len(hist) < 2:
        st.info("Not enough data yet for a full Vector OS insight on this Wave.")
    else:
        row = ws_row.iloc[0]
        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        ret_365_wave = compute_return_from_nav(nav_wave, window=len(nav_wave))
        ret_365_bm = compute_return_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_365_wave - ret_365_bm

        vol_wave = annualized_vol(wave_ret)
        vol_bm = annualized_vol(bm_ret)
        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)

        risk_bucket = "Moderate"
        if not math.isnan(vol_wave):
            if vol_wave < 0.12:
                risk_bucket = "Low"
            elif vol_wave > 0.25:
                risk_bucket = "High"

        alpha_bucket = "Neutral vs benchmark"
        if not math.isnan(alpha_365):
            if alpha_365 > 0.08:
                alpha_bucket = "Strong outperformance"
            elif alpha_365 > 0.03:
                alpha_bucket = "Outperforming"
            elif alpha_365 < -0.03:
                alpha_bucket = "Lagging"

        st.markdown(f"### Vector’s Insight — {selected_wave}")
        st.write(f"- **WaveScore (proto)**: **{row['WaveScore']:.1f}/100** (**{row['Grade']}**).")
        st.write(
            f"- **365D return**: {ret_365_wave*100:0.2f}% vs benchmark {ret_365_bm*100:0.2f}% "
            f"(alpha: {alpha_365*100:0.2f}%). → **{alpha_bucket}**."
        )
        st.write(
            f"- **Volatility (365D)**: Wave {vol_wave*100:0.2f}% vs benchmark {vol_bm*100:0.2f}% "
            f"→ **{risk_bucket}**."
        )
        st.write(f"- **Max drawdown (365D)**: Wave {mdd_wave*100:0.2f}% vs benchmark {mdd_bm*100:0.2f}%.")

        st.markdown("#### Volatility Regime Attribution (365D)")
        vra = vol_regime_attribution(hist)
        if vra.empty:
            st.info("Not enough data for VIX regime attribution.")
        else:
            fmt_vra = vra.copy()
            for c in ["Wave Return","BM Return","Alpha"]:
                fmt_vra[c] = fmt_vra[c].apply(lambda x: f"{x*100:0.2f}%" if pd.notna(x) else "—")
            st.dataframe(fmt_vra.set_index("Regime"), use_container_width=True)