# ================================
# PART 1 of 4 — Core setup + math
# Paste this at the TOP of app.py
# ================================

from __future__ import annotations

import os
import math
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------
# Optional libs
# -------------------------------
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# -------------------------------
# Engine import (guarded)
# -------------------------------
ENGINE_IMPORT_ERROR = None
try:
    import waves_engine as we  # your engine module
except Exception as e:
    we = None
    ENGINE_IMPORT_ERROR = e


# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Global UI CSS
# ============================================================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2.0rem; }
.waves-sticky {
  position: sticky; top: 0; z-index: 999;
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  padding: 10px 12px; margin: 0 0 12px 0;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(10, 15, 28, 0.66);
}
.waves-chip {
  display: inline-block;
  padding: 6px 10px;
  margin: 6px 8px 0 0;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.85rem;
  line-height: 1.0rem;
  white-space: nowrap;
}
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Helpers: formatting / safety
# ============================================================
def fmt_pct(x: Any, digits: int = 2) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x*100:0.{digits}f}%"
    except Exception:
        return "—"


def fmt_num(x: Any, digits: int = 2) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x:.{digits}f}"
    except Exception:
        return "—"


def fmt_sc(x: Any) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x:.1f}"
    except Exception:
        return "—"


def safe_series(s: Optional[pd.Series]) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return s.copy()


# ============================================================
# Core return/risk math
# ============================================================
def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return float("nan")
    window = max(2, min(int(window), len(nav)))
    sub = nav.iloc[-window:]
    start = float(sub.iloc[0])
    end = float(sub.iloc[-1])
    if start <= 0:
        return float("nan")
    return (end / start) - 1.0


def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    return float(dd.min())


def tracking_error(daily_wave: pd.Series, daily_bm: pd.Series) -> float:
    w = safe_series(daily_wave).astype(float)
    b = safe_series(daily_bm).astype(float)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    diff = df["w"] - df["b"]
    return float(diff.std() * np.sqrt(252))


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    if te is None or (isinstance(te, float) and (math.isnan(te) or te <= 0)):
        return float("nan")
    nav_wave = safe_series(nav_wave).astype(float)
    nav_bm = safe_series(nav_bm).astype(float)
    if len(nav_wave) < 2 or len(nav_bm) < 2:
        return float("nan")
    excess = ret_from_nav(nav_wave, len(nav_wave)) - ret_from_nav(nav_bm, len(nav_bm))
    return float(excess / te)


def beta_ols(y: pd.Series, x: pd.Series) -> float:
    y = safe_series(y).astype(float)
    x = safe_series(x).astype(float)
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if df.shape[0] < 20:
        return float("nan")
    vx = float(df["x"].var())
    if not math.isfinite(vx) or vx <= 0:
        return float("nan")
    cov = float(df["y"].cov(df["x"]))
    return float(cov / vx)


def sharpe_ratio(daily_ret: pd.Series, rf_annual: float = 0.0) -> float:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 20:
        return float("nan")
    rf_daily = rf_annual / 252.0
    ex = r - rf_daily
    vol = float(ex.std())
    if not math.isfinite(vol) or vol <= 0:
        return float("nan")
    return float(ex.mean() / vol * np.sqrt(252))


def downside_deviation(daily_ret: pd.Series, mar_annual: float = 0.0) -> float:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 20:
        return float("nan")
    mar_daily = mar_annual / 252.0
    d = np.minimum(0.0, (r - mar_daily).values)
    dd = float(np.sqrt(np.mean(d**2)))
    return float(dd * np.sqrt(252))


def sortino_ratio(daily_ret: pd.Series, mar_annual: float = 0.0) -> float:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 20:
        return float("nan")
    ex = float((r - (mar_annual / 252.0)).mean()) * 252.0
    dd = downside_deviation(r, mar_annual=mar_annual)
    if not math.isfinite(dd) or dd <= 0:
        return float("nan")
    return float(ex / dd)


def var_cvar(daily_ret: pd.Series, level: float = 0.95) -> Tuple[float, float]:
    r = safe_series(daily_ret).astype(float)
    if len(r) < 50:
        return (float("nan"), float("nan"))
    q = float(np.quantile(r.values, 1.0 - level))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else float("nan")
    return (q, cvar)


def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    return ((nav / peak) - 1.0).rename("drawdown")


def rolling_alpha_from_nav(wave_nav: pd.Series, bm_nav: pd.Series, window: int) -> pd.Series:
    w = safe_series(wave_nav).astype(float)
    b = safe_series(bm_nav).astype(float)
    if len(w) < window + 2 or len(b) < window + 2:
        return pd.Series(dtype=float)
    a = (w / w.shift(window) - 1.0) - (b / b.shift(window) - 1.0)
    return a.rename(f"alpha_{window}")


def _grade_from_score(score: float) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
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
    # ================================
# PART 3 of 4 — WaveScore + Heatmap + Sticky chips
# Paste this DIRECTLY UNDER Part 2
# ================================

@st.cache_data(show_spinner=False)
def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days)
        if hist is None or hist.empty or len(hist) < 20:
            rows.append({"Wave": wave, "WaveScore": np.nan, "Grade": "N/A", "IR": np.nan, "Alpha": np.nan})
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wret = hist["wave_ret"]
        bret = hist["bm_ret"]

        alpha = ret_from_nav(nav_wave, len(nav_wave)) - ret_from_nav(nav_bm, len(nav_bm))
        te = tracking_error(wret, bret)
        ir = information_ratio(nav_wave, nav_bm, te)
        mdd = max_drawdown(nav_wave)
        hit = float((wret >= bret).mean()) if len(wret) else np.nan

        # light display-only scoring
        rq = float(np.clip((np.nan_to_num(ir) / 1.5), 0.0, 1.0) * 25.0)
        rc = float(np.clip(1.0 - (abs(np.nan_to_num(mdd)) / 0.35), 0.0, 1.0) * 25.0)
        co = float(np.clip(np.nan_to_num(hit), 0.0, 1.0) * 15.0)
        rs = float(np.clip(1.0 - (abs(np.nan_to_num(te)) / 0.25), 0.0, 1.0) * 15.0)
        tr = 10.0

        total = float(np.clip(rq + rc + co + rs + tr, 0.0, 100.0))
        rows.append({"Wave": wave, "WaveScore": total, "Grade": _grade_from_score(total), "IR": ir, "Alpha": alpha})

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty:
        return df
    return df.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=365)
        if h is None or h.empty or len(h) < 2:
            rows.append({"Wave": w, "1D": np.nan, "30D": np.nan, "60D": np.nan, "365D": np.nan})
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]

        a1 = np.nan
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            a1 = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0) - float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)

        a30 = ret_from_nav(nav_w, min(30, len(nav_w))) - ret_from_nav(nav_b, min(30, len(nav_b)))
        a60 = ret_from_nav(nav_w, min(60, len(nav_w))) - ret_from_nav(nav_b, min(60, len(nav_b)))
        a365 = ret_from_nav(nav_w, len(nav_w)) - ret_from_nav(nav_b, len(nav_b))
        rows.append({"Wave": w, "1D": a1, "30D": a30, "60D": a60, "365D": a365})

    return pd.DataFrame(rows).sort_values("Wave")


def plot_alpha_heatmap(alpha_df: pd.DataFrame, title: str):
    if go is None or alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable (Plotly missing or no data).")
        return
    df = alpha_df.copy()
    cols = ["1D", "30D", "60D", "365D"]
    z = df[cols].values
    y = df["Wave"].tolist()
    v = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.10
    if not math.isfinite(v) or v <= 0:
        v = 0.10
    fig = go.Figure(data=go.Heatmap(z=z, x=cols, y=y, zmin=-v, zmax=v, colorbar=dict(title="Alpha")))
    fig.update_layout(title=title, height=min(950, 240 + 22 * max(10, len(y))), margin=dict(l=80, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Main header + discovery
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")

if ENGINE_IMPORT_ERROR is not None:
    st.error("Engine import failed. The app will use CSV fallbacks where possible.")
    st.code(str(ENGINE_IMPORT_ERROR))

all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves discovered. Check engine import + wave_config.csv / wave_weights.csv / list.csv.")
    with st.expander("Diagnostics"):
        st.write({p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py"]})
    st.stop()

modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", modes, index=0)
    selected_wave = st.selectbox("Wave", all_waves, index=0)
    days = st.slider("History window (days)", min_value=90, max_value=1500, value=365, step=30)

bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

hist = compute_wave_history(selected_wave, mode=mode, days=days)
cov = coverage_report(hist)

# Precompute common stats safely
mdd = np.nan
mdd_b = np.nan
a30 = np.nan
r30 = np.nan
a365 = np.nan
r365 = np.nan
te = np.nan
ir = np.nan

if hist is not None and (not hist.empty) and len(hist) >= 2:
    mdd = max_drawdown(hist["wave_nav"])
    mdd_b = max_drawdown(hist["bm_nav"])
    r30 = ret_from_nav(hist["wave_nav"], min(30, len(hist)))
    a30 = r30 - ret_from_nav(hist["bm_nav"], min(30, len(hist)))
    r365 = ret_from_nav(hist["wave_nav"], len(hist))
    a365 = r365 - ret_from_nav(hist["bm_nav"], len(hist))
    te = tracking_error(hist["wave_ret"], hist["bm_ret"])
    ir = information_ratio(hist["wave_nav"], hist["bm_nav"], te)

# Regime chip (VIX)
regime = "neutral"
vix_val = np.nan
if yf is not None:
    try:
        vix_df = fetch_prices_daily(["^VIX"], days=30)
        if not vix_df.empty and "^VIX" in vix_df.columns:
            vix_val = float(vix_df["^VIX"].iloc[-1])
            if vix_val >= 25:
                regime = "risk-off"
            elif vix_val <= 16:
                regime = "risk-on"
    except Exception:
        pass

# WaveScore + rank
ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=min(days, 365))
rank = None
ws_val = np.nan
if ws_df is not None and (not ws_df.empty) and selected_wave in set(ws_df["Wave"]):
    try:
        row = ws_df[ws_df["Wave"] == selected_wave].iloc[0]
        ws_val = float(row["WaveScore"]) if "WaveScore" in row else np.nan
        rank = int(ws_df.index[ws_df["Wave"] == selected_wave][0] + 1)
    except Exception:
        pass

# Sticky chips (THIS includes the line you said was missing)
chips = []
chips.append(f"BM Snapshot: {bm_id} · {'Stable' if bm_drift=='stable' else 'DRIFT'}")
chips.append(f"Coverage: {cov.get('completeness_score','—')} / 100")
chips.append(f"Rows: {cov.get('rows','—')} · Age: {cov.get('age_days','—')}")
chips.append(f"Regime: {regime}")
chips.append(f"VIX: {fmt_num(vix_val,1) if math.isfinite(vix_val) else '—'}")
chips.append(f"30D α: {fmt_pct(a30)} · 30D r: {fmt_pct(r30)}")
chips.append(f"365D α: {fmt_pct(a365)} · 365D r: {fmt_pct(r365)}")
chips.append(f"TE: {fmt_pct(te)} · IR: {fmt_num(ir,2)}")
chips.append(f"WaveScore: {fmt_sc(ws_val)} ({_grade_from_score(ws_val)}) · Rank: {rank if rank else '—'}")

st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
for c in chips:
    st.markdown(f'<span class="waves-chip">{c}</span>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
# ================================
# PART 4 of 4 — Full tabbed console (restores the "meat")
# Paste this DIRECTLY UNDER Part 3
# ================================

def _google_quote(t: str) -> str:
    t = str(t).strip().upper()
    return f"https://www.google.com/finance/quote/{t}"

def _static_basket_nav_from_holdings(hold: pd.DataFrame, days: int = 365) -> pd.Series:
    """
    Builds a NAV series for a static basket using holdings weights + yfinance daily prices.
    If yfinance unavailable or insufficient data, returns empty series.
    """
    if yf is None or hold is None or hold.empty:
        return pd.Series(dtype=float)
    if "Ticker" not in hold.columns or "Weight" not in hold.columns:
        return pd.Series(dtype=float)

    tickers = hold["Ticker"].astype(str).str.upper().str.strip().tolist()
    w = pd.to_numeric(hold["Weight"], errors="coerce").fillna(0.0).values
    if len(tickers) == 0 or float(np.sum(w)) <= 0:
        return pd.Series(dtype=float)

    # normalize
    w = w / np.sum(w)

    end = datetime.utcnow().date()
    start = end - timedelta(days=int(days) + 260)

    px = yf.download(
        tickers=sorted(list(set(tickers))),
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if px is None or len(px) == 0:
        return pd.Series(dtype=float)

    if isinstance(px.columns, pd.MultiIndex):
        if "Adj Close" in px.columns.get_level_values(0):
            px = px["Adj Close"]
        elif "Close" in px.columns.get_level_values(0):
            px = px["Close"]
        else:
            px = px[px.columns.levels[0][0]]

    if isinstance(px.columns, pd.MultiIndex):
        px = px.droplevel(0, axis=1)

    if isinstance(px, pd.Series):
        px = px.to_frame()

    px = px.sort_index().ffill().bfill()

    # align to requested weights (handle missing tickers)
    available = [t for t in tickers if t in px.columns]
    if len(available) < max(3, int(len(tickers) * 0.3)):
        return pd.Series(dtype=float)

    sub_hold = hold.copy()
    sub_hold["Ticker"] = sub_hold["Ticker"].astype(str).str.upper().str.strip()
    sub_hold = sub_hold[sub_hold["Ticker"].isin(available)].copy()
    ww = pd.to_numeric(sub_hold["Weight"], errors="coerce").fillna(0.0)
    if float(ww.sum()) <= 0:
        return pd.Series(dtype=float)
    ww = ww / ww.sum()

    ret = px[sub_hold["Ticker"].tolist()].pct_change().fillna(0.0)
    basket_daily = (ret * ww.values).sum(axis=1)
    nav = (1.0 + basket_daily).cumprod()
    nav.name = "static_basket_nav"
    if len(nav) > days:
        nav = nav.iloc[-days:]
    return nav


tabs = st.tabs([
    "Console",
    "Attribution",
    "Wave Doctor / What-If",
    "Risk Lab",
    "Correlation Matrix",
    "Market Intel",
    "Factor Decomposition",
    "WaveScore Leaderboard",
    "Vector OS Insight Layer",
    "Diagnostics",
])

# ---------------------------------
# TAB 1: Console (overview + heatmap + holdings)
# ---------------------------------
with tabs[0]:
    st.subheader("Alpha Heatmap (All Waves × Timeframe)")
    alpha_df = build_alpha_matrix(all_waves, mode=mode)
    plot_alpha_heatmap(alpha_df, title=f"Alpha Heatmap — Mode: {mode}")

    st.subheader("Coverage & Data Integrity")
    if cov.get("rows", 0) == 0:
        st.warning("No history returned for this wave/mode. The app attempted engine → CSV fallback.")
    c1, c2, c3 = st.columns(3)
    c1.metric("History Rows", cov.get("rows", 0))
    c2.metric("Last Data Age (days)", cov.get("age_days", "—"))
    c3.metric("Completeness Score", fmt_num(cov.get("completeness_score", np.nan), 1))
    with st.expander("Coverage Details"):
        st.write(cov)

    st.subheader("Top-10 Holdings (Clickable)")
    hold = get_wave_holdings(selected_wave)
    if hold is None or hold.empty:
        st.info("Holdings unavailable (engine did not return holdings and wave_weights.csv fallback did not match).")
    else:
        hold2 = hold.copy()
        hold2["Ticker"] = hold2["Ticker"].astype(str).str.upper().str.strip()
        hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce").fillna(0.0)
        tot = float(hold2["Weight"].sum())
        if tot > 0:
            hold2["Weight"] = hold2["Weight"] / tot
        hold2 = hold2.sort_values("Weight", ascending=False).reset_index(drop=True)
        hold2["Google"] = hold2["Ticker"].apply(_google_quote)

        try:
            st.dataframe(
                hold2.head(10),
                use_container_width=True,
                column_config={
                    "Weight": st.column_config.NumberColumn("Weight", format="%.4f"),
                    "Google": st.column_config.LinkColumn("Google", display_text="Open"),
                },
            )
        except Exception:
            st.dataframe(hold2.head(10), use_container_width=True)


# ---------------------------------
# TAB 2: Attribution (Engine vs Static Basket)
# ---------------------------------
with tabs[1]:
    st.subheader("Attribution — Engine vs Static Basket (Proof Layer)")
    st.caption("If the engine history is available, we compare it to a static basket built from holdings weights (shadow).")

    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough history for attribution on this wave/mode.")
    else:
        hold = get_wave_holdings(selected_wave)
        static_nav = _static_basket_nav_from_holdings(hold, days=min(days, 365))

        if static_nav.empty:
            st.info("Static basket NAV not available (yfinance missing or insufficient ticker coverage).")
            st.write("Engine NAV vs Benchmark still shown below.")
        else:
            # align dates
            df = pd.concat([
                hist["wave_nav"].rename("Engine NAV"),
                static_nav.rename("Static Basket NAV"),
                hist["bm_nav"].rename("Benchmark NAV"),
            ], axis=1).dropna()

            if df.empty or df.shape[0] < 20:
                st.info("Not enough overlapping dates to attribute.")
            else:
                # normalize to 1
                df = df / df.iloc[0]
                st.line_chart(df)

                eng_ret = ret_from_nav(df["Engine NAV"], len(df))
                stat_ret = ret_from_nav(df["Static Basket NAV"], len(df))
                bm_ret = ret_from_nav(df["Benchmark NAV"], len(df))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Engine Return", fmt_pct(eng_ret))
                c2.metric("Static Basket Return", fmt_pct(stat_ret))
                c3.metric("Benchmark Return", fmt_pct(bm_ret))
                c4.metric("Engine – Static (Attribution)", fmt_pct(eng_ret - stat_ret))

                st.write("Interpretation:")
                st.markdown("- **Engine – Static** isolates what your *engine logic* added beyond a fixed basket.")
                st.markdown("- If **Engine ≈ Static**, returns are mostly selection + weight, not adaptive logic.")


# ---------------------------------
# TAB 3: Wave Doctor / What-If (shadow simulation only)
# ---------------------------------
with tabs[2]:
    st.subheader("Wave Doctor / What-If Lab (Shadow Simulation)")
    st.caption("This does NOT change the engine. It creates a shadow NAV by scaling daily returns.")

    if hist is None or hist.empty or "wave_ret" not in hist.columns or len(hist) < 30:
        st.info("Not enough return history to run What-If Lab.")
    else:
        scale = st.slider("Risk Scaling (shadow)", min_value=0.2, max_value=1.8, value=1.0, step=0.05)
        df = hist[["wave_ret", "bm_ret"]].dropna().copy()
        df["shadow_ret"] = df["wave_ret"] * float(scale)

        shadow_nav = (1.0 + df["shadow_ret"].fillna(0.0)).cumprod()
        real_nav = (1.0 + df["wave_ret"].fillna(0.0)).cumprod()
        bm_nav = (1.0 + df["bm_ret"].fillna(0.0)).cumprod()

        nav_df = pd.concat([
            real_nav.rename("Engine (real)"),
            shadow_nav.rename("Shadow (scaled)"),
            bm_nav.rename("Benchmark"),
        ], axis=1).dropna()

        st.line_chart(nav_df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Shadow Return", fmt_pct(ret_from_nav(nav_df["Shadow (scaled)"], len(nav_df))))
        c2.metric("Shadow MaxDD", fmt_pct(max_drawdown(nav_df["Shadow (scaled)"])))
        c3.metric("Shadow – Benchmark (total)", fmt_pct(
            ret_from_nav(nav_df["Shadow (scaled)"], len(nav_df)) - ret_from_nav(nav_df["Benchmark"], len(nav_df))
        ))


# ---------------------------------
# TAB 4: Risk Lab (full)
# ---------------------------------
with tabs[3]:
    st.subheader("Risk Lab")
    if hist is None or hist.empty or len(hist) < 50:
        st.info("Not enough data to compute Risk Lab metrics.")
    else:
        r = hist["wave_ret"].dropna()
        sh = sharpe_ratio(r, 0.0)
        so = sortino_ratio(r, 0.0)
        dd = downside_deviation(r, 0.0)
        v95, c95 = var_cvar(r, 0.95)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe (0% rf)", fmt_num(sh, 2))
        c2.metric("Sortino (0% MAR)", fmt_num(so, 2))
        c3.metric("Downside Dev", fmt_pct(dd))
        c4.metric("Max Drawdown", fmt_pct(mdd))

        c5, c6 = st.columns(2)
        c5.metric("VaR 95% (daily)", fmt_pct(v95))
        c6.metric("CVaR 95% (daily)", fmt_pct(c95))

        st.write("Drawdown (Wave vs Benchmark)")
        dd_w = drawdown_series(hist["wave_nav"])
        dd_b = drawdown_series(hist["bm_nav"])
        st.line_chart(pd.concat([dd_w.rename("Wave"), dd_b.rename("Benchmark")], axis=1).dropna())

        st.write("Rolling 30D Alpha")
        ra = rolling_alpha_from_nav(hist["wave_nav"], hist["bm_nav"], 30).dropna()
        st.line_chart(ra.rename("Rolling 30D Alpha"))


# ---------------------------------
# TAB 5: Correlation Matrix
# ---------------------------------
with tabs[4]:
    st.subheader("Correlation Matrix (Wave returns)")
    rets = {}
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=min(days, 365))
        if h is not None and (not h.empty) and "wave_ret" in h.columns:
            rets[w] = h["wave_ret"]
    if len(rets) < 2:
        st.info("Not enough waves with history to compute correlations.")
    else:
        ret_df = pd.DataFrame(rets).dropna(how="all")
        corr = ret_df.corr()
        st.dataframe(corr, use_container_width=True)


# ---------------------------------
# TAB 6: Market Intel (core tickers)
# ---------------------------------
with tabs[5]:
    st.subheader("Market Intel")
    st.caption("Fast snapshot. Uses yfinance if available.")
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
    if yf is not None:
        px = fetch_prices_daily(tickers, days=120)
        if px is None or px.empty:
            st.info("Market Intel unavailable (no prices returned).")
        else:
            rets = px.pct_change().fillna(0.0)
            nav = (1.0 + rets).cumprod()
            st.line_chart(nav)
            last = (nav.iloc[-1] / nav.iloc[-21] - 1.0) if len(nav) >= 22 else pd.Series(dtype=float)
            if not last.empty:
                st.write("Approx 1-Month Returns")
                st.dataframe((last * 100).round(2).to_frame("Return %"), use_container_width=True)
    else:
        st.info("yfinance not available in this environment.")


# ---------------------------------
# TAB 7: Factor Decomposition (beta vs benchmark)
# ---------------------------------
with tabs[6]:
    st.subheader("Factor Decomposition (Baseline)")
    st.caption("This restores the factor tab: beta vs benchmark from returns (full multi-factor comes next upgrade).")
    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough history.")
    else:
        b = beta_ols(hist["wave_ret"], hist["bm_ret"])
        st.metric("Beta vs Benchmark", fmt_num(b, 2))
        st.write("Next upgrade (Copilot layer): multi-factor regression + residual alpha.")


# ---------------------------------
# TAB 8: WaveScore Leaderboard
# ---------------------------------
with tabs[7]:
    st.subheader("WaveScore Leaderboard (Display)")
    if ws_df is None or ws_df.empty:
        st.info("WaveScore table unavailable.")
    else:
        show = ws_df.copy()
        try:
            show["WaveScore"] = pd.to_numeric(show["WaveScore"], errors="coerce")
            show["IR"] = pd.to_numeric(show["IR"], errors="coerce")
            show["Alpha"] = pd.to_numeric(show["Alpha"], errors="coerce")
        except Exception:
            pass
        st.dataframe(show, use_container_width=True)


# ---------------------------------
# TAB 9: Vector OS Insight Layer (restored)
# ---------------------------------
with tabs[8]:
    st.subheader("Vector OS Insight Layer (Restored)")
    notes = []
    if cov.get("flags"):
        notes.append("**Data Integrity Flags:** " + "; ".join(cov["flags"]))
    if bm_drift != "stable":
        notes.append("**Benchmark Drift:** Snapshot changed — freeze benchmark mix for demos.")
    if math.isfinite(a30) and abs(a30) >= 0.08:
        notes.append("**Large 30D alpha:** verify benchmark mix + missing days.")
    if math.isfinite(te) and te >= 0.20:
        notes.append("**High tracking error:** active risk elevated.")
    if math.isfinite(mdd) and mdd <= -0.25:
        notes.append("**Deep drawdown:** consider SmartSafe posture in stress regimes.")

    if not notes:
        notes = ["No major anomalies detected on this window."]

    for n in notes:
        st.markdown(f"- {n}")


# ---------------------------------
# TAB 10: Diagnostics (never blank-screen)
# ---------------------------------
with tabs[9]:
    st.subheader("System Diagnostics (If something looks off)")
    st.write("Engine loaded:", we is not None)
    st.write("Engine import error:", str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else "None")
    st.write("Files present:", {p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py"]})
    st.write("Selected:", {"wave": selected_wave, "mode": mode, "days": days})
    st.write("History shape:", None if hist is None else hist.shape)
    if hist is not None and not hist.empty:
        st.write("History columns:", list(hist.columns))
        st.write("History tail:", hist.tail(5))
        