# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — RESTORE + COPILOT ENHANCEMENTS (SAFE / READ-ONLY)
#
# Restores the “meat”:
#   • Rich Console tab (NAV vs BM, rolling alpha, drawdowns, coverage)
#   • All-Waves Performance Matrix (1D/30D/60D/365D Return + Alpha)
#   • Alpha Heatmap
#   • Risk Lab (Sharpe/Sortino/VaR/CVaR/rolling vol)
#   • Correlation
#   • Holdings Top-10 w/ clickable Google Finance links
#   • Diagnostics so it never “silent dies”
#
# Adds Copilot Enhancement Pack (read-only, safe):
#   • Regime tagging (multi-signal: VIX, rates proxy, SPY trend)
#   • Explainability (factor regression betas + R²)
#   • Anomaly detection (coverage gaps, alpha spikes, drift)
#   • Committee-language narratives + risk flags
#   • Diligence export pack (download CSVs)
#   • Integration hooks section (placeholders + export objects)
#
# Notes:
#   • Engine math is NOT modified.
#   • App reads history via engine if available, else wave_history.csv.
#   • Percent formatting + red/green lighting included for the matrix.

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
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Global UI CSS (mobile-first, clean)
# ============================================================
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2.0rem; }

/* Sticky summary container */
.waves-sticky {
  position: sticky;
  top: 0;
  z-index: 999;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 10px 12px;
  margin: 0 0 12px 0;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(10, 15, 28, 0.66);
}

/* Summary chips */
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
# Formatting helpers
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


def fmt_score(x: Any) -> str:
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


def safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()


# ============================================================
# Core math
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
    daily_wave = safe_series(daily_wave).astype(float)
    daily_bm = safe_series(daily_bm).astype(float)
    df = pd.concat([daily_wave.rename("w"), daily_bm.rename("b")], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    diff = df["w"] - df["b"]
    return float(diff.std() * np.sqrt(252))


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    nav_wave = safe_series(nav_wave).astype(float)
    nav_bm = safe_series(nav_bm).astype(float)
    if len(nav_wave) < 2 or len(nav_bm) < 2:
        return float("nan")
    if te is None or (isinstance(te, float) and (math.isnan(te) or te <= 0)):
        return float("nan")
    excess = ret_from_nav(nav_wave, len(nav_wave)) - ret_from_nav(nav_bm, len(nav_bm))
    return float(excess / te)


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
    mar_daily = mar_annual / 252.0
    ex = float((r - mar_daily).mean()) * 252.0
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


def rolling_return_from_nav(nav: pd.Series, window: int) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < window + 1:
        return pd.Series(dtype=float)
    return (nav / nav.shift(window) - 1.0).rename(f"ret_{window}")


def rolling_alpha_from_nav(wave_nav: pd.Series, bm_nav: pd.Series, window: int) -> pd.Series:
    w = rolling_return_from_nav(wave_nav, window)
    b = rolling_return_from_nav(bm_nav, window)
    df = pd.concat([w, b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    return (df.iloc[:, 0] - df.iloc[:, 1]).rename(f"alpha_{window}")


def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    return ((nav / peak) - 1.0).rename("drawdown")


def alpha_persistence(alpha_series: pd.Series) -> float:
    a = safe_series(alpha_series).dropna()
    if len(a) < 30:
        return float("nan")
    return float((a > 0).mean())


# Color for pos/neg values (matrix)
def _style_posneg(v: Any) -> str:
    try:
        if v is None:
            return ""
        x = float(v)
        if math.isnan(x):
            return ""
        # text color only (safe + readable)
        if x > 0:
            return "color: green; font-weight: 700;"
        if x < 0:
            return "color: red; font-weight: 700;"
        return "color: gray;"
    except Exception:
        return ""


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
# PART 2 of 4 — History + holdings + benchmark + regime + explainability
# ================================

# ============================================================
# Optional data fetch (yfinance) — used for regime + chips
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_prices_daily(tickers: List[str], days: int = 365) -> pd.DataFrame:
    if yf is None or not tickers:
        return pd.DataFrame()
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 260)

    data = yf.download(
        tickers=sorted(list(set(tickers))),
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # Normalize yfinance shapes
    if hasattr(data, "columns") and isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]

    if hasattr(data, "columns") and isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    if len(data) > days:
        data = data.iloc[-days:]
    return data


# ============================================================
# HISTORY LOADER (engine → CSV fallback)
# ============================================================
def _standardize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    out = df.copy()

    for dc in ["date", "Date", "timestamp", "Timestamp", "datetime", "Datetime"]:
        if dc in out.columns:
            out[dc] = pd.to_datetime(out[dc], errors="coerce")
            out = out.dropna(subset=[dc]).set_index(dc)
            break

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()

    ren = {}
    for c in out.columns:
        low = str(c).strip().lower()
        if low in ["wave_nav", "nav_wave", "portfolio_nav", "nav"]:
            ren[c] = "wave_nav"
        if low in ["bm_nav", "bench_nav", "benchmark_nav", "benchmark"]:
            ren[c] = "bm_nav"
        if low in ["wave_ret", "ret_wave", "portfolio_ret", "return", "wave_return"]:
            ren[c] = "wave_ret"
        if low in ["bm_ret", "ret_bm", "benchmark_ret", "bm_return", "benchmark_return"]:
            ren[c] = "bm_ret"

    out = out.rename(columns=ren)

    if "wave_ret" not in out.columns and "wave_nav" in out.columns:
        out["wave_ret"] = pd.to_numeric(out["wave_nav"], errors="coerce").pct_change()
    if "bm_ret" not in out.columns and "bm_nav" in out.columns:
        out["bm_ret"] = pd.to_numeric(out["bm_nav"], errors="coerce").pct_change()

    for col in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[["wave_nav", "bm_nav", "wave_ret", "bm_ret"]].dropna(how="all")
    return out


@st.cache_data(show_spinner=False)
def load_wave_history_csv(path: str = "wave_history.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def history_from_csv(wave_name: str, mode: str, days: int) -> pd.DataFrame:
    raw = load_wave_history_csv("wave_history.csv")
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    wave_cols = [c for c in df.columns if c.lower() in ["wave", "wave_name", "wavename"]]
    mode_cols = [c for c in df.columns if c.lower() in ["mode", "risk_mode", "strategy_mode"]]
    date_cols = [c for c in df.columns if c.lower() in ["date", "timestamp", "datetime"]]

    wc = wave_cols[0] if wave_cols else None
    mc = mode_cols[0] if mode_cols else None
    dc = date_cols[0] if date_cols else None

    if wc:
        df[wc] = df[wc].astype(str)
        df = df[df[wc] == str(wave_name)]
    if mc:
        df[mc] = df[mc].astype(str)
        df = df[df[mc].str.lower() == str(mode).lower()]
    if dc:
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc]).sort_values(dc).set_index(dc)

    out = _standardize_history(df)
    if len(out) > days:
        out = out.iloc[-days:]
    return out


@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    if we is None:
        return history_from_csv(wave_name, mode, days)

    try:
        if hasattr(we, "compute_history_nav"):
            try:
                df = we.compute_history_nav(wave_name, mode=mode, days=days)
                df = _standardize_history(df)
                if not df.empty:
                    return df
            except TypeError:
                df = we.compute_history_nav(wave_name, mode, days)
                df = _standardize_history(df)
                if not df.empty:
                    return df
    except Exception:
        pass

    candidates = ["get_history_nav", "get_wave_history", "history_nav", "compute_nav_history", "compute_history"]
    for fn in candidates:
        if hasattr(we, fn):
            f = getattr(we, fn)
            try:
                try:
                    df = f(wave_name, mode=mode, days=days)
                except TypeError:
                    df = f(wave_name, mode, days)
                df = _standardize_history(df)
                if not df.empty:
                    return df
            except Exception:
                continue

    return history_from_csv(wave_name, mode, days)


# ============================================================
# Discover Waves (engine → files fallback)
# ============================================================
@st.cache_data(show_spinner=False)
def get_all_waves_safe() -> List[str]:
    if we is None:
        for p in ["wave_config.csv", "wave_weights.csv", "list.csv"]:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    for col in ["Wave", "wave", "wave_name"]:
                        if col in df.columns:
                            waves = sorted(list(set(df[col].astype(str).tolist())))
                            return [w for w in waves if w and w.lower() != "nan"]
                except Exception:
                    pass
        return []

    if hasattr(we, "get_all_waves"):
        try:
            waves = we.get_all_waves()
            if isinstance(waves, (list, tuple)):
                return [str(x) for x in waves]
        except Exception:
            pass

    if hasattr(we, "get_benchmark_mix_table"):
        try:
            bm = we.get_benchmark_mix_table()
            if isinstance(bm, pd.DataFrame) and "Wave" in bm.columns:
                return sorted(list(set(bm["Wave"].astype(str).tolist())))
        except Exception:
            pass

    return []


# ============================================================
# Benchmark mix (engine → empty)
# ============================================================
@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    if we is None:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])
    if hasattr(we, "get_benchmark_mix_table"):
        try:
            df = we.get_benchmark_mix_table()
            if isinstance(df, pd.DataFrame):
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


# ============================================================
# Holdings (engine → wave_weights.csv fallback)
# ============================================================
@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    if we is not None and hasattr(we, "get_wave_holdings"):
        try:
            df = we.get_wave_holdings(wave_name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                out = df.copy()
                if "Ticker" not in out.columns:
                    for alt in ["ticker", "symbol", "Symbol"]:
                        if alt in out.columns:
                            out["Ticker"] = out[alt].astype(str)
                            break
                if "Weight" not in out.columns:
                    for alt in ["weight", "w", "WeightPct"]:
                        if alt in out.columns:
                            out["Weight"] = out[alt]
                            break
                if "Name" not in out.columns:
                    out["Name"] = ""
                out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
                out["Weight"] = pd.to_numeric(out["Weight"], errors="coerce").fillna(0.0)
                tot = float(out["Weight"].sum())
                if tot > 0:
                    out["Weight"] = out["Weight"] / tot
                out = out[["Ticker", "Name", "Weight"]].sort_values("Weight", ascending=False).reset_index(drop=True)
                return out
        except Exception:
            pass

    if os.path.exists("wave_weights.csv"):
        try:
            df = pd.read_csv("wave_weights.csv")
            cols = {c.lower(): c for c in df.columns}
            if {"wave", "ticker", "weight"}.issubset(set(cols.keys())):
                wf = df[df[cols["wave"]].astype(str) == str(wave_name)].copy()
                if wf.empty:
                    return pd.DataFrame(columns=["Ticker", "Name", "Weight"])
                wf["Ticker"] = wf[cols["ticker"]].astype(str).str.upper().str.strip()
                wf["Weight"] = pd.to_numeric(wf[cols["weight"]], errors="coerce").fillna(0.0)
                wf = wf.groupby("Ticker", as_index=False)["Weight"].sum()
                tot = float(wf["Weight"].sum())
                if tot > 0:
                    wf["Weight"] = wf["Weight"] / tot
                wf["Name"] = ""
                wf = wf[["Ticker", "Name", "Weight"]].sort_values("Weight", ascending=False).reset_index(drop=True)
                return wf
        except Exception:
            pass

    return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


# ============================================================
# Benchmark snapshot + drift tracking
# ============================================================
def _normalize_bm_rows(bm_rows: pd.DataFrame) -> pd.DataFrame:
    if bm_rows is None or bm_rows.empty:
        return pd.DataFrame(columns=["Ticker", "Weight"])
    df = bm_rows.copy()
    if "Ticker" not in df.columns or "Weight" not in df.columns:
        return pd.DataFrame(columns=["Ticker", "Weight"])
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)
    df = df.groupby("Ticker", as_index=False)["Weight"].sum()
    tot = float(df["Weight"].sum())
    if tot > 0:
        df["Weight"] = df["Weight"] / tot
    df["Weight"] = df["Weight"].round(8)
    return df.sort_values("Ticker").reset_index(drop=True)[["Ticker", "Weight"]]


def benchmark_snapshot_id(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    try:
        if bm_mix_df is None or bm_mix_df.empty:
            return "BM-NA"
        rows = bm_mix_df[bm_mix_df["Wave"] == wave_name].copy() if "Wave" in bm_mix_df.columns else bm_mix_df.copy()
        rows = _normalize_bm_rows(rows)
        if rows.empty:
            return "BM-NA"
        payload = "|".join([f"{r.Ticker}:{r.Weight:.8f}" for r in rows.itertuples(index=False)])
        h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10].upper()
        return f"BM-{h}"
    except Exception:
        return "BM-ERR"


def benchmark_drift_status(wave_name: str, mode: str, snapshot_id: str) -> str:
    key = f"bm_snapshot::{mode}::{wave_name}"
    prior = st.session_state.get(key)
    if prior is None:
        st.session_state[key] = snapshot_id
        return "stable"
    if str(prior) == str(snapshot_id):
        return "stable"
    st.session_state[key] = snapshot_id
    return "drift"


# ============================================================
# Coverage / completeness scoring
# ============================================================
def _business_day_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DatetimeIndex:
    try:
        return pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq="B")
    except Exception:
        return pd.DatetimeIndex([])


def coverage_report(hist: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "rows": 0,
        "first_date": None,
        "last_date": None,
        "age_days": None,
        "missing_bdays": None,
        "missing_pct": None,
        "completeness_score": None,
        "flags": [],
    }
    try:
        if hist is None or hist.empty:
            out["flags"].append("No history returned")
            return out

        idx = pd.to_datetime(hist.index).sort_values()
        out["rows"] = int(len(idx))
        out["first_date"] = idx[0].date().isoformat() if len(idx) else None
        out["last_date"] = idx[-1].date().isoformat() if len(idx) else None

        today = datetime.utcnow().date()
        last_dt = idx[-1].date()
        out["age_days"] = int((today - last_dt).days)

        bdays = _business_day_range(idx[0], idx[-1])
        have = pd.DatetimeIndex(idx.normalize().unique())
        missing = bdays.difference(have)
        out["missing_bdays"] = int(len(missing))
        out["missing_pct"] = float(len(missing) / max(1, len(bdays)))

        score = 100.0
        score -= min(40.0, out["missing_pct"] * 200.0)
        if out["age_days"] is not None and out["age_days"] > 3:
            score -= min(25.0, float(out["age_days"] - 3) * 5.0)

        out["completeness_score"] = float(np.clip(score, 0.0, 100.0))

        if out["age_days"] is not None and out["age_days"] >= 7:
            out["flags"].append("Data is stale (>=7 days old)")
        if out["missing_pct"] is not None and out["missing_pct"] >= 0.05:
            out["flags"].append("Significant missing business days (>=5%)")
        if out["rows"] < 60:
            out["flags"].append("Limited history (<60 points)")
        return out
    except Exception:
        out["flags"].append("Coverage report error")
        return out


# ============================================================
# Regime tagging (Copilot enhancement — safe heuristic)
# ============================================================
@st.cache_data(show_spinner=False)
def compute_regime(days: int = 180) -> Dict[str, Any]:
    """
    Multi-signal regime:
      • VIX level (risk)
      • SPY trend (20d vs 100d)
      • Rate proxy (^TNX) trend (optional)
    """
    out = {"regime": "neutral", "vix": np.nan, "tnx": np.nan, "spy_trend": "—", "notes": []}
    if yf is None:
        out["notes"].append("yfinance unavailable (regime uses fallbacks).")
        return out

    px = fetch_prices_daily(["^VIX", "^TNX", "SPY"], days=max(days, 160))
    if px.empty:
        out["notes"].append("No market data returned.")
        return out

    # VIX
    if "^VIX" in px.columns:
        try:
            out["vix"] = float(px["^VIX"].iloc[-1])
        except Exception:
            pass

    # TNX
    if "^TNX" in px.columns:
        try:
            out["tnx"] = float(px["^TNX"].iloc[-1])
        except Exception:
            pass

    # SPY trend
    if "SPY" in px.columns and px["SPY"].notna().sum() > 120:
        s = px["SPY"].dropna()
        ma20 = s.rolling(20).mean().iloc[-1]
        ma100 = s.rolling(100).mean().iloc[-1]
        if math.isfinite(ma20) and math.isfinite(ma100):
            out["spy_trend"] = "up" if ma20 > ma100 else "down"

    # Regime rules
    v = out["vix"]
    spy_tr = out["spy_trend"]

    if math.isfinite(v) and v >= 25:
        out["regime"] = "risk-off"
        out["notes"].append("VIX elevated.")
    elif math.isfinite(v) and v <= 16 and spy_tr == "up":
        out["regime"] = "risk-on"
        out["notes"].append("VIX low + SPY trend up.")
    elif spy_tr == "down":
        out["regime"] = "caution"
        out["notes"].append("SPY trend down.")

    return out


# ============================================================
# Explainability: factor regression (betas + R²)
# ============================================================
def factor_regression(
    y: pd.Series,
    factor_df: pd.DataFrame,
) -> Tuple[pd.Series, float]:
    """
    OLS: y ~ factors + intercept
    Returns: betas (including intercept) and R²
    """
    y = safe_series(y).dropna()
    X = safe_df(factor_df).dropna(how="any")
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if df.shape[0] < 60 or df.shape[1] < 2:
        return (pd.Series(dtype=float), float("nan"))

    yv = df["y"].values.astype(float)
    Xv = df.drop(columns=["y"]).values.astype(float)
    # add intercept
    Xv = np.column_stack([np.ones(len(Xv)), Xv])

    try:
        beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        yhat = Xv @ beta
        ssr = float(((yv - yhat) ** 2).sum())
        sst = float(((yv - yv.mean()) ** 2).sum())
        r2 = 1.0 - (ssr / sst) if sst > 0 else float("nan")
        names = ["Intercept"] + list(df.drop(columns=["y"]).columns)
        return (pd.Series(beta, index=names), float(r2))
    except Exception:
        return (pd.Series(dtype=float), float("nan"))
        # ================================
# PART 3 of 4 — WaveScore + matrix + heatmap + anomalies + exports
# ================================

@st.cache_data(show_spinner=False)
def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days)
        if hist is None or hist.empty or len(hist) < 20:
            rows.append({"Wave": wave, "WaveScore": np.nan, "Grade": "N/A", "IR_365D": np.nan, "Alpha_365D": np.nan})
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        alpha_365 = ret_from_nav(nav_wave, min(len(nav_wave), 365)) - ret_from_nav(nav_bm, min(len(nav_bm), 365))
        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)
        mdd_wave = max_drawdown(nav_wave)
        hit_rate = float((wave_ret >= bm_ret).mean()) if len(wave_ret) else np.nan

        # light score (display-only)
        rq = float(np.clip((np.nan_to_num(ir) / 1.5), 0.0, 1.0) * 25.0)
        rc = float(np.clip(1.0 - (abs(np.nan_to_num(mdd_wave)) / 0.35), 0.0, 1.0) * 25.0)
        co = float(np.clip(np.nan_to_num(hit_rate), 0.0, 1.0) * 15.0)
        rs = float(np.clip(1.0 - (abs(np.nan_to_num(te)) / 0.25), 0.0, 1.0) * 15.0)
        tr = 10.0

        total = float(np.clip(rq + rc + co + rs + tr, 0.0, 100.0))
        rows.append({"Wave": wave, "WaveScore": total, "Grade": _grade_from_score(total), "IR_365D": ir, "Alpha_365D": alpha_365})

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("Wave") if not df.empty else df


@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist is None or hist.empty or len(hist) < 2:
            rows.append({"Wave": wname, "1D Alpha": np.nan, "30D Alpha": np.nan, "60D Alpha": np.nan, "365D Alpha": np.nan})
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]

        a1 = np.nan
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            a1 = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0) - float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)

        a30 = ret_from_nav(nav_w, min(30, len(nav_w))) - ret_from_nav(nav_b, min(30, len(nav_b)))
        a60 = ret_from_nav(nav_w, min(60, len(nav_w))) - ret_from_nav(nav_b, min(60, len(nav_b)))
        a365 = ret_from_nav(nav_w, min(365, len(nav_w))) - ret_from_nav(nav_b, min(365, len(nav_b)))

        rows.append({"Wave": wname, "1D Alpha": a1, "30D Alpha": a30, "60D Alpha": a60, "365D Alpha": a365})

    return pd.DataFrame(rows).sort_values("Wave")


def plot_alpha_heatmap(alpha_df: pd.DataFrame, title: str):
    if go is None or alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable (Plotly missing or no data).")
        return

    df = alpha_df.copy()
    cols = [c for c in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"] if c in df.columns]
    if not cols:
        st.info("No alpha columns to plot.")
        return

    z = df[cols].values
    y = df["Wave"].tolist()
    x = cols

    v = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.10
    if not math.isfinite(v) or v <= 0:
        v = 0.10

    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, zmin=-v, zmax=v, colorbar=dict(title="Alpha")))
    fig.update_layout(title=title, height=min(900, 240 + 22 * max(10, len(y))), margin=dict(l=80, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Performance Matrix (requested) — percent formatted + red/green
# ============================================================
@st.cache_data(show_spinner=False)
def build_performance_matrix(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=days)
        if h is None or h.empty or len(h) < 2:
            rows.append({
                "Wave": w,
                "1D Return": np.nan, "1D Alpha": np.nan,
                "30D Return": np.nan, "30D Alpha": np.nan,
                "60D Return": np.nan, "60D Alpha": np.nan,
                "365D Return": np.nan, "365D Alpha": np.nan,
                "Rows": 0,
            })
            continue

        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]

        r1 = np.nan
        a1 = np.nan
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1 = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            b1 = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            a1 = r1 - b1

        r30 = ret_from_nav(nav_w, min(30, len(nav_w)))
        b30 = ret_from_nav(nav_b, min(30, len(nav_b)))
        a30 = r30 - b30

        r60 = ret_from_nav(nav_w, min(60, len(nav_w)))
        b60 = ret_from_nav(nav_b, min(60, len(nav_b)))
        a60 = r60 - b60

        r365 = ret_from_nav(nav_w, min(365, len(nav_w)))
        b365 = ret_from_nav(nav_b, min(365, len(nav_b)))
        a365 = r365 - b365

        rows.append({
            "Wave": w,
            "1D Return": r1, "1D Alpha": a1,
            "30D Return": r30, "30D Alpha": a30,
            "60D Return": r60, "60D Alpha": a60,
            "365D Return": r365, "365D Alpha": a365,
            "Rows": int(len(h)),
        })

    df = pd.DataFrame(rows)
    return df


def style_perf_matrix(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    if df is None or df.empty:
        return df.style
    pct_cols = [c for c in df.columns if "Return" in c or "Alpha" in c]
    sty = df.style.format({c: "{:.2%}" for c in pct_cols})
    for c in pct_cols:
        sty = sty.applymap(_style_posneg, subset=[c])
    return sty


# ============================================================
# Anomaly detection (safe heuristics)
# ============================================================
def detect_anomalies(wave: str, mode: str, hist: pd.DataFrame, cov: Dict[str, Any], bm_drift: str) -> List[str]:
    notes: List[str] = []

    if cov.get("flags"):
        notes.append("Data integrity: " + "; ".join(cov["flags"]))

    if bm_drift != "stable":
        notes.append("Benchmark drift detected (snapshot changed during session).")

    if hist is None or hist.empty or len(hist) < 30:
        notes.append("History too short for robust anomaly checks.")
        return notes

    # Alpha spike detection on daily alpha (wave_ret - bm_ret)
    try:
        d = pd.concat([hist["wave_ret"].rename("w"), hist["bm_ret"].rename("b")], axis=1).dropna()
        if len(d) >= 40:
            a = (d["w"] - d["b"]).dropna()
            z = (a - a.mean()) / (a.std() + 1e-9)
            if float(z.abs().max()) >= 5.0:
                notes.append("Alpha spike detected (>=5σ day). Validate prices + benchmark + missing days.")
    except Exception:
        pass

    # Extreme short-window returns (sanity)
    try:
        r30 = ret_from_nav(hist["wave_nav"], min(30, len(hist)))
        if math.isfinite(r30) and abs(r30) >= 0.60:
            notes.append("Extreme 30D return magnitude (>=60%). Ensure holdings universe and data integrity.")
    except Exception:
        pass

    return notes


# ============================================================
# Exports (Diligence Pack)
# ============================================================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=True).encode("utf-8")
    # ================================
# PART 4 of 4 — Main UI (RESTORED + ENHANCED)
# ================================

st.title("WAVES Intelligence™ Institutional Console")

if ENGINE_IMPORT_ERROR is not None:
    st.error("Engine import failed. The app will use CSV fallbacks where possible.")
    st.code(str(ENGINE_IMPORT_ERROR))

all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves discovered. Check: waves_engine import, wave_config.csv, wave_weights.csv, list.csv.")
    with st.expander("Diagnostics"):
        st.write({p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py"]})
    st.stop()

modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", modes, index=0)
    selected_wave = st.selectbox("Wave", all_waves, index=0)
    days = st.slider("History window (days)", min_value=90, max_value=1500, value=365, step=30)
    st.caption("If engine history is empty, app falls back to wave_history.csv.")

bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

hist = compute_wave_history(selected_wave, mode=mode, days=days)
cov = coverage_report(hist)

# Precompute key stats (avoid NameError landmines)
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
    r365 = ret_from_nav(hist["wave_nav"], min(365, len(hist)))
    a365 = r365 - ret_from_nav(hist["bm_nav"], min(365, len(hist)))
    te = tracking_error(hist["wave_ret"], hist["bm_ret"])
    ir = information_ratio(hist["wave_nav"], hist["bm_nav"], te)

# Regime (Copilot enhancement)
reg = compute_regime(days=180)
regime = reg.get("regime", "neutral")
vix_val = reg.get("vix", np.nan)
tnx_val = reg.get("tnx", np.nan)

# WaveScore table + rank
ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=min(days, 365))
rank = None
ws_val = np.nan
if ws_df is not None and not ws_df.empty and selected_wave in set(ws_df["Wave"]):
    try:
        ws_val = float(ws_df[ws_df["Wave"] == selected_wave]["WaveScore"].iloc[0])
        ws_df_sorted = ws_df.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True)
        rank = int(ws_df_sorted.index[ws_df_sorted["Wave"] == selected_wave][0] + 1)
    except Exception:
        pass

# Sticky chips
chips = []
chips.append(f"BM Snapshot: {bm_id} · {'Stable' if bm_drift=='stable' else 'DRIFT'}")
chips.append(f"Coverage: {cov.get('completeness_score','—')} / 100")
chips.append(f"Rows: {cov.get('rows','—')} · Age: {cov.get('age_days','—')}")
chips.append(f"Regime: {regime}")
chips.append(f"VIX: {fmt_num(vix_val,1) if math.isfinite(vix_val) else '—'}")
chips.append(f"TNX: {fmt_num(tnx_val,2) if math.isfinite(tnx_val) else '—'}")
chips.append(f"30D α: {fmt_pct(a30)} · 30D r: {fmt_pct(r30)}")
chips.append(f"365D α: {fmt_pct(a365)} · 365D r: {fmt_pct(r365)}")
chips.append(f"TE: {fmt_pct(te)} · IR: {fmt_num(ir,2)}")
chips.append(f"WaveScore: {fmt_score(ws_val)} ({_grade_from_score(ws_val)}) · Rank: {rank if rank else '—'}")

st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
for c in chips:
    st.markdown(f'<span class="waves-chip">{c}</span>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs([
    "Console",
    "Risk Lab",
    "Correlation",
    "Explainability",
    "Vector OS Intelligence",
    "Exports",
    "Diagnostics",
])

# ============================================================
# TAB: Console (RESTORED RICH VIEW + PERFORMANCE MATRIX)
# ============================================================
with tabs[0]:
    st.subheader("All-Waves Performance Matrix (Returns + Alpha)")
    perf_df = build_performance_matrix(all_waves, mode=mode, days=min(days, 365))

    if perf_df is None or perf_df.empty:
        st.info("Performance matrix unavailable (no history).")
    else:
        # Put selected wave first for scan
        if "Wave" in perf_df.columns and selected_wave in set(perf_df["Wave"]):
            top = perf_df[perf_df["Wave"] == selected_wave]
            rest = perf_df[perf_df["Wave"] != selected_wave]
            perf_df = pd.concat([top, rest], axis=0).reset_index(drop=True)

        try:
            st.dataframe(style_perf_matrix(perf_df), use_container_width=True)
        except Exception:
            # fallback without styling
            show = perf_df.copy()
            pct_cols = [c for c in show.columns if "Return" in c or "Alpha" in c]
            for c in pct_cols:
                show[c] = show[c].apply(lambda x: fmt_pct(x, 2) if pd.notna(x) else "—")
            st.dataframe(show, use_container_width=True)

        st.caption("All values shown as percentages. Green = positive, Red = negative.")

    st.divider()

    st.subheader("Alpha Heatmap (All Waves × Timeframe)")
    alpha_df = build_alpha_matrix(all_waves, mode=mode)
    plot_alpha_heatmap(alpha_df, title=f"Alpha Heatmap — Mode: {mode}")

    st.divider()

    st.subheader("Selected Wave — NAV vs Benchmark + Rolling Alpha + Drawdowns")
    if hist is None or hist.empty or len(hist) < 5:
        st.warning("Not enough history for charts for this wave/mode.")
    else:
        nav_df = pd.concat(
            [hist["wave_nav"].rename("Wave NAV"), hist["bm_nav"].rename("Benchmark NAV")],
            axis=1
        ).dropna()
        if not nav_df.empty:
            nav_df = nav_df / nav_df.iloc[0]
            st.line_chart(nav_df)

        st.write("Rolling 30D Alpha")
        ra = rolling_alpha_from_nav(hist["wave_nav"], hist["bm_nav"], window=30).dropna()
        st.line_chart(ra.rename("Rolling 30D Alpha")) if len(ra) else st.info("Not enough data for rolling 30D alpha.")

        st.write("Drawdown (Wave vs Benchmark)")
        dd_w = drawdown_series(hist["wave_nav"])
        dd_b = drawdown_series(hist["bm_nav"])
        dd_df = pd.concat([dd_w.rename("Wave"), dd_b.rename("Benchmark")], axis=1).dropna()
        st.line_chart(dd_df) if not dd_df.empty else st.info("Drawdown chart unavailable.")

    st.divider()

    st.subheader("Coverage & Data Integrity")
    if cov.get("rows", 0) == 0:
        st.warning("No history returned for this wave/mode. Engine → CSV fallback attempted.")
    c1, c2, c3 = st.columns(3)
    c1.metric("History Rows", cov.get("rows", 0))
    c2.metric("Last Data Age (days)", cov.get("age_days", "—"))
    c3.metric("Completeness Score", fmt_num(cov.get("completeness_score", np.nan), 1))
    with st.expander("Coverage Details"):
        st.write(cov)

    st.divider()

    st.subheader("Top-10 Holdings (Clickable)")
    hold = get_wave_holdings(selected_wave)
    if hold is None or hold.empty:
        st.info("Holdings unavailable.")
    else:
        hold2 = hold.copy()
        hold2["Ticker"] = hold2["Ticker"].astype(str).str.upper().str.strip()
        hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce").fillna(0.0)
        tot = float(hold2["Weight"].sum())
        if tot > 0:
            hold2["Weight"] = hold2["Weight"] / tot
        hold2 = hold2.sort_values("Weight", ascending=False).reset_index(drop=True)
        hold2["Google"] = hold2["Ticker"].apply(lambda t: f"https://www.google.com/finance/quote/{t}")

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

# ============================================================
# TAB: Risk Lab
# ============================================================
with tabs[1]:
    st.subheader("Risk Lab")
    if hist is None or hist.empty or len(hist) < 50:
        st.info("Not enough data to compute risk lab metrics.")
    else:
        r = hist["wave_ret"].dropna()
        sh = sharpe_ratio(r, 0.0)
        so = sortino_ratio(r, 0.0)
        dd = downside_deviation(r, 0.0)
        v95, c95 = var_cvar(r, 0.95)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Sharpe (0% rf)", fmt_num(sh, 2))
        a2.metric("Sortino (0% MAR)", fmt_num(so, 2))
        a3.metric("Downside Dev", fmt_pct(dd))
        a4.metric("Max Drawdown", fmt_pct(mdd))

        b1, b2 = st.columns(2)
        b1.metric("VaR 95% (daily)", fmt_pct(v95))
        b2.metric("CVaR 95% (daily)", fmt_pct(c95))

# ============================================================
# TAB: Correlation
# ============================================================
with tabs[2]:
    st.subheader("Correlation (All Waves)")
    rets = {}
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=min(days, 365))
        if h is not None and not h.empty and "wave_ret" in h.columns:
            rets[w] = h["wave_ret"]

    if len(rets) < 2:
        st.info("Not enough waves with history to compute correlations.")
    else:
        ret_df = pd.DataFrame(rets).dropna(how="all")
        corr = ret_df.corr()
        st.dataframe(corr, use_container_width=True)

# ============================================================
# TAB: Explainability (Copilot enhancement)
# ============================================================
with tabs[3]:
    st.subheader("Explainability — Factor Betas + R²")
    st.caption("Simple regression on daily returns. Safe + committee-friendly.")

    if hist is None or hist.empty or len(hist) < 90:
        st.info("Not enough history for explainability regression.")
    else:
        # Factor set (US-friendly)
        factors = ["SPY", "QQQ", "IWM", "TLT", "GLD"]
        if yf is None:
            st.info("yfinance unavailable — explainability needs factor price series.")
        else:
            px = fetch_prices_daily(factors, days=min(days, 365))
            if px.empty:
                st.info("No factor prices returned.")
            else:
                fac_ret = px.pct_change().dropna()
                y = hist["wave_ret"].dropna()
                df = pd.concat([y.rename("wave"), fac_ret], axis=1).dropna()
                if df.shape[0] < 90:
                    st.info("Not enough overlapping days with factor series.")
                else:
                    betas, r2 = factor_regression(df["wave"], df[factors])
                    st.metric("Model R²", fmt_num(r2, 2))
                    if betas is None or betas.empty:
                        st.info("Regression unavailable.")
                    else:
                        st.dataframe(betas.rename("beta").to_frame(), use_container_width=True)
                        st.caption("Interpretation: betas show sensitivity to factor sleeves; R² shows explainable share of returns.")

# ============================================================
# TAB: Vector OS Intelligence (Copilot enhancement pack)
# ============================================================
with tabs[4]:
    st.subheader("Vector OS Intelligence Layer")
    st.caption("Read-only intelligence: regime, anomalies, committee narrative, governance posture.")

    anomalies = detect_anomalies(selected_wave, mode, hist, cov, bm_drift)
    if anomalies:
        st.markdown("### Flags / Anomalies")
        for n in anomalies:
            st.markdown(f"- {n}")
    else:
        st.markdown("### Flags / Anomalies")
        st.markdown("- None detected on this window.")

    st.divider()

    st.markdown("### Committee Narrative (Auto)")
    narrative = []
    narrative.append(f"Mode: **{mode}**. Benchmark snapshot: **{bm_id}** ({'stable' if bm_drift=='stable' else 'drift'}).")
    narrative.append(f"Regime tag: **{regime}** (VIX={fmt_num(vix_val,1)}, TNX={fmt_num(tnx_val,2)}).")
    narrative.append(f"30D: return **{fmt_pct(r30)}**, alpha **{fmt_pct(a30)}**.")
    narrative.append(f"365D: return **{fmt_pct(r365)}**, alpha **{fmt_pct(a365)}**.")
    narrative.append(f"Tracking Error **{fmt_pct(te)}**, IR **{fmt_num(ir,2)}**, MaxDD **{fmt_pct(mdd)}**.")

    for line in narrative:
        st.markdown(f"- {line}")

    st.divider()

    st.markdown("### Governance Posture (Demo-safe)")
    st.markdown("- This console reads from reproducible inputs (engine + CSV fallbacks).")
    st.markdown("- Benchmark snapshot IDs help prevent accidental benchmark drift during demos.")
    st.markdown("- Coverage scoring highlights stale/missing data risk.")

# ============================================================
# TAB: Exports (Copilot enhancement)
# ============================================================
with tabs[5]:
    st.subheader("Diligence Export Pack (CSV)")
    st.caption("Download artifacts buyers/ICs ask for. Safe: exports are read-only outputs.")

    perf_df = build_performance_matrix(all_waves, mode=mode, days=min(days, 365))
    alpha_df = build_alpha_matrix(all_waves, mode=mode)
    hold_df = get_wave_holdings(selected_wave)
    hist_df = hist.copy() if hist is not None else pd.DataFrame()

    st.download_button("Download Performance Matrix (CSV)", data=df_to_csv_bytes(perf_df), file_name="performance_matrix.csv", mime="text/csv")
    st.download_button("Download Alpha Heatmap Table (CSV)", data=df_to_csv_bytes(alpha_df), file_name="alpha_matrix.csv", mime="text/csv")
    st.download_button("Download WaveScore Table (CSV)", data=df_to_csv_bytes(ws_df), file_name="wavescore_table.csv", mime="text/csv")
    st.download_button("Download Selected Wave History (CSV)", data=df_to_csv_bytes(hist_df), file_name=f"{selected_wave}_history.csv", mime="text/csv")
    st.download_button("Download Selected Wave Holdings (CSV)", data=df_to_csv_bytes(hold_df), file_name=f"{selected_wave}_holdings.csv", mime="text/csv")

    st.divider()
    st.subheader("Integration Hooks (Placeholder)")
    st.code(
        "Suggested exports endpoint pattern (future):\n"
        "/exports/performance_matrix\n"
        "/exports/wavescore\n"
        "/exports/wave/{wave_name}/history\n"
        "/exports/wave/{wave_name}/holdings\n"
    )

# ============================================================
# TAB: Diagnostics (never silent-die)
# ============================================================
with tabs[6]:
    st.subheader("System Diagnostics")
    st.write("Engine loaded:", we is not None)
    st.write("Engine import error:", str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else "None")
    st.write("Files present:", {p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py"]})
    st.write("Selected:", {"wave": selected_wave, "mode": mode, "days": days})
    st.write("History shape:", None if hist is None else hist.shape)
    if hist is not None and not hist.empty:
        st.write("History columns:", list(hist.columns))
        st.write("History tail:", hist.tail(3))