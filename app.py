# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION FILE (NO PATCHES) — RESTORED RICH CONSOLE + MATRIX + SAFE FALLBACKS
#
# Core: Engine math NOT modified. App is display + analytics only.
# Fixes: NameError/blank screens, robust history fallback, no DeltaGenerator spam.
# Adds back: Performance Matrix (1D/30D/60D/365D r + α), Attribution tab, Market Intel tab,
# Wave Doctor (shadow), WaveScore table, Correlation, Risk Lab, Factor Decomp, Vector Insight.

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
# Global UI CSS
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
  padding: 10px 12px 10px 12px;
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

/* Tighter tables */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Helpers: formatting
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
# Basic return/risk math
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

def rolling_vol(daily_ret: pd.Series, window: int = 20) -> pd.Series:
    r = safe_series(daily_ret).astype(float)
    if len(r) < window + 5:
        return pd.Series(dtype=float)
    return (r.rolling(window).std() * np.sqrt(252)).rename(f"vol_{window}")

def drawdown_series(nav: pd.Series) -> pd.Series:
    nav = safe_series(nav).astype(float)
    if len(nav) < 2:
        return pd.Series(dtype=float)
    peak = nav.cummax()
    return ((nav / peak) - 1.0).rename("drawdown")
    # ================================
# PART 2 of 4 — History + holdings + benchmark mix (engine → CSV fallbacks)
# ================================

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
                return out[["Ticker", "Name", "Weight"]].sort_values("Weight", ascending=False).reset_index(drop=True)
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
                return wf[["Ticker", "Name", "Weight"]].sort_values("Weight", ascending=False).reset_index(drop=True)
        except Exception:
            pass

    return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


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
        # ================================
# PART 3 of 4 — WaveScore + Heatmap + Performance Matrix (pct + red/green)
# ================================

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


@st.cache_data(show_spinner=False)
def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days)
        if hist.empty or len(hist) < 20:
            rows.append({"Wave": wave, "WaveScore": np.nan, "Grade": "N/A", "IR_365D": np.nan, "Alpha_365D": np.nan})
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        alpha_365 = ret_from_nav(nav_wave, len(nav_wave)) - ret_from_nav(nav_bm, len(nav_bm))
        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)
        mdd_wave = max_drawdown(nav_wave)
        hit_rate = float((wave_ret >= bm_ret).mean()) if len(wave_ret) else np.nan

        rq = float(np.clip((np.nan_to_num(ir) / 1.5), 0.0, 1.0) * 25.0)
        rc = float(np.clip(1.0 - (abs(np.nan_to_num(mdd_wave)) / 0.35), 0.0, 1.0) * 25.0)
        co = float(np.clip(np.nan_to_num(hit_rate), 0.0, 1.0) * 15.0)
        rs = float(np.clip(1.0 - (abs(np.nan_to_num(te)) / 0.25), 0.0, 1.0) * 15.0)
        tr = 10.0

        total = float(np.clip(rq + rc + co + rs + tr, 0.0, 100.0))
        rows.append({"Wave": wave, "WaveScore": total, "Grade": _grade_from_score(total), "IR_365D": ir, "Alpha_365D": alpha_365})

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("WaveScore", ascending=False, na_position="last").reset_index(drop=True) if not df.empty else df


@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist.empty or len(hist) < 2:
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

    return pd.DataFrame(rows).sort_values("Wave").reset_index(drop=True)


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

    return pd.DataFrame(rows).reset_index(drop=True)


def _style_pos_neg_pct(df: pd.DataFrame, pct_cols: List[str]) -> "pd.io.formats.style.Styler":
    def colorize(v):
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            v = float(v)
            if v > 0:
                return "color: #2ecc71; font-weight: 700;"  # green
            if v < 0:
                return "color: #e74c3c; font-weight: 700;"  # red
            return "color: #bdc3c7;"
        except Exception:
            return ""

    sty = df.style
    for c in pct_cols:
        if c in df.columns:
            sty = sty.applymap(colorize, subset=[c]).format({c: "{:.2%}"})
    if "Rows" in df.columns:
        sty = sty.format({"Rows": "{:d}"})
    return sty
    # ================================
# PART 4 of 4 — Main UI (RESTORED tabs: Console, Attribution, Wave Doctor, Market Intel, Factor, Risk, Corr, WaveScore, Vector)
# ================================

st.title("WAVES Intelligence™ Institutional Console")

if ENGINE_IMPORT_ERROR is not None:
    st.error("Engine import failed. The app will use CSV fallbacks where possible.")
    st.code(str(ENGINE_IMPORT_ERROR))

all_waves = get_all_waves_safe()
if not all_waves:
    st.warning("No waves discovered yet. Check: engine import, wave_config.csv, wave_weights.csv, list.csv, wave_history.csv.")
    with st.expander("Diagnostics"):
        st.write({p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py"]})
    st.stop()

modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", modes, index=0)
    selected_wave = st.selectbox("Wave", all_waves, index=0)
    days = st.slider("History window (days)", min_value=90, max_value=1500, value=365, step=30)
    st.caption("History loader: engine → wave_history.csv fallback. If short history, some charts won't render.")

bm_mix = get_benchmark_mix()
bm_id = benchmark_snapshot_id(selected_wave, bm_mix)
bm_drift = benchmark_drift_status(selected_wave, mode, bm_id)

hist = compute_wave_history(selected_wave, mode=mode, days=days)
cov = coverage_report(hist)

# Precompute stats used across tabs
mdd = np.nan
mdd_b = np.nan
r30 = np.nan
a30 = np.nan
r60 = np.nan
a60 = np.nan
r365 = np.nan
a365 = np.nan
te = np.nan
ir = np.nan

if hist is not None and (not hist.empty) and len(hist) >= 2:
    mdd = max_drawdown(hist["wave_nav"])
    mdd_b = max_drawdown(hist["bm_nav"])
    r30 = ret_from_nav(hist["wave_nav"], min(30, len(hist)))
    a30 = r30 - ret_from_nav(hist["bm_nav"], min(30, len(hist)))
    r60 = ret_from_nav(hist["wave_nav"], min(60, len(hist)))
    a60 = r60 - ret_from_nav(hist["bm_nav"], min(60, len(hist)))
    r365 = ret_from_nav(hist["wave_nav"], min(365, len(hist)))
    a365 = r365 - ret_from_nav(hist["bm_nav"], min(365, len(hist)))
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

# WaveScore table + rank
ws_df = compute_wavescore_for_all_waves(all_waves, mode=mode, days=min(days, 365))
rank = None
ws_val = np.nan
if not ws_df.empty and selected_wave in set(ws_df["Wave"]):
    ws_val = float(ws_df[ws_df["Wave"] == selected_wave]["WaveScore"].iloc[0])
    try:
        rank = int(ws_df.index[ws_df["Wave"] == selected_wave][0] + 1)
    except Exception:
        rank = None

# Sticky chips (NOTE: uses fmt_score — NOT fmt_sc)
chips = []
chips.append(f"BM Snapshot: {bm_id} · {'Stable' if bm_drift=='stable' else 'DRIFT'}")
chips.append(f"Coverage: {fmt_num(cov.get('completeness_score', np.nan), 1)} / 100")
chips.append(f"Rows: {cov.get('rows','—')} · Age: {cov.get('age_days','—')}")
chips.append(f"Regime: {regime}")
chips.append(f"VIX: {fmt_num(vix_val,1) if math.isfinite(vix_val) else '—'}")
chips.append(f"30D α: {fmt_pct(a30)} · 30D r: {fmt_pct(r30)}")
chips.append(f"60D α: {fmt_pct(a60)} · 60D r: {fmt_pct(r60)}")
chips.append(f"365D α: {fmt_pct(a365)} · 365D r: {fmt_pct(r365)}")
chips.append(f"TE: {fmt_pct(te)} · IR: {fmt_num(ir,2)}")
chips.append(f"WaveScore: {fmt_score(ws_val)} ({_grade_from_score(ws_val)}) · Rank: {rank if rank else '—'}")

st.markdown('<div class="waves-sticky">', unsafe_allow_html=True)
for c in chips:
    st.markdown(f'<span class="waves-chip">{c}</span>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs([
    "Console",
    "Attribution",
    "Wave Doctor (Shadow)",
    "Market Intel",
    "Factor Decomposition",
    "Risk Lab",
    "Correlation",
    "WaveScore Leaderboard",
    "Vector OS Insight Layer",
])

# -------------------------
# TAB: Console (RESTORED RICH VIEW + MATRIX)
# -------------------------
with tabs[0]:
    st.subheader("All-Waves Performance Matrix (Returns + Alpha) — Pct + Red/Green")
    perf_df = build_performance_matrix(all_waves, mode=mode, days=min(days, 365))

    if perf_df is None or perf_df.empty:
        st.info("Performance matrix unavailable (no history).")
    else:
        # Put selected wave first for scan readability
        if "Wave" in perf_df.columns and selected_wave in set(perf_df["Wave"]):
            top = perf_df[perf_df["Wave"] == selected_wave]
            rest = perf_df[perf_df["Wave"] != selected_wave]
            perf_df = pd.concat([top, rest], axis=0).reset_index(drop=True)

        pct_cols = ["1D Return","1D Alpha","30D Return","30D Alpha","60D Return","60D Alpha","365D Return","365D Alpha"]
        try:
            st.dataframe(_style_pos_neg_pct(perf_df, pct_cols), use_container_width=True)
        except Exception:
            # fallback (no Styler support)
            show = perf_df.copy()
            for c in pct_cols:
                if c in show.columns:
                    show[c] = show[c].apply(lambda v: fmt_pct(v))
            st.dataframe(show, use_container_width=True)

    st.divider()

    st.subheader("Alpha Heatmap (All Waves × Timeframe)")
    alpha_df = build_alpha_matrix(all_waves, mode=mode)
    plot_alpha_heatmap(alpha_df, title=f"Alpha Heatmap — Mode: {mode}")

    st.divider()

    st.subheader("Selected Wave — NAV vs Benchmark (Normalized) + Rolling Alpha + Drawdown")
    if hist is None or hist.empty or len(hist) < 10:
        st.warning("Not enough history for charts for this wave/mode (likely short history).")
    else:
        nav_df = pd.concat([hist["wave_nav"].rename("Wave NAV"), hist["bm_nav"].rename("Benchmark NAV")], axis=1).dropna()
        if not nav_df.empty:
            nav_df = nav_df / nav_df.iloc[0]
            st.line_chart(nav_df)  # IMPORTANT: no st.write(st.line_chart(...)) — avoids DeltaGenerator spam

        st.write("Rolling 30D Alpha")
        ra = rolling_alpha_from_nav(hist["wave_nav"], hist["bm_nav"], window=30).dropna()
        if len(ra):
            st.line_chart(ra.rename("Rolling 30D Alpha"))
        else:
            st.info("Not enough data for rolling 30D alpha.")

        st.write("Drawdown (Wave vs Benchmark)")
        dd_df = pd.concat([drawdown_series(hist["wave_nav"]).rename("Wave"),
                           drawdown_series(hist["bm_nav"]).rename("Benchmark")], axis=1).dropna()
        if not dd_df.empty:
            st.line_chart(dd_df)
        else:
            st.info("Drawdown chart unavailable.")

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
                    "Weight": st.column_config.NumberColumn("Weight", format="%.2%"),
                    "Google": st.column_config.LinkColumn("Google", display_text="Open"),
                },
            )
        except Exception:
            st.dataframe(hold2.head(10), use_container_width=True)

# -------------------------
# TAB: Attribution (Engine vs Static Basket)
# -------------------------
with tabs[1]:
    st.subheader("Alpha Attribution — Engine vs Static Basket (Holdings)")
    st.caption("Static basket is computed from holdings weights + yfinance prices. This is display-only (shadow attribution).")

    hold = get_wave_holdings(selected_wave)
    if yf is None:
        st.info("Attribution unavailable: yfinance not installed in this environment.")
    elif hold is None or hold.empty:
        st.info("Attribution unavailable: holdings missing.")
    elif hist is None or hist.empty or len(hist) < 30:
        st.info("Attribution needs at least ~30 points of history.")
    else:
        tickers = hold["Ticker"].astype(str).str.upper().str.strip().tolist()
        wts = hold["Weight"].astype(float).tolist()
        prices = fetch_prices_daily(tickers, days=min(days, 365))
        if prices is None or prices.empty:
            st.info("No price data returned for holdings (static basket).")
        else:
            prices = prices.dropna(how="all").ffill().bfill()
            # align to hist dates
            px = prices.reindex(pd.to_datetime(hist.index)).dropna(how="all").ffill().bfill()
            if px.empty:
                st.info("Static basket could not align to history dates.")
            else:
                w = np.array(wts, dtype=float)
                w = w / w.sum() if w.sum() > 0 else w
                # basket nav (start 1.0)
                px_norm = px / px.iloc[0]
                basket_nav = (px_norm * w).sum(axis=1).rename("Static Basket NAV")
                # wave nav normalized
                wave_nav = (hist["wave_nav"] / hist["wave_nav"].iloc[0]).rename("Engine Wave NAV")
                bm_nav = (hist["bm_nav"] / hist["bm_nav"].iloc[0]).rename("Benchmark NAV")

                df = pd.concat([wave_nav, basket_nav, bm_nav], axis=1).dropna()
                st.line_chart(df)

                engine_alpha_365 = ret_from_nav(hist["wave_nav"], min(365, len(hist))) - ret_from_nav(hist["bm_nav"], min(365, len(hist)))
                basket_alpha_365 = ret_from_nav(basket_nav, min(365, len(basket_nav))) - ret_from_nav(bm_nav, min(365, len(bm_nav)))
                delta = engine_alpha_365 - basket_alpha_365

                c1, c2, c3 = st.columns(3)
                c1.metric("Engine α (365D)", fmt_pct(engine_alpha_365))
                c2.metric("Static Basket α (365D)", fmt_pct(basket_alpha_365))
                c3.metric("Engine − Basket", fmt_pct(delta))

# -------------------------
# TAB: Wave Doctor (Shadow Simulation)
# -------------------------
with tabs[2]:
    st.subheader("Wave Doctor — What-If Lab (Shadow)")
    st.caption("This does NOT change engine math. It simulates a simple tilt on holdings weights and recomputes a shadow basket NAV.")

    hold = get_wave_holdings(selected_wave)
    if yf is None:
        st.info("Wave Doctor unavailable: yfinance not installed.")
    elif hold is None or hold.empty:
        st.info("Wave Doctor unavailable: holdings missing.")
    else:
        tilt_top = st.slider("Tilt Top-10 weights (%)", min_value=-25, max_value=25, value=0, step=1)
        tickers = hold["Ticker"].astype(str).str.upper().str.strip().tolist()
        wts = hold["Weight"].astype(float).values
        # Apply tilt to top names (simple)
        w2 = wts.copy()
        top_n = min(10, len(w2))
        if top_n > 0:
            bump = (tilt_top / 100.0)
            w2[:top_n] = np.clip(w2[:top_n] * (1.0 + bump), 0.0, None)
            w2 = w2 / w2.sum() if w2.sum() > 0 else w2

        prices = fetch_prices_daily(tickers, days=min(days, 365))
        if prices is None or prices.empty:
            st.info("No price data for holdings.")
        else:
            prices = prices.dropna(how="all").ffill().bfill()
            px_norm = prices / prices.iloc[0]
            basket_base = (px_norm * wts).sum(axis=1)
            basket_tilt = (px_norm * w2).sum(axis=1)

            out = pd.concat([
                basket_base.rename("Base Basket"),
                basket_tilt.rename("Tilted Basket"),
            ], axis=1).dropna()
            st.line_chart(out)

            st.write("Shadow result (not engine):")
            c1, c2 = st.columns(2)
            c1.metric("Base Basket Return", fmt_pct(out["Base Basket"].iloc[-1] / out["Base Basket"].iloc[0] - 1.0))
            c2.metric("Tilted Basket Return", fmt_pct(out["Tilted Basket"].iloc[-1] / out["Tilted Basket"].iloc[0] - 1.0))

# -------------------------
# TAB: Market Intel
# -------------------------
with tabs[3]:
    st.subheader("Market Intel")
    st.caption("Quick read on SPY / QQQ / IWM / TLT / GLD / BTC / VIX / TNX (if available).")

    if yf is None:
        st.info("Market Intel unavailable: yfinance not installed.")
    else:
        symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"]
        px = fetch_prices_daily(symbols, days=90)
        if px is None or px.empty:
            st.info("No market prices returned.")
        else:
            rets = (px / px.shift(1) - 1.0).iloc[-1]
            rows = []
            for s in symbols:
                v = float(rets.get(s, np.nan))
                rows.append({"Symbol": s, "1D Return": v})
            df = pd.DataFrame(rows)
            try:
                st.dataframe(_style_pos_neg_pct(df, ["1D Return"]), use_container_width=True)
            except Exception:
                df["1D Return"] = df["1D Return"].apply(fmt_pct)
                st.dataframe(df, use_container_width=True)

# -------------------------
# TAB: Factor Decomposition
# -------------------------
with tabs[4]:
    st.subheader("Factor Decomposition (Light)")
    st.caption("Beta vs benchmark from daily returns.")

    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough history.")
    else:
        b = beta_ols(hist["wave_ret"], hist["bm_ret"])
        st.metric("Beta vs Benchmark", fmt_num(b, 2))

# -------------------------
# TAB: Risk Lab
# -------------------------
with tabs[5]:
    st.subheader("Risk Lab")
    if hist is None or hist.empty or len(hist) < 50:
        st.info("Not enough data to compute risk lab metrics.")
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

        st.write("Rolling 30D Alpha + Rolling Vol")
        ra = rolling_alpha_from_nav(hist["wave_nav"], hist["bm_nav"], window=30)
        rv = rolling_vol(hist["wave_ret"], window=20)
        roll_df = pd.concat([ra.rename("Rolling 30D Alpha"), rv.rename("Rolling Vol (20D)")], axis=1).dropna()
        if not roll_df.empty:
            st.line_chart(roll_df)
        else:
            st.info("Not enough data for rolling series.")

# -------------------------
# TAB: Correlation
# -------------------------
with tabs[6]:
    st.subheader("Correlation (Daily Returns)")
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

# -------------------------
# TAB: WaveScore Leaderboard
# -------------------------
with tabs[7]:
    st.subheader("WaveScore Leaderboard (Console-side)")
    if ws_df is None or ws_df.empty:
        st.info("WaveScore unavailable (insufficient history).")
    else:
        st.dataframe(ws_df, use_container_width=True)

# -------------------------
# TAB: Vector OS Insight Layer
# -------------------------
with tabs[8]:
    st.subheader("Vector OS Insight Layer")
    if hist is None or hist.empty or len(hist) < 20:
        st.info("Not enough data for insights yet.")
    else:
        notes = []
        if cov.get("flags"):
            notes.append("**Data Integrity Flags:** " + "; ".join(cov["flags"]))
        if bm_drift != "stable":
            notes.append("**Benchmark Drift:** Snapshot changed during session — freeze benchmark mix for demos.")
        if math.isfinite(a30) and abs(a30) >= 0.08:
            notes.append("**Large 30D alpha:** verify benchmark mix + missing days; big alpha can be real or coverage-driven.")
        if math.isfinite(te) and te >= 0.20:
            notes.append("**High tracking error:** wave behaving very differently than benchmark (active risk elevated).")
        if math.isfinite(mdd) and mdd <= -0.25:
            notes.append("**Deep drawdown:** consider stronger SmartSafe posture in stress regimes.")
        if not notes:
            notes.append("No major anomalies detected on this window.")
        for n in notes:
            st.markdown(f"- {n}")

# ============================================================
# Footer diagnostics (prevents silent deaths)
# ============================================================
with st.expander("System Diagnostics (if something looks off)"):
    st.write("Engine loaded:", we is not None)
    st.write("Engine import error:", str(ENGINE_IMPORT_ERROR) if ENGINE_IMPORT_ERROR else "None")
    st.write("Files present:", {p: os.path.exists(p) for p in ["wave_config.csv", "wave_weights.csv", "wave_history.csv", "list.csv", "waves_engine.py"]})
    st.write("Selected:", {"wave": selected_wave, "mode": mode, "days": days})
    st.write("History shape:", None if hist is None else hist.shape)
    if hist is not None and not hist.empty:
        st.write("History columns:", list(hist.columns))
        st.write("History tail:", hist.tail(5))