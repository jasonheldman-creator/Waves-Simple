# waves_engine.py
# WAVES Intelligence™ — Engine
# HARD RULE: Wave discovery is ONLY from wave_weights.csv (no hardcoded list).
# Includes: Conditional Attribution Grid + Vol Regime Attribution + Recommendations + Safe Apply + Persistent logs

from __future__ import annotations

import os
import json
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


# -----------------------------
# FILES / DIRS
# -----------------------------
WEIGHTS_FILE = "wave_weights.csv"

LOG_DIR = "logs"
COND_DIR = os.path.join(LOG_DIR, "conditional")
RECO_DIR = os.path.join(LOG_DIR, "recommendations")
OVR_DIR = os.path.join(LOG_DIR, "overrides")

RECO_EVENTS_CSV = os.path.join(RECO_DIR, "reco_events.csv")
PERSIST_OVERRIDES_JSON = os.path.join(OVR_DIR, "persistent_overrides.json")

VIX_TICKER = "^VIX"


# -----------------------------
# MODES
# -----------------------------
MODE_BASE_EXPOSURE = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.85,
    "Private Logic": 1.00,
}
MODE_SMARTSAFE_BASE = {
    "Standard": 0.00,
    "Alpha-Minus-Beta": 0.05,
    "Private Logic": 0.00,
}

SAFE_APPLY_LIMITS = {
    "min_confidence_to_apply": "Medium",
    "max_abs_exposure_step": 0.10,
    "max_abs_smartsafe_step": 0.10,
    "exposure_bounds": (0.0, 1.25),
    "smartsafe_bounds": (0.0, 0.90),
}
CONF_RANK = {"Low": 1, "Medium": 2, "High": 3}

WINDOWS = {"1D": 1, "30D": 30, "60D": 60, "365D": 365}
HISTORY_LOOKBACK_DAYS = 450


# -----------------------------
# UTIL
# -----------------------------
def _ensure_dirs() -> None:
    for p in [LOG_DIR, COND_DIR, RECO_DIR, OVR_DIR]:
        os.makedirs(p, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _key_wave_mode(wave: str, mode: str) -> str:
    return f"{wave}__{mode}"


# -----------------------------
# WEIGHTS LOADER (ONLY SOURCE OF WAVES)
# -----------------------------
_WEIGHTS_CACHE: Optional[pd.DataFrame] = None


def refresh_weights() -> None:
    global _WEIGHTS_CACHE
    _WEIGHTS_CACHE = None


def _load_weights() -> pd.DataFrame:
    global _WEIGHTS_CACHE
    if _WEIGHTS_CACHE is not None:
        return _WEIGHTS_CACHE.copy()

    if not os.path.exists(WEIGHTS_FILE):
        _WEIGHTS_CACHE = pd.DataFrame()
        return _WEIGHTS_CACHE.copy()

    df = pd.read_csv(WEIGHTS_FILE)

    # normalize col names
    cols = {c.lower().strip(): c for c in df.columns}
    wave_col = cols.get("wave") or cols.get("wavename") or cols.get("wave_name")
    tick_col = cols.get("ticker") or cols.get("symbol")
    wgt_col = cols.get("weight") or cols.get("wgt") or cols.get("pct")

    if not wave_col or not tick_col or not wgt_col:
        _WEIGHTS_CACHE = pd.DataFrame()
        return _WEIGHTS_CACHE.copy()

    df = df[[wave_col, tick_col, wgt_col]].rename(columns={
        wave_col: "Wave",
        tick_col: "Ticker",
        wgt_col: "Weight",
    })

    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").fillna(0.0)

    df = df[(df["Wave"] != "") & (df["Ticker"] != "")]
    _WEIGHTS_CACHE = df.copy()
    return _WEIGHTS_CACHE.copy()


def get_all_waves() -> List[str]:
    df = _load_weights()
    if df.empty:
        return []
    # exact unique set from weights file
    waves = sorted(df["Wave"].dropna().unique().tolist())
    return waves


def get_wave_holdings(wave: str) -> pd.DataFrame:
    df = _load_weights()
    if df.empty:
        return pd.DataFrame(columns=["ticker", "weight"])

    sub = df[df["Wave"] == wave].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ticker", "weight"])

    sub["Weight"] = pd.to_numeric(sub["Weight"], errors="coerce").fillna(0.0)
    agg = sub.groupby("Ticker", as_index=False)["Weight"].sum()
    agg = agg.rename(columns={"Ticker": "ticker", "Weight": "weight"})

    total = float(agg["weight"].sum()) if not agg.empty else 0.0
    if total > 0:
        agg["weight"] = agg["weight"] / total * 100.0

    agg = agg.sort_values("weight", ascending=False).reset_index(drop=True)
    return agg


# -----------------------------
# BENCHMARK (simple default)
# -----------------------------
def get_auto_benchmark_holdings(wave: str) -> pd.DataFrame:
    nm = wave.lower()

    # crypto waves -> BTC proxy
    if "crypto" in nm or "bitcoin" in nm:
        return pd.DataFrame([{"ticker": "BTC-USD", "weight": 100.0}])

    # cash-ish waves -> SGOV
    if "smartsafe" in nm or "treasury" in nm or "muni" in nm or "cash" in nm or "ladder" in nm:
        return pd.DataFrame([{"ticker": "SGOV", "weight": 100.0}])

    # default equity -> SPY
    return pd.DataFrame([{"ticker": "SPY", "weight": 100.0}])


# -----------------------------
# MARKET DATA
# -----------------------------
def _download_history(tickers: List[str], days: int = HISTORY_LOOKBACK_DAYS) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()

    tickers = [t for t in tickers if isinstance(t, str) and t.strip() != ""]
    tickers = sorted(list(dict.fromkeys(tickers)))
    if not tickers:
        return pd.DataFrame()

    try:
        df = yf.download(
            tickers=tickers,
            period="2y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        close = pd.DataFrame({t: df[(t, "Close")] for t in tickers if (t, "Close") in df.columns})
    else:
        close = pd.DataFrame({tickers[0]: df["Close"]})

    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    if len(close) > days:
        close = close.iloc[-days:]

    return close


def _portfolio_nav(price_df: pd.DataFrame, holdings: pd.DataFrame) -> pd.Series:
    if price_df.empty or holdings.empty:
        return pd.Series(dtype=float)

    h = holdings.copy()
    h["ticker"] = h["ticker"].astype(str).str.upper()
    h = h[h["ticker"].isin(price_df.columns)]
    if h.empty:
        return pd.Series(dtype=float)

    w = (h.set_index("ticker")["weight"] / 100.0).astype(float)
    s = float(w.sum())
    if s <= 0:
        return pd.Series(dtype=float)
    w = w / s

    rets = price_df[w.index].pct_change().fillna(0.0)
    port_ret = (rets * w.values).sum(axis=1)
    nav = (1.0 + port_ret).cumprod()
    return nav


# -----------------------------
# PERSISTENT OVERRIDES
# -----------------------------
def load_persistent_overrides() -> Dict[str, dict]:
    _ensure_dirs()
    if not os.path.exists(PERSIST_OVERRIDES_JSON):
        return {}
    try:
        with open(PERSIST_OVERRIDES_JSON, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_persistent_overrides(state: Dict[str, dict]) -> None:
    _ensure_dirs()
    try:
        with open(PERSIST_OVERRIDES_JSON, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
    except Exception:
        pass


def get_persistent_override(wave: str, mode: str) -> dict:
    state = load_persistent_overrides()
    return state.get(_key_wave_mode(wave, mode), {})


def persist_apply(
    wave: str,
    mode: str,
    new_exposure: float,
    new_smartsafe: float,
    reason: str,
    reco_id: str = "",
    confidence: str = ""
) -> None:
    state = load_persistent_overrides()
    k = _key_wave_mode(wave, mode)
    state[k] = {
        "exposure": float(new_exposure),
        "smartsafe": float(new_smartsafe),
        "meta": {"ts": _now_iso(), "reason": reason, "reco_id": reco_id, "confidence": confidence},
    }
    save_persistent_overrides(state)

    log_event({
        "ts": _now_iso(),
        "type": "persist_apply",
        "wave": wave,
        "mode": mode,
        "reco_id": reco_id,
        "confidence": confidence,
        "reason": reason,
        "new_exposure": float(new_exposure),
        "new_smartsafe": float(new_smartsafe),
    })


def persist_clear(wave: str, mode: str) -> None:
    state = load_persistent_overrides()
    k = _key_wave_mode(wave, mode)
    if k in state:
        del state[k]
        save_persistent_overrides(state)
        log_event({"ts": _now_iso(), "type": "persist_clear", "wave": wave, "mode": mode})


def log_event(row: dict) -> None:
    _ensure_dirs()
    df = pd.DataFrame([row])
    if os.path.exists(RECO_EVENTS_CSV):
        try:
            df.to_csv(RECO_EVENTS_CSV, mode="a", header=False, index=False)
            return
        except Exception:
            pass
    try:
        df.to_csv(RECO_EVENTS_CSV, index=False)
    except Exception:
        pass


# -----------------------------
# CORE: NAV / ALPHA
# -----------------------------
def compute_history_nav(
    wave: str,
    mode: str,
    exposure_override: Optional[float] = None,
    smartsafe_override: Optional[float] = None,
) -> pd.DataFrame:
    _ensure_dirs()

    base_expo = MODE_BASE_EXPOSURE.get(mode, 1.0)
    base_ss = MODE_SMARTSAFE_BASE.get(mode, 0.0)

    expo = base_expo if exposure_override is None else float(exposure_override)
    ss = base_ss if smartsafe_override is None else float(smartsafe_override)

    expo = _clamp(expo, *SAFE_APPLY_LIMITS["exposure_bounds"])
    ss = _clamp(ss, *SAFE_APPLY_LIMITS["smartsafe_bounds"])

    wave_h = get_wave_holdings(wave)
    bm_h = get_auto_benchmark_holdings(wave)

    # SmartSafe blend: allocate ss to SGOV, remainder to wave holdings
    if ss > 0:
        cash = pd.DataFrame([{"ticker": "SGOV", "weight": 100.0}])
        if not wave_h.empty:
            wave_h = wave_h.copy()
            wave_h["weight"] = wave_h["weight"] * (1.0 - ss)
            cash["weight"] = cash["weight"] * ss
            wave_h = pd.concat([wave_h, cash], ignore_index=True)
            total = float(wave_h["weight"].sum())
            if total > 0:
                wave_h["weight"] = wave_h["weight"] / total * 100.0

    tickers = sorted(list(dict.fromkeys(
        wave_h["ticker"].tolist() + bm_h["ticker"].tolist() + [VIX_TICKER]
    )))

    px = _download_history(tickers, days=HISTORY_LOOKBACK_DAYS)
    if px.empty:
        return pd.DataFrame(columns=["date", "wave_nav", "bm_nav", "alpha", "vix"])

    vix = px[VIX_TICKER].copy() if VIX_TICKER in px.columns else pd.Series(index=px.index, data=np.nan)
    px2 = px.drop(columns=[VIX_TICKER], errors="ignore")

    wave_nav_raw = _portfolio_nav(px2, wave_h)
    bm_nav = _portfolio_nav(px2, bm_h)

    if wave_nav_raw.empty or bm_nav.empty:
        return pd.DataFrame(columns=["date", "wave_nav", "bm_nav", "alpha", "vix"])

    wave_ret = _pct_change(wave_nav_raw) * expo
    wave_nav = (1.0 + wave_ret).cumprod()

    bm_ret = _pct_change(bm_nav)
    alpha_daily = (wave_ret - bm_ret).fillna(0.0)

    out = pd.DataFrame({
        "date": wave_nav.index,
        "wave_nav": wave_nav.values,
        "bm_nav": bm_nav.reindex(wave_nav.index).values,
        "alpha": alpha_daily.values,
        "vix": vix.reindex(wave_nav.index).ffill().values,
    })

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    if len(out) > 365:
        out = out.iloc[-365:].reset_index(drop=True)

    return out


def compute_multi_window_summary(wave: str, mode: str) -> dict:
    df = compute_history_nav(wave, mode)
    if df.empty or len(df) < 2:
        return {f"{k}_return": None for k in WINDOWS} | {f"{k}_alpha": None for k in WINDOWS}

    nav = df.copy()
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.set_index("date")

    wave_nav = nav["wave_nav"]
    alpha = nav["alpha"]

    def window_return(s: pd.Series, d: int) -> Optional[float]:
        if len(s) < d + 1:
            return None
        a = float(s.iloc[-(d + 1)])
        b = float(s.iloc[-1])
        return None if a == 0 else (b / a - 1.0)

    def window_alpha(a: pd.Series, d: int) -> Optional[float]:
        if len(a) < d:
            return None
        sub = a.iloc[-d:]
        return float((1.0 + sub).prod() - 1.0)

    out = {}
    for k, d in WINDOWS.items():
        out[f"{k}_return"] = window_return(wave_nav, d)
        out[f"{k}_alpha"] = window_alpha(alpha, d)
    return out


# -----------------------------
# REGIMES + CONDITIONAL GRID
# -----------------------------
def _vix_regime(vix: pd.Series) -> pd.Series:
    v = vix.copy().astype(float)
    r = pd.Series(index=v.index, dtype=object)
    r[v < 16] = "Low"
    r[(v >= 16) & (v < 22)] = "Medium"
    r[(v >= 22) & (v < 30)] = "High"
    r[v >= 30] = "Stress"
    return r.ffill().fillna("Medium")


def _trend_bucket(bm_nav: pd.Series) -> pd.Series:
    s = bm_nav.copy().astype(float)
    ma20 = s.rolling(20).mean()
    ma50 = s.rolling(50).mean()
    slope = ma20.diff(5)

    trend = pd.Series(index=s.index, dtype=object)
    trend[(s > ma50) & (slope > 0)] = "Uptrend"
    trend[(s < ma50) & (slope < 0)] = "Downtrend"
    return trend.fillna("Sideways")


def compute_conditional_grid(df365: pd.DataFrame) -> pd.DataFrame:
    if df365 is None or df365.empty or len(df365) < 80:
        return pd.DataFrame()

    nav = df365.copy()
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.set_index("date")

    regime = _vix_regime(nav["vix"])
    trend = _trend_bucket(nav["bm_nav"])
    alpha = nav["alpha"].astype(float)

    tmp = pd.DataFrame({"regime": regime.values, "trend": trend.values, "alpha": alpha.values}, index=nav.index)

    g = tmp.groupby(["regime", "trend"], as_index=False).agg(
        days=("alpha", "count"),
        mean_daily_alpha=("alpha", "mean"),
        cum_alpha=("alpha", lambda x: float((1.0 + x).prod() - 1.0)),
    )

    regime_order = ["Low", "Medium", "High", "Stress"]
    trend_order = ["Uptrend", "Sideways", "Downtrend"]
    g["regime"] = pd.Categorical(g["regime"], categories=regime_order, ordered=True)
    g["trend"] = pd.Categorical(g["trend"], categories=trend_order, ordered=True)
    return g.sort_values(["regime", "trend"]).reset_index(drop=True)


# -----------------------------
# DIAGNOSTICS + RECOMMENDATIONS
# -----------------------------
def compute_diagnostics(holdings: pd.DataFrame) -> List[dict]:
    if holdings is None or holdings.empty:
        return [{"level": "WARN", "msg": "No holdings found for this Wave."}]

    diags = []
    top1 = float(holdings["weight"].iloc[0])
    top3 = float(holdings["weight"].iloc[:3].sum()) if len(holdings) >= 3 else float(holdings["weight"].sum())
    if top1 >= 60:
        diags.append({"level": "WARN", "msg": f"High single-name concentration (top holding = {top1:.1f}%)."})
    if top3 >= 85:
        diags.append({"level": "WARN", "msg": f"High top-3 concentration (top-3 = {top3:.1f}%)."})
    if not diags:
        diags.append({"level": "PASS", "msg": "No issues detected."})
    return diags


def _recommend_from_conditional(cond: pd.DataFrame) -> List[dict]:
    recos: List[dict] = []
    if cond is None or cond.empty:
        return recos

    def pick(reg: str, tr: str):
        sub = cond[(cond["regime"] == reg) & (cond["trend"] == tr)]
        return None if sub.empty else sub.iloc[0].to_dict()

    benign = pick("Low", "Uptrend") or pick("Medium", "Uptrend")
    stress = pick("Stress", "Downtrend") or pick("High", "Downtrend")

    if benign and benign.get("days", 0) >= 25:
        m = float(benign.get("mean_daily_alpha", 0.0))
        if m >= 0.0008:
            recos.append({
                "id": "benign_exposure_up",
                "title": "Slightly increase exposure in benign regimes",
                "confidence": "Medium" if m < 0.0012 else "High",
                "why": f"{benign['regime']}+{benign['trend']} mean daily alpha is {m*10000:.1f} bp/day over {int(benign['days'])} days.",
                "deltas": {"exposure_delta": +0.05, "smartsafe_delta": 0.00},
            })

    if stress and stress.get("days", 0) >= 15:
        m = float(stress.get("mean_daily_alpha", 0.0))
        if m <= -0.0008:
            recos.append({
                "id": "stress_smartsafe_up",
                "title": "Increase SmartSafe in panic / stress",
                "confidence": "Medium" if m > -0.0015 else "High",
                "why": f"{stress['regime']}+{stress['trend']} mean daily alpha is {m*10000:.1f} bp/day over {int(stress['days'])} days.",
                "deltas": {"exposure_delta": 0.00, "smartsafe_delta": +0.05},
            })

    return recos


def compute_recommendations(df365: pd.DataFrame, holdings: pd.DataFrame) -> List[dict]:
    recos: List[dict] = []
    cond = compute_conditional_grid(df365)
    recos.extend(_recommend_from_conditional(cond))

    if holdings is not None and not holdings.empty:
        top1 = float(holdings["weight"].iloc[0])
        if top1 >= 35:
            recos.append({
                "id": "reduce_concentration",
                "title": "Reduce single-name concentration (holding design)",
                "confidence": "Medium",
                "why": f"Top holding weight is {top1:.1f}%. Consider widening holdings or capping single-name weights.",
                "deltas": {},
            })
    return recos


def apply_recommendation_preview(
    wave: str,
    mode: str,
    current_exposure: float,
    current_smartsafe: float,
    deltas: dict,
) -> Tuple[float, float, dict]:
    exp_delta = float(deltas.get("exposure_delta", 0.0) or 0.0)
    ss_delta = float(deltas.get("smartsafe_delta", 0.0) or 0.0)

    exp_delta_c = _clamp(exp_delta, -SAFE_APPLY_LIMITS["max_abs_exposure_step"], SAFE_APPLY_LIMITS["max_abs_exposure_step"])
    ss_delta_c = _clamp(ss_delta, -SAFE_APPLY_LIMITS["max_abs_smartsafe_step"], SAFE_APPLY_LIMITS["max_abs_smartsafe_step"])

    new_exp = _clamp(current_exposure + exp_delta_c, *SAFE_APPLY_LIMITS["exposure_bounds"])
    new_ss = _clamp(current_smartsafe + ss_delta_c, *SAFE_APPLY_LIMITS["smartsafe_bounds"])

    return new_exp, new_ss, {"exposure_delta": exp_delta_c, "smartsafe_delta": ss_delta_c}


# -----------------------------
# BUNDLE FOR UI
# -----------------------------
def wave_detail_bundle(
    wave: str,
    mode: str,
    exposure_override: Optional[float] = None,
    smartsafe_override: Optional[float] = None,
    include_persistent_defaults: bool = True,
) -> dict:
    _ensure_dirs()

    if include_persistent_defaults:
        povr = get_persistent_override(wave, mode)
        if exposure_override is None and "exposure" in povr:
            exposure_override = float(povr["exposure"])
        if smartsafe_override is None and "smartsafe" in povr:
            smartsafe_override = float(povr["smartsafe"])

    holdings = get_wave_holdings(wave)
    df365 = compute_history_nav(wave, mode, exposure_override=exposure_override, smartsafe_override=smartsafe_override)

    vol_attr = pd.DataFrame()
    cond_grid = pd.DataFrame()
    cond_log_path = ""

    if df365 is not None and not df365.empty and len(df365) >= 80:
        nav = df365.copy()
        nav["date"] = pd.to_datetime(nav["date"])
        nav = nav.set_index("date")

        regime = _vix_regime(nav["vix"])
        wave_ret = _pct_change(nav["wave_nav"])
        bm_ret = _pct_change(nav["bm_nav"])
        alpha = nav["alpha"]

        tmp = pd.DataFrame({"regime": regime.values, "wave_ret": wave_ret.values, "bm_ret": bm_ret.values, "alpha": alpha.values}, index=nav.index)
        vol_attr = tmp.groupby("regime", as_index=False).agg(
            days=("alpha", "count"),
            wave_ret=("wave_ret", lambda x: float((1.0 + x).prod() - 1.0)),
            bm_ret=("bm_ret", lambda x: float((1.0 + x).prod() - 1.0)),
            alpha=("alpha", lambda x: float((1.0 + x).prod() - 1.0)),
        )

        cond_grid = compute_conditional_grid(df365)
        cond_log_path = os.path.join(COND_DIR, f"{wave}__{mode.replace(' ', '_')}__conditional.csv")
        try:
            if cond_grid is not None and not cond_grid.empty:
                cond_grid.to_csv(cond_log_path, index=False)
        except Exception:
            pass

    diagnostics = compute_diagnostics(holdings)
    recommendations = compute_recommendations(df365, holdings)

    return {
        "wave": wave,
        "mode": mode,
        "holdings": holdings,
        "df365": df365,
        "vol_attr": vol_attr,
        "cond_grid": cond_grid,
        "cond_log_path": cond_log_path,
        "diagnostics": diagnostics,
        "recommendations": recommendations,
    }