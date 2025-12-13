# waves_engine.py
# WAVES Intelligence™ — Engine
# Adds: Conditional Attribution Grid + persistent logging + safe recommendation preview/apply
# Keeps: Wave discovery via wave_weights.csv (so ALL waves appear)

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


# -----------------------------
# CONFIG
# -----------------------------
WEIGHTS_FILE = "wave_weights.csv"   # must exist in repo root (or adjust path)
LOG_DIR = "logs"

COND_DIR = os.path.join(LOG_DIR, "conditional")
RECO_DIR = os.path.join(LOG_DIR, "recommendations")
OVR_DIR = os.path.join(LOG_DIR, "overrides")

RECO_EVENTS_CSV = os.path.join(RECO_DIR, "reco_events.csv")
PERSIST_OVERRIDES_JSON = os.path.join(OVR_DIR, "persistent_overrides.json")

VIX_TICKER = "^VIX"
BTC_TICKER = "BTC-USD"

# Mode defaults (safe, simple)
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

# Recommendation safety limits
SAFE_APPLY_LIMITS = {
    "min_confidence_to_apply": "Medium",
    "max_abs_exposure_step": 0.10,      # per apply
    "max_abs_smartsafe_step": 0.10,     # per apply
    "exposure_bounds": (0.0, 1.25),
    "smartsafe_bounds": (0.0, 0.90),
}

CONF_RANK = {"Low": 1, "Medium": 2, "High": 3}

# For history windows
WINDOWS = {
    "1D": 1,
    "30D": 30,
    "60D": 60,
    "365D": 365,
}

# Download window (days). We fetch a bit more than 365 to compute regimes/trend cleanly.
HISTORY_LOOKBACK_DAYS = 450


# -----------------------------
# UTIL
# -----------------------------
def _ensure_dirs() -> None:
    for p in [LOG_DIR, COND_DIR, RECO_DIR, OVR_DIR]:
        os.makedirs(p, exist_ok=True)

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None

def _pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _key_wave_mode(wave: str, mode: str) -> str:
    return f"{wave}__{mode}"

def _read_csv_if_exists(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# -----------------------------
# WEIGHTS LOADER
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

    # Normalize columns
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

    # If weights in 0-100 range, keep as-is (we normalize later). If in 0-1, also ok.
    df = df[df["Wave"] != ""]
    df = df[df["Ticker"] != ""]
    _WEIGHTS_CACHE = df.copy()
    return _WEIGHTS_CACHE.copy()

def get_all_waves() -> List[str]:
    """
    Source of truth = wave_weights.csv.
    Also adds any orphan wave logs if present (optional).
    """
    df = _load_weights()
    waves = sorted(df["Wave"].dropna().unique().tolist()) if not df.empty else []

    # Optional: include orphan waves from conditional logs
    if os.path.exists(COND_DIR):
        try:
            for fn in os.listdir(COND_DIR):
                if fn.endswith(".csv") and "__" in fn:
                    w = fn.split("__")[0]
                    if w and w not in waves:
                        waves.append(w)
        except Exception:
            pass

    return sorted(list(dict.fromkeys(waves)))

def get_wave_holdings(wave: str) -> pd.DataFrame:
    df = _load_weights()
    if df.empty:
        return pd.DataFrame(columns=["ticker", "weight"])

    sub = df[df["Wave"] == wave].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ticker", "weight"])

    # Aggregate duplicates
    sub["Weight"] = pd.to_numeric(sub["Weight"], errors="coerce").fillna(0.0)
    agg = sub.groupby("Ticker", as_index=False)["Weight"].sum()
    agg = agg.rename(columns={"Ticker": "ticker", "Weight": "weight"})

    # Normalize to 100% (weight displayed as %)
    total = float(agg["weight"].sum()) if not agg.empty else 0.0
    if total > 0:
        agg["weight"] = agg["weight"] / total * 100.0

    agg = agg.sort_values("weight", ascending=False).reset_index(drop=True)
    return agg


# -----------------------------
# BENCHMARK (simple, transparent)
# -----------------------------
def get_auto_benchmark_holdings(wave: str) -> pd.DataFrame:
    """
    Simple, transparent baseline:
    - If wave contains 'Crypto' -> benchmark = BTC-USD
    - If wave contains 'Muni'/'Treasury'/'SmartSafe' -> benchmark = SGOV (or SHY if missing)
    - Else benchmark = SPY
    """
    nm = wave.lower()
    if "crypto" in nm or "bitcoin" in nm:
        return pd.DataFrame([{"ticker": BTC_TICKER, "weight": 100.0}])

    if "muni" in nm or "treasury" in nm or "smartsafe" in nm or "cash" in nm or "ladder" in nm:
        # Use SGOV as cash proxy (works better than VIX logic)
        return pd.DataFrame([{"ticker": "SGOV", "weight": 100.0}])

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

    # Use 2y to be safe; then clip to requested range
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

    # yfinance returns different shapes for single vs multi ticker
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        # We want Adj Close-like (auto_adjust True => Close is adjusted)
        close = pd.DataFrame({t: df[(t, "Close")] for t in tickers if (t, "Close") in df.columns})
    else:
        # single ticker
        close = pd.DataFrame({tickers[0]: df["Close"]})

    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    if len(close) > days:
        close = close.iloc[-days:]

    return close


# -----------------------------
# NAV + ALPHA CORE
# -----------------------------
def _portfolio_nav(price_df: pd.DataFrame, holdings: pd.DataFrame) -> pd.Series:
    """
    holdings: columns ticker, weight (in %)
    """
    if price_df.empty or holdings.empty:
        return pd.Series(dtype=float)

    h = holdings.copy()
    h["ticker"] = h["ticker"].astype(str).str.upper()
    h = h[h["ticker"].isin(price_df.columns)]
    if h.empty:
        return pd.Series(dtype=float)

    w = (h.set_index("ticker")["weight"] / 100.0).astype(float)
    # normalize in case benchmark uses 100 and wave uses <100 etc
    s = float(w.sum())
    if s <= 0:
        return pd.Series(dtype=float)
    w = w / s

    # daily returns and weighted sum
    rets = price_df[w.index].pct_change().fillna(0.0)
    port_ret = (rets * w.values).sum(axis=1)

    nav = (1.0 + port_ret).cumprod()
    return nav

def compute_history_nav(
    wave: str,
    mode: str,
    exposure_override: Optional[float] = None,
    smartsafe_override: Optional[float] = None,
) -> pd.DataFrame:
    """
    Returns df with columns: date, wave_nav, bm_nav, alpha (daily), vix
    """
    _ensure_dirs()

    base_expo = MODE_BASE_EXPOSURE.get(mode, 1.0)
    base_ss = MODE_SMARTSAFE_BASE.get(mode, 0.0)

    expo = base_expo if exposure_override is None else float(exposure_override)
    ss = base_ss if smartsafe_override is None else float(smartsafe_override)

    expo = _clamp(expo, *SAFE_APPLY_LIMITS["exposure_bounds"])
    ss = _clamp(ss, *SAFE_APPLY_LIMITS["smartsafe_bounds"])

    wave_h = get_wave_holdings(wave)
    bm_h = get_auto_benchmark_holdings(wave)

    # SmartSafe implemented as cash-like SGOV blend (simple & stable)
    # If ss>0, we blend wave holdings with SGOV.
    if ss > 0:
        cash = pd.DataFrame([{"ticker": "SGOV", "weight": 100.0}])
        # Reduce wave weights by (1-ss), allocate ss to cash
        if not wave_h.empty:
            wave_h = wave_h.copy()
            wave_h["weight"] = wave_h["weight"] * (1.0 - ss)
            cash["weight"] = cash["weight"] * ss
            wave_h = pd.concat([wave_h, cash], ignore_index=True)
            # renormalize to 100
            total = float(wave_h["weight"].sum())
            if total > 0:
                wave_h["weight"] = wave_h["weight"] / total * 100.0

    tickers = sorted(list(dict.fromkeys(
        wave_h["ticker"].tolist() + bm_h["ticker"].tolist() + [VIX_TICKER]
    )))

    px = _download_history(tickers, days=HISTORY_LOOKBACK_DAYS)
    if px.empty:
        return pd.DataFrame(columns=["date", "wave_nav", "bm_nav", "alpha", "vix"])

    # Separate VIX
    vix = px[VIX_TICKER].copy() if VIX_TICKER in px.columns else pd.Series(index=px.index, data=np.nan)
    px2 = px.drop(columns=[VIX_TICKER], errors="ignore")

    wave_nav_raw = _portfolio_nav(px2, wave_h)
    bm_nav = _portfolio_nav(px2, bm_h)

    if wave_nav_raw.empty or bm_nav.empty:
        return pd.DataFrame(columns=["date", "wave_nav", "bm_nav", "alpha", "vix"])

    # Apply exposure as a leverage scalar on daily returns (simple)
    wave_ret = _pct_change(wave_nav_raw)
    wave_ret = wave_ret * expo
    wave_nav = (1.0 + wave_ret).cumprod()

    bm_ret = _pct_change(bm_nav)
    alpha_daily = (wave_ret - bm_ret).fillna(0.0)

    out = pd.DataFrame({
        "date": wave_nav.index,
        "wave_nav": wave_nav.values,
        "bm_nav": bm_nav.values,
        "alpha": alpha_daily.values,
        "vix": vix.reindex(wave_nav.index).fillna(method="ffill").fillna(np.nan).values,
    })

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    # clip last 365 for the UI bundle computations
    if len(out) > 365:
        out = out.iloc[-365:].reset_index(drop=True)

    return out


# -----------------------------
# MULTI-WINDOW SUMMARY
# -----------------------------
def compute_multi_window_summary(wave: str, mode: str) -> dict:
    """
    Returns dict with keys: 1D_return, 1D_alpha, 30D_return, ... 365D_alpha
    Graceful if insufficient history.
    """
    df = compute_history_nav(wave, mode)
    if df.empty or len(df) < 2:
        return {
            "wave": wave, "mode": mode,
            "1D_return": None, "1D_alpha": None,
            "30D_return": None, "30D_alpha": None,
            "60D_return": None, "60D_alpha": None,
            "365D_return": None, "365D_alpha": None,
        }

    nav = df.copy()
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.set_index("date")

    wave_nav = nav["wave_nav"]
    bm_nav = nav["bm_nav"]
    alpha = nav["alpha"]

    def window_return(s: pd.Series, d: int) -> Optional[float]:
        if len(s) < d + 1:
            return None
        a = s.iloc[-(d+1)]
        b = s.iloc[-1]
        if a == 0:
            return None
        return float(b / a - 1.0)

    def window_alpha(alpha_daily: pd.Series, d: int) -> Optional[float]:
        if len(alpha_daily) < d:
            return None
        # compound alpha daily
        sub = alpha_daily.iloc[-d:]
        return float((1.0 + sub).prod() - 1.0)

    out = {"wave": wave, "mode": mode}
    for k, d in WINDOWS.items():
        out[f"{k}_return"] = window_return(wave_nav, d)
        out[f"{k}_alpha"] = window_alpha(alpha, d)

    return out


# -----------------------------
# REGIMES + TREND + CONDITIONAL GRID
# -----------------------------
def _vix_regime(vix: pd.Series) -> pd.Series:
    """
    Simple regime buckets:
    Low: < 16
    Medium: 16-22
    High: 22-30
    Stress: >= 30
    """
    v = vix.copy().astype(float)
    r = pd.Series(index=v.index, dtype=object)
    r[v < 16] = "Low"
    r[(v >= 16) & (v < 22)] = "Medium"
    r[(v >= 22) & (v < 30)] = "High"
    r[v >= 30] = "Stress"
    r = r.fillna(method="ffill").fillna("Medium")
    return r

def _trend_bucket(bm_nav: pd.Series) -> pd.Series:
    """
    Trend based on benchmark momentum:
    Uptrend if 20d MA slope positive and bm above 50d MA.
    Downtrend if bm below 50d MA and slope negative.
    Else Sideways.
    """
    s = bm_nav.copy().astype(float)
    ma20 = s.rolling(20).mean()
    ma50 = s.rolling(50).mean()
    slope = ma20.diff(5)

    trend = pd.Series(index=s.index, dtype=object)
    trend[(s > ma50) & (slope > 0)] = "Uptrend"
    trend[(s < ma50) & (slope < 0)] = "Downtrend"
    trend = trend.fillna("Sideways")
    return trend

def compute_conditional_grid(df365: pd.DataFrame) -> pd.DataFrame:
    if df365 is None or df365.empty or len(df365) < 80:
        return pd.DataFrame()

    nav = df365.copy()
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.set_index("date")

    regime = _vix_regime(nav["vix"])
    trend = _trend_bucket(nav["bm_nav"])

    alpha = nav["alpha"].copy().astype(float)

    tmp = pd.DataFrame({
        "regime": regime.values,
        "trend": trend.values,
        "alpha": alpha.values,
    }, index=nav.index)

    g = tmp.groupby(["regime", "trend"], as_index=False).agg(
        days=("alpha", "count"),
        mean_daily_alpha=("alpha", "mean"),
        cum_alpha=("alpha", lambda x: float((1.0 + x).prod() - 1.0)),
    )

    # stable ordering
    regime_order = ["Low", "Medium", "High", "Stress"]
    trend_order = ["Uptrend", "Sideways", "Downtrend"]
    g["regime"] = pd.Categorical(g["regime"], categories=regime_order, ordered=True)
    g["trend"] = pd.Categorical(g["trend"], categories=trend_order, ordered=True)
    g = g.sort_values(["regime", "trend"]).reset_index(drop=True)

    return g


# -----------------------------
# PRACTICAL ATTRIBUTION (ENGINE vs STATIC)
# -----------------------------
def compute_static_basket_return(wave: str) -> Optional[float]:
    """
    Static basket = wave holdings, no SmartSafe, no exposure scaling (expo=1.0).
    """
    df = compute_history_nav(wave, "Standard", exposure_override=1.0, smartsafe_override=0.0)
    if df.empty or len(df) < 2:
        return None
    a = float(df["wave_nav"].iloc[0])
    b = float(df["wave_nav"].iloc[-1])
    if a == 0:
        return None
    return b / a - 1.0


# -----------------------------
# DIAGNOSTICS
# -----------------------------
def compute_diagnostics(holdings: pd.DataFrame) -> List[dict]:
    diags = []
    if holdings is None or holdings.empty:
        return [{"level": "WARN", "msg": "No holdings found for this Wave."}]

    w = holdings.copy()
    if "weight" in w.columns:
        top1 = float(w["weight"].iloc[0])
        top3 = float(w["weight"].iloc[:3].sum()) if len(w) >= 3 else float(w["weight"].sum())
        if top1 >= 60:
            diags.append({"level": "WARN", "msg": f"High single-name concentration (top holding = {top1:.1f}%)."})
        if top3 >= 85:
            diags.append({"level": "WARN", "msg": f"High top-3 concentration (top-3 = {top3:.1f}%)."})

    if not diags:
        diags.append({"level": "PASS", "msg": "No issues detected."})
    return diags


# -----------------------------
# RECOMMENDATION ENGINE (simple, robust, safe)
# -----------------------------
def _recommend_from_conditional(cond: pd.DataFrame) -> List[dict]:
    """
    Generates a small set of conservative recommendations from conditional grid.
    """
    recos: List[dict] = []
    if cond is None or cond.empty:
        return recos

    # helper to find row
    def get(reg, tr):
        sub = cond[(cond["regime"] == reg) & (cond["trend"] == tr)]
        if sub.empty:
            return None
        return sub.iloc[0].to_dict()

    benign = get("Low", "Uptrend") or get("Medium", "Uptrend")
    stress = get("Stress", "Downtrend") or get("High", "Downtrend")

    # If benign regimes have strong positive mean alpha -> suggest slightly higher exposure
    if benign and benign.get("days", 0) >= 25:
        m = float(benign.get("mean_daily_alpha", 0.0))
        # 8bp/day ~ strong
        if m >= 0.0008:
            recos.append({
                "id": "benign_exposure_up",
                "title": "Slightly increase exposure in benign regimes",
                "confidence": "Medium" if m < 0.0012 else "High",
                "why": f"{benign['regime']}+{benign['trend']} mean daily alpha is {m*10000:.1f} bp/day over {int(benign['days'])} days.",
                "deltas": {"exposure_delta": +0.05, "smartsafe_delta": 0.00},
            })

    # If stress regimes are meaningfully negative -> suggest more SmartSafe in panic
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
    """
    Recommendations based on conditional attribution + basic diagnostics.
    """
    recos: List[dict] = []
    cond = compute_conditional_grid(df365)
    recos.extend(_recommend_from_conditional(cond))

    # Concentration safety reco (only if extreme)
    if holdings is not None and not holdings.empty:
        top1 = float(holdings["weight"].iloc[0])
        if top1 >= 35:
            recos.append({
                "id": "reduce_concentration",
                "title": "Reduce single-name concentration (holding design)",
                "confidence": "Medium",
                "why": f"Top holding weight is {top1:.1f}%. Consider widening holdings or capping single-name weights.",
                "deltas": {},  # informational (no auto-apply)
            })

    return recos


# -----------------------------
# APPLY / PREVIEW (safe caps)
# -----------------------------
def apply_recommendation_preview(
    wave: str,
    mode: str,
    current_exposure: float,
    current_smartsafe: float,
    deltas: dict,
) -> Tuple[float, float, dict]:
    """
    Returns (new_exposure, new_smartsafe, applied_deltas_after_caps)
    """
    exp_delta = float(deltas.get("exposure_delta", 0.0) or 0.0)
    ss_delta = float(deltas.get("smartsafe_delta", 0.0) or 0.0)

    # cap the step sizes
    exp_delta_c = _clamp(exp_delta, -SAFE_APPLY_LIMITS["max_abs_exposure_step"], SAFE_APPLY_LIMITS["max_abs_exposure_step"])
    ss_delta_c = _clamp(ss_delta, -SAFE_APPLY_LIMITS["max_abs_smartsafe_step"], SAFE_APPLY_LIMITS["max_abs_smartsafe_step"])

    new_exp = _clamp(current_exposure + exp_delta_c, *SAFE_APPLY_LIMITS["exposure_bounds"])
    new_ss = _clamp(current_smartsafe + ss_delta_c, *SAFE_APPLY_LIMITS["smartsafe_bounds"])

    applied = {"exposure_delta": exp_delta_c, "smartsafe_delta": ss_delta_c}
    return new_exp, new_ss, applied


# -----------------------------
# PERSISTENT OVERRIDES + EVENT LOGGING
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

def persist_apply(wave: str, mode: str, new_exposure: float, new_smartsafe: float, reason: str, reco_id: str = "", confidence: str = "") -> None:
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
        log_event({
            "ts": _now_iso(),
            "type": "persist_clear",
            "wave": wave,
            "mode": mode,
        })

def log_event(row: dict) -> None:
    _ensure_dirs()
    # append to CSV
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
# MAIN BUNDLE for UI
# -----------------------------
def wave_detail_bundle(
    wave: str,
    mode: str,
    exposure_override: Optional[float] = None,
    smartsafe_override: Optional[float] = None,
    include_persistent_defaults: bool = True,
) -> dict:
    """
    One-stop shop for app.py.
    - Applies persistent overrides by default (unless overridden)
    - Computes df365, holdings, conditional grid, diagnostics, recommendations
    - Writes conditional grid log
    """
    _ensure_dirs()

    # persistent override acts as baseline unless explicit overrides passed
    if include_persistent_defaults:
        povr = get_persistent_override(wave, mode)
        if exposure_override is None and "exposure" in povr:
            exposure_override = float(povr["exposure"])
        if smartsafe_override is None and "smartsafe" in povr:
            smartsafe_override = float(povr["smartsafe"])

    holdings = get_wave_holdings(wave)
    df365 = compute_history_nav(wave, mode, exposure_override=exposure_override, smartsafe_override=smartsafe_override)

    vol_attr = None
    cond_grid = None
    cond_log_path = ""

    if df365 is not None and not df365.empty and len(df365) >= 80:
        # Volatility regime attribution (simple)
        nav = df365.copy()
        nav["date"] = pd.to_datetime(nav["date"])
        nav = nav.set_index("date")

        regime = _vix_regime(nav["vix"])
        wave_ret = _pct_change(nav["wave_nav"])
        bm_ret = _pct_change(nav["bm_nav"])
        alpha = nav["alpha"]

        tmp = pd.DataFrame({
            "regime": regime.values,
            "wave_ret": wave_ret.values,
            "bm_ret": bm_ret.values,
            "alpha": alpha.values,
        }, index=nav.index)

        vol_attr = tmp.groupby("regime", as_index=False).agg(
            days=("alpha", "count"),
            wave_ret=("wave_ret", lambda x: float((1.0 + x).prod() - 1.0)),
            bm_ret=("bm_ret", lambda x: float((1.0 + x).prod() - 1.0)),
            alpha=("alpha", lambda x: float((1.0 + x).prod() - 1.0)),
        )

        cond_grid = compute_conditional_grid(df365)

        # Persistent conditional log
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