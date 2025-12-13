# waves_engine.py
# WAVES Intelligence™ — Engine (Console Support)
# Features:
# - Robust wave discovery (weights + orphan logs)
# - Multi-window summary (1D/30D/60D/365D)
# - Wave detail bundle (NAV, volatility regime attribution, conditional attribution grid)
# - Persistent logging (conditional attribution + recommendation events)
# - Safe preview-only auto recommendations apply (session-only; app controls it)

from __future__ import annotations

import os
import glob
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


# -----------------------------
# Config / constants
# -----------------------------
WAVE_WEIGHTS_CSV = "wave_weights.csv"

LOG_DIR = "logs"
COND_LOG_DIR = os.path.join(LOG_DIR, "attribution", "conditional")
RECO_LOG_DIR = os.path.join(LOG_DIR, "recommendations")
RECO_EVENT_LOG = os.path.join(RECO_LOG_DIR, "recommendation_events.csv")

# "SmartSafe" ticker candidate (cash proxy)
SMARTSAFE_TICKER = "SGOV"

# VIX proxy for regimes
VIX_TICKER = "^VIX"
SPY_TICKER = "SPY"

# Modes
MODE_BASE_EXPOSURE = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.85,
    "Private Logic": 1.05,
}
MODE_SMARTSAFE_BASE = {
    "Standard": 0.00,
    "Alpha-Minus-Beta": 0.05,
    "Private Logic": 0.00,
}

# Safe apply guardrails (used by app.py UI)
SAFE_APPLY_LIMITS = {
    "min_confidence_to_apply": "Medium",  # Low blocked
    "max_abs_exposure_delta": 0.10,
    "max_abs_smartsafe_delta": 0.10,
    "exposure_min": 0.00,
    "exposure_max": 1.25,
    "smartsafe_min": 0.00,
    "smartsafe_max": 0.90,
}

CONF_RANK = {"Low": 0, "Medium": 1, "High": 2}


# -----------------------------
# In-memory weights cache
# -----------------------------
WAVE_WEIGHTS: Dict[str, Dict[str, float]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dirs() -> None:
    os.makedirs(COND_LOG_DIR, exist_ok=True)
    os.makedirs(RECO_LOG_DIR, exist_ok=True)


def refresh_weights() -> None:
    """Reload wave_weights.csv into WAVE_WEIGHTS cache."""
    global WAVE_WEIGHTS
    WAVE_WEIGHTS = load_wave_weights(WAVE_WEIGHTS_CSV)


def load_wave_weights(path: str) -> Dict[str, Dict[str, float]]:
    """
    Expected columns (case-insensitive tolerant):
      Wave, Ticker, Weight
    Weight can be 0-1 or 0-100. We normalize.
    """
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    if "wave" not in cols or "ticker" not in cols or "weight" not in cols:
        # Try common alternates
        raise ValueError("wave_weights.csv must have Wave, Ticker, Weight columns")

    df = df.rename(columns={
        cols["wave"]: "wave",
        cols["ticker"]: "ticker",
        cols["weight"]: "weight",
    })

    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    # Heuristic: if weights look like 0-100, convert
    if df["weight"].max() > 2.0:
        df["weight"] = df["weight"] / 100.0

    out: Dict[str, Dict[str, float]] = {}
    for w, g in df.groupby("wave"):
        weights = {}
        for _, r in g.iterrows():
            t = r["ticker"]
            wt = float(r["weight"])
            if wt <= 0:
                continue
            weights[t] = weights.get(t, 0.0) + wt
        weights = _normalize_weights(weights)
        out[w] = weights
    return out


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    if not w:
        return {}
    s = float(sum(w.values()))
    if s <= 0:
        return {}
    return {k: float(v) / s for k, v in w.items()}


def _discover_orphan_waves_from_logs() -> List[str]:
    """
    Finds wave names from:
      logs/positions/<Wave>_positions_YYYYMMDD.csv
      logs/performance/<Wave>_performance_daily.csv
    """
    waves = set()

    # positions logs
    pos_glob = os.path.join(LOG_DIR, "positions", "*_positions_*.csv")
    for p in glob.glob(pos_glob):
        base = os.path.basename(p)
        if "_positions_" in base:
            waves.add(base.split("_positions_")[0])

    # performance logs
    perf_glob = os.path.join(LOG_DIR, "performance", "*_performance_daily.csv")
    for p in glob.glob(perf_glob):
        base = os.path.basename(p)
        if base.endswith("_performance_daily.csv"):
            waves.add(base.replace("_performance_daily.csv", ""))

    return sorted([w for w in waves if w.strip()])


def get_all_waves(include_orphans: bool = True) -> List[str]:
    """
    Primary: waves present in weights.
    Also include 'orphan' waves present in logs (so they don't disappear).
    """
    if not WAVE_WEIGHTS:
        # Lazy load
        try:
            refresh_weights()
        except Exception:
            pass

    w_from_weights = set(WAVE_WEIGHTS.keys())

    if include_orphans:
        for w in _discover_orphan_waves_from_logs():
            w_from_weights.add(w)

    return sorted(list(w_from_weights))


# -----------------------------
# Market data
# -----------------------------
def _download_history(tickers: List[str], days: int = 420) -> pd.DataFrame:
    """
    Download adjusted close for tickers. Returns dataframe indexed by date.
    Needs yfinance.
    """
    if yf is None:
        return pd.DataFrame()

    tickers = [t for t in tickers if t]
    tickers = sorted(list(dict.fromkeys(tickers)))  # unique preserve

    try:
        data = yf.download(
            tickers=tickers,
            period=f"{days}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if data is None or len(data) == 0:
        return pd.DataFrame()

    # yfinance shape varies for single vs multi ticker
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close" in data.columns.get_level_values(0)) or ("Adj Close" in data.columns.get_level_values(0)):
            # Prefer Close if present after auto_adjust
            lvl0 = data.columns.get_level_values(0)
            key = "Close" if "Close" in lvl0 else "Adj Close"
            px = data[key].copy()
        else:
            # fallback: try last level
            px = data.xs(data.columns.levels[0][0], axis=1, level=0)
    else:
        # single series
        px = data.copy()
        if isinstance(px, pd.Series):
            px = px.to_frame(name=tickers[0])
        elif px.shape[1] == 1 and px.columns[0] != tickers[0]:
            px.columns = [tickers[0]]

    px = px.dropna(how="all")
    px.index = pd.to_datetime(px.index)
    return px


def _pct_change(px: pd.DataFrame) -> pd.DataFrame:
    ret = px.pct_change().fillna(0.0)
    ret = ret.replace([np.inf, -np.inf], 0.0)
    return ret


# -----------------------------
# Benchmark construction (simple + stable)
# -----------------------------
def get_auto_benchmark_holdings(wave_name: str) -> Dict[str, float]:
    """
    Simple stable benchmark:
      SPY 60% + VTV 40%
    (This keeps a tougher benchmark than SPY alone and stays consistent.)
    """
    return {"SPY": 0.60, "VTV": 0.40}


# -----------------------------
# NAV simulation (practical)
# -----------------------------
def compute_history_nav(
    wave_name: str,
    mode: str,
    days: int = 365,
    exposure_override: Optional[float] = None,
    smartsafe_override: Optional[float] = None,
) -> pd.DataFrame:
    """
    Builds 365D history for:
      - wave_nav (dynamic overlay via VIX regimes + smartsafe)
      - bm_nav (static benchmark basket)
      - alpha (daily wave_ret - bm_ret)
    """
    if wave_name not in WAVE_WEIGHTS:
        # Lazy refresh attempt (covers "all waves always visible")
        refresh_weights()

    wave_holdings = WAVE_WEIGHTS.get(wave_name, {})
    if not wave_holdings:
        return pd.DataFrame(columns=["date", "wave_nav", "bm_nav", "alpha", "wave_ret", "bm_ret", "vix"])

    bm_holdings = get_auto_benchmark_holdings(wave_name)

    base_exp = MODE_BASE_EXPOSURE.get(mode, 1.0)
    base_ss = MODE_SMARTSAFE_BASE.get(mode, 0.0)

    exp = float(exposure_override) if exposure_override is not None else float(base_exp)
    ss = float(smartsafe_override) if smartsafe_override is not None else float(base_ss)

    # Guardrails
    exp = float(np.clip(exp, SAFE_APPLY_LIMITS["exposure_min"], SAFE_APPLY_LIMITS["exposure_max"]))
    ss = float(np.clip(ss, SAFE_APPLY_LIMITS["smartsafe_min"], SAFE_APPLY_LIMITS["smartsafe_max"]))

    # We simulate "overlay" as: VIX regime reduces exposure and increases SmartSafe weight.
    tickers = set(wave_holdings.keys()) | set(bm_holdings.keys())
    tickers.add(SMARTSAFE_TICKER)
    tickers.add(SPY_TICKER)
    tickers.add(VIX_TICKER)

    px = _download_history(sorted(list(tickers)), days=max(days + 60, 420))
    if px.empty:
        return pd.DataFrame(columns=["date", "wave_nav", "bm_nav", "alpha", "wave_ret", "bm_ret", "vix"])

    # Trim to last days
    px = px.iloc[-(days + 1):].copy()
    rets = _pct_change(px)

    # VIX series (may be missing; if missing treat as "Low")
    vix = px[VIX_TICKER] if VIX_TICKER in px.columns else pd.Series(index=px.index, data=10.0)

    # Regime-based exposure adjustment (simple but effective)
    # Low: 1.00, Med: 0.95, High: 0.85, Stress: 0.70
    def exposure_mult(v):
        if v >= 35:
            return 0.70
        if v >= 25:
            return 0.85
        if v >= 15:
            return 0.95
        return 1.00

    mult = vix.map(exposure_mult).reindex(px.index).fillna(1.0)

    # Wave returns:
    # risky sleeve = wave holdings normalized
    risky_w = _normalize_weights(wave_holdings)
    risky_ret = pd.Series(0.0, index=px.index)
    for t, w in risky_w.items():
        if t in rets.columns:
            risky_ret += w * rets[t]

    cash_ret = rets[SMARTSAFE_TICKER] if SMARTSAFE_TICKER in rets.columns else pd.Series(0.0, index=px.index)

    # dynamic exposure & smartsafe:
    # portion in risky = (1 - ss) * exp * mult
    # remainder goes to SmartSafe cash proxy
    risky_alloc = ((1.0 - ss) * exp * mult).clip(0.0, 1.25)
    cash_alloc = (1.0 - risky_alloc).clip(0.0, 1.0)

    wave_ret = risky_alloc * risky_ret + cash_alloc * cash_ret

    # Benchmark returns (static basket; no overlay)
    bm_w = _normalize_weights(bm_holdings)
    bm_ret = pd.Series(0.0, index=px.index)
    for t, w in bm_w.items():
        if t in rets.columns:
            bm_ret += w * rets[t]

    # Build NAV
    wave_nav = (1.0 + wave_ret).cumprod()
    bm_nav = (1.0 + bm_ret).cumprod()
    alpha = wave_ret - bm_ret

    out = pd.DataFrame({
        "date": px.index.astype("datetime64[ns]"),
        "wave_nav": wave_nav.values,
        "bm_nav": bm_nav.values,
        "alpha": alpha.values,
        "wave_ret": wave_ret.values,
        "bm_ret": bm_ret.values,
        "vix": vix.reindex(px.index).fillna(10.0).values,
    })
    return out.reset_index(drop=True)


# -----------------------------
# Multi-window summary
# -----------------------------
def compute_multi_window_summary(wave_name: str, mode: str) -> Dict[str, Optional[float]]:
    """
    Returns dict with returns/alpha for 1D/30D/60D/365D.
    Values are decimals (0.12 = 12%).
    """
    df = compute_history_nav(wave_name, mode, days=365)
    if df is None or df.empty:
        return {
            "1D_return": None, "1D_alpha": None,
            "30D_return": None, "30D_alpha": None,
            "60D_return": None, "60D_alpha": None,
            "365D_return": None, "365D_alpha": None,
        }

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    def window_metrics(n: int) -> Tuple[Optional[float], Optional[float]]:
        if len(df) < n + 1:
            return None, None
        tail = df.iloc[-(n + 1):]
        r = float(tail["wave_nav"].iloc[-1] / tail["wave_nav"].iloc[0] - 1.0)
        a = float((1.0 + tail["alpha"]).prod() - 1.0)
        return r, a

    r1, a1 = window_metrics(1)
    r30, a30 = window_metrics(30)
    r60, a60 = window_metrics(60)
    r365, a365 = window_metrics(365)

    return {
        "1D_return": r1, "1D_alpha": a1,
        "30D_return": r30, "30D_alpha": a30,
        "60D_return": r60, "60D_alpha": a60,
        "365D_return": r365, "365D_alpha": a365,
    }


# -----------------------------
# Volatility regime attribution
# -----------------------------
def _vol_regime(vix_val: float) -> str:
    if vix_val >= 35:
        return "Stress"
    if vix_val >= 25:
        return "High"
    if vix_val >= 15:
        return "Medium"
    return "Low"


def volatility_regime_attribution(df365: pd.DataFrame) -> pd.DataFrame:
    if df365 is None or df365.empty:
        return pd.DataFrame()

    d = df365.copy()
    d["regime"] = d["vix"].map(_vol_regime)

    rows = []
    for reg, g in d.groupby("regime"):
        wave_ret = float((1.0 + g["wave_ret"]).prod() - 1.0)
        bm_ret = float((1.0 + g["bm_ret"]).prod() - 1.0)
        alpha = float((1.0 + (g["wave_ret"] - g["bm_ret"])).prod() - 1.0)
        rows.append({"regime": reg, "days": int(len(g)), "wave_ret": wave_ret, "bm_ret": bm_ret, "alpha": alpha})

    order = ["Low", "Medium", "High", "Stress"]
    out = pd.DataFrame(rows)
    if not out.empty:
        out["regime"] = pd.Categorical(out["regime"], categories=order, ordered=True)
        out = out.sort_values("regime")
    return out


# -----------------------------
# Conditional attribution: Regime × Trend
# -----------------------------
def _trend_bucket(df365: pd.DataFrame) -> pd.Series:
    """
    Trend classification based on SPY 20-day return (Uptrend/Downtrend/Sideways).
    """
    if df365 is None or df365.empty:
        return pd.Series(dtype=str)

    # derive SPY proxy from bm_nav if SPY not explicitly in df; use bm_ret as market-ish
    # here we approximate trend from bm_nav slope over 20d
    nav = df365["bm_nav"].astype(float)
    r20 = nav.pct_change(20).fillna(0.0)

    def bucket(x):
        if x >= 0.03:
            return "Uptrend"
        if x <= -0.03:
            return "Downtrend"
        return "Sideways"

    return r20.map(bucket)


def conditional_attribution_grid(df365: pd.DataFrame) -> pd.DataFrame:
    if df365 is None or df365.empty:
        return pd.DataFrame()

    d = df365.copy()
    d["regime"] = d["vix"].map(_vol_regime)
    d["trend"] = _trend_bucket(d)

    # daily alpha series
    d["daily_alpha"] = d["wave_ret"] - d["bm_ret"]

    rows = []
    for (reg, tr), g in d.groupby(["regime", "trend"]):
        days = int(len(g))
        mean_daily_alpha = float(g["daily_alpha"].mean()) if days > 0 else 0.0
        cum_alpha = float((1.0 + g["daily_alpha"]).prod() - 1.0) if days > 0 else 0.0
        rows.append({
            "regime": reg,
            "trend": tr,
            "days": days,
            "mean_daily_alpha": mean_daily_alpha,
            "cum_alpha": cum_alpha,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    reg_order = ["Low", "Medium", "High", "Stress"]
    tr_order = ["Uptrend", "Sideways", "Downtrend"]
    out["regime"] = pd.Categorical(out["regime"], categories=reg_order, ordered=True)
    out["trend"] = pd.Categorical(out["trend"], categories=tr_order, ordered=True)
    out = out.sort_values(["regime", "trend"])
    return out


def _log_conditional_grid_event(wave: str, mode: str, grid: pd.DataFrame) -> str:
    """
    Writes a daily CSV snapshot of the conditional grid for this wave/mode.
    """
    _ensure_dirs()
    dt = datetime.now(timezone.utc).strftime("%Y%m%d")
    safe_wave = wave.replace("/", "-")
    safe_mode = mode.replace(" ", "_").replace("/", "-")
    path = os.path.join(COND_LOG_DIR, f"{safe_wave}__{safe_mode}__{dt}.csv")
    try:
        grid.to_csv(path, index=False)
    except Exception:
        pass
    return path


def _log_recommendation_event(event: Dict[str, Any]) -> None:
    """
    Appends reco events to a single CSV (persistent).
    """
    _ensure_dirs()
    cols = [
        "ts", "wave", "mode", "type", "title", "confidence",
        "why", "applied", "new_params", "reverted_to"
    ]
    row = {c: "" for c in cols}
    for k, v in event.items():
        if k in row:
            row[k] = v

    # stringify dict payloads
    for k in ["applied", "new_params", "reverted_to"]:
        if isinstance(row.get(k), (dict, list)):
            row[k] = str(row[k])

    df = pd.DataFrame([row])
    header = not os.path.exists(RECO_EVENT_LOG)
    try:
        df.to_csv(RECO_EVENT_LOG, mode="a", header=header, index=False)
    except Exception:
        pass


# -----------------------------
# Recommendations (simple, safe)
# -----------------------------
def build_recommendations(df365: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Creates small, conservative recommendations based on conditional grid.
    This is intentionally simple + capped — app controls apply and can undo.
    """
    if df365 is None or df365.empty or len(df365) < 80:
        return []

    grid = conditional_attribution_grid(df365)
    if grid.empty:
        return []

    # Example: if Low+Uptrend has strongly positive mean daily alpha, suggest small exposure increase
    pick = grid[(grid["regime"] == "Low") & (grid["trend"] == "Uptrend")]
    if pick.empty:
        return []

    days = int(pick["days"].iloc[0])
    mean_daily_alpha = float(pick["mean_daily_alpha"].iloc[0])

    # threshold: 8 bp/day over at least 30 days -> Medium
    if days >= 30 and mean_daily_alpha >= 0.0008:
        return [{
            "title": "Slightly increase exposure in benign regimes",
            "confidence": "Medium",
            "why": f"Low+Uptrend mean daily alpha is {mean_daily_alpha*10000:.1f} bp/day over {days} days.",
            "deltas": {"exposure_delta": 0.05, "smartsafe_delta": 0.00},
        }]

    return []


def apply_recommendation_preview(
    wave: str,
    mode: str,
    current_exposure: float,
    current_smartsafe: float,
    deltas: Dict[str, Any],
) -> Tuple[float, float, Dict[str, float]]:
    """
    Returns (new_exposure, new_smartsafe, applied_deltas) after applying caps.
    This does NOT persist changes; the app stores overrides session-only.
    """
    exp_delta = float(deltas.get("exposure_delta", 0.0))
    ss_delta = float(deltas.get("smartsafe_delta", 0.0))

    # Cap deltas
    exp_delta = float(np.clip(exp_delta, -SAFE_APPLY_LIMITS["max_abs_exposure_delta"], SAFE_APPLY_LIMITS["max_abs_exposure_delta"]))
    ss_delta = float(np.clip(ss_delta, -SAFE_APPLY_LIMITS["max_abs_smartsafe_delta"], SAFE_APPLY_LIMITS["max_abs_smartsafe_delta"]))

    new_exp = float(np.clip(current_exposure + exp_delta, SAFE_APPLY_LIMITS["exposure_min"], SAFE_APPLY_LIMITS["exposure_max"]))
    new_ss = float(np.clip(current_smartsafe + ss_delta, SAFE_APPLY_LIMITS["smartsafe_min"], SAFE_APPLY_LIMITS["smartsafe_max"]))

    return new_exp, new_ss, {"exposure_delta": exp_delta, "smartsafe_delta": ss_delta}


# -----------------------------
# Diagnostics
# -----------------------------
def diagnostics_for_holdings(holdings: Dict[str, float]) -> List[Dict[str, str]]:
    if not holdings:
        return [{"level": "WARN", "msg": "No holdings found for this Wave."}]

    wts = sorted(holdings.values(), reverse=True)
    top1 = wts[0] if wts else 0.0
    top3 = sum(wts[:3]) if len(wts) >= 3 else sum(wts)

    diags = []
    if top1 >= 0.70:
        diags.append({"level": "WARN", "msg": f"High single-name concentration. Top holding is {top1*100:.1f}% of the Wave."})
    if top3 >= 0.85:
        diags.append({"level": "WARN", "msg": f"High top-3 concentration. Top-3 holdings sum to {top3*100:.1f}%."})

    if not diags:
        diags.append({"level": "PASS", "msg": "No issues detected."})
    return diags


# -----------------------------
# Public bundle API used by app.py
# -----------------------------
def wave_detail_bundle(
    wave_name: str,
    mode: str,
    exposure_override: Optional[float] = None,
    smartsafe_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns dict consumed by app.py:
      df365, vol_attr, cond_grid, cond_log_path, holdings, diagnostics, recommendations
    """
    # Ensure weights loaded
    if not WAVE_WEIGHTS:
        try:
            refresh_weights()
        except Exception:
            pass

    holdings = WAVE_WEIGHTS.get(wave_name, {})
    df365 = compute_history_nav(
        wave_name,
        mode,
        days=365,
        exposure_override=exposure_override,
        smartsafe_override=smartsafe_override,
    )

    vol_attr = volatility_regime_attribution(df365) if df365 is not None and not df365.empty else pd.DataFrame()
    cond_grid = conditional_attribution_grid(df365) if df365 is not None and not df365.empty else pd.DataFrame()

    cond_log_path = ""
    if cond_grid is not None and not cond_grid.empty:
        cond_log_path = _log_conditional_grid_event(wave_name, mode, cond_grid)

    diags = diagnostics_for_holdings(holdings)
    recos = build_recommendations(df365) if df365 is not None and not df365.empty else []

    return {
        "df365": df365 if df365 is not None and not df365.empty else None,
        "vol_attr": vol_attr if vol_attr is not None and not vol_attr.empty else None,
        "cond_grid": cond_grid if cond_grid is not None and not cond_grid.empty else None,
        "cond_log_path": cond_log_path,
        "holdings": pd.DataFrame(
            [{"ticker": t, "weight": w * 100.0} for t, w in sorted(holdings.items(), key=lambda x: x[1], reverse=True)][:50]
        ) if holdings else pd.DataFrame(),
        "diagnostics": diags,
        "recommendations": recos,
    }