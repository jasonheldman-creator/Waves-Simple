# waves_engine.py
# WAVES Intelligence™ — Engine (All-waves discovery + Conditional Attribution logging + safe auto-apply)
# Designed to be resilient: NEVER hide waves; show placeholders when data is insufficient.

from __future__ import annotations

import os
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


# -----------------------------
# Paths / Config
# -----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
WAVE_WEIGHTS_CSV = os.path.join(ROOT, "wave_weights.csv")
LIST_CSV = os.path.join(ROOT, "list.csv")

LOG_DIR = os.path.join(ROOT, "logs")
LOG_PERF_DIR = os.path.join(LOG_DIR, "performance")
LOG_POS_DIR = os.path.join(LOG_DIR, "positions")
LOG_DIAG_DIR = os.path.join(LOG_DIR, "diagnostics")
LOG_ATTR_DIR = os.path.join(LOG_DIR, "attribution")
LOG_RECO_DIR = os.path.join(LOG_DIR, "recommendations")

for _d in [LOG_DIR, LOG_PERF_DIR, LOG_POS_DIR, LOG_DIAG_DIR, LOG_ATTR_DIR, LOG_RECO_DIR]:
    os.makedirs(_d, exist_ok=True)


# -----------------------------
# Regime configuration (VIX-based)
# -----------------------------
VIX_TICKER = "^VIX"
BTC_TICKER = "BTC-USD"

REGIMES = [
    ("Low", 0.0, 15.0),
    ("Medium", 15.0, 25.0),
    ("High", 25.0, 35.0),
    ("Stress", 35.0, float("inf")),
]


MODE_BASE_EXPOSURE = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.85,
    "Private Logic": 0.70,
}

MODE_SMARTSAFE_BASE = {
    "Standard": 0.00,
    "Alpha-Minus-Beta": 0.10,
    "Private Logic": 0.20,
}

SAFE_APPLY_LIMITS = {
    "max_exposure_delta": 0.10,      # cap per apply
    "max_smartsafe_delta": 0.15,     # cap per apply
    "min_confidence_to_apply": "Medium",  # Low/Medium/High
}

CONF_RANK = {"Low": 0, "Medium": 1, "High": 2}


# -----------------------------
# Utility
# -----------------------------
def _today_yyyymmdd() -> str:
    return datetime.utcnow().strftime("%Y%m%d")


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _pct(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return x * 100.0


def _ensure_yf():
    if yf is None:
        raise RuntimeError("yfinance is not available. Add it to requirements.txt")


# -----------------------------
# Parsing wave_weights.csv flexibly
# Expected common formats:
# 1) columns: Wave, Ticker, Weight
# 2) columns: wave_name, ticker, weight
# 3) wide format: first col Wave, then tickers as columns with weights (rare)
# -----------------------------
def _load_wave_weights() -> Dict[str, pd.Series]:
    if not os.path.exists(WAVE_WEIGHTS_CSV):
        return {}

    df = pd.read_csv(WAVE_WEIGHTS_CSV)

    # normalize columns
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    lower = {c.lower(): c for c in df.columns}

    # long format
    if ("wave" in lower or "wave_name" in lower) and ("ticker" in lower) and ("weight" in lower):
        wave_col = lower.get("wave", lower.get("wave_name"))
        tick_col = lower["ticker"]
        w_col = lower["weight"]

        df[wave_col] = df[wave_col].astype(str).str.strip()
        df[tick_col] = df[tick_col].astype(str).str.strip().str.upper()
        df[w_col] = pd.to_numeric(df[w_col], errors="coerce").fillna(0.0)

        weights: Dict[str, pd.Series] = {}
        for wave, g in df.groupby(wave_col):
            g = g.copy()
            g = g[g[tick_col].notna() & (g[tick_col] != "")]
            if len(g) == 0:
                weights[wave] = pd.Series(dtype=float)
                continue
            s = g.groupby(tick_col)[w_col].sum()
            s = s[s != 0.0]
            # normalize
            if s.sum() != 0:
                s = s / s.sum()
            weights[wave] = s.sort_values(ascending=False)
        return weights

    # fallback wide: first column is Wave; other columns tickers
    first = df.columns[0]
    weights: Dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        wave = str(row[first]).strip()
        vals = {}
        for c in df.columns[1:]:
            t = str(c).strip().upper()
            v = _safe_float(row[c])
            if v is None or v == 0:
                continue
            vals[t] = v
        s = pd.Series(vals, dtype=float)
        if s.sum() != 0:
            s = s / s.sum()
        weights[wave] = s.sort_values(ascending=False)
    return weights


WAVE_WEIGHTS: Dict[str, pd.Series] = _load_wave_weights()


def refresh_weights() -> None:
    global WAVE_WEIGHTS
    WAVE_WEIGHTS = _load_wave_weights()


def _discover_log_waves() -> List[str]:
    waves = set()
    if os.path.isdir(LOG_PERF_DIR):
        for fn in os.listdir(LOG_PERF_DIR):
            if fn.endswith("_performance_daily.csv"):
                wave = fn.replace("_performance_daily.csv", "")
                waves.add(wave)
    if os.path.isdir(LOG_POS_DIR):
        for fn in os.listdir(LOG_POS_DIR):
            if "_positions_" in fn and fn.endswith(".csv"):
                wave = fn.split("_positions_")[0]
                waves.add(wave)
    return sorted(waves)


def get_all_waves() -> List[str]:
    """
    SOURCE OF TRUTH:
      1) all Waves present in wave_weights.csv
      2) plus any waves present in logs (so nothing is orphaned)
    """
    refresh_weights()
    waves_from_weights = set(WAVE_WEIGHTS.keys())
    waves_from_logs = set(_discover_log_waves())
    all_waves = sorted(waves_from_weights.union(waves_from_logs))
    return all_waves


# -----------------------------
# Benchmark mapping (simple default)
# If you already have custom mappings elsewhere, you can replace this.
# -----------------------------
DEFAULT_BENCHMARK = "SPY"

def get_benchmark_ticker_for_wave(wave_name: str) -> str:
    # Keep it simple + resilient. If you have per-wave mappings, add here.
    # You can also encode benchmark tickers in wave_weights.csv if desired.
    return DEFAULT_BENCHMARK


# -----------------------------
# Price history
# -----------------------------
_PRICE_CACHE: Dict[Tuple[Tuple[str, ...], str, str], pd.DataFrame] = {}

def _download_history(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    _ensure_yf()
    tickers = sorted(list({t for t in tickers if isinstance(t, str) and t.strip() != ""}))
    if not tickers:
        return pd.DataFrame()

    key = (tuple(tickers), start, end)
    if key in _PRICE_CACHE:
        return _PRICE_CACHE[key].copy()

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yf returns multiindex for many tickers, series for one
    if isinstance(data.columns, pd.MultiIndex):
        # use Adj Close equivalent: with auto_adjust True, "Close" is adjusted
        if ("Close" in data.columns.get_level_values(0)):
            close = data["Close"].copy()
        else:
            close = data.xs("Close", axis=1, level=0, drop_level=True)
    else:
        # single ticker
        close = data[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.dropna(how="all")
    _PRICE_CACHE[key] = close.copy()
    return close


def _returns_from_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    ret = price_df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return ret


# -----------------------------
# NAV simulation (lightweight)
# -----------------------------
@dataclass
class SimConfig:
    mode: str = "Standard"
    exposure: float = 1.0
    smartsafe: float = 0.0


def _normalize_weights(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s = s.copy()
    s.index = s.index.astype(str).str.upper()
    s = s.groupby(s.index).sum()
    s = s[s != 0.0]
    if s.sum() != 0:
        s = s / s.sum()
    return s.sort_values(ascending=False)


def _wave_holdings(wave_name: str) -> pd.Series:
    refresh_weights()
    s = WAVE_WEIGHTS.get(wave_name, pd.Series(dtype=float))
    return _normalize_weights(s)


def compute_history_nav(
    wave_name: str,
    mode: str,
    days: int = 365,
    end_date: Optional[datetime] = None,
    exposure_override: Optional[float] = None,
    smartsafe_override: Optional[float] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ['date','wave_nav','bm_nav','wave_ret','bm_ret','alpha','vix','regime']
    If insufficient data, returns empty df with those columns.
    """
    if end_date is None:
        end_date = datetime.utcnow()

    holdings = _wave_holdings(wave_name)
    bm_ticker = get_benchmark_ticker_for_wave(wave_name)

    cols = ["date", "wave_nav", "bm_nav", "wave_ret", "bm_ret", "alpha", "vix", "regime"]
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=cols)

    tickers = list(holdings.index)
    # include benchmark + vix
    all_tickers = list(set(tickers + [bm_ticker, VIX_TICKER]))

    start_date = (end_date - timedelta(days=int(days * 2.2))).strftime("%Y-%m-%d")
    end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    price = _download_history(all_tickers, start=start_date, end=end_str)
    if price.empty or len(price) < 30:
        return pd.DataFrame(columns=cols)

    # trim to last N trading rows (approx)
    price = price.tail(max(days + 10, 60))

    rets = _returns_from_prices(price)

    # align weights to available columns
    avail = [t for t in tickers if t in rets.columns]
    if len(avail) < 3:
        return pd.DataFrame(columns=cols)

    w = holdings.reindex(avail).fillna(0.0)
    w = _normalize_weights(w)
    if w.empty:
        return pd.DataFrame(columns=cols)

    # base mode parameters
    base_exposure = MODE_BASE_EXPOSURE.get(mode, 1.0)
    base_smartsafe = MODE_SMARTSAFE_BASE.get(mode, 0.0)

    exposure = float(exposure_override) if exposure_override is not None else base_exposure
    smartsafe = float(smartsafe_override) if smartsafe_override is not None else base_smartsafe

    # wave daily return: (1 - smartsafe)*exposure*(w•r)  (simple)
    wave_core = (rets[avail] * w.values).sum(axis=1)
    wave_ret = (1.0 - smartsafe) * exposure * wave_core

    # benchmark return
    if bm_ticker not in rets.columns:
        return pd.DataFrame(columns=cols)
    bm_ret = rets[bm_ticker].fillna(0.0)

    # alpha
    alpha = wave_ret - bm_ret

    # NAVs
    wave_nav = (1.0 + wave_ret).cumprod()
    bm_nav = (1.0 + bm_ret).cumprod()

    # vix + regime
    vix = None
    if VIX_TICKER in price.columns:
        vix = price[VIX_TICKER].copy().fillna(method="ffill").fillna(method="bfill")
    else:
        vix = pd.Series(index=wave_nav.index, data=np.nan)

    regime = pd.Series(index=wave_nav.index, dtype=str)
    for name, lo, hi in REGIMES:
        mask = (vix >= lo) & (vix < hi)
        regime.loc[mask] = name
    regime = regime.fillna("Unknown")

    out = pd.DataFrame({
        "date": wave_nav.index,
        "wave_nav": wave_nav.values,
        "bm_nav": bm_nav.values,
        "wave_ret": wave_ret.values,
        "bm_ret": bm_ret.values,
        "alpha": alpha.values,
        "vix": vix.reindex(wave_nav.index).values,
        "regime": regime.values
    })

    # hard-trim to last `days`
    if len(out) > days:
        out = out.tail(days).reset_index(drop=True)
    return out


# -----------------------------
# Summaries
# -----------------------------
def _window_return(nav: pd.Series) -> Optional[float]:
    if nav is None or len(nav) < 2:
        return None
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def compute_multi_window_summary(wave_name: str, mode: str) -> Dict[str, Optional[float]]:
    """
    Returns fractional returns (not %): intraday (1d), 30d, 60d, 365d and alpha versions.
    """
    windows = {"1D": 2, "30D": 30, "60D": 60, "365D": 365}
    out: Dict[str, Optional[float]] = {}

    for k, d in windows.items():
        df = compute_history_nav(wave_name, mode, days=d)
        if df.empty or len(df) < min(10, d // 2 + 1):
            out[f"{k}_return"] = None
            out[f"{k}_alpha"] = None
            continue
        out[f"{k}_return"] = _window_return(df["wave_nav"])
        out[f"{k}_alpha"] = _window_return((df["alpha"] + 1.0).cumprod())  # synthetic alpha NAV
    return out


def top_holdings(wave_name: str, n: int = 10) -> pd.DataFrame:
    s = _wave_holdings(wave_name)
    if s is None or s.empty:
        return pd.DataFrame(columns=["ticker", "weight"])
    s = s.head(n)
    return pd.DataFrame({"ticker": s.index, "weight": (s.values * 100.0)})


# -----------------------------
# Attribution (Vol regime + Conditional grid)
# -----------------------------
def volatility_regime_attribution(df_nav: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs df_nav from compute_history_nav() which includes alpha + regime.
    Returns table: regime, days, wave_ret, bm_ret, alpha
    """
    if df_nav is None or df_nav.empty or "regime" not in df_nav.columns:
        return pd.DataFrame(columns=["regime", "days", "wave_ret", "bm_ret", "alpha"])

    g = df_nav.groupby("regime")
    rows = []
    for rg, sub in g:
        if len(sub) < 5:
            continue
        wave = float((1.0 + sub["wave_ret"]).prod() - 1.0)
        bm = float((1.0 + sub["bm_ret"]).prod() - 1.0)
        a = float((1.0 + sub["alpha"]).prod() - 1.0)
        rows.append({"regime": rg, "days": int(len(sub)), "wave_ret": wave, "bm_ret": bm, "alpha": a})
    if not rows:
        return pd.DataFrame(columns=["regime", "days", "wave_ret", "bm_ret", "alpha"])
    return pd.DataFrame(rows).sort_values("days", ascending=False).reset_index(drop=True)


def conditional_attribution_grid(df_nav: pd.DataFrame) -> pd.DataFrame:
    """
    Simple conditional grid: regime x trend(Up/Down) using benchmark 10d slope sign.
    Outputs mean daily alpha and days.
    """
    cols = ["regime", "trend", "days", "mean_daily_alpha", "cum_alpha"]
    if df_nav is None or df_nav.empty:
        return pd.DataFrame(columns=cols)

    if "bm_nav" not in df_nav.columns or "alpha" not in df_nav.columns:
        return pd.DataFrame(columns=cols)

    bm = df_nav["bm_nav"].astype(float)
    slope = bm.pct_change(10).fillna(0.0)
    trend = np.where(slope >= 0, "Uptrend", "Downtrend")

    work = df_nav.copy()
    work["trend"] = trend

    rows = []
    for (rg, tr), sub in work.groupby(["regime", "trend"]):
        if len(sub) < 10:
            continue
        mean_alpha = float(sub["alpha"].mean())
        cum_alpha = float((1.0 + sub["alpha"]).prod() - 1.0)
        rows.append({
            "regime": rg,
            "trend": tr,
            "days": int(len(sub)),
            "mean_daily_alpha": mean_alpha,
            "cum_alpha": cum_alpha
        })

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values(["regime", "trend"]).reset_index(drop=True)


def log_conditional_attribution(wave_name: str, mode: str, grid: pd.DataFrame, df_nav: pd.DataFrame) -> str:
    """
    Writes:
      logs/attribution/<Wave>_conditional_<mode>_YYYYMMDD.csv
      logs/attribution/<Wave>_nav_<mode>_YYYYMMDD.csv  (light)
    """
    date = _today_yyyymmdd()
    safe_mode = mode.replace(" ", "_").replace("/", "_")
    grid_path = os.path.join(LOG_ATTR_DIR, f"{wave_name}_conditional_{safe_mode}_{date}.csv")
    nav_path = os.path.join(LOG_ATTR_DIR, f"{wave_name}_nav_{safe_mode}_{date}.csv")

    try:
        if grid is not None and not grid.empty:
            grid.to_csv(grid_path, index=False)
        else:
            pd.DataFrame(columns=["regime","trend","days","mean_daily_alpha","cum_alpha"]).to_csv(grid_path, index=False)

        # save a light nav file (date + alpha + regime)
        if df_nav is not None and not df_nav.empty:
            slim = df_nav[["date","wave_nav","bm_nav","alpha","regime"]].copy()
            slim.to_csv(nav_path, index=False)
        else:
            pd.DataFrame(columns=["date","wave_nav","bm_nav","alpha","regime"]).to_csv(nav_path, index=False)
    except Exception:
        pass

    return grid_path


# -----------------------------
# Diagnostics + Recommendations (preview-first)
# -----------------------------
def diagnostics_for_wave(wave_name: str) -> List[Dict]:
    """
    Simple structural checks from weights.
    """
    s = _wave_holdings(wave_name)
    diags = []

    if s is None or s.empty:
        diags.append({"level": "WARN", "msg": "No holdings found for this Wave (check wave_weights.csv)."})
        return diags

    top1 = float(s.iloc[0])
    top3 = float(s.iloc[:3].sum())

    if top1 >= 0.60:
        diags.append({"level": "WARN", "msg": f"High single-name concentration: top holding is {top1*100:.1f}%."})
    if top3 >= 0.85:
        diags.append({"level": "WARN", "msg": f"High top-3 concentration: top-3 sum to {top3*100:.1f}%."})
    if len(s) < 8:
        diags.append({"level": "INFO", "msg": f"Low breadth: only {len(s)} holdings in weights file."})

    if not diags:
        diags.append({"level": "PASS", "msg": "No issues detected."})
    return diags


def _log_recommendation_event(event: Dict) -> None:
    date = _today_yyyymmdd()
    path = os.path.join(LOG_RECO_DIR, f"recommendation_events_{date}.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass


def generate_recommendations(
    wave_name: str,
    mode: str,
    df_nav: pd.DataFrame,
    grid: pd.DataFrame
) -> List[Dict]:
    """
    Simple, safe, wave-scoped suggestions based on conditional grid.
    Output: list of dicts with preview deltas only (no persistent auto-apply unless user clicks).
    """
    recos: List[Dict] = []
    if df_nav is None or df_nav.empty or grid is None or grid.empty:
        return recos

    # Find "benign regime" (Low/Medium + Uptrend) edge: positive mean daily alpha
    benign = grid[(grid["regime"].isin(["Low","Medium"])) & (grid["trend"] == "Uptrend")].copy()
    if len(benign) >= 1:
        best = benign.sort_values("mean_daily_alpha", ascending=False).iloc[0]
        mean_bp = float(best["mean_daily_alpha"] * 10000.0)
        days = int(best["days"])

        if days >= 25 and mean_bp >= 6.0:
            # Suggest a slight exposure increase in benign regimes (preview)
            recos.append({
                "title": "Slightly increase exposure in benign regimes",
                "confidence": "Medium" if mean_bp < 12 else "High",
                "why": f"Low/Medium + Uptrend mean daily alpha is {mean_bp:.1f} bp/day over {days} days.",
                "deltas": {"exposure_delta": 0.05, "smartsafe_delta": 0.00},
                "guardrails": SAFE_APPLY_LIMITS,
            })

    return recos


def apply_recommendation_preview(
    wave_name: str,
    mode: str,
    current_exposure: float,
    current_smartsafe: float,
    deltas: Dict[str, float],
) -> Tuple[float, float, Dict]:
    """
    Applies guardrails and returns (new_exposure, new_smartsafe, applied_deltas)
    """
    exp_d = float(deltas.get("exposure_delta", 0.0))
    ss_d = float(deltas.get("smartsafe_delta", 0.0))

    # cap deltas
    exp_d = float(np.clip(exp_d, -SAFE_APPLY_LIMITS["max_exposure_delta"], SAFE_APPLY_LIMITS["max_exposure_delta"]))
    ss_d = float(np.clip(ss_d, -SAFE_APPLY_LIMITS["max_smartsafe_delta"], SAFE_APPLY_LIMITS["max_smartsafe_delta"]))

    new_exposure = float(np.clip(current_exposure + exp_d, 0.0, 1.25))
    new_smartsafe = float(np.clip(current_smartsafe + ss_d, 0.0, 0.90))

    applied = {"exposure_delta": exp_d, "smartsafe_delta": ss_d}
    return new_exposure, new_smartsafe, applied


# -----------------------------
# Optional: compute static basket return for attribution
# -----------------------------
def compute_static_basket_nav(wave_name: str, days: int = 365) -> pd.DataFrame:
    """
    Static basket = fixed weights, no smartsafe, exposure=1.0
    Useful as a baseline for overlay contribution.
    """
    df = compute_history_nav(wave_name, mode="Standard", days=days, exposure_override=1.0, smartsafe_override=0.0)
    if df.empty:
        return df
    df = df.copy()
    df["static_nav"] = df["wave_nav"]
    return df


# -----------------------------
# Convenience for app: wave detail bundle
# -----------------------------
def wave_detail_bundle(
    wave_name: str,
    mode: str,
    exposure_override: Optional[float] = None,
    smartsafe_override: Optional[float] = None,
) -> Dict:
    df365 = compute_history_nav(wave_name, mode, days=365, exposure_override=exposure_override, smartsafe_override=smartsafe_override)
    summary = compute_multi_window_summary(wave_name, mode)
    holds = top_holdings(wave_name, n=10)
    diags = diagnostics_for_wave(wave_name)

    vol_attr = volatility_regime_attribution(df365)
    cond_grid = conditional_attribution_grid(df365)

    # persistent log for conditional attribution
    log_path = log_conditional_attribution(wave_name, mode, cond_grid, df365)

    # recommendations
    recos = generate_recommendations(wave_name, mode, df365, cond_grid)

    return {
        "wave": wave_name,
        "mode": mode,
        "summary": summary,
        "holdings": holds,
        "diagnostics": diags,
        "df365": df365,
        "vol_attr": vol_attr,
        "cond_grid": cond_grid,
        "cond_log_path": log_path,
        "recommendations": recos,
    }