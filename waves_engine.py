# waves_engine.py — WAVES Intelligence™ Engine (Private Logic Phase-1 Refactor)
#
# Implements:
# 1) Private Logic regime gate (ON/OFF)
# 2) Private Logic outputs only when ON (otherwise None -> "—" in console)
# 3) Exclude defensive waves from Private Logic (SmartSafe + ladders always None)
#
# Notes:
# - This file is intentionally conservative and avoids "juicing" results.
# - It can run standalone or be called from app.py.
# - It auto-discovers waves from wave_weights.csv.

from __future__ import annotations

import os
import re
import math
import json
import time
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None


# ---------------------------
# Config
# ---------------------------

DATA_DIR = "."
DEFAULT_WEIGHTS_FILE = os.path.join(DATA_DIR, "wave_weights.csv")
DEFAULT_LIST_FILE = os.path.join(DATA_DIR, "list.csv")

LOG_DIR = os.path.join(DATA_DIR, "logs")
POS_DIR = os.path.join(LOG_DIR, "positions")
PERF_DIR = os.path.join(LOG_DIR, "performance")
META_DIR = os.path.join(LOG_DIR, "meta")

os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(PERF_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

MODE_STANDARD = "Standard"
MODE_AMB = "Alpha-Minus-Beta"
MODE_PL = "Private Logic"

# Defensive waves excluded from PL
DEFENSIVE_KEYWORDS = [
    "smartsafe",
    "money market",
    "treasury cash",
    "cash",
    "muni ladder",
    "treasury ladder",
    "ladder",
    "tax-free",
]

# Default benchmark mapping (fallback when not provided in CSV)
DEFAULT_BENCHMARKS = {
    # wave name keywords -> benchmark ticker
    "bitcoin": "BTC-USD",
    "crypto": "BTC-USD",
    "gold": "GLD",
    "energy": "XLE",
    "semis": "SOXX",
    "ai": "QQQ",
    "cloud": "QQQ",
    "mega": "SPY",
    "small": "IWM",
    "mid": "IWM",
    "infinity": "SPY",
    "muni": "MUB",
    "treasury": "SHY",
    "money market": "BIL",
    "cash": "BIL",
}


# ---------------------------
# Helpers
# ---------------------------

def _now_yyyymmdd() -> str:
    return dt.datetime.now().strftime("%Y%m%d")

def _safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _clean_ticker(t: str) -> str:
    t = str(t).strip().upper()
    t = t.replace(" ", "")
    return t

def _normalize_wave_name(name: str) -> str:
    # Keep as-is but strip weird whitespace
    return re.sub(r"\s+", " ", str(name).strip())

def _is_defensive_wave(wave_name: str) -> bool:
    s = wave_name.lower()
    return any(k in s for k in DEFENSIVE_KEYWORDS)

def _pick_benchmark_for_wave(wave_name: str) -> str:
    s = wave_name.lower()
    for k, b in DEFAULT_BENCHMARKS.items():
        if k in s:
            return b
    return "SPY"


def _download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Returns Adj Close prices indexed by date with tickers as columns.
    """
    if yf is None:
        raise RuntimeError("yfinance is not available in this environment.")

    tickers = list(dict.fromkeys([_clean_ticker(t) for t in tickers if str(t).strip()]))
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True
    )

    # yfinance shape varies for single vs multi ticker
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer Adj Close; fallback Close
        if ("Adj Close" in df.columns.get_level_values(0)):
            prices = df["Adj Close"].copy()
        else:
            prices = df["Close"].copy()
    else:
        # Single ticker series (columns are OHLC)
        if "Adj Close" in df.columns:
            prices = df[["Adj Close"]].copy()
            prices.columns = [tickers[0]]
        elif "Close" in df.columns:
            prices = df[["Close"]].copy()
            prices.columns = [tickers[0]]
        else:
            prices = pd.DataFrame()

    prices = prices.dropna(how="all")
    return prices


def _calc_return(prices: pd.Series, lookback_days: int) -> Optional[float]:
    """
    Computes simple return over lookback_days using last available <= lookback.
    Returns None if insufficient data.
    """
    if prices is None or prices.empty:
        return None
    prices = prices.dropna()
    if len(prices) < 2:
        return None

    end_val = prices.iloc[-1]
    # Find value closest to lookback_days ago
    end_date = prices.index[-1]
    start_date = end_date - pd.Timedelta(days=lookback_days)
    past = prices[prices.index <= start_date]
    if past.empty:
        return None
    start_val = past.iloc[-1]
    if start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
        return None
    return float((end_val / start_val) - 1.0)


def _weighted_portfolio_series(price_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Creates a normalized portfolio value series from prices and weights.
    Uses a simple weighted normalized price approach.
    """
    if price_df is None or price_df.empty:
        return pd.Series(dtype=float)

    # Ensure all tickers exist in price_df
    cols = [c for c in price_df.columns if c in weights]
    if not cols:
        return pd.Series(dtype=float)

    # Normalize each series to 1.0 at first valid point
    p = price_df[cols].copy()
    p = p.dropna(how="all")
    if p.empty:
        return pd.Series(dtype=float)

    # Forward fill for continuity (conservative)
    p = p.ffill()

    base = p.iloc[0]
    base = base.replace(0, pd.NA)
    norm = p.divide(base)

    w = pd.Series({c: weights.get(c, 0.0) for c in cols}, dtype=float)
    if w.abs().sum() == 0:
        return pd.Series(dtype=float)

    # Normalize weights to sum to 1
    w = w / w.sum()

    port = norm.mul(w, axis=1).sum(axis=1)
    return port


# ---------------------------
# Private Logic Regime Gate
# ---------------------------

@dataclass
class PLGateStatus:
    is_on: bool
    reasons: List[str]
    vix_last: Optional[float] = None
    vix_percentile_1y: Optional[float] = None
    corr_spy_tlt_60d: Optional[float] = None


def compute_private_logic_gate(end_date: Optional[str] = None) -> PLGateStatus:
    """
    Conservative regime gate:
      - VIX must be in the lower ~60% of its 1y distribution (i.e., not stressed)
      - 60d correlation between SPY and TLT must not be strongly positive (risk-off stress)
    If data missing -> OFF.
    """
    reasons = []
    if yf is None:
        return PLGateStatus(False, ["yfinance unavailable -> gate OFF"])

    if end_date is None:
        end_dt = dt.datetime.now().date()
    else:
        end_dt = pd.to_datetime(end_date).date()

    start_dt = end_dt - dt.timedelta(days=420)  # enough for 1y + buffer
    start = start_dt.isoformat()
    end = (end_dt + dt.timedelta(days=1)).isoformat()

    tickers = ["^VIX", "SPY", "TLT"]
    try:
        prices = _download_prices(tickers, start=start, end=end)
    except Exception as e:
        return PLGateStatus(False, [f"price fetch failed -> gate OFF ({e})"])

    # VIX
    vix = prices.get("^VIX")
    if vix is None or vix.dropna().empty:
        return PLGateStatus(False, ["no VIX data -> gate OFF"])

    vix = vix.dropna()
    vix_last = float(vix.iloc[-1])
    vix_1y = vix.iloc[-252:] if len(vix) >= 252 else vix

    # percentile (0..1)
    vix_percentile = float((vix_1y.rank(pct=True).iloc[-1])) if len(vix_1y) >= 30 else None

    # SPY/TLT 60d correlation
    spy = prices.get("SPY")
    tlt = prices.get("TLT")
    corr_60 = None
    if spy is not None and tlt is not None:
        df = pd.DataFrame({"SPY": spy, "TLT": tlt}).dropna().ffill()
        if len(df) >= 65:
            corr_60 = float(df["SPY"].pct_change().iloc[-60:].corr(df["TLT"].pct_change().iloc[-60:]))

    # Gate rules
    is_on = True

    # Rule 1: VIX percentile must be <= 0.60 (lower stress)
    if vix_percentile is None:
        is_on = False
        reasons.append("insufficient VIX history -> gate OFF")
    elif vix_percentile > 0.60:
        is_on = False
        reasons.append(f"VIX elevated (percentile={vix_percentile:.2f}) -> gate OFF")

    # Rule 2: SPY/TLT correlation must not be strongly positive (> 0.35)
    # (Strong positive correlation can signal risk-off / crisis regimes)
    if corr_60 is None:
        is_on = False
        reasons.append("insufficient SPY/TLT history -> gate OFF")
    elif corr_60 > 0.35:
        is_on = False
        reasons.append(f"risk-off correlation high (SPY/TLT corr60={corr_60:.2f}) -> gate OFF")

    if is_on and not reasons:
        reasons.append("conditions OK -> gate ON")

    return PLGateStatus(
        is_on=is_on,
        reasons=reasons,
        vix_last=vix_last,
        vix_percentile_1y=vix_percentile,
        corr_spy_tlt_60d=corr_60
    )


# ---------------------------
# Load weights
# ---------------------------

def load_wave_weights(weights_file: str = DEFAULT_WEIGHTS_FILE) -> pd.DataFrame:
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Missing wave weights file: {weights_file}")

    df = pd.read_csv(weights_file)

    # Flexible column mapping
    cols = {c.lower().strip(): c for c in df.columns}

    wave_col = cols.get("wave") or cols.get("wavename") or cols.get("wave_name") or cols.get("portfolio") or cols.get("name")
    ticker_col = cols.get("ticker") or cols.get("symbol")
    weight_col = cols.get("weight") or cols.get("pct") or cols.get("percentage") or cols.get("allocation")
    bench_col = cols.get("benchmark") or cols.get("bench") or cols.get("benchmark_ticker")

    if wave_col is None or ticker_col is None or weight_col is None:
        raise ValueError(
            "wave_weights.csv must include columns for Wave, Ticker, Weight "
            "(accepted variants: wave/wavename, ticker/symbol, weight/pct/allocation)."
        )

    df["_Wave"] = df[wave_col].apply(_normalize_wave_name)
    df["_Ticker"] = df[ticker_col].apply(_clean_ticker)
    df["_Weight"] = df[weight_col].apply(_safe_float)

    if bench_col is not None:
        df["_Benchmark"] = df[bench_col].astype(str).str.strip().replace({"": pd.NA})
    else:
        df["_Benchmark"] = pd.NA

    # Drop empties
    df = df[df["_Wave"].astype(str).str.len() > 0]
    df = df[df["_Ticker"].astype(str).str.len() > 0]
    df = df[df["_Weight"].abs() > 0]

    return df[["_Wave", "_Ticker", "_Weight", "_Benchmark"]].copy()


def build_wave_positions(weights_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Returns {Wave: {Ticker: Weight}} with deduped tickers and normalized weights.
    """
    out: Dict[str, Dict[str, float]] = {}
    for wave, g in weights_df.groupby("_Wave"):
        tick_w = {}
        for _, r in g.iterrows():
            t = r["_Ticker"]
            w = float(r["_Weight"])
            tick_w[t] = tick_w.get(t, 0.0) + w
        # Normalize
        total = sum(abs(v) for v in tick_w.values())
        if total == 0:
            continue
        # Keep sign if you ever short; normalize by sum of weights
        s = sum(tick_w.values())
        if s == 0:
            # fallback to abs normalization
            tick_w = {k: v / total for k, v in tick_w.items()}
        else:
            tick_w = {k: v / s for k, v in tick_w.items()}
        out[wave] = tick_w
    return out


def build_wave_benchmarks(weights_df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for wave, g in weights_df.groupby("_Wave"):
        b = None
        if g["_Benchmark"].notna().any():
            # take first non-null
            b = str(g[g["_Benchmark"].notna()]["_Benchmark"].iloc[0]).strip()
            if b == "" or b.lower() == "nan":
                b = None
        out[wave] = _clean_ticker(b) if b else _pick_benchmark_for_wave(wave)
    return out


# ---------------------------
# Core compute
# ---------------------------

@dataclass
class WaveMetrics:
    wave: str
    mode: str
    benchmark: str
    ret_365: Optional[float]
    alpha_365: Optional[float]
    meta: Dict[str, object]


def compute_wave_metrics(
    positions: Dict[str, Dict[str, float]],
    benchmarks: Dict[str, str],
    mode: str,
    pl_gate: Optional[PLGateStatus] = None,
) -> List[WaveMetrics]:
    """
    Computes 365D return and 365D alpha (return - benchmark return).
    For Private Logic:
      - If gate OFF: all waves return None
      - If wave defensive: None
    """
    # Determine date window (a bit more than 365 for safety)
    end_dt = dt.datetime.now().date()
    start_dt = end_dt - dt.timedelta(days=430)
    start = start_dt.isoformat()
    end = (end_dt + dt.timedelta(days=1)).isoformat()

    # Pre-decide PL behavior
    pl_on = True
    pl_reasons = []
    if mode == MODE_PL:
        if pl_gate is None:
            pl_gate = compute_private_logic_gate()
        pl_on = bool(pl_gate.is_on)
        pl_reasons = pl_gate.reasons

    # Gather all tickers to fetch
    all_tickers = set()
    for wave, wmap in positions.items():
        for t in wmap.keys():
            all_tickers.add(t)
    for wave, b in benchmarks.items():
        all_tickers.add(b)

    prices = _download_prices(sorted(list(all_tickers)), start=start, end=end)

    metrics: List[WaveMetrics] = []
    for wave, wmap in positions.items():
        bench = benchmarks.get(wave, _pick_benchmark_for_wave(wave))

        # Private Logic exclusions / gate
        if mode == MODE_PL:
            if not pl_on:
                metrics.append(WaveMetrics(
                    wave=wave,
                    mode=mode,
                    benchmark=bench,
                    ret_365=None,
                    alpha_365=None,
                    meta={"pl_active": False, "pl_reasons": pl_reasons, "excluded": False}
                ))
                continue
            if _is_defensive_wave(wave):
                metrics.append(WaveMetrics(
                    wave=wave,
                    mode=mode,
                    benchmark=bench,
                    ret_365=None,
                    alpha_365=None,
                    meta={"pl_active": True, "pl_reasons": pl_reasons, "excluded": True}
                ))
                continue

        # Compute portfolio series and benchmark series
        port_series = _weighted_portfolio_series(prices, wmap)
        bench_series = prices.get(bench)

        ret365 = _calc_return(port_series, 365) if not port_series.empty else None
        bret365 = _calc_return(bench_series, 365) if bench_series is not None else None

        alpha365 = None
        if ret365 is not None and bret365 is not None:
            alpha365 = ret365 - bret365

        metrics.append(WaveMetrics(
            wave=wave,
            mode=mode,
            benchmark=bench,
            ret_365=ret365,
            alpha_365=alpha365,
            meta={
                "pl_active": (pl_on if mode == MODE_PL else None),
                "pl_reasons": pl_reasons if mode == MODE_PL else None,
                "excluded": (_is_defensive_wave(wave) if mode == MODE_PL else None),
            }
        ))

    return metrics


def write_logs(positions: Dict[str, Dict[str, float]], metrics: List[WaveMetrics]) -> None:
    today = _now_yyyymmdd()

    # positions logs
    for wave, wmap in positions.items():
        rows = [{"Wave": wave, "Ticker": t, "Weight": w} for t, w in sorted(wmap.items())]
        dfp = pd.DataFrame(rows)
        safe_wave = re.sub(r"[^A-Za-z0-9_\-]+", "_", wave)
        dfp.to_csv(os.path.join(POS_DIR, f"{safe_wave}_positions_{today}.csv"), index=False)

    # performance log (one consolidated file for the day)
    rows = []
    for m in metrics:
        rows.append({
            "Date": today,
            "Wave": m.wave,
            "Mode": m.mode,
            "Benchmark": m.benchmark,
            "Return_365D": m.ret_365,
            "Alpha_365D": m.alpha_365,
            "MetaJSON": json.dumps(m.meta, default=str),
        })
    dfo = pd.DataFrame(rows)
    dfo.to_csv(os.path.join(PERF_DIR, f"portfolio_overview_{today}.csv"), index=False)


def run_engine(
    weights_file: str = DEFAULT_WEIGHTS_FILE,
    mode: str = MODE_STANDARD,
) -> Tuple[Dict[str, Dict[str, float]], List[WaveMetrics], Dict[str, object]]:
    """
    Main entrypoint.
    Returns (positions, metrics, diagnostics)
    """
    weights_df = load_wave_weights(weights_file)
    positions = build_wave_positions(weights_df)
    benchmarks = build_wave_benchmarks(weights_df)

    pl_gate = None
    if mode == MODE_PL:
        pl_gate = compute_private_logic_gate()

    metrics = compute_wave_metrics(positions, benchmarks, mode, pl_gate=pl_gate)

    diagnostics = {
        "mode": mode,
        "timestamp": dt.datetime.now().isoformat(),
        "pl_gate": (pl_gate.__dict__ if pl_gate else None),
        "waves_count": len(positions),
    }

    # Write logs (safe to keep)
    try:
        write_logs(positions, metrics)
        # write meta
        with open(os.path.join(META_DIR, f"diagnostics_{_now_yyyymmdd()}.json"), "w") as f:
            json.dump(diagnostics, f, indent=2, default=str)
    except Exception:
        pass

    return positions, metrics, diagnostics


if __name__ == "__main__":
    # Quick manual run
    for m in [MODE_STANDARD, MODE_AMB, MODE_PL]:
        print(f"\nRunning mode: {m}")
        _, metrics, diag = run_engine(mode=m)
        print(diag)
        for x in metrics[:5]:
            print(x.wave, x.ret_365, x.alpha_365)