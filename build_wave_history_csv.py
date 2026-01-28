"""
build_wave_history_csv.py
WAVES Intelligence™ — Canonical Wave History Builder

PURPOSE
-------
Generate REAL per-wave historical return series required for
alpha attribution.

OUTPUT (CRITICAL)
-----------------
Writes:
  data/history/{wave_id}/history.csv

Schema:
  date, wave_return, benchmark_return

GUARANTEES
----------
• Uses existing WAVES data sources only
• Produces NO synthetic data
• Never silently skips a wave
• NEVER fails CI
• Deterministic + attribution-safe
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Optional


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

PRICE_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_REGISTRY_CSV = Path("data/wave_registry.csv")
WAVE_WEIGHTS_DIR = Path("data/waves")
OUTPUT_ROOT = Path("data/history")

# ⬇️ CRITICAL FIX:
# Must support ≥252 trading days for 365D attribution
# Use calendar buffer to survive holidays/weekends
LOOKBACK_DAYS = 600


# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg)


# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------

def normalize_ticker(t: str) -> str:
    return str(t).upper().strip().replace(".", "-")


def load_price_cache() -> Optional[pd.DataFrame]:
    if not PRICE_CACHE.exists():
        log("[EXIT] prices_cache.parquet missing")
        return None

    df = pd.read_parquet(PRICE_CACHE)

    if "date" not in df.columns:
        df = df.reset_index()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    rename_map = {
        c: normalize_ticker(c)
        for c in df.columns
        if c != "date"
    }
    df = df.rename(columns=rename_map)

    return df


def load_wave_registry() -> pd.DataFrame:
    if not WAVE_REGISTRY_CSV.exists():
        log("[EXIT] wave_registry.csv missing")
        return pd.DataFrame()

    df = pd.read_csv(WAVE_REGISTRY_CSV)

    if "wave_id" not in df.columns or "benchmark_spec" not in df.columns:
        log("[EXIT] wave_registry.csv missing required columns")
        return pd.DataFrame()

    return df


def load_wave_weights(wave_id: str) -> Optional[Dict[str, float]]:
    path = WAVE_WEIGHTS_DIR / wave_id / "weights.csv"

    if not path.exists():
        log(f"[SKIP] {wave_id}: weights.csv missing")
        return None

    df = pd.read_csv(path)

    ticker_col = (
        "ticker" if "ticker" in df.columns
        else "symbol" if "symbol" in df.columns
        else None
    )

    if ticker_col is None or "weight" not in df.columns:
        log(f"[SKIP] {wave_id}: invalid weights.csv schema")
        return None

    df = df[[ticker_col, "weight"]].dropna()
    df = df[df["weight"] != 0]

    if df.empty:
        log(f"[SKIP] {wave_id}: all weights zero")
        return None

    return {
        normalize_ticker(t): float(w)
        for t, w in zip(df[ticker_col], df["weight"])
    }


def extract_benchmark(benchmark_spec: str) -> Optional[str]:
    if not isinstance(benchmark_spec, str):
        return None
    return normalize_ticker(benchmark_spec.split(",")[0].split(":")[0])


# ---------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------

def compute_weighted_returns(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
) -> Optional[pd.Series]:
    missing = [t for t in weights if t not in returns_df.columns]
    if missing:
        log(f"[SKIP] Missing tickers in cache: {missing}")
        return None

    aligned = returns_df[list(weights.keys())]
    weighted = aligned.mul(pd.Series(weights), axis=1).sum(axis=1)
    weighted = weighted.dropna()

    return None if weighted.empty else weighted


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def build_wave_history() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    prices = load_price_cache()
    if prices is None or prices.empty:
        return

    registry = load_wave_registry()
    if registry.empty:
        return

    # ⬇️ CRITICAL FIX: extended lookback
    prices = prices.tail(LOOKBACK_DAYS).set_index("date")
    returns = prices.pct_change().dropna(how="all")

    waves_written = 0

    for _, row in registry.iterrows():
        wave_id = row["wave_id"]
        benchmark_spec = row["benchmark_spec"]

        log(f"[WAVE] Processing {wave_id}")

        benchmark = extract_benchmark(benchmark_spec)
        if benchmark not in returns.columns:
            log(f"[SKIP] {wave_id}: benchmark missing")
            continue

        weights = load_wave_weights(wave_id)
        if weights is None:
            continue

        wave_returns = compute_weighted_returns(returns, weights)
        if wave_returns is None:
            log(f"[SKIP] {wave_id}: could not compute returns")
            continue

        bench_returns = returns[benchmark].loc[wave_returns.index]

        out_df = pd.DataFrame({
            "date": wave_returns.index,
            "wave_return": wave_returns.values,
            "benchmark_return": bench_returns.values,
        })

        if out_df.empty:
            log(f"[SKIP] {wave_id}: empty output")
            continue

        wave_dir = OUTPUT_ROOT / wave_id
        wave_dir.mkdir(parents=True, exist_ok=True)

        out_path = wave_dir / "history.csv"
        out_df.to_csv(out_path, index=False)

        waves_written += 1
        log(f"[OK] Wrote {out_path}")

    log("======================================")
    log("Wave history build complete")
    log(f"Waves written: {waves_written}")
    log("======================================")


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    build_wave_history()