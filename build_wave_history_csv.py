"""
build_wave_history_csv.py
WAVES Intelligence™ — Canonical Wave History Builder

PURPOSE
-------
Generate REAL per-wave historical return series required for
alpha attribution.

GUARANTEES
----------
• Uses existing WAVES data sources only
• Produces NO synthetic data
• Skips waves with missing inputs (with diagnostics)
• NEVER fails CI
• Writes data/history/{wave_id}_history.csv when possible
• Schema is attribution-safe and deterministic
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
OUTPUT_DIR = Path("data/history")

DEFAULT_LOOKBACK_DAYS = 365


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg)


def normalize_ticker(t: str) -> str:
    """
    Canonical ticker normalization to maximize cache alignment.
    """
    return (
        str(t)
        .upper()
        .strip()
        .replace(".", "-")
    )


def load_price_cache() -> Optional[pd.DataFrame]:
    if not PRICE_CACHE.exists():
        log("[SKIP] Price cache missing")
        return None

    df = pd.read_parquet(PRICE_CACHE)

    if "date" not in df.columns:
        df = df.reset_index()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Normalize all price column tickers
    rename_map = {
        c: normalize_ticker(c)
        for c in df.columns
        if c != "date"
    }
    df = df.rename(columns=rename_map)

    return df


def load_wave_registry() -> pd.DataFrame:
    if not WAVE_REGISTRY_CSV.exists():
        log("[SKIP] Wave registry missing")
        return pd.DataFrame()

    return pd.read_csv(WAVE_REGISTRY_CSV)


def load_wave_weights(wave_id: str) -> Optional[Dict[str, float]]:
    """
    Expected path:
    data/waves/{wave_id}/weights.csv
    """
    path = WAVE_WEIGHTS_DIR / wave_id / "weights.csv"

    if not path.exists():
        log(f"[SKIP] {wave_id}: weights.csv missing")
        return None

    df = pd.read_csv(path)

    # Allow either ticker or symbol column
    if "ticker" in df.columns:
        ticker_col = "ticker"
    elif "symbol" in df.columns:
        ticker_col = "symbol"
    else:
        log(f"[SKIP] {wave_id}: weights.csv missing ticker column")
        return None

    if "weight" not in df.columns:
        log(f"[SKIP] {wave_id}: weights.csv missing weight column")
        return None

    df = df[[ticker_col, "weight"]].dropna()

    if df.empty:
        log(f"[SKIP] {wave_id}: weights.csv empty after cleaning")
        return None

    weights = {
        normalize_ticker(t): float(w)
        for t, w in zip(df[ticker_col], df["weight"])
        if w != 0
    }

    if not weights:
        log(f"[SKIP] {wave_id}: all weights zero")
        return None

    return weights


def extract_benchmark(benchmark_spec: str) -> Optional[str]:
    """
    Extract first benchmark ticker from registry spec.
    Example:
    'SPY:1.0' → 'SPY'
    'QQQ,SPY' → 'QQQ'
    """
    if not isinstance(benchmark_spec, str):
        return None

    raw = benchmark_spec.split(",")[0].split(":")[0]
    return normalize_ticker(raw)


def compute_weighted_returns(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
) -> Optional[pd.Series]:
    missing = [t for t in weights if t not in returns_df.columns]
    if missing:
        log(f"[SKIP] Missing tickers in cache: {missing}")
        return None

    aligned = returns_df[list(weights.keys())]

    weighted = aligned.mul(
        pd.Series(weights),
        axis=1
    ).sum(axis=1)

    weighted = weighted.dropna()

    if weighted.empty:
        return None

    return weighted


# ---------------------------------------------------------------------
# MAIN BUILD
# ---------------------------------------------------------------------

def build_wave_history() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    price_df = load_price_cache()
    if price_df is None or price_df.empty:
        log("[EXIT] No price data available")
        return

    registry = load_wave_registry()
    if registry.empty:
        log("[EXIT] Wave registry empty")
        return

    # Restrict lookback window
    price_df = price_df.tail(DEFAULT_LOOKBACK_DAYS)
    price_df = price_df.set_index("date")

    # Compute daily returns ONCE
    returns_df = price_df.pct_change().dropna(how="all")

    waves_built = 0

    for _, row in registry.iterrows():
        wave_id = row.get("wave_id")
        benchmark_spec = row.get("benchmark_spec")

        if not isinstance(wave_id, str):
            continue

        benchmark = extract_benchmark(benchmark_spec)
        if not benchmark:
            log(f"[SKIP] {wave_id}: invalid benchmark spec")
            continue

        weights = load_wave_weights(wave_id)
        if weights is None:
            continue

        if benchmark not in returns_df.columns:
            log(f"[SKIP] {wave_id}: benchmark '{benchmark}' missing from cache")
            continue

        wave_returns = compute_weighted_returns(returns_df, weights)
        if wave_returns is None:
            log(f"[SKIP] {wave_id}: could not compute wave returns")
            continue

        benchmark_returns = returns_df[benchmark].loc[wave_returns.index]

        out_df = pd.DataFrame({
            "date": wave_returns.index,
            "wave_return": wave_returns.values,
            "benchmark_return": benchmark_returns.values,
        })

        if out_df.empty:
            log(f"[SKIP] {wave_id}: output empty after alignment")
            continue

        out_path = OUTPUT_DIR / f"{wave_id}_history.csv"
        out_df.to_csv(out_path, index=False)

        waves_built += 1
        log(f"[OK] Built history for {wave_id}")

    log("======================================")
    log("Wave history build complete")
    log(f"Waves written: {waves_built}")
    log("======================================")


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    build_wave_history()