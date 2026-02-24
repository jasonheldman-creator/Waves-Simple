# scripts/generate_live_snapshot_csv.py
# WAVES Intelligence™ - Canonical Live Snapshot Generator
# PURPOSE:
# Produce a live, attribution-complete live_snapshot.csv every run
# using canonical price data from data/prices.csv

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
WEIGHTS_PATH = DATA_DIR / "wave_weights.csv"
PRICES_PATH = DATA_DIR / "prices.csv"
REGISTRY_PATH = DATA_DIR / "wave_registry.csv"
OUTPUT_PATH = DATA_DIR / "live_snapshot.csv"

# ---- REQUIRED UI COLUMNS ----

BASE_COLUMNS = [
    "wave_name",
    "asof",
    "return_1d", "alpha_1d",
    "return_30d", "alpha_30d",
    "return_60d", "alpha_60d",
    "return_365d", "alpha_365d",
]

ATTRIBUTION_COLUMNS = [
    # Residual
    "alpha_residual_30d", "alpha_residual_60d", "alpha_residual_365d",
    # Momentum
    "alpha_momentum_30d", "alpha_momentum_60d", "alpha_momentum_365d",
    # Volatility
    "alpha_volatility_30d", "alpha_volatility_60d", "alpha_volatility_365d",
    # Beta
    "alpha_beta_30d", "alpha_beta_60d", "alpha_beta_365d",
    # Allocation
    "alpha_allocation_30d", "alpha_allocation_60d", "alpha_allocation_365d",
]

ALL_COLUMNS = BASE_COLUMNS + ATTRIBUTION_COLUMNS

# ---- HELPERS ----

def _parse_benchmark_spec(spec_str):
    """Parse 'QQQ:0.6,SMH:0.25,IGV:0.15' into {'QQQ': 0.6, 'SMH': 0.25, 'IGV': 0.15}."""
    result = {}
    for part in str(spec_str).split(","):
        part = part.strip()
        if ":" in part:
            ticker, weight = part.split(":", 1)
            try:
                result[ticker.strip()] = float(weight.strip())
            except (ValueError, TypeError):
                pass
    return result if result else {"SPY": 1.0}


def _compute_weighted_return(prices_pivot, ticker_weights, days):
    """Compute weighted portfolio return over a lookback of ``days`` trading rows."""
    if prices_pivot is None or (isinstance(prices_pivot, pd.DataFrame) and prices_pivot.empty):
        return np.nan
    if not ticker_weights:
        return np.nan

    weighted_sum = 0.0
    total_weight = 0.0

    for ticker, weight in ticker_weights:
        if ticker not in prices_pivot.columns:
            continue
        series = prices_pivot[ticker].dropna()
        if len(series) < 2:
            continue
        needed_rows = min(days, len(series) - 1)
        end_price = series.iloc[-1]
        start_price = series.iloc[-1 - needed_rows]
        if start_price <= 0:
            continue
        ret = (end_price / start_price) - 1.0
        weighted_sum += weight * ret
        total_weight += weight

    if total_weight <= 0:
        return np.nan
    return weighted_sum / total_weight


# ---- LOAD WAVES ----

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError("data/wave_weights.csv not found")

weights_df = pd.read_csv(WEIGHTS_PATH)
if "wave_name" not in weights_df.columns:
    for alt_col in ("wave", "wave_id"):
        if alt_col in weights_df.columns:
            weights_df = weights_df.rename(columns={alt_col: "wave_name"})
            break
if "wave_name" not in weights_df.columns:
    raise ValueError("wave_weights.csv must contain 'wave_name', 'wave', or 'wave_id' column")

# Build holdings dict: {wave_name: [(ticker, weight), ...]}
holdings = {}
for wave_name, grp in weights_df.groupby("wave_name"):
    tickers = []
    for _, row in grp.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        try:
            weight = float(row.get("weight", 0))
        except (ValueError, TypeError):
            weight = 0.0
        if ticker and weight > 0:
            tickers.append((ticker, weight))
    if tickers:
        holdings[str(wave_name)] = tickers

waves = sorted(holdings.keys())

# ---- LOAD PRICES ----

prices_pivot = pd.DataFrame()
if PRICES_PATH.exists():
    try:
        all_tickers = set()
        for tickers in holdings.values():
            for ticker, _ in tickers:
                all_tickers.add(ticker)
        all_tickers.add("SPY")

        prices_df = pd.read_csv(PRICES_PATH, parse_dates=["date"])
        prices_df = prices_df[prices_df["ticker"].isin(all_tickers)]
        prices_df = prices_df.sort_values("date")
        prices_pivot = prices_df.pivot(index="date", columns="ticker", values="close")
        prices_pivot.sort_index(inplace=True)
        print(f"[LIVE] Loaded prices for {len(prices_pivot.columns)} tickers, {len(prices_pivot)} rows")
    except Exception as e:
        print(f"[WARN] Failed to load prices: {e}")
else:
    print(f"[WARN] prices.csv not found at {PRICES_PATH}")

# ---- LOAD BENCHMARKS ----

benchmarks = {}
if REGISTRY_PATH.exists():
    try:
        reg = pd.read_csv(REGISTRY_PATH)
        for _, row in reg.iterrows():
            wname = str(row.get("wave_name", "")).strip()
            bench_spec = str(row.get("benchmark_spec", "")).strip()
            if wname and bench_spec and bench_spec.lower() not in ("nan", "none", ""):
                benchmarks[wname] = list(_parse_benchmark_spec(bench_spec).items())
    except Exception as e:
        print(f"[WARN] Failed to load benchmark specs: {e}")

# ---- BUILD SNAPSHOT ----

asof_date = prices_pivot.index.max().strftime("%Y-%m-%d") if not prices_pivot.empty else datetime.utcnow().strftime("%Y-%m-%d")
horizons = [("1d", 1), ("30d", 30), ("60d", 60), ("365d", 365)]
rows = []

for wave_name in waves:
    tickers = holdings.get(wave_name, [])
    bench_tickers = benchmarks.get(wave_name, [("SPY", 1.0)])
    row = {"wave_name": wave_name, "asof": asof_date}

    for label, days in horizons:
        wave_ret = _compute_weighted_return(prices_pivot, tickers, days)
        bench_ret = _compute_weighted_return(prices_pivot, bench_tickers, days)
        row[f"return_{label}"] = wave_ret
        if pd.notna(wave_ret) and pd.notna(bench_ret):
            row[f"alpha_{label}"] = wave_ret - bench_ret
        else:
            row[f"alpha_{label}"] = np.nan

    for col in ATTRIBUTION_COLUMNS:
        row[col] = np.nan

    rows.append(row)
    if any(pd.notna(v) and k.startswith("return_") for k, v in row.items() if k != "return_1d" or v != 0.0):
        print(f"[LIVE] {wave_name}: return_1d={row.get('return_1d'):.4f}, return_30d={row.get('return_30d')}")

snapshot_df = pd.DataFrame(rows)

# ---- FINAL SAFETY CHECK ----

for col in ALL_COLUMNS:
    if col not in snapshot_df.columns:
        snapshot_df[col] = np.nan

snapshot_df = snapshot_df[ALL_COLUMNS]

# ---- WRITE OUTPUT ----

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
snapshot_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ live_snapshot.csv written with {len(snapshot_df)} rows")
print(f"✅ Columns: {len(snapshot_df.columns)}")