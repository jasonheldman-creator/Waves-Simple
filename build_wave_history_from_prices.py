import pandas as pd
import numpy as np


# ------------------------
# Configuration
# ------------------------

PRICES_FILE = "prices.csv"
WAVE_WEIGHTS_FILE = "wave_weights.csv"
OUTPUT_FILE = "wave_history.csv"

# Map each Wave to its benchmark ticker (edit this as needed)
BENCHMARK_BY_WAVE = {
    "S&P Wave": "SPY",
    "Growth Wave": "SPY",
    "Small Cap Growth Wave": "IWM",
    "Small-Mid Cap Growth Wave": "IJH",
    "Quantum Computing Wave": "QQQ",
    "Crypto Income Wave": "BTC-USD",  # example only
    "Future Power & Energy Wave": "XLE",
    "Clean Transit-Infrastructure Wave": "ICLN",
    "Income Wave": "AGG",
    # add the rest of your 15 Waves here
}


# ------------------------
# Load data
# ------------------------

print("Loading prices...")
prices = pd.read_csv(PRICES_FILE, parse_dates=["date"])
prices = prices.sort_values(["ticker", "date"])

print("Loading wave weights...")
weights = pd.read_csv(WAVE_WEIGHTS_FILE)

if not {"wave", "ticker", "weight"}.issubset(weights.columns):
    raise ValueError("wave_weights.csv must contain columns: wave, ticker, weight")

# Pivot prices to wide format and compute daily returns
print("Computing daily returns...")
price_wide = prices.pivot_table(index="date", columns="ticker", values="close").sort_index()
rets = price_wide.pct_change().dropna(how="all")  # daily returns, NaN where missing


# ------------------------
# Build wave history
# ------------------------

records = []

for wave, wdf in weights.groupby("wave"):
    if wave not in BENCHMARK_BY_WAVE:
        print(f"[WARN] No benchmark defined for wave '{wave}'. Skipping.")
        continue

    bench_ticker = BENCHMARK_BY_WAVE[wave]
    if bench_ticker not in rets.columns:
        print(f"[WARN] Benchmark ticker '{bench_ticker}' not found in prices. Skipping wave '{wave}'.")
        continue

    wdf = wdf.copy()
    wdf["weight"] = wdf["weight"].astype(float)

    # Normalize weights so they sum to 1 by absolute weight (per wave)
    total_abs = wdf["weight"].abs().sum()
    if total_abs == 0:
        print(f"[WARN] Wave '{wave}' has zero total weight. Skipping.")
        continue
    wdf["norm_weight"] = wdf["weight"] / total_abs

    tickers = list(wdf["ticker"])
    missing = [t for t in tickers if t not in rets.columns]
    if missing:
        print(f"[WARN] Some tickers for wave '{wave}' not in prices: {missing}")

    # Use only tickers that we actually have returns for
    tickers = [t for t in tickers if t in rets.columns]
    if not tickers:
        print(f"[WARN] No valid tickers for wave '{wave}'. Skipping.")
        continue

    wdf = wdf[wdf["ticker"].isin(tickers)].set_index("ticker")
    wave_rets = (rets[tickers] * wdf["norm_weight"]).sum(axis=1)

    bench_rets = rets[bench_ticker]

    df_wave = pd.DataFrame({
        "date": wave_rets.index,
        "wave": wave,
        "portfolio_return": wave_rets.values,
        "benchmark_return": bench_rets.reindex(wave_rets.index).values,
    }).dropna()

    records.append(df_wave)

if not records:
    raise RuntimeError("No wave history could be built. Check inputs and mappings.")

wave_history = pd.concat(records, ignore_index=True).sort_values(["wave", "date"])

print(f"Writing {OUTPUT_FILE} with {len(wave_history)} rows...")
wave_history.to_csv(OUTPUT_FILE, index=False)
print("Done.")
