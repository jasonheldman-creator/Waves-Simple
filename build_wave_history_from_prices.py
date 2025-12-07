import pandas as pd
import numpy as np

# ============================================================================
# CONFIG — adjust filenames / benchmark mapping if needed
# ============================================================================

WAVE_WEIGHTS_FILE = "wave_weights.csv"   # wave,ticker,weight
PRICES_FILE = "prices.csv"              # date,ticker,close
OUTPUT_FILE = "wave_history.csv"

# Optional: map each Wave to a benchmark ticker.
# If a Wave is not in this dict, DEFAULT_BENCHMARK will be used.
BENCHMARK_BY_WAVE = {
    "S&P Wave": "SPY",
    "Growth Wave": "QQQ",
    "Future Power & Energy Wave": "XLE",   # or SPY
    "Small Cap Growth Wave": "IWM",
    "Small-Mid Cap Growth Wave": "IJH",
    "Clean Transit-Infrastructure Wave": "XTN",  # or SPY
    "Quantum Computing Wave": "QQQ",
    "Crypto Income Wave": "BTC-USD",      # only if present in prices.csv
    "Income Wave": "AGG",                 # or TLT/LQD
}

DEFAULT_BENCHMARK = "SPY"  # fallback benchmark


# ============================================================================
# CORE FUNCTION
# ============================================================================

def build_wave_history(
    wave_weights_file=WAVE_WEIGHTS_FILE,
    prices_file=PRICES_FILE,
    output_file=OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Build wave_history.csv from:
      - wave_weights.csv (wave,ticker,weight)
      - prices.csv (date,ticker,close)

    Output: wave_history.csv with columns:
      date, wave, portfolio_return, benchmark_return
    """

    # -----------------------------
    # 1) Load & clean wave_weights
    # -----------------------------
    wave_weights = pd.read_csv(wave_weights_file)
    required_ww = {"wave", "ticker", "weight"}
    if not required_ww.issubset(wave_weights.columns):
        raise ValueError(f"{wave_weights_file} must have columns {required_ww}")

    wave_weights["wave"] = wave_weights["wave"].astype(str)
    wave_weights["ticker"] = wave_weights["ticker"].astype(str).str.upper()

    # -----------------------------
    # 2) Load & clean prices
    # -----------------------------
    prices = pd.read_csv(prices_file, parse_dates=["date"])
    required_p = {"date", "ticker", "close"}
    if not required_p.issubset(prices.columns):
        raise ValueError(f"{prices_file} must have columns {required_p}")

    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices = prices.sort_values(["ticker", "date"])

    # Compute daily returns per ticker
    prices["return"] = prices.groupby("ticker")["close"].pct_change()
    returns = prices[["date", "ticker", "return"]].dropna()

    # -----------------------------
    # 3) Compute Wave portfolio returns
    # -----------------------------
    merged = returns.merge(
        wave_weights[["wave", "ticker", "weight"]],
        on="ticker",
        how="inner",
        validate="many_to_many",
    )

    merged["weighted_ret"] = merged["return"] * merged["weight"]

    wave_ret = (
        merged.groupby(["date", "wave"], as_index=False)["weighted_ret"]
        .sum()
        .rename(columns={"weighted_ret": "portfolio_return"})
    )

    # -----------------------------
    # 4) Compute benchmark returns
    # -----------------------------
    # Re-use prices for benchmarks (simple & code-only)
    bench_prices = prices[["date", "ticker", "close"]].copy()
    bench_prices = bench_prices.sort_values(["ticker", "date"])
    bench_prices["benchmark_return"] = bench_prices.groupby("ticker")["close"].pct_change()
    bench_returns = bench_prices[["date", "ticker", "benchmark_return"]].dropna()

    waves = wave_weights["wave"].unique()
    all_dates = wave_ret["date"].drop_duplicates().sort_values().to_list()

    bench_rows = []
    for wave in waves:
        bench_ticker = BENCHMARK_BY_WAVE.get(wave, DEFAULT_BENCHMARK)
        for d in all_dates:
            bench_rows.append(
                {"date": d, "wave": wave, "bench_ticker": bench_ticker}
            )

    bench_map_df = pd.DataFrame(bench_rows)

    bench_returns["ticker"] = bench_returns["ticker"].astype(str).str.upper()
    bench_map_df = bench_map_df.merge(
        bench_returns,
        left_on=["date", "bench_ticker"],
        right_on=["date", "ticker"],
        how="left",
    )

    bench_map_df = bench_map_df[["date", "wave", "benchmark_return"]]

    # -----------------------------
    # 5) Combine portfolio + benchmark returns
    # -----------------------------
    wave_hist = wave_ret.merge(
        bench_map_df,
        on=["date", "wave"],
        how="left",
        validate="one_to_one",
    )

    # Drop rows missing returns (optional)
    wave_hist = wave_hist.dropna(subset=["portfolio_return", "benchmark_return"])

    wave_hist = wave_hist.sort_values(["wave", "date"])

    # -----------------------------
    # 6) Save to CSV
    # -----------------------------
    wave_hist.to_csv(output_file, index=False)

    print(f"✔ Wrote {output_file}")
    print(f"  Waves: {wave_hist['wave'].nunique()}")
    print(f"  Dates: {wave_hist['date'].nunique()}")
    print(f"  Rows:  {len(wave_hist)}")

    return wave_hist


if __name__ == "__main__":
    build_wave_history()
