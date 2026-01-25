"""
generate_all_wave_weights.py

Creates canonical weights.csv files for ALL waves based on wave_registry.csv.
This is the missing bridge between registry-defined holdings and
alpha-attribution history building.

• No trial-and-error
• No per-wave edits
• Deterministic
• Idempotent
"""

import pandas as pd
from pathlib import Path

REGISTRY_PATH = Path("data/wave_registry.csv")
WAVES_DIR = Path("data/waves")

def parse_tickers(ticker_str: str) -> list[str]:
    return [t.strip() for t in ticker_str.split(",") if t.strip()]

def main():
    df = pd.read_csv(REGISTRY_PATH)

    created = 0
    skipped = 0

    for _, row in df.iterrows():
        wave_id = row["wave_id"]
        active = bool(row.get("active", False))
        holdings_source = row.get("holdings_source")

        if not active or holdings_source != "canonical":
            skipped += 1
            continue

        tickers_raw = row.get("ticker_normalized")
        if not isinstance(tickers_raw, str) or not tickers_raw.strip():
            skipped += 1
            continue

        tickers = parse_tickers(tickers_raw)
        if not tickers:
            skipped += 1
            continue

        wave_dir = WAVES_DIR / wave_id
        wave_dir.mkdir(parents=True, exist_ok=True)

        weights_path = wave_dir / "weights.csv"

        # If weights already exist, do not overwrite
        if weights_path.exists():
            continue

        # Equal-weight fallback (fractional, not synthetic)
        w = round(1.0 / len(tickers), 6)
        weights_df = pd.DataFrame({
            "ticker": tickers,
            "weight": [w] * len(tickers)
        })

        weights_df.to_csv(weights_path, index=False)
        created += 1

    print("======================================")
    print("Wave weights generation complete")
    print(f"Created weights.csv: {created}")
    print(f"Skipped waves: {skipped}")
    print("======================================")

if __name__ == "__main__":
    main()