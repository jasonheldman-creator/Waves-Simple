import sys
from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd

# ------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH (CI-safe)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Canonical imports (must exist in CI)
# ------------------------------------------------------------------
from helpers.price_book import get_price_book
from helpers.wave_registry import get_wave_registry


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
HORIZONS = {
    "1D": 1,
    "30D": 30,
    "60D": 60,
    "365D": 365,
}


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def safe_return(series: pd.Series, days: int):
    """
    Compute simple return over N days.
    Returns None if insufficient data.
    """
    if series is None or len(series) <= days:
        return None

    start = series.iloc[-days - 1]
    end = series.iloc[-1]

    if start == 0 or pd.isna(start) or pd.isna(end):
        return None

    return (end / start) - 1.0


# ------------------------------------------------------------------
# TruthFrame Builder
# ------------------------------------------------------------------
def build_truthframe(days: int = 365) -> dict:
    truthframe = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": days,
            "status": "INIT",
        },
        "waves": {},
    }

    # --------------------------------------------------------------
    # Load price book (gatekeeper)
    # --------------------------------------------------------------
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        truthframe["_meta"]["status"] = "PRICE_BOOK_MISSING"
        return truthframe

    price_book = price_book.tail(days)

    truthframe["_meta"]["price_book_rows"] = len(price_book)
    truthframe["_meta"]["price_book_cols"] = len(price_book.columns)

    # --------------------------------------------------------------
    # Load wave registry
    # --------------------------------------------------------------
    registry = get_wave_registry()
    if registry is None or registry.empty:
        truthframe["_meta"]["status"] = "WAVE_REGISTRY_MISSING"
        return truthframe

    # --------------------------------------------------------------
    # Build per-wave TruthFrame
    # --------------------------------------------------------------
    validated_waves = 0

    for _, row in registry.iterrows():
        wave_id = row["wave_id"]
        tickers = row.get("tickers", [])
        benchmark = row.get("benchmark")

        # Defensive defaults
        wave_block = {
            "performance": {},
            "alpha": {
                "total": None,
                "selection": None,
                "overlay": 0.0,
                "cash": 0.0,
            },
            "health": {
                "status": "UNKNOWN",
            },
            "learning": {},
        }

        # ----------------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------------
        if not tickers or benchmark not in price_book.columns:
            truthframe["waves"][wave_id] = wave_block
            continue

        missing = [t for t in tickers if t not in price_book.columns]
        if missing:
            truthframe["waves"][wave_id] = wave_block
            continue

        # ----------------------------------------------------------
        # Build wave price series (equal-weighted)
        # ----------------------------------------------------------
        wave_prices = price_book[tickers].mean(axis=1)
        benchmark_prices = price_book[benchmark]

        alpha_sum = 0.0
        alpha_count = 0

        for label, d in HORIZONS.items():
            wave_ret = safe_return(wave_prices, d)
            bench_ret = safe_return(benchmark_prices, d)

            wave_block["performance"][label] = {
                "return": wave_ret,
                "benchmark": bench_ret,
            }

            if wave_ret is not None and bench_ret is not None:
                alpha = wave_ret - bench_ret
                alpha_sum += alpha
                alpha_count += 1

        # ----------------------------------------------------------
        # Alpha attribution
        # ----------------------------------------------------------
        if alpha_count > 0:
            wave_block["alpha"]["total"] = alpha_sum
            wave_block["alpha"]["selection"] = alpha_sum
            wave_block["health"]["status"] = "OK"
            validated_waves += 1
        else:
            wave_block["health"]["status"] = "NO_DATA"

        truthframe["waves"][wave_id] = wave_block

    # --------------------------------------------------------------
    # Final meta status
    # --------------------------------------------------------------
    truthframe["_meta"]["wave_count"] = len(truthframe["waves"])
    truthframe["_meta"]["validated_waves"] = validated_waves

    if validated_waves > 0:
        truthframe["_meta"]["status"] = "OK"
        truthframe["_meta"]["validated_performance"] = True
    else:
        truthframe["_meta"]["status"] = "DEGRADED"
        truthframe["_meta"]["validated_performance"] = False

    return truthframe


# ------------------------------------------------------------------
# CLI entrypoint (used by GitHub Actions)
# ------------------------------------------------------------------
if __name__ == "__main__":
    tf = build_truthframe()

    output_path = ROOT / "data" / "truthframe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tf, f, indent=2)

    print(f"✅ TruthFrame written to {output_path}")
    print(f"Status: {tf['_meta']['status']}")