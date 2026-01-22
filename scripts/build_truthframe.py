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
# Canonical imports ONLY (must exist in CI)
# ------------------------------------------------------------------
from helpers.price_book import get_price_book
from helpers.wave_registry import get_wave_registry


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def compute_return(series: pd.Series, days: int) -> float | None:
    """
    Compute simple return over N trading days.
    """
    if series is None or len(series) <= days:
        return None

    start = series.iloc[-days - 1]
    end = series.iloc[-1]

    if start == 0 or pd.isna(start) or pd.isna(end):
        return None

    return (end / start) - 1.0


# ------------------------------------------------------------------
# TruthFrame Builder (validated returns only)
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

    validated_wave_count = 0

    # --------------------------------------------------------------
    # Build per-wave performance
    # --------------------------------------------------------------
    for _, row in registry.iterrows():
        wave_id = row["wave_id"]

        # Parse tickers safely
        tickers = []
        if "tickers" in row and isinstance(row["tickers"], str):
            tickers = [t.strip() for t in row["tickers"].split(",")]

        available = [t for t in tickers if t in price_book.columns]

        performance = {
            "1D": None,
            "30D": None,
            "60D": None,
            "365D": None,
        }

        if available:
            prices = price_book[available].dropna(how="all")

            if not prices.empty:
                # Equal-weight wave price
                wave_series = prices.mean(axis=1)

                performance["1D"] = compute_return(wave_series, 1)
                performance["30D"] = compute_return(wave_series, 30)
                performance["60D"] = compute_return(wave_series, 60)
                performance["365D"] = compute_return(wave_series, 365)

                # Validate if any horizon exists
                if any(v is not None for v in performance.values()):
                    validated_wave_count += 1

        truthframe["waves"][wave_id] = {
            "performance": performance,
            "alpha": {
                # Alpha intentionally zero until benchmarks are wired
                "total": 0.0,
                "selection": 0.0,
                "overlay": 0.0,
                "cash": 0.0,
            },
            "health": {
                "status": "OK" if any(v is not None for v in performance.values()) else "NO_DATA"
            },
            "learning": {},
        }

    # --------------------------------------------------------------
    # Final system status
    # --------------------------------------------------------------
    truthframe["_meta"]["wave_count"] = len(truthframe["waves"])
    truthframe["_meta"]["validated_waves"] = validated_wave_count

    if validated_wave_count > 0:
        truthframe["_meta"]["status"] = "OK"
    else:
        truthframe["_meta"]["status"] = "NO_VALIDATED_PERFORMANCE"

    return truthframe


# ------------------------------------------------------------------
# CLI Entrypoint (GitHub Actions)
# ------------------------------------------------------------------
if __name__ == "__main__":
    tf = build_truthframe()

    output_path = ROOT / "data" / "truthframe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tf, f, indent=2)

    print("✅ TruthFrame built")
    print(f"   Status: {tf['_meta']['status']}")
    print(f"   Validated waves: {tf['_meta'].get('validated_waves', 0)}")