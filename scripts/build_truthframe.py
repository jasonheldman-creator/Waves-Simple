import sys
from pathlib import Path
import json
from datetime import datetime, timezone

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
# TruthFrame Builder (CI-safe, non-fatal)
# ------------------------------------------------------------------
def build_truthframe(days: int = 60) -> dict:
    truthframe = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": days,
        },
        "waves": {},
    }

    # Load price book (gatekeeper)
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        truthframe["_meta"]["status"] = "PRICE_BOOK_MISSING"
        return truthframe

    truthframe["_meta"]["price_book_rows"] = len(price_book)
    truthframe["_meta"]["price_book_cols"] = len(price_book.columns)

    # Load canonical wave registry
    registry = get_wave_registry()
    if registry is None or registry.empty:
        truthframe["_meta"]["status"] = "WAVE_REGISTRY_MISSING"
        return truthframe

    # Build per-wave placeholders (alpha attribution comes later)
    for _, row in registry.iterrows():
        wave_id = row["wave_id"]

        truthframe["waves"][wave_id] = {
            "alpha": {
                "total": 0.0,
                "selection": 0.0,
                "overlay": 0.0,
                "cash": 0.0,
            },
            "health": {
                "status": "OK",
            },
            "learning": {},
        }

    truthframe["_meta"]["status"] = "OK"
    truthframe["_meta"]["wave_count"] = len(truthframe["waves"])

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