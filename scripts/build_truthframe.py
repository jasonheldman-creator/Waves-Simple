import sys
from pathlib import Path
import json
from datetime import datetime, timezone

# ------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH (CI-safe)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Canonical imports ONLY (must exist in CI)
# ------------------------------------------------------------------
from helpers.price_book import get_price_book
from helpers.wave_registry import get_wave_registry


# ------------------------------------------------------------------
# TruthFrame Builder (CI-safe, non-fatal, schema-stable)
# ------------------------------------------------------------------
def build_truthframe(days: int = 60) -> dict:
    """
    Build the canonical TruthFrame.
    Read-only, CI-safe, no app-side math.
    """

    truthframe = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": days,
            "status": "INIT",
        },
        "waves": {},
    }

    # --------------------------------------------------------------
    # Load price book (hard gate)
    # --------------------------------------------------------------
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        truthframe["_meta"]["status"] = "PRICE_BOOK_MISSING"
        return truthframe

    truthframe["_meta"]["price_book_rows"] = len(price_book)
    truthframe["_meta"]["price_book_cols"] = len(price_book.columns)

    # --------------------------------------------------------------
    # Load canonical wave registry
    # --------------------------------------------------------------
    registry = get_wave_registry()
    if registry is None or registry.empty:
        truthframe["_meta"]["status"] = "WAVE_REGISTRY_MISSING"
        return truthframe

    # --------------------------------------------------------------
    # Build per-wave schema placeholders (NO math yet)
    # --------------------------------------------------------------
    for _, row in registry.sort_values("wave_id").iterrows():
        wave_id = row["wave_id"]

        truthframe["waves"][wave_id] = {
            "alpha": {
                "total": 0.0,
                "selection": 0.0,
                "overlay": 0.0,
                "cash": 0.0,
            },
            "performance": {
                "1D": {},
                "30D": {},
                "60D": {},
                "365D": {},
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

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tf, f, indent=2)

    # Honest CI signal
    if tf["_meta"].get("status") != "OK":
        print("❌ TruthFrame build incomplete:", tf["_meta"]["status"])
        sys.exit(1)

    print(f"✅ TruthFrame written to {output_path}")