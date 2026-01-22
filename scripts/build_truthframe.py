import sys
from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Dict, Any

import pandas as pd

# ------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH (CI-safe)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Canonical imports (must exist)
# ------------------------------------------------------------------
from helpers.price_book import get_price_book
from helpers.wave_registry import get_wave_registry
from helpers.return_pipeline import compute_wave_returns_pipeline


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
HORIZONS = {
    "1D": 1,
    "30D": 30,
    "60D": 60,
    "365D": 365,
}


# ------------------------------------------------------------------
# TruthFrame Builder (WIRED)
# ------------------------------------------------------------------
def build_truthframe(days: int = 365) -> Dict[str, Any]:
    truthframe: Dict[str, Any] = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": days,
            "status": "INITIALIZING",
            "validated": False,
        },
        "waves": {},
    }

    # -----------------------------
    # Load price book
    # -----------------------------
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        truthframe["_meta"]["status"] = "PRICE_BOOK_MISSING"
        return truthframe

    truthframe["_meta"]["price_book_rows"] = len(price_book)
    truthframe["_meta"]["price_book_cols"] = len(price_book.columns)

    # -----------------------------
    # Load wave registry
    # -----------------------------
    registry = get_wave_registry()
    if registry is None or registry.empty:
        truthframe["_meta"]["status"] = "WAVE_REGISTRY_MISSING"
        return truthframe

    # -----------------------------
    # Compute returns + alpha
    # -----------------------------
    try:
        returns_df = compute_wave_returns_pipeline(
            price_book=price_book,
            wave_registry=registry,
            horizons=list(HORIZONS.values()),
        )
    except Exception as e:
        truthframe["_meta"]["status"] = "RETURN_PIPELINE_FAILED"
        truthframe["_meta"]["error"] = str(e)
        return truthframe

    if returns_df is None or returns_df.empty:
        truthframe["_meta"]["status"] = "NO_COMPUTED_RETURNS"
        return truthframe

    # Expected columns (defensive)
    required_cols = {
        "wave_id",
        "horizon",
        "wave_return",
        "benchmark_return",
        "alpha",
    }
    if not required_cols.issubset(set(returns_df.columns)):
        truthframe["_meta"]["status"] = "RETURN_SCHEMA_INVALID"
        truthframe["_meta"]["missing_columns"] = sorted(
            list(required_cols - set(returns_df.columns))
        )
        return truthframe

    # -----------------------------
    # Build per-wave TruthFrame
    # -----------------------------
    validated_waves = 0

    for _, row in registry.iterrows():
        wave_id = row["wave_id"]

        wave_perf = returns_df[returns_df["wave_id"] == wave_id]
        performance_block = {}

        for label, days in HORIZONS.items():
            slice_df = wave_perf[wave_perf["horizon"] == days]

            if slice_df.empty:
                continue

            r = slice_df.iloc[0]

            performance_block[label] = {
                "return": float(r["wave_return"]),
                "alpha": float(r["alpha"]),
                "benchmark_return": float(r["benchmark_return"]),
            }

        truthframe["waves"][wave_id] = {
            "alpha": {
                "total": float(wave_perf["alpha"].sum())
                if not wave_perf.empty
                else 0.0,
                "selection": 0.0,   # reserved for attribution layer
                "overlay": 0.0,     # reserved
                "cash": 0.0,        # reserved
            },
            "performance": performance_block,
            "health": {
                "status": "OK" if performance_block else "DEGRADED"
            },
            "learning": {},
        }

        if performance_block:
            validated_waves += 1

    # -----------------------------
    # Final validation
    # -----------------------------
    truthframe["_meta"]["wave_count"] = len(truthframe["waves"])
    truthframe["_meta"]["validated_waves"] = validated_waves

    if validated_waves > 0:
        truthframe["_meta"]["validated"] = True
        truthframe["_meta"]["status"] = "OK"
    else:
        truthframe["_meta"]["status"] = "NO_VALIDATED_WAVES"

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

    print(f"✅ TruthFrame written to {output_path}")
    print(f"Status: {tf['_meta'].get('status')}")
    print(f"Validated: {tf['_meta'].get('validated')}")