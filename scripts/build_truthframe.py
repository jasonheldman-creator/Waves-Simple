import sys
from pathlib import Path

# ------------------------------------------------------------
# Ensure repo root is on PYTHONPATH (CI-safe)
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from datetime import datetime, timezone

import pandas as pd

from helpers.price_book import get_price_book
from helpers.wave_registry import get_wave_registry
from helpers.wave_data import get_wave_data_filtered


def build_truthframe(days: int = 60):
    price_book = get_price_book()

    if price_book is None or price_book.empty:
        raise RuntimeError("PRICE_BOOK is empty — cannot build TruthFrame")

    wave_registry = get_wave_registry()
    if wave_registry is None or wave_registry.empty:
        raise RuntimeError("Wave registry is empty — cannot build TruthFrame")

    wave_ids = wave_registry["wave_id"].tolist()

    truth = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": days,
            "price_book_rows": len(price_book),
            "price_book_cols": len(price_book.columns),
            "wave_count": len(wave_ids),
        },
        "waves": {},
    }

    for wave_id in wave_ids:
        try:
            df = get_wave_data_filtered(wave_id, days)

            if df is None or df.empty:
                raise ValueError("No wave data available")

            df = df.copy()
            df["alpha"] = df["portfolio_return"] - df["benchmark_return"]

            total_alpha = float(df["alpha"].sum())
            exposure = float(df["exposure"].mean()) if "exposure" in df.columns else 1.0

            truth["waves"][wave_id] = {
                "alpha": {
                    "total": total_alpha,
                    "selection": total_alpha * exposure,
                    "overlay": total_alpha * (1 - exposure) * 0.7,
                    "cash": total_alpha * (1 - exposure) * 0.3,
                },
                "health": {"status": "OK"},
                "learning": {},
            }

        except Exception as e:
            truth["waves"][wave_id] = {
                "alpha": {
                    "total": 0.0,
                    "selection": 0.0,
                    "overlay": 0.0,
                    "cash": 0.0,
                },
                "health": {
                    "status": "DEGRADED",
                    "error": str(e),
                },
                "learning": {},
            }

    return truth


if __name__ == "__main__":
    truthframe = build_truthframe()

    output_path = Path("data/truthframe.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(truthframe, f, indent=2)

    print(f"✅ TruthFrame written to {output_path.resolve()}")