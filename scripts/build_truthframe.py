import sys
from pathlib import Path

# ------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH (CI + local safe)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from datetime import datetime, timezone
from typing import Dict, Any

import pandas as pd

from helpers.price_book import get_price_book
from helpers.wave_registry import get_wave_registry
from helpers.wave_data import get_wave_data_filtered


# ------------------------------------------------------------------
# TruthFrame Builder
# ------------------------------------------------------------------
def build_truthframe(days: int = 60) -> Dict[str, Any]:
    """
    Build canonical TruthFrame.
    - CI-safe
    - App-safe
    - Never depends on Streamlit
    """

    # --- Validate PRICE_BOOK ---
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        raise RuntimeError("PRICE_BOOK unavailable — aborting TruthFrame build")

    # --- Load wave registry (canonical list of waves) ---
    wave_registry = get_wave_registry()
    if wave_registry is None or wave_registry.empty:
        raise RuntimeError("Wave registry unavailable")

    wave_ids = sorted(wave_registry["wave_id"].unique().tolist())

    truth: Dict[str, Any] = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": days,
            "price_book_rows": len(price_book),
            "price_book_cols": len(price_book.columns),
            "wave_count": len(wave_ids),
        },
        "waves": {},
    }

    # --- Per-wave attribution ---
    for wave_id in wave_ids:
        try:
            df = get_wave_data_filtered(wave_id, days)

            if df is None or df.empty:
                raise ValueError("No wave data available")

            if not {"portfolio_return", "benchmark_return"}.issubset(df.columns):
                raise ValueError("Missing required return columns")

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
            # Graceful degradation (never break build)
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


# ------------------------------------------------------------------
# CLI Entry (GitHub Actions)
# ------------------------------------------------------------------
if __name__ == "__main__":
    truthframe = build_truthframe(days=60)

    output_path = ROOT / "data" / "truthframe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(truthframe, f, indent=2)

    print(f"✅ TruthFrame written to {output_path}")