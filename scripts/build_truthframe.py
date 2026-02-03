import sys
from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Dict, Any

import pandas as pd
import pandas_market_calendars as mcal

# ------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH (CI-safe)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Canonical imports
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

EQUITY_CALENDAR = mcal.get_calendar("NYSE")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def market_is_open_now() -> bool:
    now = pd.Timestamp.utcnow()
    schedule = EQUITY_CALENDAR.schedule(
        start_date=now.normalize(),
        end_date=now.normalize(),
    )
    if schedule.empty:
        return False

    market_open = schedule.iloc[0]["market_open"]
    market_close = schedule.iloc[0]["market_close"]
    return market_open <= now <= market_close


# ------------------------------------------------------------------
# TruthFrame Builder
# ------------------------------------------------------------------
def build_truthframe(days: int = 365) -> Dict[str, Any]:
    market_open = market_is_open_now()

    truthframe: Dict[str, Any] = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": days,
            "status": "INITIALIZING",
            "validated": False,
            "market_open": market_open,
        },
        "waves": {},
    }

    # ------------------------------------------------------------
    # Load price book
    # ------------------------------------------------------------
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        truthframe["_meta"]["status"] = "PRICE_BOOK_MISSING"
        return truthframe

    truthframe["_meta"]["price_book_rows"] = len(price_book)
    truthframe["_meta"]["price_book_cols"] = len(price_book.columns)

    # ------------------------------------------------------------
    # Load wave registry
    # ------------------------------------------------------------
    registry = get_wave_registry()
    if registry is None or registry.empty:
        truthframe["_meta"]["status"] = "WAVE_REGISTRY_MISSING"
        return truthframe

    # ------------------------------------------------------------
    # Compute returns
    # ------------------------------------------------------------
    returns_df = compute_wave_returns_pipeline(
        price_book=price_book,
        wave_registry=registry,
        horizons=list(HORIZONS.values()),
    )

    if returns_df is None or returns_df.empty:
        truthframe["_meta"]["status"] = "NO_COMPUTED_RETURNS"
        return truthframe

    validated_waves = 0

    # ------------------------------------------------------------
    # Build per-wave blocks
    # ------------------------------------------------------------
    for _, row in registry.iterrows():
        wave_id = row["wave_id"]
        wave_perf = returns_df[returns_df["wave_id"] == wave_id]

        performance = {}

        for label, days_h in HORIZONS.items():
            slice_df = wave_perf[wave_perf["horizon"] == days_h]
            if slice_df.empty:
                continue

            r = slice_df.iloc[0]

            # Skip 1D validation if market is closed AND equity wave
            if label == "1D" and not market_open and row["asset_class"] == "equity":
                continue

            performance[label] = {
                "return": float(r["wave_return"]),
                "alpha": float(r["alpha"]),
                "benchmark_return": float(r["benchmark_return"]),
            }

        health_ok = bool(performance)

        truthframe["waves"][wave_id] = {
            "alpha": {
                "total": float(wave_perf["alpha"].sum()) if not wave_perf.empty else 0.0,
                "selection": 0.0,
                "overlay": 0.0,
                "cash": 0.0,
            },
            "performance": performance,
            "health": {
                "status": "OK" if health_ok else "DEGRADED"
            },
            "learning": {},
        }

        if health_ok:
            validated_waves += 1

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------
    truthframe["_meta"]["wave_count"] = len(truthframe["waves"])
    truthframe["_meta"]["validated_waves"] = validated_waves
    truthframe["_meta"]["validated"] = validated_waves > 0
    truthframe["_meta"]["status"] = "OK" if validated_waves > 0 else "DEGRADED"
    truthframe["_meta"]["validation_source"] = "MARKET_AWARE_VALIDATION"

    return truthframe


# ------------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    tf = build_truthframe()

    output_path = ROOT / "data" / "truthframe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tf, f, indent=2)

    print(f"TruthFrame written to {output_path}")
    print(f"Status: {tf['_meta'].get('status')}")
    print(f"Validated: {tf['_meta'].get('validated')}")