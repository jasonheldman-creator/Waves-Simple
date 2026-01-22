import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import pandas as pd
from datetime import datetime, timezone

from helpers.wave_registry import get_active_wave_registry


TRUTHFRAME_OUTPUT_PATH = "data/truthframe.json"
LIVE_SNAPSHOT_PATH = "data/live_snapshot.csv"


def build_truthframe() -> dict:
    """
    Build TruthFrame from canonical live_snapshot.csv.
    This NEVER raises — degraded waves are allowed.
    """

    truth = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "live_snapshot.csv",
        },
        "waves": {},
    }

    # Load wave registry (canonical wave list)
    try:
        registry = get_active_wave_registry()
        wave_ids = registry["wave_id"].tolist()
    except Exception:
        wave_ids = []

    # Load live snapshot
    if not Path(LIVE_SNAPSHOT_PATH).exists():
        truth["_meta"]["error"] = "live_snapshot.csv missing"
        return truth

    snapshot = pd.read_csv(LIVE_SNAPSHOT_PATH)

    for wave_id in wave_ids:
        wave_rows = snapshot[snapshot["wave_id"] == wave_id]

        if wave_rows.empty:
            truth["waves"][wave_id] = {
                "alpha": {
                    "total": 0.0,
                    "selection": 0.0,
                    "overlay": 0.0,
                    "cash": 0.0,
                },
                "health": {"status": "MISSING"},
                "learning": {},
            }
            continue

        try:
            total_alpha = float(wave_rows["alpha"].sum())

            selection = float(
                wave_rows.get("selection_alpha", wave_rows["alpha"]).sum()
            )
            overlay = float(
                wave_rows.get("overlay_alpha", 0.0).sum()
            )
            cash = float(
                wave_rows.get("cash_alpha", 0.0).sum()
            )

            truth["waves"][wave_id] = {
                "alpha": {
                    "total": total_alpha,
                    "selection": selection,
                    "overlay": overlay,
                    "cash": cash,
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
                "health": {"status": "DEGRADED", "error": str(e)},
                "learning": {},
            }

    return truth


if __name__ == "__main__":
    truthframe = build_truthframe()

    with open(TRUTHFRAME_OUTPUT_PATH, "w") as f:
        json.dump(truthframe, f, indent=2)

    print("✅ TruthFrame written to data/truthframe.json")