# scripts/generate_live_snapshot_csv.py
"""
WAVES Intelligence™
Generate Live Snapshot CSV (Authoritative)

This script is the SINGLE SOURCE OF TRUTH for:
- data/live_snapshot.csv
- Portfolio Snapshot (Overview tab)
- Alpha Attribution (drivers + totals)

Rules:
- Never hardcode wave names
- Never crash on individual wave failure
- Always emit a structurally valid CSV
"""

import os
import sys
import pandas as pd
from typing import List

# -------------------------------------------------------------------
# Repo path setup
# -------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "live_snapshot.csv")

# -------------------------------------------------------------------
# Imports from engine (authoritative)
# -------------------------------------------------------------------
from waves_engine import (
    compute_history_nav,
    get_all_waves,
)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
HORIZONS = {
    "1d": 1,
    "30d": 30,
    "60d": 60,
    "365d": 365,
}

REQUIRED_COLUMNS = [
    "wave",
    "ret_1d",
    "ret_30d",
    "ret_60d",
    "ret_365d",
    "alpha_1d",
    "alpha_30d",
    "alpha_60d",
    "alpha_365d",
]

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _calc_return(nav: pd.Series, days: int):
    if nav is None or len(nav) <= days:
        return None
    try:
        return float(nav.iloc[-1] / nav.iloc[-(days + 1)] - 1.0)
    except Exception:
        return None


# -------------------------------------------------------------------
# Main generator
# -------------------------------------------------------------------
def generate_live_snapshot():
    waves: List[str] = get_all_waves()

    rows = []

    for wave in waves:
        try:
            hist = compute_history_nav(
                wave_name=wave,
                mode="Standard",
                days=400,  # buffer so 365d math never underflows
                include_diagnostics=False,
            )

            if hist is None or hist.empty:
                print(f"[WARN] {wave} skipped: empty history")
                continue

            if "wave_nav" not in hist.columns or "bm_nav" not in hist.columns:
                print(f"[WARN] {wave} skipped: missing NAV columns")
                continue

            wave_nav = hist["wave_nav"]
            bm_nav = hist["bm_nav"]

            row = {"wave": wave}

            for label, d in HORIZONS.items():
                w_ret = _calc_return(wave_nav, d)
                b_ret = _calc_return(bm_nav, d)

                row[f"ret_{label}"] = w_ret
                row[f"alpha_{label}"] = (
                    (w_ret - b_ret) if w_ret is not None and b_ret is not None else None
                )

            rows.append(row)

        except Exception as e:
            # HARD RULE: never kill snapshot for one wave
            print(f"[WARN] {wave} skipped: {e}")
            continue

    if not rows:
        raise RuntimeError("No valid waves produced snapshot output")

    df = pd.DataFrame(rows)

    # Ensure all expected columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[REQUIRED_COLUMNS]

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Live snapshot written: {OUTPUT_PATH}")
    print(f"   Waves included: {len(df)}")


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    generate_live_snapshot()