import os
import sys
import pandas as pd
from datetime import datetime

# ------------------------------------------------------------
# Ensure repo root is in sys.path
# ------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ------------------------------------------------------------
# Imports from engine
# ------------------------------------------------------------
from waves_engine import compute_history_nav, get_all_waves

# ------------------------------------------------------------
# Output path
# ------------------------------------------------------------
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "live_snapshot.csv")

# ------------------------------------------------------------
# Horizons (UI expects these)
# ------------------------------------------------------------
HORIZONS = {
    "1d": 1,
    "30d": 30,
    "60d": 60,
    "365d": 365,
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def compute_return(nav: pd.Series, days: int):
    if nav is None or len(nav) <= days:
        return None
    try:
        return float(nav.iloc[-1] / nav.iloc[-days] - 1.0)
    except Exception:
        return None


# ------------------------------------------------------------
# Main snapshot builder
# ------------------------------------------------------------
def generate_live_snapshot():
    os.makedirs(DATA_DIR, exist_ok=True)

    waves = get_all_waves()
    rows = []

    print(f"[INFO] Generating snapshot for {len(waves)} waves")

    for wave_name in waves:
        try:
            hist = compute_history_nav(
                wave_name=wave_name,
                mode="Standard",
                days=365,
                include_diagnostics=True,
            )

            if hist is None or hist.empty:
                print(f"[WARN] {wave_name} skipped: empty history")
                continue

            if "wave_nav" not in hist or "bm_nav" not in hist:
                print(f"[WARN] {wave_name} skipped: missing NAV columns")
                continue

            wave_nav = hist["wave_nav"]
            bm_nav = hist["bm_nav"]

            row = {
                "wave_name": wave_name,
                "asof": hist.index[-1] if len(hist.index) else None,
            }

            # Returns + alpha per horizon
            for label, days in HORIZONS.items():
                w_ret = compute_return(wave_nav, days)
                b_ret = compute_return(bm_nav, days)

                row[f"return_{label}"] = w_ret
                row[f"alpha_{label}"] = (
                    w_ret - b_ret if w_ret is not None and b_ret is not None else None
                )

            rows.append(row)

        except Exception as e:
            print(f"[WARN] {wave_name} skipped: {e}")

    if not rows:
        raise RuntimeError("No valid waves produced snapshot")

    df = pd.DataFrame(rows)

    df.sort_values("wave_name", inplace=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[SUCCESS] Snapshot written → {OUTPUT_PATH}")
    print(df.head(3))


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        generate_live_snapshot()
    except Exception as e:
        print(f"[FATAL] Failed to generate live snapshot: {e}")
        sys.exit(1)