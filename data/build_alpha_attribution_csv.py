"""
Build Alpha Attribution Summary CSV

This script aggregates 365-day alpha attribution data per Wave.

CRITICAL GUARANTEES:
- At least one row per eligible wave OR the script FAILS
- No header-only CSVs are allowed
- Every exclusion is logged explicitly
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# ==============================
# CONFIG
# ==============================

DAYS_LOOKBACK = 365

REGISTRY_PATH = Path("data/wave_registry.csv")
HISTORY_DIR = Path("data/history")
OUTPUT_PATH = Path("data/alpha_attribution_summary.csv")

OUTPUT_COLUMNS = [
    "wave_name",
    "mode",
    "days",
    "total_alpha",
    "total_wave_return",
    "total_benchmark_return",
]

# ==============================
# HELPERS
# ==============================

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return pd.read_csv(path)

def load_history(wave_id: str) -> pd.DataFrame:
    hist_path = HISTORY_DIR / wave_id / "history.csv"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history.csv for wave: {wave_id}")
    df = pd.read_csv(hist_path)
    if "date" not in df.columns:
        raise ValueError(f"'date' column missing in history for {wave_id}")
    df["date"] = pd.to_datetime(df["date"])
    return df

# ==============================
# MAIN
# ==============================

def main():
    print("▶ Starting Build Alpha Attribution CSV")

    registry = load_csv(REGISTRY_PATH)

    if "wave_id" not in registry.columns or "wave_name" not in registry.columns:
        raise ValueError("wave_registry.csv must contain wave_id and wave_name columns")

    cutoff_date = datetime.utcnow() - timedelta(days=DAYS_LOOKBACK)
    rows = []
    skipped = []

    for _, row in registry.iterrows():
        wave_id = row["wave_id"]
        wave_name = row["wave_name"]
        mode = row.get("mode", "STANDARD")

        try:
            hist = load_history(wave_id)

            hist = hist[hist["date"] >= cutoff_date]

            if hist.empty:
                skipped.append((wave_name, "No data in lookback window"))
                continue

            required_cols = {"wave_return", "benchmark_return"}
            if not required_cols.issubset(hist.columns):
                skipped.append((wave_name, "Missing return columns"))
                continue

            wave_total = (1 + hist["wave_return"]).prod() - 1
            bench_total = (1 + hist["benchmark_return"]).prod() - 1
            alpha_total = wave_total - bench_total

            rows.append({
                "wave_name": wave_name,
                "mode": mode,
                "days": DAYS_LOOKBACK,
                "total_alpha": round(alpha_total, 6),
                "total_wave_return": round(wave_total, 6),
                "total_benchmark_return": round(bench_total, 6),
            })

            print(f"✓ Added row for {wave_name}")

        except Exception as e:
            skipped.append((wave_name, str(e)))
            print(f"✗ Skipped {wave_name}: {e}")

    print(f"▶ Waves processed: {len(registry)}")
    print(f"▶ Rows generated: {len(rows)}")
    print(f"▶ Waves skipped: {len(skipped)}")

    for w, reason in skipped:
        print(f"  - {w}: {reason}")

    # 🚨 HARD FAIL if no rows
    if len(rows) == 0:
        raise RuntimeError(
            "CRITICAL FAILURE: alpha_attribution_summary.csv would be empty.\n"
            "This is a hard stop to prevent silent header-only output."
        )

    df_out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Wrote {len(df_out)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()