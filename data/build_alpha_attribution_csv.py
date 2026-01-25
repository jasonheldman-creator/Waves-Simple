"""
build_alpha_attribution_csv.py

Builds alpha_attribution_summary.csv with GUARANTEED rows.

Rules:
- Every active wave with history MUST produce a row
- No silent skips
- No header-only output
- If nothing is written, the script raises
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
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

# =========================
# HELPERS
# =========================
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def ensure_history_exists(wave_id: str) -> Path:
    hist_path = HISTORY_DIR / wave_id / "history.csv"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history for wave: {wave_id}")
    return hist_path


# =========================
# MAIN
# =========================
def main():
    registry = load_csv(REGISTRY_PATH)

    rows = []
    cutoff_date = datetime.utcnow() - timedelta(days=DAYS_LOOKBACK)

    print(f"Building alpha attribution for {len(registry)} waves")
    print(f"Lookback window: {DAYS_LOOKBACK} days")

    for _, row in registry.iterrows():
        wave_id = row.get("wave_id")
        wave_name = row.get("display_name", wave_id)
        active = bool(row.get("active", False))
        mode = row.get("mode", "standard")

        if not active:
            print(f"SKIP (inactive): {wave_id}")
            continue

        # ---- Load history ----
        hist_path = ensure_history_exists(wave_id)
        history = pd.read_csv(hist_path, parse_dates=["date"])

        history = history[history["date"] >= cutoff_date]

        if history.empty:
            raise RuntimeError(f"No history rows after cutoff for wave: {wave_id}")

        # ---- Required columns ----
        required_cols = {"wave_return", "benchmark_return"}
        missing = required_cols - set(history.columns)
        if missing:
            raise RuntimeError(
                f"Missing columns {missing} in history for wave: {wave_id}"
            )

        # ---- Compute totals ----
        total_wave_return = history["wave_return"].sum()
        total_benchmark_return = history["benchmark_return"].sum()
        total_alpha = total_wave_return - total_benchmark_return

        rows.append(
            {
                "wave_name": wave_name,
                "mode": mode,
                "days": len(history),
                "total_alpha": round(total_alpha, 6),
                "total_wave_return": round(total_wave_return, 6),
                "total_benchmark_return": round(total_benchmark_return, 6),
            }
        )

        print(f"OK: {wave_id} → rows={len(history)} alpha={total_alpha:.4f}")

    # =========================
    # FINAL VALIDATION
    # =========================
    if not rows:
        raise RuntimeError(
            "FATAL: alpha_attribution_summary.csv would be empty. Aborting write."
        )

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print("======================================")
    print(f"Wrote {len(out_df)} rows to {OUTPUT_PATH}")
    print("======================================")


if __name__ == "__main__":
    main()