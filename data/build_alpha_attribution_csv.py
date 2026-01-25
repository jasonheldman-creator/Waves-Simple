"""
build_alpha_attribution_csv.py

AUTHORITATIVE, GUARANTEED alpha attribution builder.

Rules:
- ALWAYS writes data/alpha_attribution_summary.csv
- ALWAYS emits ≥1 row per active wave (even if history is missing)
- NEVER silently skips a wave
- NEVER produces header-only output
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

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

def load_registry() -> pd.DataFrame:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing registry: {REGISTRY_PATH}")
    return pd.read_csv(REGISTRY_PATH)

def history_path(wave_id: str) -> Path:
    return HISTORY_DIR / wave_id / "history.csv"

def safe_sum(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    return float(series.sum())

# =========================
# MAIN
# =========================

def main() -> None:
    registry = load_registry()

    rows = []
    now = datetime.utcnow().date()

    for _, wave in registry.iterrows():
        wave_id = wave["wave_id"]
        wave_name = wave.get("display_name", wave_id)
        active = bool(wave.get("active", True))

        if not active:
            continue

        hist_file = history_path(wave_id)

        if hist_file.exists():
            hist = pd.read_csv(hist_file)
            hist["date"] = pd.to_datetime(hist["date"]).dt.date

            cutoff = now - pd.Timedelta(days=DAYS_LOOKBACK)
            hist = hist[hist["date"] >= cutoff]

            wave_ret = safe_sum(hist.get("wave_return"))
            bench_ret = safe_sum(hist.get("benchmark_return"))
            alpha = wave_ret - bench_ret
            days = len(hist)

        else:
            # HARD GUARANTEE: still emit a row
            wave_ret = 0.0
            bench_ret = 0.0
            alpha = 0.0
            days = 0

        # LIVE row
        rows.append([
            wave_name,
            "LIVE",
            days,
            alpha,
            wave_ret,
            bench_ret,
        ])

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # ABSOLUTE SAFETY CHECK
    if df.empty:
        raise RuntimeError("Alpha attribution produced zero rows — this is a bug.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[OK] Wrote {len(df)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()