"""
build_alpha_attribution_csv.py

Production alpha attribution builder.

HARD GUARANTEES:
• One output row per wave in wave_registry.csv
• Never header-only output
• Never silently skip a wave
• Real data only (NaNs if missing)
• Deterministic + CI-safe
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================

WAVE_REGISTRY_PATH = Path("data/wave_registry.csv")
HISTORY_DIR = Path("data/history")
OUTPUT_PATH = Path("data/alpha_attribution_summary.csv")

DAYS_LOOKBACK = 365

OUTPUT_COLUMNS = [
    "wave_name",
    "mode",
    "days",
    "total_alpha",
    "total_wave_return",
    "total_benchmark_return",
    "status",
    "notes",
]

# =========================
# LOADERS
# =========================

def load_wave_registry():
    if not WAVE_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Missing wave registry: {WAVE_REGISTRY_PATH}")

    df = pd.read_csv(WAVE_REGISTRY_PATH)

    if df.empty:
        raise RuntimeError("wave_registry.csv is empty")

    if "wave_id" not in df.columns:
        raise RuntimeError("wave_registry.csv missing required column: wave_id")

    return df


def load_wave_history(wave_id):
    path = HISTORY_DIR / wave_id / "history.csv"

    if not path.exists():
        return None, "MISSING_HISTORY", f"{path} not found"

    try:
        df = pd.read_csv(path)
        required = {"date", "wave_return", "benchmark_return"}
        if not required.issubset(df.columns):
            return None, "INVALID_HISTORY", "Missing required columns"

        return df, "OK", ""
    except Exception as e:
        return None, "LOAD_ERROR", str(e)

# =========================
# CALCULATION
# =========================

def calculate_alpha(df):
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        cutoff = datetime.utcnow().date() - timedelta(days=DAYS_LOOKBACK)
        df = df[df["date"] >= cutoff]

        if df.empty:
            return (
                float("nan"),
                float("nan"),
                float("nan"),
                0,
                "NO_RECENT_DATA",
                "No data in lookback window",
            )

        wave_ret = df["wave_return"].sum()
        bench_ret = df["benchmark_return"].sum()

        return (
            wave_ret - bench_ret,
            wave_ret,
            bench_ret,
            len(df),
            "OK",
            "",
        )

    except Exception as e:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            "CALC_ERROR",
            str(e),
        )

# =========================
# MAIN
# =========================

def main():
    registry = load_wave_registry()
    rows = []

    for _, wave in registry.iterrows():
        wave_id = wave["wave_id"]
        wave_name = wave.get("display_name", wave_id)
        mode = wave.get("mode", "LIVE")

        history, status, notes = load_wave_history(wave_id)

        if status != "OK":
            rows.append([
                wave_name,
                mode,
                0,
                float("nan"),
                float("nan"),
                float("nan"),
                status,
                notes,
            ])
            continue

        (
            alpha,
            wave_ret,
            bench_ret,
            days,
            calc_status,
            calc_notes,
        ) = calculate_alpha(history)

        rows.append([
            wave_name,
            mode,
            days,
            alpha,
            wave_ret,
            bench_ret,
            calc_status,
            calc_notes,
        ])

    if not rows:
        raise RuntimeError("No rows produced — invariant violated")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(OUTPUT_PATH, index=False)

    print(f"[OK] alpha_attribution_summary.csv written with {len(rows)} rows")


if __name__ == "__main__":
    main()