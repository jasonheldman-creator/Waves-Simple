import os
import pandas as pd
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------

WAVE_ID = "sp500_wave"
WAVE_NAME = "S&P 500 Wave"
MODE = "Standard"
DAYS_LOOKBACK = 365

HISTORY_PATH = Path(f"data/history/{WAVE_ID}_history.csv")
OUTPUT_PATH = Path("data/alpha_attribution_summary.csv")

OUTPUT_COLUMNS = [
    "wave_name",
    "mode",
    "days",
    "total_alpha",
    "total_wave_return",
    "total_benchmark_return",
]

# -----------------------------
# Guardrails
# -----------------------------

if not HISTORY_PATH.exists():
    print(f"[alpha attribution] history file not found: {HISTORY_PATH}")
    print("[alpha attribution] skipping build (non-fatal)")
    exit(0)

# -----------------------------
# Load history
# -----------------------------

df = pd.read_csv(HISTORY_PATH)

required_cols = {"date", "wave_return", "benchmark_return"}
if not required_cols.issubset(df.columns):
    print("[alpha attribution] missing required columns")
    print(f"Found: {df.columns.tolist()}")
    exit(0)

df = df.sort_values("date").tail(DAYS_LOOKBACK)

# -----------------------------
# Math (real, cumulative)
# -----------------------------

df["wave_growth"] = (1 + df["wave_return"]).cumprod()
df["benchmark_growth"] = (1 + df["benchmark_return"]).cumprod()

total_wave_return = df["wave_growth"].iloc[-1] - 1
total_benchmark_return = df["benchmark_growth"].iloc[-1] - 1
total_alpha = total_wave_return - total_benchmark_return

# -----------------------------
# Build output row
# -----------------------------

row = {
    "wave_name": WAVE_NAME,
    "mode": MODE,
    "days": len(df),
    "total_alpha": round(total_alpha, 6),
    "total_wave_return": round(total_wave_return, 6),
    "total_benchmark_return": round(total_benchmark_return, 6),
}

output_df = pd.DataFrame([row], columns=OUTPUT_COLUMNS)

# -----------------------------
# Write CSV (overwrite-safe)
# -----------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
output_df.to_csv(OUTPUT_PATH, index=False)

print("[alpha attribution] build complete")
print(output_df)