import pandas as pd
from pathlib import Path

# --- Config ---
WAVE_ID = "sp500_wave"
HISTORY_PATH = Path(f"data/history/{WAVE_ID}_history.csv")
OUTPUT_DIR = Path("data/diagnostics")
OUTPUT_PATH = OUTPUT_DIR / f"{WAVE_ID}_diagnostics.csv"

# --- Safety checks ---
if not HISTORY_PATH.exists():
    print(f"[WARN] Missing history file: {HISTORY_PATH}")
    print("Exiting gracefully (CI-safe).")
    exit(0)

# --- Load history ---
df = pd.read_csv(HISTORY_PATH)

required_cols = {"date", "wave_return", "benchmark_return"}
if not required_cols.issubset(df.columns):
    print(f"[WARN] Missing required columns in {HISTORY_PATH}")
    print(f"Found: {list(df.columns)}")
    exit(0)

# --- Core math ---
df["alpha"] = df["wave_return"] - df["benchmark_return"]
df["cum_wave_return"] = (1 + df["wave_return"]).cumprod() - 1
df["cum_benchmark_return"] = (1 + df["benchmark_return"]).cumprod() - 1
df["cum_alpha"] = df["cum_wave_return"] - df["cum_benchmark_return"]

# --- Normalize schema (important for attribution) ---
diagnostics = df[[
    "date",
    "wave_return",
    "benchmark_return",
    "alpha",
    "cum_alpha"
]].copy()

diagnostics.insert(0, "wave_id", WAVE_ID)
diagnostics.insert(1, "mode", "Standard")

# --- Write output ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
diagnostics.to_csv(OUTPUT_PATH, index=False)

print(f"[OK] Wrote diagnostics → {OUTPUT_PATH}")
print(f"[ROWS] {len(diagnostics)}")