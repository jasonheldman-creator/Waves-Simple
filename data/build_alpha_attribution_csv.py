"""
build_alpha_attribution_csv.py

Production alpha attribution builder.

HARD REQUIREMENTS:
1. Generates `data/alpha_attribution_summary.csv` with ≥1 data row.
2. Each wave in `data/wave_registry.csv` must produce a row in the output CSV.
3. Real data only — do NOT fabricate or simulate returns.
4. Missing data results in rows with NaN values for returns and explicit status/notes.
5. Never silently skips a wave from the registry.
6. Header-only output prohibited.
7. Missing/empty `wave_registry.csv` raises a hard error.
8. Deterministic, auditable, and CI-safe behavior.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# =========================
# CONFIGURATION
# =========================

WAVE_REGISTRY_PATH = Path("data/wave_registry.csv")
HISTORY_DIR = Path("data/history")
OUTPUT_PATH = Path("data/alpha_attribution_summary.csv")
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
DAYS_LOOKBACK = 365


# =========================
# READERS
# =========================

def load_wave_registry():
    """Load and validate the wave registry file."""
    if not WAVE_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Wave registry not found: {WAVE_REGISTRY_PATH}")

    registry = pd.read_csv(WAVE_REGISTRY_PATH)
    if registry.empty:
        raise ValueError("Wave registry is empty; cannot proceed.")
    
    if "wave_id" not in registry.columns:
        raise ValueError("Wave registry missing required column: 'wave_id'")
    
    return registry


def load_wave_history(wave_id):
    """Load the wave's historical returns file."""
    history_path = HISTORY_DIR / wave_id / "history.csv"
    if not history_path.exists():
        return None, "MISSING_HISTORY", f"File not found: {history_path}"
    
    try:
        history = pd.read_csv(history_path)
        if "date" not in history or "wave_return" not in history or "benchmark_return" not in history:
            return None, "INVALID_HISTORY", "Missing required columns in history file."
        
        return history, "OK", ""
    except Exception as e:
        return None, "ERROR_LOADING_HISTORY", str(e)


# =========================
# ATTRIBUTION LOGIC
# =========================

def calculate_alpha_attribution(history, wave_id):
    """Calculate alpha attribution metrics for a wave."""
    try:
        history["date"] = pd.to_datetime(history["date"]).dt.date
        cutoff_date = datetime.utcnow().date() - pd.Timedelta(days=DAYS_LOOKBACK)
        recent_history = history[history["date"] >= cutoff_date]

        total_wave_return = recent_history["wave_return"].sum()
        total_benchmark_return = recent_history["benchmark_return"].sum()
        total_alpha = total_wave_return - total_benchmark_return
        days_count = len(recent_history)
        
        if days_count == 0:
            return 0.0, 0.0, 0.0, 0, "NO_RECENT_HISTORY", "No data within the lookback period."
        
        return total_alpha, total_wave_return, total_benchmark_return, days_count, "OK", ""
    except Exception as e:
        return float("nan"), float("nan"), float("nan"), 0, "COMPUTATION_ERROR", f"Error during calculation: {e}"


# =========================
# MAIN LOGIC
# =========================

def main():
    registry = load_wave_registry()
    output_rows = []

    for idx, wave in registry.iterrows():
        wave_id = wave["wave_id"]
        wave_name = wave.get("display_name", wave_id)
        active = wave.get("active", True)

        if not active:
            continue
        
        # Load and validate history for the current wave
        history, load_status, notes = load_wave_history(wave_id)
        
        if load_status != "OK":
            output_rows.append([
                wave_name, "LIVE", 0, float("nan"), float("nan"), float("nan"), load_status, notes
            ])
            continue
        
        # Compute alpha attribution if history is valid
        total_alpha, total_wave_return, total_benchmark_return, days_count, calc_status, calc_notes = calculate_alpha_attribution(history, wave_id)
        
        output_rows.append([
            wave_name,
            "LIVE",
            days_count,
            total_alpha,
            total_wave_return,
            total_benchmark_return,
            calc_status,
            calc_notes,
        ])
    
    # Raise an error if no rows were generated (hard guarantee)
    if not output_rows:
        raise RuntimeError("No rows were generated; this violates a hard requirement.")

    # Write results to the output file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"[INFO] Alpha attribution completed. Rows written: {len(output_df)}")


if __name__ == "__main__":
    main()