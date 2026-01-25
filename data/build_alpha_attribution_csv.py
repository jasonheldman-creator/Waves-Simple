"""
build_alpha_attribution_csv.py

Builds a canonical alpha attribution summary CSV for all waves.

This file intentionally ALWAYS emits a row for each wave + mode
once sufficient history exists — even if alpha is zero.

The UI depends on row existence, not alpha magnitude.
"""

import pandas as pd
from pathlib import Path

# =========================
# Configuration
# =========================

DAYS_LOOKBACK = 365

HISTORY_DIR = Path("data/history")
REGISTRY_PATH = Path("data/wave_registry.csv")
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
# Helpers
# =========================

def load_history(wave_id: str, mode: str) -> pd.DataFrame:
    """
    Load history CSV for a wave + mode.
    """
    path = HISTORY_DIR / wave_id / f"{mode}_history.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_totals