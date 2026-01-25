#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

AUTHORITATIVE live snapshot generator.

Reads:
- data/cache/prices_cache.parquet
- data/wave_weights.csv

Writes:
- data/live_snapshot.csv

This file is schema-normalized and defensive by design.
"""

from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

# --------------------
# Paths
# --------------------
PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")

# --------------------
# Output Columns
# --------------------
OUTPUT_COLUMNS = [
    "Wave_ID",
    "Wave",
    "Return_1D",
    "Return_30D",
    "Return_60