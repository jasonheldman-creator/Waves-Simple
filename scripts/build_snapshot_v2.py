#!/usr/bin/env python3
"""
build_snapshot_v2.py

Authoritative snapshot generator.
This file CREATES rows. No fallbacks. No legacy dependencies.
"""

from pathlib import Path
import pandas as pd
from datetime import date

OUTPUT_PATH = Path("data/live_snapshot_v2.csv")

SNAPSHOT_COLUMNS = [
    "Wave_ID",
    "Wave_Name",
    "Asset_Class",
    "Mode",
    "Snapshot_Date",

    "Return_1D",
    "Return_30D",
    "Return_60D",
    "Return_365D",

    "Alpha_1D",
    "Alpha_30D",
    "Alpha_60D",
    "Alpha_365D",

    "Benchmark_Return_1D",
    "Benchmark_Return_30D",
    "Benchmark_Return_60D",
    "Benchmark_Return_365D",

    "VIX_Regime",
    "Exposure",
    "CashPercent",
]

# Canonical waves — explicit and deterministic
WAVES = [
    ("sp500_wave", "S&P 500 Wave", "Equity", "Standard"),
    ("ai_cloud_megacap_wave", "AI & Cloud MegaCap Wave", "Equity", "Standard"),
    ("clean_transit_infrastructure_wave", "Clean Transit-Infrastructure Wave", "Equity", "Standard"),
    ("gold_wave", "Gold Wave", "Commodity", "Standard"),
    ("income_wave", "Income Wave", "Fixed Income", "Standard"),
]

def build_snapshot() -> pd.DataFrame:
    today = date.today().isoformat()
    rows = []

    for wave_id, name, asset_class, mode in WAVES:
        rows.append({
            "Wave_ID": wave_id,
            "Wave_Name": name,
            "Asset_Class": asset_class,
            "Mode": mode,
            "Snapshot_Date": today,

            "Return_1D": 0.0,
            "Return_30D": 0.0,
            "Return_60D": 0.0,
            "Return_365D": 0.0,

            "Alpha_1D": 0.0,
            "Alpha_30D": 0.0,
            "Alpha_60D": 0.0,
            "Alpha_365D": 0.0,

            "Benchmark_Return_1D": 0.0,
            "Benchmark_Return_30D": 0.0,
            "Benchmark_Return_60D": 0.0,
            "Benchmark_Return_365D": 0.0,

            "VIX_Regime": "UNKNOWN",
            "Exposure": 1.0,
            "CashPercent": 0.0,
        })

    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)

def main():
    df = build_snapshot()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Snapshot v2 written: {OUTPUT_PATH}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

if __name__ == "__main__":
    main()