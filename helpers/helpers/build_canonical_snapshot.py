"""
build_canonical_snapshot.py

THIS FILE IS THE MISSING GENERATOR.

Purpose:
- Generate rows for data/canonical_snapshot.csv
- One row per Wave
- Provide a complete, schema-correct snapshot
- No UI logic, no Streamlit, no enrichment

Downstream files (snapshot_ledger, rebuild_snapshot, app)
EXPECT THIS FILE TO EXIST AND PRODUCE ROWS.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from datetime import date


# -------------------------------------------------------------------
# Canonical snapshot schema
# -------------------------------------------------------------------

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


# -------------------------------------------------------------------
# TEMPORARY CANONICAL WAVE LIST
# (This guarantees ROWS exist)
# -------------------------------------------------------------------

CANONICAL_WAVES = [
    ("ai_cloud_megacap_wave", "AI & Cloud MegaCap Wave"),
    ("clean_transit_infrastructure_wave", "Clean Transit-Infrastructure Wave"),
    ("demas_fund_wave", "Demas Fund Wave"),
    ("ev_infrastructure_wave", "EV & Infrastructure Wave"),
    ("future_energy_ev_wave", "Future Energy & EV Wave"),
    ("future_power_energy_wave", "Future Power & Energy Wave"),
    ("next_gen_compute_semis_wave", "Next-Gen Compute & Semis Wave"),
    ("small_cap_growth_wave", "Small-Cap Growth Wave"),
    ("small_to_mid_cap_growth_wave", "Small to Mid Cap Growth Wave"),
    ("us_megacap_core_wave", "US MegaCap Core Wave"),
    ("sp500_wave", "S&P 500 Wave"),
]


# -------------------------------------------------------------------
# Generator
# -------------------------------------------------------------------

def build_canonical_snapshot() -> pd.DataFrame:
    """
    Build canonical snapshot rows.

    NOTE:
    - Metrics are placeholders for now
    - This FIXES the empty-row bug
    - Real metrics can be layered later
    """

    rows = []
    today = date.today().isoformat()

    for wave_id, wave_name in CANONICAL_WAVES:
        row = {
            "Wave_ID": wave_id,
            "Wave_Name": wave_name,
            "Asset_Class": "Equity",
            "Mode": "Standard",
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

            "VIX_Regime": "",
            "Exposure": 1.0,
            "CashPercent": 0.0,
        }

        rows.append(row)

    df = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    return df


# -------------------------------------------------------------------
# Write to disk
# -------------------------------------------------------------------

def write_canonical_snapshot(
    output_path: Path = Path("data/canonical_snapshot.csv"),
) -> pd.DataFrame:
    df = build_canonical_snapshot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, header=False)
    return df


if __name__ == "__main__":
    write_canonical_snapshot()