from pathlib import Path
import pandas as pd

SNAPSHOT_PATH = Path("data/live_snapshot_v2.csv")

REQUIRED_COLUMNS = [
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

def load_snapshot_v2() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"Snapshot not found: {SNAPSHOT_PATH}")

    df = pd.read_csv(SNAPSHOT_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Snapshot missing columns: {missing}")

    if df.empty:
        raise ValueError("Snapshot is empty")

    return df