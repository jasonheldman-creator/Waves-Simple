# scripts/generate_live_snapshot_csv.py
# Build data/live_snapshot.csv from engine outputs

import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH when run from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd

from attribution_engine import compute_horizon_attribution


DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"


def generate_live_snapshot_csv(
    waves: dict,
    benchmarks: dict,
    engine_weights: dict,
) -> None:
    """
    Generate live_snapshot.csv with multi-horizon attribution.
    """

    rows = []

    horizon_meta = {
        "30d": 30,
        "60d": 60,
        "365d": 252,
    }

    for wave_id, wave_series in waves.items():
        benchmark_series = benchmarks.get(wave_id)
        if benchmark_series is None:
            continue

        row = {
            "wave_id": wave_id,
        }

        # Track which horizons have sufficient history
        horizon_ok = {}

        # Returns and total alpha per horizon
        for suffix, days in horizon_meta.items():
            wave_h = wave_series.tail(days)
            bench_h = benchmark_series.tail(days)

            if len(wave_h) < 2 or len(bench_h) < 2:
                row[f"return_{suffix}"] = np.nan
                row[f"benchmark_return_{suffix}"] = np.nan
                row[f"alpha_{suffix}"] = np.nan
                horizon_ok[suffix] = False
                continue

            wave_ret = wave_h.iloc[-1] / wave_h.iloc[0] - 1.0
            bench_ret = bench_h.iloc[-1] / bench_h.iloc[0] - 1.0
            total_alpha = wave_ret - bench_ret

            row[f"return_{suffix}"] = float(wave_ret)
            row[f"benchmark_return_{suffix}"] = float(bench_ret)
            row[f"alpha_{suffix}"] = float(total_alpha)
            horizon_ok[suffix] = True

        # Attribution components per horizon
        for suffix, days in horizon_meta.items():

            # FIX: Only compute attribution when returns were computable
            if not horizon_ok[suffix]:
                row[f"alpha_beta_{suffix}"] = np.nan
                row[f"alpha_momentum_{suffix}"] = np.nan
                row[f"alpha_volatility_{suffix}"] = np.nan
                row[f"alpha_allocation_{suffix}"] = np.nan
                row[f"alpha_residual_{suffix}"] = np.nan

                if suffix == "365d":
                    row["Alpha_Momentum_365D"] = np.nan
                    row["Alpha_Volatility_365D"] = np.nan
                    row["Alpha_Residual_365D"] = np.nan

                continue

            # Safe to compute attribution
            attribution = compute_horizon_attribution(
                wave_series,
                benchmark_series,
                engine_weights,
                days,
            )

            beta = attribution.get("beta", np.nan)
            momentum = attribution.get("momentum", np.nan)
            volatility = attribution.get("volatility", np.nan)
            allocation = attribution.get("allocation", np.nan)
            residual = attribution.get("residual", np.nan)

            # Canonical lowercase (engine-native)
            row[f"alpha_beta_{suffix}"] = beta
            row[f"alpha_momentum_{suffix}"] = momentum
            row[f"alpha_volatility_{suffix}"] = volatility
            row[f"alpha_allocation_{suffix}"] = allocation
            row[f"alpha_residual_{suffix}"] = residual

            # Legacy UI contract (TitleCase snapshot schema)
            if suffix == "365d":
                row["Alpha_Momentum_365D"] = momentum
                row["Alpha_Volatility_365D"] = volatility
                row["Alpha_Residual_365D"] = residual

        rows.append(row)

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LIVE_SNAPSHOT_PATH, index=False)


if __name__ == "__main__":
    pass
