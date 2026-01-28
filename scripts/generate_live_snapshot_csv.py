# ============================================================
# generate_live_snapshot_csv.py
# Build data/live_snapshot.csv from engine outputs
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

from attribution_engine import compute_horizon_attribution

DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"


def generate_live_snapshot_csv(
    waves: dict,
    benchmarks: dict,
    engine_weights: dict,
) -> None:

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

        row = {"wave_id": wave_id}

        for suffix, days in horizon_meta.items():
            wave_h = wave_series.tail(days)
            bench_h = benchmark_series.tail(days)

            if len(wave_h) < 2 or len(bench_h) < 2:
                row[f"return_{suffix}"] = np.nan
                row[f"benchmark_return_{suffix}"] = np.nan
                row[f"alpha_{suffix}"] = np.nan
                continue

            wave_ret = wave_h.iloc[-1] / wave_h.iloc[0] - 1.0
            bench_ret = bench_h.iloc[-1] / bench_h.iloc[0] - 1.0

            row[f"return_{suffix}"] = float(wave_ret)
            row[f"benchmark_return_{suffix}"] = float(bench_ret)
            row[f"alpha_{suffix}"] = float(wave_ret - bench_ret)

        for suffix, days in horizon_meta.items():
            attribution = compute_horizon_attribution(
                wave_series,
                benchmark_series,
                engine_weights,
                days,
            )

            row[f"alpha_beta_{suffix}"] = attribution.get("beta", np.nan)
            row[f"alpha_momentum_{suffix}"] = attribution.get("momentum", np.nan)
            row[f"alpha_volatility_{suffix}"] = attribution.get("volatility", np.nan)
            row[f"alpha_allocation_{suffix}"] = attribution.get("allocation", np.nan)
            row[f"alpha_residual_{suffix}"] = attribution.get("residual", np.nan)

        rows.append(row)

    pd.DataFrame(rows).to_csv(LIVE_SNAPSHOT_PATH, index=False)