# ============================================================
# PORTFOLIO SNAPSHOT HELPER
# Canonical, TruthFrame-backed, SAFE
# ============================================================

from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Canonical horizons
HORIZONS = [
    ("1D", 1),
    ("30D", 30),
    ("60D", 60),
    ("365D", 365),
]


def _safe_number(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def build_portfolio_snapshot_from_truthframe(
    truthframe: Dict[str, Any]
) -> pd.DataFrame:
    """
    Build a portfolio-level snapshot table from TruthFrame.

    Output columns:
        Wave
        Return_1D / Alpha_1D
        Return_30D / Alpha_30D
        Return_60D / Alpha_60D
        Return_365D / Alpha_365D
        Alpha_Total
        Health

    SAFE:
        • Never raises
        • Missing data → NaN
        • Always returns a DataFrame
    """

    rows: List[Dict[str, Any]] = []

    if not isinstance(truthframe, dict):
        return pd.DataFrame()

    waves = truthframe.get("waves", {})
    if not isinstance(waves, dict) or not waves:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Per-wave rows
    # --------------------------------------------------------
    for wave_name, wave_data in waves.items():
        row = {
            "Wave": wave_name,
        }

        performance = wave_data.get("performance", {})
        alpha_block = wave_data.get("alpha", {})

        for label, _days in HORIZONS:
            perf = performance.get(label, {})

            row[f"Return_{label}"] = _safe_number(
                perf.get("return")
            )
            row[f"Alpha_{label}"] = _safe_number(
                perf.get("alpha")
            )

        row["Alpha_Total"] = _safe_number(
            alpha_block.get("total")
        )

        row["Health"] = (
            wave_data.get("health", {}).get("status", "UNKNOWN")
        )

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # --------------------------------------------------------
    # Portfolio roll-up row
    # --------------------------------------------------------
    portfolio_row = {"Wave": "PORTFOLIO"}

    for label, _ in HORIZONS:
        portfolio_row[f"Return_{label}"] = _safe_number(
            df[f"Return_{label}"].mean()
        )
        portfolio_row[f"Alpha_{label}"] = _safe_number(
            df[f"Alpha_{label}"].sum()
        )

    portfolio_row["Alpha_Total"] = _safe_number(
        df["Alpha_Total"].sum()
    )

    portfolio_row["Health"] = "OK"

    df = pd.concat(
        [df, pd.DataFrame([portfolio_row])],
        ignore_index=True,
    )

    return df