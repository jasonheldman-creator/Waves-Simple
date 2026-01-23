import pandas as pd
from typing import Dict, Any, List


def wave_performance_diagnostics(wave_row: Dict[str, Any]) -> List[str]:
    """
    Diagnoses validation issues for a single wave performance row.
    Does NOT block the system — only reports issues.
    """
    issues = []

    if wave_row is None:
        return ["No wave performance data"]

    # Accept ANY valid numeric return as sufficient
    return_fields = [
        "return_1d",
        "return_30d",
        "return_60d",
        "return_365d",
    ]

    has_valid_return = False

    for field in return_fields:
        val = wave_row.get(field)
        if isinstance(val, (int, float)) and not pd.isna(val):
            has_valid_return = True
            break

    if not has_valid_return:
        issues.append("No valid return horizons")

    return issues


def compute_global_health(performance_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes global performance health based on wave-level diagnostics.
    SAFE:
    - Accepts partial horizons
    - Requires only ONE valid return anywhere
    """

    total = len(performance_rows)
    validated = 0
    diagnostics = []

    for row in performance_rows:
        issues = wave_performance_diagnostics(row)
        diagnostics.append({
            "wave_id": row.get("wave_id"),
            "issues": issues,
        })

        if not issues:
            validated += 1

    status = "OK" if validated > 0 else "DEGRADED"

    return {
        "status": status,
        "waves_total": total,
        "waves_validated": validated,
        "diagnostics": diagnostics,
    }