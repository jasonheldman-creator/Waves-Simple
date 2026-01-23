import pandas as pd
from typing import Dict, Any, List, Optional


# ============================================================================
# WAVE-LEVEL DIAGNOSTICS (NON-BLOCKING)
# ============================================================================

def wave_performance_diagnostics(wave_row: Dict[str, Any]) -> List[str]:
    """
    Diagnoses validation issues for a single wave performance row.

    GUARANTEES:
    - Does NOT block the system
    - Accepts partial horizons
    - Requires only ONE valid numeric return
    """
    issues: List[str] = []

    if not isinstance(wave_row, dict):
        return ["No wave performance data"]

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


# ============================================================================
# PUBLIC API — REQUIRED BY APP (DO NOT REMOVE)
# ============================================================================

def compute_all_waves_performance(
    price_book: pd.DataFrame,
    periods: Optional[List[int]] = None,
    only_validated: bool = False,
) -> pd.DataFrame:
    """
    Computes performance metrics for all waves.

    CONTRACT (CRITICAL):
    - MUST return a pandas DataFrame
    - MUST include return_* columns
    - MUST NOT hard-fail
    - UI + System Health depend on this
    """

    if periods is None:
        periods = [1, 30, 60, 365]

    rows: List[Dict[str, Any]] = []

    # SAFETY: empty or missing price book
    if price_book is None or price_book.empty:
        return pd.DataFrame(columns=["wave_id"] + [f"return_{p}d" for p in periods])

    # NOTE:
    # This assumes price_book columns are wave identifiers
    # and index is datetime-like.
    for wave_id in price_book.columns:
        series = price_book[wave_id].dropna()

        row: Dict[str, Any] = {"wave_id": wave_id}

        for p in periods:
            col_name = f"return_{p}d"

            try:
                if len(series) > p:
                    ret = (series.iloc[-1] / series.iloc[-(p + 1)] - 1)
                    row[col_name] = float(ret)
                else:
                    row[col_name] = None
            except Exception:
                row[col_name] = None

        # Diagnostics (non-blocking)
        issues = wave_performance_diagnostics(row)
        row["validation_issues"] = issues

        if only_validated and issues:
            continue

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


# ============================================================================
# GLOBAL HEALTH (OPTIONAL, SAFE)
# ============================================================================

def compute_global_health(performance_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes global performance health from wave diagnostics.

    SAFE LOGIC:
    - Requires only ONE validated wave
    - Partial horizons allowed
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