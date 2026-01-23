import pandas as pd
from typing import Dict, Any, List, Optional

# ============================================================================
# WAVE-LEVEL DIAGNOSTICS (NON-BLOCKING, CSV + COMPUTED SAFE)
# ============================================================================

def wave_performance_diagnostics(wave_row: Dict[str, Any]) -> List[str]:
    """
    Diagnoses validation issues for a single wave performance row.

    RULES (CRITICAL):
    - Non-blocking
    - ONE numeric return is sufficient
    - Accepts CSV or computed formats
    """

    if not isinstance(wave_row, dict):
        return ["No wave performance data"]

    has_valid_return = False

    for key, val in wave_row.items():
        if "return" in key.lower():
            if isinstance(val, (int, float)) and not pd.isna(val):
                has_valid_return = True
                break

    if not has_valid_return:
        return ["No validated performance horizons"]

    return []


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

    CONTRACT:
    - MUST return a DataFrame
    - MUST include return_* columns
    - MUST NOT hard-fail
    """

    if periods is None:
        periods = [1, 30, 60, 365]

    if price_book is None or price_book.empty:
        return pd.DataFrame(columns=["wave_id"] + [f"return_{p}d" for p in periods])

    rows: List[Dict[str, Any]] = []

    for wave_id in price_book.columns:
        series = price_book[wave_id].dropna()
        row: Dict[str, Any] = {"wave_id": wave_id}

        for p in periods:
            col = f"return_{p}d"
            try:
                if len(series) > p:
                    row[col] = float(series.iloc[-1] / series.iloc[-(p + 1)] - 1)
                else:
                    row[col] = None
            except Exception:
                row[col] = None

        issues = wave_performance_diagnostics(row)
        row["validation_issues"] = issues

        if only_validated and issues:
            continue

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# GLOBAL HEALTH — THIS IS WHAT DRIVES THE DASHBOARD STATUS
# ============================================================================

def compute_global_health(performance_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Global system health.

    HEALTHY IF:
    - At least ONE wave has a validated return
    """

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

    return {
        "status": "OK" if validated > 0 else "DEGRADED",
        "waves_total": len(performance_rows),
        "waves_validated": validated,
        "diagnostics": diagnostics,
    }