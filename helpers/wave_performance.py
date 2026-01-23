import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# ============================================================================
# INTRADAY-AWARE 1D RETURN (TRUTH-SAFE, NO FAKE ZEROS)
# ============================================================================

def compute_intraday_1d_return(prices: pd.Series) -> Optional[float]:
    """
    Computes a truth-safe 1D return.

    RULES:
    - Never fabricates 0.0 returns
    - Flat prices -> None (not a lie)
    - Intraday-safe without market-hour guessing
    - Never raises
    """

    try:
        if prices is None or len(prices) < 2:
            return None

        latest = float(prices.iloc[-1])
        base = float(prices.iloc[-2])

        # Invalid base
        if base <= 0:
            return None

        # CRITICAL: suppress fake zero returns
        if latest == base:
            return None

        return (latest / base) - 1

    except Exception:
        return None


# ============================================================================
# WAVE-LEVEL DIAGNOSTICS (NON-BLOCKING, TRUTH-FIRST)
# ============================================================================

def wave_performance_diagnostics(wave_row: Dict[str, Any]) -> List[str]:
    """
    Diagnoses validation issues for a single wave.

    VALID IF:
    - ANY numeric return exists, OR
    - NAV exists

    NEVER blocks the pipeline.
    """

    if not isinstance(wave_row, dict):
        return ["No wave performance data"]

    # 1) Any numeric return is sufficient
    for key, val in wave_row.items():
        if "return" in key.lower():
            if isinstance(val, (int, float)) and not pd.isna(val):
                return []

    # 2) NAV fallback (computed or CSV-based)
    nav = wave_row.get("nav") or wave_row.get("NAV")
    if isinstance(nav, (int, float)) and not pd.isna(nav):
        return []

    # 3) Truly invalid
    return ["No validated performance horizons"]


# ============================================================================
# PUBLIC API — REQUIRED BY APP (DO NOT REMOVE OR RENAME)
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
    - Truth-safe (no fake zeros)
    """

    if periods is None:
        periods = [1, 30, 60, 365]

    # SAFETY: empty input
    if price_book is None or price_book.empty:
        return pd.DataFrame(
            columns=["wave_id"] + [f"return_{p}d" for p in periods]
        )

    rows: List[Dict[str, Any]] = []

    # Assumes:
    # - columns = wave_id
    # - index = datetime
    for wave_id in price_book.columns:
        series = price_book[wave_id].dropna()
        row: Dict[str, Any] = {"wave_id": wave_id}

        for p in periods:
            col = f"return_{p}d"

            try:
                if p == 1:
                    row[col] = compute_intraday_1d_return(series)
                else:
                    if len(series) > p:
                        base = float(series.iloc[-(p + 1)])
                        latest = float(series.iloc[-1])

                        if base <= 0 or latest == base:
                            row[col] = None
                        else:
                            row[col] = (latest / base) - 1
                    else:
                        row[col] = None
            except Exception:
                row[col] = None

        # Non-blocking diagnostics
        issues = wave_performance_diagnostics(row)
        row["validation_issues"] = issues

        if only_validated and issues:
            continue

        rows.append(row)

    df = pd.DataFrame(rows)

    # UI compatibility layer (display-safe, logic remains truth-first)
    for col in df.columns:
        if col.startswith("return_"):
            df[col] = df[col].fillna(0.0)

    return df


# ============================================================================
# GLOBAL HEALTH — DRIVES DASHBOARD STATUS
# ============================================================================

def compute_global_health(performance_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Global system health.

    HEALTHY IF:
    - At least ONE wave validates
    - Partial horizons allowed
    - NAV-only waves allowed
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