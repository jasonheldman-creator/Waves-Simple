import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, time

# ============================================================================
# INTRADAY-AWARE 1D RETURN (OPTION B — REAL EQUITY BEHAVIOR)
# ============================================================================

def _is_us_equity_market_open(now_utc: Optional[datetime] = None) -> bool:
    """
    Lightweight US equity market open check (UTC).
    Conservative by design — avoids false positives.
    """

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    # Weekend → closed
    if now_utc.weekday() >= 5:
        return False

    # NYSE regular hours ≈ 14:30–21:00 UTC
    market_open = time(14, 30)
    market_close = time(21, 0)

    return market_open <= now_utc.time() <= market_close


def compute_intraday_1d_return(prices: pd.Series) -> Optional[float]:
    """
    Option B — Intraday-aware 1D return.

    LOGIC:
    - If market OPEN → (last price / prior close) - 1
    - If market CLOSED → (last close / prior close) - 1
    - Suppresses fake zero returns
    - Never raises
    """

    try:
        if prices is None or len(prices) < 2:
            return None

        prices = prices.dropna()
        if len(prices) < 2:
            return None

        now_utc = datetime.now(timezone.utc)
        market_open = _is_us_equity_market_open(now_utc)

        latest = float(prices.iloc[-1])
        prior = float(prices.iloc[-2])

        if prior <= 0:
            return None

        ret = (latest / prior) - 1

        # Suppress meaningless flat values
        if abs(ret) < 1e-6:
            return None

        return ret

    except Exception:
        return None


# ============================================================================
# WAVE-LEVEL DIAGNOSTICS (NON-BLOCKING)
# ============================================================================

def wave_performance_diagnostics(wave_row: Dict[str, Any]) -> List[str]:
    """
    Diagnoses validation issues for a single wave.

    VALID IF:
    - ANY numeric return exists OR
    - NAV exists
    """

    if not isinstance(wave_row, dict):
        return ["No wave data"]

    for k, v in wave_row.items():
        if "return" in k.lower():
            if isinstance(v, (int, float)) and not pd.isna(v):
                return []

    nav = wave_row.get("nav") or wave_row.get("NAV")
    if isinstance(nav, (int, float)) and not pd.isna(nav):
        return []

    return ["No validated performance horizons"]


# ============================================================================
# PUBLIC API — REQUIRED BY APP
# ============================================================================

def compute_all_waves_performance(
    price_book: pd.DataFrame,
    periods: Optional[List[int]] = None,
    only_validated: bool = False,
) -> pd.DataFrame:
    """
    Computes performance for all waves.

    CONTRACT:
    - Always returns DataFrame
    - Never hard-fails
    - Includes return_* columns
    """

    if periods is None:
        periods = [1, 30, 60, 365]

    if price_book is None or price_book.empty:
        return pd.DataFrame(
            columns=["wave_id"] + [f"return_{p}d" for p in periods]
        )

    rows: List[Dict[str, Any]] = []

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
# GLOBAL HEALTH
# ============================================================================

def compute_global_health(performance_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    System health logic.

    HEALTHY IF:
    - At least ONE wave validates
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