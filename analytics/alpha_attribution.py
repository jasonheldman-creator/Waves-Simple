# analytics/alpha_attribution.py

import os
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

LIVE_SNAPSHOT_PATH = "data/live_snapshot.csv"
ATTRIBUTION_PATH = "data/alpha_attribution_snapshot.csv"

# -------------------------------------------------------------------
# Canonical driver binding (UI-facing keys → snapshot columns)
# -------------------------------------------------------------------
# This is the explicit, non-inferred binding layer that the Alpha Attribution
# UI can import/use to resolve canonical driver keys:
#
#   "beta"        → alpha_market
#   "momentum"    → alpha_momentum
#   "volatility"  → alpha_volatility
#   "allocation"  → alpha_rotation
#   "residual"    → alpha_stock_selection
#
# NOTE:
# - This does NOT rename any columns.
# - This does NOT change attribution math.
# - This does NOT change the CSV schema written below.
CANONICAL_ALPHA_DRIVER_MAP = {
    "beta": "alpha_market",
    "momentum": "alpha_momentum",
    "volatility": "alpha_volatility",
    "allocation": "alpha_rotation",
    "residual": "alpha_stock_selection",
}


def get_canonical_alpha_driver_map():
    """
    Accessor for the canonical alpha driver mapping.

    This is intentionally a thin wrapper so that downstream consumers
    (TruthFrame, UI adapters, etc.) can import a single, authoritative
    mapping from canonical driver keys to the columns emitted in
    alpha_attribution_snapshot.csv, without inferring or hard-coding
    their own bindings.
    """
    return CANONICAL_ALPHA_DRIVER_MAP.copy()


# -------------------------------------------------------------------
# Core generator (SAFE to call from app.py)
# -------------------------------------------------------------------

def build_alpha_attribution_snapshot():
    """
    Build alpha attribution snapshot directly from live_snapshot.csv.

    HARD RULES (non-negotiable):
    - Only real Waves (must have Wave_ID)
    - No benchmark placeholder rows
    - No cash-only waves
    - Alpha must be numeric and finite
    - One row in → one row out (never all-or-nothing failure)

    Alpha sources (canonical):
    - Market / Beta
    - Momentum
    - Volatility (VIX / convexity)
    - Rotation / Allocation
    - Stock Selection (Residual)
    """

    # ---------------------------------------------------------------
    # Preconditions
    # ---------------------------------------------------------------
    if not os.path.exists(LIVE_SNAPSHOT_PATH):
        return False, "live_snapshot.csv not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)

    if df.empty:
        return False, "live_snapshot.csv is empty"

    required_cols = ["Wave_ID", "Wave", "Category"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    # ---------------------------------------------------------------
    # Determine preferred alpha column (institutional priority)
    # ---------------------------------------------------------------
    if "Alpha_60D" in df.columns:
        alpha_col = "Alpha_60D"
    elif "Alpha_30D" in df.columns:
        alpha_col = "Alpha_30D"
    elif "Alpha_1D" in df.columns:
        alpha_col = "Alpha_1D"
    else:
        return False, "No Alpha column found (Alpha_60D / Alpha_30D / Alpha_1D)"

    rows = []
    skipped = 0

    # ---------------------------------------------------------------
    # Attribution logic (ROW-SAFE)
    # ---------------------------------------------------------------
    for _, r in df.iterrows():
        try:
            wave_id = r.get("Wave_ID")
            wave_name = r.get("Wave")
            category = str(r.get("Category", "")).lower()

            # -------------------------------------------------------
            # HARD FILTERS
            # -------------------------------------------------------
            if pd.isna(wave_id) or str(wave_id).strip() == "":
                skipped += 1
                continue

            if "benchmark" in str(wave_name).lower():
                skipped += 1
                continue

            if category in {"cash", "money market", "money_market"}:
                skipped += 1
                continue

            # -------------------------------------------------------
            # Alpha (must be valid)
            # -------------------------------------------------------
            alpha_total = r.get(alpha_col, None)
            if alpha_total is None or pd.isna(alpha_total):
                skipped += 1
                continue

            try:
                alpha_total = float(alpha_total)
            except Exception:
                skipped += 1
                continue

            # -------------------------------------------------------
            # Market beta contribution
            # -------------------------------------------------------
            try:
                beta_real = float(r.get("Beta_Real", 1.0))
            except Exception:
                beta_real = 1.0

            alpha_market = (beta_real - 1.0) * alpha_total * 0.20

            # -------------------------------------------------------
            # Volatility / VIX overlay (canonical VOLATILITY driver)
            # -------------------------------------------------------
            alpha_volatility = 0.0
            vix_regime = str(r.get("VIX_Regime", "")).lower()

            if "low" in vix_regime or "risk" in vix_regime:
                try:
                    adj = float(r.get("VIX_Adjustment_Pct", 0.0))
                    alpha_volatility = adj if adj != 0 else alpha_total * 0.25
                except Exception:
                    alpha_volatility = alpha_total * 0.25

            # -------------------------------------------------------
            # Momentum / trend overlays
            # -------------------------------------------------------
            alpha_momentum = 0.0
            stack = f"{r.get('strategy_stack', '')} {r.get('strategy_stack_applied', '')}".lower()

            if "momentum" in stack or "trend" in stack:
                alpha_momentum = alpha_total * 0.35

            # -------------------------------------------------------
            # Rotation / beta drift
            # -------------------------------------------------------
            try:
                beta_drift = float(r.get("Beta_Drift", 0.0))
            except Exception:
                beta_drift = 0.0

            alpha_rotation = -beta_drift * 0.15 if beta_drift != 0 else 0.0

            # -------------------------------------------------------
            # Residual = stock selection
            # -------------------------------------------------------
            alpha_stock_selection = (
                alpha_total
                - alpha_market
                - alpha_volatility
                - alpha_momentum
                - alpha_rotation
            )

            rows.append(
                {
                    "wave_id": wave_id,
                    "wave_name": wave_name,

                    # Canonical attribution drivers (UI-visible)
                    "alpha_market": alpha_market,
                    "alpha_momentum": alpha_momentum,
                    "alpha_volatility": alpha_volatility,
                    "alpha_rotation": alpha_rotation,
                    "alpha_stock_selection": alpha_stock_selection,

                    # Total alpha for reconciliation
                    "alpha_total": alpha_total,
                }
            )

        except Exception:
            skipped += 1
            continue

    # ---------------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------------
    if not rows:
        return False, f"No valid attribution rows produced (skipped={skipped})"

    out_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(ATTRIBUTION_PATH), exist_ok=True)
    out_df.to_csv(ATTRIBUTION_PATH, index=False)

    return True, f"Wrote {len(out_df)} rows to {ATTRIBUTION_PATH} (skipped={skipped})"


# -------------------------------------------------------------------
# CLI support
# -------------------------------------------------------------------

if __name__ == "__main__":
    ok, msg = build_alpha_attribution_snapshot()
    print(msg)
