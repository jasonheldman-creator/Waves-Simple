# analytics/alpha_attribution.py

import os
import pandas as pd

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

LIVE_SNAPSHOT_PATH = "data/live_snapshot.csv"
ATTRIBUTION_PATH = "data/alpha_attribution_snapshot.csv"


# -------------------------------------------------------------------
# Core generator (SAFE to call from app.py)
# -------------------------------------------------------------------

def build_alpha_attribution_snapshot():
    """
    Build alpha attribution snapshot directly from live_snapshot.csv.

    Attribution logic:
    - Uses Alpha_60D as primary institutional signal (falls back to Alpha_1D)
    - Decomposes alpha into:
        * Market beta contribution
        * VIX / volatility overlay
        * Momentum / trend overlays
        * Rotation / beta drift effects
        * Residual stock selection

    Output columns:
        - wave_id
        - wave_name
        - alpha_market
        - alpha_vix
        - alpha_momentum
        - alpha_rotation
        - alpha_stock_selection
        - alpha_total

    Returns:
        (bool, str)
    """

    # ---------------------------------------------------------------
    # Preconditions
    # ---------------------------------------------------------------
    if not os.path.exists(LIVE_SNAPSHOT_PATH):
        return False, "live_snapshot.csv not found"

    df = pd.read_csv(LIVE_SNAPSHOT_PATH)

    if df.empty:
        return False, "live_snapshot.csv is empty"

    # Required identifiers
    required_cols = ["Wave_ID", "Wave"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    # Determine best alpha column (institutional preference)
    if "Alpha_60D" in df.columns:
        alpha_col = "Alpha_60D"
    elif "Alpha_30D" in df.columns:
        alpha_col = "Alpha_30D"
    elif "Alpha_1D" in df.columns:
        alpha_col = "Alpha_1D"
    else:
        return False, "No Alpha column found (Alpha_60D / Alpha_30D / Alpha_1D)"

    rows = []

    # ---------------------------------------------------------------
    # Attribution logic
    # ---------------------------------------------------------------
    for _, r in df.iterrows():
        try:
            wave_id = r.get("Wave_ID")
            wave_name = r.get("Wave")

            alpha_total = float(r.get(alpha_col, 0.0))
            if pd.isna(alpha_total):
                alpha_total = 0.0

            # -------------------------------------------------------
            # Market beta contribution
            # -------------------------------------------------------
            beta_real = r.get("Beta_Real", 1.0)
            try:
                beta_real = float(beta_real)
            except Exception:
                beta_real = 1.0

            alpha_market = (beta_real - 1.0) * alpha_total * 0.20

            # -------------------------------------------------------
            # VIX / volatility overlay
            # -------------------------------------------------------
            alpha_vix = 0.0
            vix_regime = str(r.get("VIX_Regime", "")).lower()

            if "low" in vix_regime or "risk-off" in vix_regime:
                # Use explicit adjustment if present, else proportional
                adj = r.get("VIX_Adjustment_Pct", None)
                if pd.notna(adj):
                    try:
                        alpha_vix = float(adj)
                    except Exception:
                        alpha_vix = alpha_total * 0.25
                else:
                    alpha_vix = alpha_total * 0.25

            # -------------------------------------------------------
            # Momentum / trend overlays
            # -------------------------------------------------------
            alpha_momentum = 0.0
            alpha_rotation = 0.0

            stack = str(r.get("strategy_stack", "")).lower()
            stack_applied = str(r.get("strategy_stack_applied", "")).lower()
            combined_stack = f"{stack} {stack_applied}"

            if "momentum" in combined_stack or "trend" in combined_stack:
                alpha_momentum = alpha_total * 0.35

            # -------------------------------------------------------
            # Rotation / beta drift effects
            # -------------------------------------------------------
            beta_drift = r.get("Beta_Drift", 0.0)
            try:
                beta_drift = float(beta_drift)
            except Exception:
                beta_drift = 0.0

            if beta_drift != 0:
                alpha_rotation = -beta_drift * 0.15

            # -------------------------------------------------------
            # Residual = stock selection
            # -------------------------------------------------------
            alpha_stock_selection = (
                alpha_total
                - alpha_market
                - alpha_vix
                - alpha_momentum
                - alpha_rotation
            )

            rows.append(
                {
                    "wave_id": wave_id,
                    "wave_name": wave_name,
                    "alpha_market": alpha_market,
                    "alpha_vix": alpha_vix,
                    "alpha_momentum": alpha_momentum,
                    "alpha_rotation": alpha_rotation,
                    "alpha_stock_selection": alpha_stock_selection,
                    "alpha_total": alpha_total,
                }
            )

        except Exception:
            # Never allow a single bad row to break attribution
            continue

    if not rows:
        return False, "No valid rows produced for attribution"

    # ---------------------------------------------------------------
    # Write output
    # ---------------------------------------------------------------
    out_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(ATTRIBUTION_PATH), exist_ok=True)
    out_df.to_csv(ATTRIBUTION_PATH, index=False)

    return True, f"Wrote {len(out_df)} rows to {ATTRIBUTION_PATH}"


# -------------------------------------------------------------------
# CLI support
# -------------------------------------------------------------------

if __name__ == "__main__":
    ok, msg = build_alpha_attribution_snapshot()
    print(msg)