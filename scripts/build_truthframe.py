import sys
from pathlib import Path
import json
from datetime import datetime, timezone

# ------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH (CI-safe)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Canonical imports (must exist)
# ------------------------------------------------------------------
from helpers.price_book import get_price_book
from helpers.wave_registry import get_wave_registry

# Return + alpha computation
from helpers.return_pipeline import compute_wave_returns_pipeline
from helpers.wave_performance import compute_portfolio_alpha_ledger


# ------------------------------------------------------------------
# TruthFrame Builder
# ------------------------------------------------------------------
def build_truthframe(lookbacks=(1, 30, 60, 365)) -> dict:
    truthframe = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookbacks": list(lookbacks),
            "status": "INITIALIZING",
            "validated": False,
        },
        "waves": {},
    }

    # --------------------------------------------------------------
    # Load price book
    # --------------------------------------------------------------
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        truthframe["_meta"]["status"] = "PRICE_BOOK_MISSING"
        return truthframe

    truthframe["_meta"]["price_book_rows"] = len(price_book)
    truthframe["_meta"]["price_book_cols"] = len(price_book.columns)

    # --------------------------------------------------------------
    # Load wave registry
    # --------------------------------------------------------------
    registry = get_wave_registry()
    if registry is None or registry.empty:
        truthframe["_meta"]["status"] = "WAVE_REGISTRY_MISSING"
        return truthframe

    wave_ids = registry["wave_id"].tolist()

    # --------------------------------------------------------------
    # STEP 1: Compute raw returns + alpha
    # --------------------------------------------------------------
    try:
        returns_df = compute_wave_returns_pipeline(
            price_book=price_book,
            wave_registry=registry,
            lookbacks=lookbacks,
        )
    except Exception as e:
        truthframe["_meta"]["status"] = f"RETURN_PIPELINE_ERROR: {e}"
        return truthframe

    if returns_df is None or returns_df.empty:
        truthframe["_meta"]["status"] = "NO_RETURN_DATA"
        return truthframe

    # --------------------------------------------------------------
    # STEP 2: Compute alpha attribution + validation
    # --------------------------------------------------------------
    try:
        alpha_ledger, validation = compute_portfolio_alpha_ledger(
            returns_df=returns_df,
            tolerance=0.001,
        )
    except Exception as e:
        truthframe["_meta"]["status"] = f"ALPHA_PIPELINE_ERROR: {e}"
        return truthframe

    # --------------------------------------------------------------
    # STEP 3: Build per-wave TruthFrame blocks
    # --------------------------------------------------------------
    all_valid = True

    for wave_id in wave_ids:
        wave_perf = returns_df[returns_df["wave_id"] == wave_id]
        wave_alpha = alpha_ledger.get(wave_id, {})

        # Performance by horizon
        performance_block = {}
        for lb in lookbacks:
            row = wave_perf[wave_perf["lookback"] == lb]
            if row.empty:
                performance_block[f"{lb}D"] = {"return": None, "alpha": None}
                all_valid = False
            else:
                performance_block[f"{lb}D"] = {
                    "return": float(row.iloc[0]["wave_return"]),
                    "alpha": float(row.iloc[0]["alpha"]),
                }

        # Alpha attribution
        alpha_block = {
            "total": float(wave_alpha.get("total", 0.0)),
            "selection": float(wave_alpha.get("selection", 0.0)),
            "overlay": float(wave_alpha.get("overlay", 0.0)),
            "cash": float(wave_alpha.get("cash", 0.0)),
        }

        wave_valid = validation.get(wave_id, False)
        if not wave_valid:
            all_valid = False

        truthframe["waves"][wave_id] = {
            "alpha": alpha_block,
            "performance": performance_block,
            "health": {
                "status": "OK" if wave_valid else "DEGRADED",
            },
            "learning": {},
        }

    # --------------------------------------------------------------
    # STEP 4: Final TruthFrame status
    # --------------------------------------------------------------
    truthframe["_meta"]["validated"] = all_valid
    truthframe["_meta"]["status"] = "OK" if all_valid else "DEGRADED"
    truthframe["_meta"]["wave_count"] = len(truthframe["waves"])

    return truthframe


# ------------------------------------------------------------------
# CLI Entrypoint (used by GitHub Actions)
# ------------------------------------------------------------------
if __name__ == "__main__":
    tf = build_truthframe()

    output_path = ROOT / "data" / "truthframe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tf, f, indent=2)

    print(f"✅ TruthFrame written to {output_path}")