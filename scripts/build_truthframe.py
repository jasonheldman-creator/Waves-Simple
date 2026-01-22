import sys
from pathlib import Path
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Ensure repo root is on PYTHONPATH (CI-safe)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Canonical imports ONLY (must exist in CI)
# ------------------------------------------------------------------
from helpers.price_book import get_price_book
from helpers.wave_registry import get_wave_registry


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
HORIZONS = {
    "1D": 1,
    "30D": 30,
    "60D": 60,
    "365D": 365,
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _compound_return(series: pd.Series) -> float | None:
    """
    Compound daily returns: (1+r1)(1+r2)... - 1
    """
    if series is None or series.empty:
        return None

    series = series.dropna()
    if series.empty:
        return None

    try:
        return float((1.0 + series).prod() - 1.0)
    except Exception:
        return None


# ------------------------------------------------------------------
# TruthFrame Builder (CANONICAL)
# ------------------------------------------------------------------
def build_truthframe(days: int = 365) -> dict:
    truthframe = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": days,
            "status": "OK",
        },
        "waves": {},
    }

    # --------------------------------------------------------------
    # Load price book (gatekeeper)
    # --------------------------------------------------------------
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        truthframe["_meta"]["status"] = "PRICE_BOOK_MISSING"
        return truthframe

    price_book = price_book.tail(days)

    truthframe["_meta"]["price_book_rows"] = len(price_book)
    truthframe["_meta"]["price_book_cols"] = len(price_book.columns)

    # --------------------------------------------------------------
    # Load wave registry
    # --------------------------------------------------------------
    registry = get_wave_registry()
    if registry is None or registry.empty:
        truthframe["_meta"]["status"] = "WAVE_REGISTRY_MISSING"
        return truthframe

    # --------------------------------------------------------------
    # Build per-wave TruthFrame
    # --------------------------------------------------------------
    for _, row in registry.iterrows():
        wave_id = row["wave_id"]

        wave_col = f"wave:{wave_id}"
        bench_col = f"benchmark:{wave_id}"

        wave_block = {
            "alpha": {
                "total": None,
                "selection": None,
                "overlay": None,
                "cash": None,
            },
            "performance": {},
            "health": {
                "status": "OK",
            },
            "learning": {},
        }

        # Skip if data missing
        if wave_col not in price_book.columns or bench_col not in price_book.columns:
            truthframe["waves"][wave_id] = wave_block
            continue

        wave_returns = price_book[wave_col]
        bench_returns = price_book[bench_col]

        alpha_total = 0.0
        alpha_valid = False

        for label, lookback in HORIZONS.items():
            wave_slice = wave_returns.tail(lookback)
            bench_slice = bench_returns.tail(lookback)

            wave_ret = _compound_return(wave_slice)
            bench_ret = _compound_return(bench_slice)

            if wave_ret is not None and bench_ret is not None:
                alpha = wave_ret - bench_ret
                alpha_valid = True
            else:
                alpha = None

            wave_block["performance"][label] = {
                "return": _safe_float(wave_ret),
                "benchmark_return": _safe_float(bench_ret),
                "alpha": _safe_float(alpha),
            }

            if alpha is not None:
                alpha_total += alpha

        wave_block["alpha"]["total"] = _safe_float(alpha_total) if alpha_valid else None

        truthframe["waves"][wave_id] = wave_block

    truthframe["_meta"]["wave_count"] = len(truthframe["waves"])
    return truthframe


# ------------------------------------------------------------------
# CLI Entrypoint (GitHub Actions)
# ------------------------------------------------------------------
if __name__ == "__main__":
    tf = build_truthframe()

    output_path = ROOT / "data" / "truthframe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tf, f, indent=2)

    print(f"✅ TruthFrame written to {output_path}")