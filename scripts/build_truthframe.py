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
# Canonical horizons
# ------------------------------------------------------------------
HORIZONS = {
    "1D": 1,
    "30D": 30,
    "60D": 60,
    "365D": 365,
}


# ------------------------------------------------------------------
# Utility: safe return calculation
# ------------------------------------------------------------------
def compute_return(series: pd.Series) -> float | None:
    """
    Computes fractional return: (last / first - 1)
    Returns None if insufficient data.
    """
    if series is None or len(series) < 2:
        return None

    start = series.iloc[0]
    end = series.iloc[-1]

    if start == 0 or pd.isna(start) or pd.isna(end):
        return None

    return float(end / start - 1)


# ------------------------------------------------------------------
# TruthFrame Builder (canonical, CI-safe)
# ------------------------------------------------------------------
def build_truthframe(days: int = 365) -> dict:
    truthframe = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": days,
            "status": "INIT",
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

    # Use last N days only
    price_book = price_book.tail(days + 2)

    truthframe["_meta"]["price_book_rows"] = int(len(price_book))
    truthframe["_meta"]["price_book_cols"] = int(len(price_book.columns))

    # --------------------------------------------------------------
    # Load wave registry
    # Expected columns:
    # - wave_id
    # - tickers (comma-separated)
    # - benchmark_ticker
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

        tickers = [
            t.strip()
            for t in str(row.get("tickers", "")).split(",")
            if t.strip() in price_book.columns
        ]

        benchmark = row.get("benchmark_ticker")
        benchmark_ok = benchmark in price_book.columns

        wave_block = {
            "alpha": {
                "total": 0.0,
                "selection": 0.0,
                "overlay": 0.0,
                "cash": 0.0,
            },
            "performance": {},
            "health": {
                "status": "OK",
            },
            "learning": {},
        }

        # If no usable tickers, mark unhealthy
        if not tickers:
            wave_block["health"]["status"] = "NO_TICKERS"
            truthframe["waves"][wave_id] = wave_block
            continue

        # ----------------------------------------------------------
        # Compute returns by horizon
        # ----------------------------------------------------------
        alpha_sum = 0.0
        alpha_count = 0

        for label, d in HORIZONS.items():
            if len(price_book) < d + 1:
                continue

            # Wave return = mean of constituent returns
            wave_returns = []
            for t in tickers:
                r = compute_return(price_book[t].tail(d + 1))
                if r is not None:
                    wave_returns.append(r)

            wave_ret = float(np.mean(wave_returns)) if wave_returns else None

            # Benchmark return
            bench_ret = None
            if benchmark_ok:
                bench_ret = compute_return(price_book[benchmark].tail(d + 1))

            # Alpha
            alpha = None
            if wave_ret is not None and bench_ret is not None:
                alpha = wave_ret - bench_ret
                alpha_sum += alpha
                alpha_count += 1

            wave_block["performance"][label] = {
                "return": wave_ret,
                "benchmark_return": bench_ret,
                "alpha": alpha,
            }

        # ----------------------------------------------------------
        # Aggregate alpha
        # ----------------------------------------------------------
        if alpha_count > 0:
            wave_block["alpha"]["total"] = float(alpha_sum / alpha_count)

        truthframe["waves"][wave_id] = wave_block

    truthframe["_meta"]["wave_count"] = len(truthframe["waves"])
    truthframe["_meta"]["status"] = "OK"

    return truthframe


# ------------------------------------------------------------------
# CLI entrypoint (GitHub Actions)
# ------------------------------------------------------------------
if __name__ == "__main__":
    tf = build_truthframe()

    output_path = ROOT / "data" / "truthframe.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tf, f, indent=2)

    print(f"✅ TruthFrame written to {output_path}")