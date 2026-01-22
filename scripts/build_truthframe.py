import sys
from pathlib import Path
import json
from datetime import datetime, timezone
import pandas as pd

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
from helpers.benchmarks import get_wave_benchmark


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def compute_return(series: pd.Series, days: int):
    if len(series) < days + 1:
        return None
    start = series.iloc[-(days + 1)]
    end = series.iloc[-1]
    if start == 0 or pd.isna(start) or pd.isna(end):
        return None
    return (end / start) - 1.0


# ------------------------------------------------------------------
# TruthFrame Builder
# ------------------------------------------------------------------
def build_truthframe(days: int = 365) -> dict:
    truthframe = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": days,
            "status": "DEGRADED",
            "validated_performance": False,
        },
        "waves": {},
    }

    # ------------------------------------------------------------------
    # Load price book (gatekeeper)
    # ------------------------------------------------------------------
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        truthframe["_meta"]["status"] = "PRICE_BOOK_MISSING"
        return truthframe

    price_book = price_book.tail(days)
    truthframe["_meta"]["price_book_rows"] = len(price_book)
    truthframe["_meta"]["price_book_cols"] = len(price_book.columns)

    # ------------------------------------------------------------------
    # Load wave registry
    # ------------------------------------------------------------------
    registry = get_wave_registry()
    if registry is None or registry.empty:
        truthframe["_meta"]["status"] = "WAVE_REGISTRY_MISSING"
        return truthframe

    validated_waves = 0

    # ------------------------------------------------------------------
    # Build per-wave TruthFrame
    # ------------------------------------------------------------------
    for _, row in registry.iterrows():
        wave_id = row["wave_id"]

        # ---- Normalize tickers (THIS FIXES DEGRADED MODE)
        raw_tickers = row.get("tickers", [])
        if isinstance(raw_tickers, str):
            tickers = [t.strip() for t in raw_tickers.split(",") if t.strip()]
        elif isinstance(raw_tickers, list):
            tickers = raw_tickers
        else:
            tickers = []

        if not tickers:
            continue

        available = [t for t in tickers if t in price_book.columns]
        if not available:
            continue

        wave_prices = price_book[available].mean(axis=1)

        # ---- Benchmark
        benchmark_ticker = get_wave_benchmark(wave_id)
        benchmark_prices = (
            price_book[benchmark_ticker]
            if benchmark_ticker in price_book.columns
            else None
        )

        # ---- Returns
        returns = {
            "1D": compute_return(wave_prices, 1),
            "30D": compute_return(wave_prices, 30),
            "60D": compute_return(wave_prices, 60),
            "365D": compute_return(wave_prices, 365),
        }

        # ---- Alpha
        alpha = {}
        if benchmark_prices is not None:
            alpha = {
                k: (
                    returns[k]
                    - compute_return(benchmark_prices, int(k.replace("D", "")))
                    if returns[k] is not None
                    else None
                )
                for k in returns
            }
        else:
            alpha = {k: None for k in returns}

        # ---- Validation
        if any(v is not None for v in returns.values()):
            validated_waves += 1

        truthframe["waves"][wave_id] = {
            "performance": returns,
            "alpha": {
                "total": alpha.get("365D"),
                "selection": alpha.get("30D"),
                "overlay": alpha.get("1D"),
                "cash": 0.0,
            },
            "health": {
                "status": "OK" if any(v is not None for v in returns.values()) else "UNKNOWN"
            },
            "learning": {},
        }

    # ------------------------------------------------------------------
    # Final system validation
    # ------------------------------------------------------------------
    truthframe["_meta"]["wave_count"] = len(truthframe["waves"])

    if validated_waves > 0:
        truthframe["_meta"]["status"] = "OK"
        truthframe["_meta"]["validated_performance"] = True
    else:
        truthframe["_meta"]["status"] = "DEGRADED"
        truthframe["_meta"]["validated_performance"] = False

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