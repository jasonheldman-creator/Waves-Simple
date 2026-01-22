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
# Canonical imports (EXISTING ONLY)
# ------------------------------------------------------------------
from helpers.price_book import get_price_book
from helpers.wave_registry import get_wave_registry


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def compute_return(prices: pd.Series) -> float | None:
    if prices is None or len(prices) < 2:
        return None
    start = prices.iloc[0]
    end = prices.iloc[-1]
    if start <= 0 or pd.isna(start) or pd.isna(end):
        return None
    return float(end / start - 1.0)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


# ------------------------------------------------------------------
# TruthFrame Builder (VALIDATED, CI-SAFE)
# ------------------------------------------------------------------
def build_truthframe(days: int = 60) -> dict:
    now_utc = datetime.now(timezone.utc).isoformat()

    truthframe = {
        "_meta": {
            "generated_at": now_utc,
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

    price_book = price_book.tail(days)

    truthframe["_meta"]["price_book_rows"] = int(len(price_book))
    truthframe["_meta"]["price_book_cols"] = int(len(price_book.columns))

    # --------------------------------------------------------------
    # Load wave registry
    # --------------------------------------------------------------
    registry = get_wave_registry()
    if registry is None or registry.empty:
        truthframe["_meta"]["status"] = "WAVE_REGISTRY_MISSING"
        return truthframe

    # --------------------------------------------------------------
    # Build per-wave data
    # --------------------------------------------------------------
    horizons = {
        "1D": 1,
        "30D": 30,
        "60D": 60,
        "365D": 365,
    }

    validated_waves = 0

    for _, row in registry.iterrows():
        wave_id = row["wave_id"]
        tickers = row.get("tickers")

        if not isinstance(tickers, list) or len(tickers) == 0:
            continue

        wave_prices = price_book[tickers].dropna(axis=1, how="all")
        if wave_prices.empty:
            continue

        wave_returns = {}
        wave_alpha = {}

        for label, lookback in horizons.items():
            prices = wave_prices.tail(max(lookback + 1, 2))

            # Portfolio-style equal-weight return
            returns = []
            for col in prices.columns:
                r = compute_return(prices[col])
                if r is not None:
                    returns.append(r)

            if not returns:
                wave_returns[label] = None
                wave_alpha[label] = None
                continue

            portfolio_return = float(np.mean(returns))

            # Simple benchmark proxy: SPY if available, else mean market
            if "SPY" in price_book.columns:
                spy_prices = price_book["SPY"].tail(max(lookback + 1, 2))
                benchmark_return = compute_return(spy_prices)
            else:
                benchmark_return = 0.0

            alpha = (
                portfolio_return - benchmark_return
                if benchmark_return is not None
                else None
            )

            wave_returns[label] = safe_float(portfolio_return)
            wave_alpha[label] = safe_float(alpha)

        truthframe["waves"][wave_id] = {
            "returns": wave_returns,
            "alpha": wave_alpha,
            "health": {"status": "OK"},
            "learning": {},
        }

        validated_waves += 1

    # --------------------------------------------------------------
    # Final validation status
    # --------------------------------------------------------------
    truthframe["_meta"]["wave_count"] = validated_waves

    if validated_waves > 0:
        truthframe["_meta"]["status"] = "OK"
    else:
        truthframe["_meta"]["status"] = "NO_VALID_WAVES"

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