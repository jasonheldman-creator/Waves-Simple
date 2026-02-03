"""
benchmark_returns.py — Canonical Benchmark Return Engine

Purpose:
• Construct benchmark return series per Wave_ID
• Support static and future dynamic benchmarks
• Serve as the ONLY producer of Benchmark_Return fields

Design Guarantees:
• Stateless
• Snapshot-driven
• Import-safe
• No Streamlit dependency
• No network access
• Deterministic outputs
"""

from typing import Dict, List
import pandas as pd
import numpy as np


# -----------------------------------------------------------------------------
# Core benchmark return computation
# -----------------------------------------------------------------------------

def compute_benchmark_returns(
    price_book: pd.DataFrame,
    benchmark_config: dict,
) -> pd.DataFrame:
    """
    Compute benchmark returns per Wave_ID.

    Inputs:
        price_book:
            DataFrame indexed by Date, columns = tickers (prices)
        benchmark_config:
            Parsed equity_benchmarks.json

    Output:
        DataFrame with columns:
            • Date
            • Wave_ID
            • Benchmark_Return
            • Dynamic_Benchmark_Return
    """

    if price_book is None or price_book.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Wave_ID",
                "Benchmark_Return",
                "Dynamic_Benchmark_Return",
            ]
        )

    if "benchmarks" not in benchmark_config:
        raise ValueError("benchmark_config missing 'benchmarks' key")

    # Ensure datetime index
    prices = price_book.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Compute daily returns once
    returns = prices.pct_change().fillna(0.0)

    records: List[pd.DataFrame] = []

    for wave_id, spec in benchmark_config["benchmarks"].items():
        components = spec.get("components", [])

        if not components:
            continue

        tickers = [c["ticker"] for c in components]
        weights = np.array([c["weight"] for c in components], dtype=float)

        # Normalize defensively
        if weights.sum() != 0:
            weights = weights / weights.sum()

        # Ensure all tickers exist
        for t in tickers:
            if t not in returns.columns:
                returns[t] = 0.0

        component_returns = returns[tickers]

        # Weighted benchmark return
        benchmark_return = component_returns.dot(weights)

        df = pd.DataFrame({
            "Date": benchmark_return.index,
            "Wave_ID": wave_id,
            "Benchmark_Return": benchmark_return.values,
            # Phase 1B: dynamic == static (future regime logic plugs here)
            "Dynamic_Benchmark_Return": benchmark_return.values,
        })

        records.append(df)

    if not records:
        return pd.DataFrame(
            columns=[
                "Date",
                "Wave_ID",
                "Benchmark_Return",
                "Dynamic_Benchmark_Return",
            ]
        )

    out = pd.concat(records, ignore_index=True)

    return out