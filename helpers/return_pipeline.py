"""
Return Pipeline (Batch)

Computes standardized wave returns, benchmark returns,
and alpha across multiple horizons for ALL waves.

This module is the canonical bridge between:
- price_book
- wave_registry
- TruthFrame

Output schema is intentionally simple and explicit.
"""

from typing import List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_wave_returns_pipeline(
    *,
    price_book: pd.DataFrame,
    wave_registry: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    """
    Compute returns + alpha for all waves across horizons.

    Parameters
    ----------
    price_book : pd.DataFrame
        Wide price dataframe indexed by date, columns = tickers
    wave_registry : pd.DataFrame
        Wave definitions (must include wave_id, ticker_normalized, benchmark_recipe)
    horizons : list[int]
        Lookback windows in trading days (e.g. [1, 30, 60, 365])

    Returns
    -------
    pd.DataFrame with columns:
        wave_id
        horizon
        wave_return
        benchmark_return
        alpha
    """

    if price_book is None or price_book.empty:
        raise ValueError("price_book is empty")

    if wave_registry is None or wave_registry.empty:
        raise ValueError("wave_registry is empty")

    results = []

    price_book = price_book.sort_index()

    for _, wave in wave_registry.iterrows():
        wave_id = wave.get("wave_id")
        ticker_str = wave.get("ticker_normalized", "")

        if not wave_id or not ticker_str:
            continue

        wave_tickers = _parse_tickers(ticker_str)
        wave_tickers = [t for t in wave_tickers if t in price_book.columns]

        if not wave_tickers:
            continue

        # --- Wave daily returns ---
        wave_prices = price_book[wave_tickers]
        wave_daily_returns = wave_prices.pct_change().mean(axis=1).dropna()

        if wave_daily_returns.empty:
            continue

        # --- Benchmark daily returns ---
        benchmark_recipe = wave.get("benchmark_recipe") or {}
        benchmark_tickers = [
            t for t in benchmark_recipe.keys() if t in price_book.columns
        ]

        if benchmark_tickers:
            weights = np.array(
                [benchmark_recipe[t] for t in benchmark_tickers], dtype=float
            )
            if weights.sum() != 0:
                weights = weights / weights.sum()

            bench_prices = price_book[benchmark_tickers]
            bench_returns = bench_prices.pct_change()
            benchmark_daily_returns = (bench_returns * weights).sum(axis=1).reindex(
                wave_daily_returns.index
            )
        else:
            benchmark_daily_returns = pd.Series(
                0.0, index=wave_daily_returns.index
            )

        # --- Horizon aggregation (tolerant, non-dropping) ---
        emitted_any = False

        for h in horizons:
            if len(wave_daily_returns) >= h:
                w_ret = (1 + wave_daily_returns.tail(h)).prod() - 1
                b_ret = (1 + benchmark_daily_returns.tail(h)).prod() - 1
                emitted_any = True
            else:
                # Not enough history — emit explicit NaNs (truthful + non-breaking)
                w_ret = np.nan
                b_ret = np.nan

            results.append(
                {
                    "wave_id": wave_id,
                    "horizon": h,
                    "wave_return": float(w_ret) if pd.notna(w_ret) else np.nan,
                    "benchmark_return": float(b_ret) if pd.notna(b_ret) else np.nan,
                    "alpha": (
                        float(w_ret - b_ret)
                        if pd.notna(w_ret) and pd.notna(b_ret)
                        else np.nan
                    ),
                }
            )

        # Safety net: if for some reason nothing emitted, force a 1D NaN row
        if not emitted_any:
            results.append(
                {
                    "wave_id": wave_id,
                    "horizon": min(horizons),
                    "wave_return": np.nan,
                    "benchmark_return": np.nan,
                    "alpha": np.nan,
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _parse_tickers(ticker_string: str) -> list:
    if not ticker_string:
        return []
    return [t.strip() for t in ticker_string.split(",") if t.strip()]