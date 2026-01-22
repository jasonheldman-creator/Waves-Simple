"""
Return Pipeline (Canonical)

This module computes standardized wave returns, benchmark returns,
and alpha across multiple horizons for downstream TruthFrame ingestion.

This version is WIRED for TruthFrame aggregation and supports
batch computation across all waves.
"""

import logging
from typing import List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API (TruthFrame expects THIS signature)
# ------------------------------------------------------------------
def compute_wave_returns_pipeline(
    price_book: pd.DataFrame,
    wave_registry: pd.DataFrame,
    horizons: List[int],
) -> pd.DataFrame:
    """
    Compute wave + benchmark returns and alpha across multiple horizons.

    Args:
        price_book: Canonical price dataframe (index=date, cols=tickers)
        wave_registry: Wave registry dataframe (must include wave_id,
                       ticker_normalized, benchmark_recipe)
        horizons: List of lookback horizons in days (e.g. [1, 30, 60, 365])

    Returns:
        DataFrame with columns:
        - wave_id
        - horizon
        - wave_return
        - benchmark_return
        - alpha
    """

    if price_book is None or price_book.empty:
        raise ValueError("price_book is empty")

    if wave_registry is None or wave_registry.empty:
        raise ValueError("wave_registry is empty")

    results = []

    # Ensure sorted by date
    price_book = price_book.sort_index()

    for _, wave_row in wave_registry.iterrows():
        wave_id = wave_row.get("wave_id")

        try:
            wave_tickers = _parse_ticker_list(
                wave_row.get("ticker_normalized", "")
            )

            benchmark_recipe = wave_row.get("benchmark_recipe", {}) or {}
            benchmark_tickers = list(benchmark_recipe.keys())

            if not wave_tickers:
                continue

            all_tickers = list(set(wave_tickers + benchmark_tickers))
            available = [t for t in all_tickers if t in price_book.columns]

            if not available:
                continue

            price_slice = price_book[available].dropna(how="all")
            if price_slice.empty:
                continue

            for horizon in horizons:
                if len(price_slice) < horizon + 1:
                    continue

                window = price_slice.iloc[-(horizon + 1):]

                wave_return = _compute_portfolio_return(
                    window,
                    wave_tickers,
                    equal_weight=True,
                )

                benchmark_return = (
                    _compute_portfolio_return(
                        window,
                        benchmark_tickers,
                        weights=[
                            benchmark_recipe.get(t, 0.0)
                            for t in benchmark_tickers
                        ],
                    )
                    if benchmark_tickers
                    else np.nan
                )

                alpha = (
                    wave_return - benchmark_return
                    if not pd.isna(benchmark_return)
                    else np.nan
                )

                results.append(
                    {
                        "wave_id": wave_id,
                        "horizon": horizon,
                        "wave_return": float(wave_return),
                        "benchmark_return": float(benchmark_return)
                        if not pd.isna(benchmark_return)
                        else np.nan,
                        "alpha": float(alpha)
                        if not pd.isna(alpha)
                        else np.nan,
                    }
                )

        except Exception as e:
            logger.error(
                f"Return pipeline failed for wave {wave_id}: {e}",
                exc_info=True,
            )
            continue

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _parse_ticker_list(ticker_string: str) -> List[str]:
    if not ticker_string or pd.isna(ticker_string):
        return []
    return [t.strip() for t in ticker_string.split(",") if t.strip()]


def _compute_portfolio_return(
    price_df: pd.DataFrame,
    tickers: List[str],
    weights: List[float] = None,
    equal_weight: bool = False,
) -> float:
    available = [t for t in tickers if t in price_df.columns]
    if not available:
        return np.nan

    prices = price_df[available]

    returns = prices.pct_change(fill_method=None).iloc[1:]

    if equal_weight or weights is None:
        w = np.ones(len(available)) / len(available)
    else:
        raw = []
        for t in available:
            idx = tickers.index(t)
            raw.append(weights[idx] if idx < len(weights) else 0.0)
        s = sum(raw)
        w = np.array(raw) / s if s > 0 else np.ones(len(raw)) / len(raw)

    portfolio_returns = returns.mul(w, axis=1).sum(axis=1)

    cumulative = (1 + portfolio_returns).prod() - 1
    return cumulative