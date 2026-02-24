# helpers/wave_holdings_loader.py
# Load wave holdings (source of truth) from wave_weights.csv.

from __future__ import annotations

from pathlib import Path

import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent.parent


def load_wave_holdings(wave_name: str) -> list[str]:
    """Return unique non-null tickers for the given wave from data/wave_weights.csv.

    Parameters
    ----------
    wave_name:
        Name of the wave as it appears in the wave_id / wave_name column.

    Returns
    -------
    list[str]
        Sorted list of unique ticker symbols.  Empty list when the wave is not
        found or the CSV is unavailable.
    """
    weights_path = _BASE_DIR / "data" / "wave_weights.csv"
    fallback_path = _BASE_DIR / "wave_weights.csv"

    path = (
        weights_path
        if weights_path.exists()
        else (fallback_path if fallback_path.exists() else None)
    )
    if path is None:
        return []

    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        wave_col = next(
            (c for c in ["wave_id", "wave_name", "wave"] if c in df.columns), None
        )
        if wave_col is None:
            return []

        ticker_col = next(
            (c for c in ["ticker", "symbol"] if c in df.columns), None
        )
        if ticker_col is None:
            return []

        mask = df[wave_col].astype(str).str.strip() == str(wave_name).strip()
        subset = df[mask][ticker_col].dropna().astype(str).str.strip()
        tickers = [t for t in subset.unique().tolist() if t]
        return sorted(tickers)
    except Exception:
        return []
