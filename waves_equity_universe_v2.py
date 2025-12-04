# waves_equity_universe_v2.py
"""
Minimal wave engine helpers for the Equity Waves Console.
More advanced NAV / alpha logic can be plugged in later.
"""

import io
import requests
import pandas as pd


def load_holdings_from_csv(url: str) -> pd.DataFrame:
    """
    Load holdings from a CSV URL (Google Sheets published link, etc.).
    Assumes the first row is headers.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def compute_wave_nav(wave_config, holdings: pd.DataFrame) -> dict:
    """
    Placeholder NAV/return/alpha calculator.

    For the Franklin demo, we don't need real-time pricing yet – we just need
    a consistent, non-crashing snapshot. This returns a flat NAV and N/A for
    returns/alpha. You can upgrade this later to use live prices.
    """
    notional = getattr(wave_config, "default_notional", 100_000.0)

    return {
        "nav": float(notional),
        "wave_return": float("nan"),
        "benchmark_return": float("nan"),
        "alpha": float("nan"),
    }