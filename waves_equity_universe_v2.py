"""
waves_equity_universe_v2.py

Simple equity-wave engine for the WAVES Intelligence™ – Portfolio Wave Console.

- Loads a single master universe CSV: Master_Stock_Sheet.csv
- Standardizes columns: ticker, name, weight, sector, market_value, price
- Defines 10 equity Waves, all carved from the same master universe
- Provides helpers for the Streamlit app to load holdings + stats per Wave
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
#  Master universe location
# ---------------------------------------------------------------------------

# Assumes the CSV is in the root of the repo next to app.py.
# If you rename/move it, update this path.
MASTER_CSV_PATH = "Master_Stock_Sheet.csv"


# ---------------------------------------------------------------------------
#  Wave config
# ---------------------------------------------------------------------------

@dataclass
class WaveConfig:
    wave_id: str
    name: str
    benchmark: str
    style: str
    wave_type: str  # e.g. "AI-Managed Wave"
    description: str


# 10 equity Waves – all carved out of the same universe
WAVES_CONFIG: List[WaveConfig] = [
    WaveConfig(
        wave_id="SPX",
        name="S&P 500 Core Equity Wave",
        benchmark="SPY",
        style="Core – Large Cap",
        wave_type="AI-Managed Wave",
        description="Core US large-cap leaders, approximated by the top 500 names by market value.",
    ),
    WaveConfig(
        wave_id="GROW",
        name="US Growth & Innovation Wave",
        benchmark="QQQ",
        style="Growth – Large Cap",
        wave_type="AI-Managed Wave",
        description="US innovation leaders, tilted to growth sectors using the top slice of the universe.",
    ),
    WaveConfig(
        wave_id="MID",
        name="US Mid Cap Core Wave",
        benchmark="IJH",
        style="Core – Mid Cap",
        wave_type="AI-Managed Wave",
        description="Mid-cap core allocation carved from the middle of the market-cap spectrum.",
    ),
    WaveConfig(
        wave_id="SMGX",
        name="US Small & Mid Cap Growth Wave",
        benchmark="IWO",
        style="Growth – Small/Mid",
        wave_type="AI-Managed Wave",
        description="Smaller, higher-growth names from the lower half of the market-cap spectrum.",
    ),
    WaveConfig(
        wave_id="VAL",
        name="US Value Tilt Wave",
        benchmark="IWD",
        style="Value – All Cap",
        wave_type="AI-Managed Wave",
        description="Value-tilted slice of the universe; falls back to broad market if no value data is present.",
    ),
    WaveConfig(
        wave_id="FIN",
        name="US Financials Leadership Wave",
        benchmark="XLF",
        style="Sector – Financials",
        wave_type="AI-Managed Wave",
        description="Financials-centric allocation; falls back to top financial-adjacent names if sectors are missing.",
    ),
    WaveConfig(
        wave_id="HC",
        name="US Health Care Leadership Wave",
        benchmark="XLV",
        style="Sector – Health Care",
        wave_type="AI-Managed Wave",
        description="Health-care-centric allocation with resilience tilt.",
    ),
    WaveConfig(
        wave_id="TECH",
        name="Future Tech & AI Leaders Wave",
        benchmark="XLK",
        style="Thematic – Technology / AI",
        wave_type="AI-Managed Wave",
        description="Tech & AI-tilted selection from the universe; uses sector where available, otherwise top tech-adjacent names.",
    ),
    WaveConfig(
        wave_id="CORE",
        name="US Total Market Core Wave",
        benchmark="VTI",
        style="Core – All Cap",
        wave_type="AI-Managed Wave",
        description="Broad US equity market representation using the full master universe.",
    ),
    WaveConfig(
        wave_id="EQX",
        name="US Quality Equity Wave",
        benchmark="QUAL",
        style="Quality – All Cap",
        wave_type="AI-Managed Wave",
        description="Quality-tilted equity slice; approximated using size and equal-weighting when fundamentals are unavailable.",
    ),
]


def get_wave_config(wave_id: str) -> WaveConfig:
    for cfg in WAVES_CONFIG:
        if cfg.wave_id == wave_id:
            return cfg
    raise KeyError(f"Unknown wave_id: {wave_id}")


# ---------------------------------------------------------------------------
#  Universe loader + normalization helpers
# ---------------------------------------------------------------------------

def load_master_universe(path: str = MASTER_CSV_PATH) -> pd.DataFrame:
    """
    Load the master universe CSV and standardize column names.

    Expected headers (case-insensitive):
      - 'Ticker'
      - 'Company'
      - 'Weight'          (optional – will be recomputed if missing/zero)
      - 'Sector'          (optional)
      - 'Market Value'    (optional – used if present)
      - 'Price'           (optional – not required for console)

    Anything extra is ignored.
    """
    df = pd.read_csv(path)

    # Normalize column names
    rename_map = {}
    for col in df.columns:
        c = col.strip().lower()
        if c == "ticker":
            rename_map[col] = "ticker"
        elif c in ("company", "name"):
            rename_map[col] = "name"
        elif c.startswith("weight"):
            rename_map[col] = "weight"
        elif c.startswith("sector"):
            rename_map[col] = "sector"
        elif c.startswith("market value"):
            rename_map[col] = "market_value"
        elif c.startswith("price"):
            rename_map[col] = "price"

    df = df.rename(columns=rename_map)

    # Enforce minimal required columns
    if "ticker" not in df.columns:
        raise ValueError("Master_Stock_Sheet CSV must have a 'Ticker' column.")
    if "name" not in df.columns:
        df["name"] = df["ticker"]

    # Ensure market_value numeric (optional)
    if "market_value" in df.columns:
        df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
    else:
        df["market_value"] = pd.NA

    # Ensure weight numeric (optional; will recompute)
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    else:
        df["weight"] = pd.NA

    # Sector is optional
    if "sector" not in df.columns:
        df["sector"] = pd.NA

    # Basic cleaning
    df = df.dropna(subset=["ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["name"] = df["name"].astype(str).str.strip()

    return df


def normalize_weights(df: pd.DataFrame, weight_col: str = "weight") -> pd.DataFrame:
    """
    Ensure weights are positive and sum to 1.0.

    If existing weights are missing or invalid, derive from market_value.
    """
    out = df.copy()

    # If weights are mostly missing or zero, try to infer from market_value
    w = pd.to_numeric(out.get(weight_col), errors="coerce")
    if w.isna().all() or (w.fillna(0) <= 0).all():
        if "market_value" in out.columns and not out["market_value"].isna().all():
            mv = out["market_value"].fillna(0).clip(lower=0)
            total_mv = mv.sum()
            if total_mv > 0:
                out[weight_col] = mv / total_mv
            else:
                out[weight_col] = 1.0 / len(out)
        else:
            out[weight_col] = 1.0 / len(out)
    else:
        w = w.fillna(0).clip(lower=0)
        total_w = w.sum()
        if total_w > 0:
            out[weight_col] = w / total_w
        else:
            out[weight_col] = 1.0 / len(out)

    return out


# ---------------------------------------------------------------------------
#  Wave carving logic
# ---------------------------------------------------------------------------

def _sector_available(df: pd.DataFrame) -> bool:
    if "sector" not in df.columns:
        return False
    non_empty = df["sector"].astype(str).str.strip().replace({"": pd.NA}).dropna()
    return len(non_empty) > 0


def carve_wave_holdings(master: pd.DataFrame, wave_id: str) -> pd.DataFrame:
    """
    Take the full master universe and carve out holdings for a specific Wave.

    This is intentionally simple + robust: no external data, just your CSV.
    """
    df = master.copy()

    # Rank by market value (fallback: equal)
    if "market_value" in df.columns and not df["market_value"].isna().all():
        df = df.sort_values("market_value", ascending=False)
    else:
        df = df.sort_values("ticker")

    n = len(df)
    top_500 = min(500, n)
    top_300 = min(300, n)
    mid_start = max(0, min(top_500, n // 3))
    mid_end = min(n, mid_start + 400)  # 400-name mid-cap band
    lower_half_start = n // 2

    sector_ok = _sector_available(df)

    if wave_id == "SPX":
        # Top 500 by market cap
        subset = df.head(top_500)

    elif wave_id == "GROW":
        if sector_ok:
            growth_sectors = ["Information Technology", "Technology", "Communication Services",
                              "Consumer Discretionary"]
            mask = df["sector"].astype(str).isin(growth_sectors)
            subset = df[mask].head(top_300)
            if subset.empty:
                subset = df.head(top_300)
        else:
            subset = df.head(top_300)

    elif wave_id == "MID":
        subset = df.iloc[mid_start:mid_end]

    elif wave_id == "SMGX":
        # Smaller names, growth tilt
        subset = df.iloc[lower_half_start:].head(600)

    elif wave_id == "VAL":
        if sector_ok:
            val_sectors = ["Financials", "Energy", "Utilities", "Real Estate", "Consumer Staples"]
            mask = df["sector"].astype(str).isin(val_sectors)
            subset = df[mask]
            if subset.empty:
                subset = df.iloc[top_500:top_500 + 600]
        else:
            subset = df.iloc[top_500:top_500 + 600]

    elif wave_id == "FIN":
        if sector_ok:
            mask = df["sector"].astype(str).str.contains("Financial", case=False, na=False)
            subset = df[mask]
            if subset.empty:
                subset = df.iloc[top_500:top_500 + 200]
        else:
            subset = df.iloc[top_500:top_500 + 200]

    elif wave_id == "HC":
        if sector_ok:
            mask = df["sector"].astype(str).str.contains("Health", case=False, na=False)
            subset = df[mask]
            if subset.empty:
                subset = df.iloc[top_500:top_500 + 200]
        else:
            subset = df.iloc[top_500:top_500 + 200]

    elif wave_id == "TECH":
        if sector_ok:
            mask = df["sector"].astype(str).str.contains("Tech", case=False, na=False)
            subset = df[mask].head(200)
            if subset.empty:
                subset = df.head(200)
        else:
            subset = df.head(200)

    elif wave_id == "CORE":
        subset = df  # full universe

    elif wave_id == "EQX":
        # Quality tilt approximation: take mid-ish sized names, equal weight
        mid_q_start = n // 4
        mid_q_end = min(n, mid_q_start + 400)
        subset = df.iloc[mid_q_start:mid_q_end]

    else:
        raise KeyError(f"Unknown wave_id: {wave_id}")

    subset = normalize_weights(subset)
    return subset.reset_index(drop=True)


# ---------------------------------------------------------------------------
#  Wave stats helpers
# ---------------------------------------------------------------------------

def compute_wave_stats(holdings: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Compute simple snapshot stats for the wave from holdings.
    """
    if holdings.empty:
        return {
            "num_holdings": 0,
            "largest_weight": None,
            "top10_weight": None,
        }

    w = pd.to_numeric(holdings["weight"], errors="coerce").fillna(0)
    w_sorted = w.sort_values(ascending=False)
    num_holdings = (w > 0).sum()
    largest_weight = float(w_sorted.iloc[0]) if num_holdings > 0 else None
    top10_weight = float(w_sorted.head(10).sum())

    return {
        "num_holdings": int(num_holdings),
        "largest_weight": largest_weight,
        "top10_weight": top10_weight,
    }


def load_wave_holdings_and_stats(
    wave_id: str,
    master_path: str = MASTER_CSV_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    """
    Convenience wrapper used by the Streamlit app.
    """
    master = load_master_universe(master_path)
    holdings = carve_wave_holdings(master, wave_id)
    stats = compute_wave_stats(holdings)
    return holdings, stats