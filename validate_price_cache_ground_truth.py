#!/usr/bin/env python3
"""
Ground-truth validation of the price cache contents.

Usage:
  python validate_price_cache_ground_truth.py

What it checks:
- Cache file exists & is readable
- Shape, date range, ticker coverage
- Required anchors (SPY/QQQ/IWM), VIX proxies, T-bill proxies
- Cross-check against tickers referenced by wave registry (if available)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Set, Tuple, Optional

import pandas as pd


DEFAULT_CACHE_CANDIDATES = [
    "data/cache/prices_cache.parquet",  # Canonical path (first priority)
    "data/cache/prices_cache_v2.parquet",  # Legacy v2 path (fallback for compatibility)
]


ANCHORS_ALL = ["SPY", "QQQ", "IWM"]
VIX_ANY = ["^VIX", "VIXY", "VXX"]
TBILL_ANY = ["BIL", "SHY"]


def _pick_cache_path() -> Path:
    for p in DEFAULT_CACHE_CANDIDATES:
        pp = Path(p)
        if pp.exists() and pp.is_file():
            return pp
    # If none exist, return first candidate for error messaging
    return Path(DEFAULT_CACHE_CANDIDATES[0])


def _load_parquet(cache_path: Path) -> pd.DataFrame:
    # Force parquet engine selection; pyarrow preferred
    try:
        df = pd.read_parquet(cache_path)
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet at {cache_path}. "
            f"This usually means pyarrow isn't installed or the file is corrupt.\n"
            f"Error: {e}"
        ) from e


def _infer_layout(df: pd.DataFrame) -> Tuple[str, pd.DatetimeIndex, Set[str]]:
    """
    Detect whether df is:
      A) wide: index = dates, columns = tickers
      B) long: columns include ['date','ticker','close'] or similar
    Returns: (layout, dates_index, tickers_set)
    """
    # Try wide format first: datetime-like index and many columns
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index
        tickers = set(map(str, df.columns))
        return "wide", dates, tickers

    # Try long format
    cols_lower = {c.lower(): c for c in df.columns}
    date_col = cols_lower.get("date") or cols_lower.get("datetime") or cols_lower.get("timestamp")
    ticker_col = cols_lower.get("ticker") or cols_lower.get("symbol")

    if date_col and ticker_col:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        # use unique on non-null tickers
        tickers = set(map(str, df[ticker_col].dropna().unique().tolist()))
        return "long", pd.DatetimeIndex(dates.unique()).sort_values(), tickers

    # Fallback: can't infer
    raise RuntimeError(
        "Could not infer cache layout. Expected either:\n"
        "- wide parquet with DatetimeIndex and ticker columns, OR\n"
        "- long parquet with date + ticker columns.\n"
        f"Index type: {type(df.index)}; columns: {list(df.columns)[:30]}"
    )


def _load_wave_registry_tickers() -> Optional[Set[str]]:
    """
    Best-effort pull tickers from common locations.
    If it can't find anything, returns None (still validates cache standalone).
    """
    candidates = [
        Path("data/wave_registry.csv"),
        Path("data/waves/wave_registry.csv"),
        Path("data/waves/registry.csv"),
        Path("data/waves/waves.csv"),
    ]
    reg_path = next((p for p in candidates if p.exists() and p.is_file()), None)
    if not reg_path:
        return None

    try:
        reg = pd.read_csv(reg_path)
    except Exception:
        return None

    # Try common column names
    possible_cols = []
    for c in reg.columns:
        cl = c.lower()
        if "ticker" in cl or "symbol" in cl:
            possible_cols.append(c)

    if not possible_cols:
        return None

    tickers: Set[str] = set()
    for c in possible_cols:
        series = reg[c].dropna().astype(str)
        for v in series.tolist():
            # split comma/space separated lists
            parts = [p.strip() for p in v.replace(";", ",").replace("|", ",").split(",")]
            for p in parts:
                if not p or p.lower() in ("nan", "none"):
                    continue
                tickers.add(p.upper())

    # Remove obvious non-ticker noise
    tickers.discard("")
    return tickers if tickers else None


def _report_presence(label: str, tickers_present: Set[str], required_all: Iterable[str], required_any: Iterable[str]) -> None:
    required_all = [t.upper() for t in required_all]
    required_any = [t.upper() for t in required_any]

    missing_all = [t for t in required_all if t not in tickers_present]
    any_ok = any(t in tickers_present for t in required_any) if required_any else True
    any_found = [t for t in required_any if t in tickers_present]

    print(f"\n[{label}]")
    if required_all:
        print(f"  ALL required: {required_all}")
        print(f"  Missing ALL:  {missing_all if missing_all else 'None ✅'}")
    if required_any:
        print(f"  ANY required: {required_any}")
        print(f"  Found ANY:    {any_found if any_found else 'None ❌'}")
        print(f"  ANY satisfied: {'YES ✅' if any_ok else 'NO ❌'}")


def main() -> int:
    cache_path = _pick_cache_path()

    print("\n=== PRICE CACHE GROUND-TRUTH VALIDATION ===")
    print(f"Cache path candidate: {cache_path}")

    if not cache_path.exists():
        print(f"❌ Cache file not found at {cache_path}")
        print("Checked candidates:")
        for p in DEFAULT_CACHE_CANDIDATES:
            print(f"  - {p}")
        return 2

    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Cache exists ✅ | size: {size_mb:.2f} MB")

    try:
        df = _load_parquet(cache_path)
    except Exception as e:
        print(f"❌ Read failed: {e}")
        return 3

    print(f"Loaded parquet ✅ | df.shape = {df.shape}")

    try:
        layout, dates, tickers = _infer_layout(df)
    except Exception as e:
        print(f"❌ Layout inference failed: {e}")
        return 4

    print(f"Layout: {layout}")
    if len(dates) == 0:
        print("❌ No valid dates found in cache")
        return 5

    dmin = pd.to_datetime(dates.min()).date()
    dmax = pd.to_datetime(dates.max()).date()
    print(f"Date range: {dmin} → {dmax}  | distinct dates: {len(pd.Index(dates).unique())}")

    tickers_upper = {t.upper() for t in tickers if isinstance(t, str)}
    print(f"Tickers present: {len(tickers_upper)}")

    # Core presence checks
    _report_presence("Anchors", tickers_upper, required_all=ANCHORS_ALL, required_any=[])
    _report_presence("VIX Proxy", tickers_upper, required_all=[], required_any=VIX_ANY)
    _report_presence("T-Bill Proxy", tickers_upper, required_all=[], required_any=TBILL_ANY)

    # Registry cross-check
    reg_tickers = _load_wave_registry_tickers()
    if reg_tickers:
        reg_tickers = {t.upper() for t in reg_tickers}
        missing = sorted(list(reg_tickers - tickers_upper))
        present = sorted(list(reg_tickers & tickers_upper))

        print("\n[Wave Registry Cross-check]")
        print(f"Tickers referenced by registry: {len(reg_tickers)}")
        print(f"Registry tickers PRESENT in cache: {len(present)}")
        print(f"Registry tickers MISSING from cache: {len(missing)}")

        if missing:
            print("Sample missing tickers (up to 50):")
            for t in missing[:50]:
                print(f"  - {t}")
        else:
            print("✅ All registry tickers are present in the cache.")
    else:
        print("\n[Wave Registry Cross-check]")
        print("⚠️ Could not locate/load wave registry tickers (skipping registry cross-check).")
        print("This is OK — cache validation above still stands.")

    # Quick sanity thresholds (informational)
    print("\n[Sanity Heuristics]")
    if size_mb < 5:
        print(f"⚠️ Cache size is small ({size_mb:.2f} MB). For long history + many tickers, you'd often expect larger.")
    if len(tickers_upper) < 30:
        print(f"⚠️ Ticker count is low ({len(tickers_upper)}). If you expect 100+ symbols, cache likely incomplete.")
    if len(pd.Index(dates).unique()) < 260:
        print(f"⚠️ Date count is low ({len(pd.Index(dates).unique())}). 365D returns may be impossible.")

    print("\n✅ Ground-truth validation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())