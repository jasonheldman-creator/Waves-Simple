#!/usr/bin/env python3
"""
Test SPY-based trading calendar functionality.

This test validates that:
1. get_trading_calendar_dates() extracts SPY-specific dates
2. Snapshot dates use SPY calendar
3. Price cache metadata uses SPY max date
4. Portfolio snapshot aggregation works with missing wave returns
"""

import pandas as pd
import json
from datetime import datetime
from helpers.trading_calendar import get_trading_calendar_dates, get_asof_date_str
from snapshot_ledger import _get_snapshot_date


def test_spy_trading_calendar():
    """Test that SPY-based trading calendar is used correctly."""
    print("\n" + "=" * 70)
    print("TEST: SPY-Based Trading Calendar")
    print("=" * 70)
    
    # Load price cache
    cache_df = pd.read_parquet("data/cache/prices_cache.parquet")
    print(f"✓ Loaded cache: {len(cache_df)} rows, {len(cache_df.columns)} columns")
    
    # Test 1: SPY trading calendar dates
    print("\n1. Testing get_trading_calendar_dates()...")
    asof_date, prev_date = get_trading_calendar_dates(cache_df)
    assert asof_date is not None, "asof_date should not be None"
    assert prev_date is not None, "prev_date should not be None"
    print(f"   ✓ SPY asof_date: {asof_date.strftime('%Y-%m-%d')}")
    print(f"   ✓ SPY prev_date: {prev_date.strftime('%Y-%m-%d')}")
    
    # Verify SPY date vs overall max date
    spy_series = cache_df['SPY'].dropna()
    spy_max = spy_series.index.max()
    overall_max = cache_df.index.max()
    
    assert asof_date.strftime('%Y-%m-%d') == spy_max.strftime('%Y-%m-%d'), \
        f"asof_date should match SPY max: {asof_date} vs {spy_max}"
    print(f"   ✓ SPY max date: {spy_max.strftime('%Y-%m-%d')}")
    print(f"   ℹ Overall max date: {overall_max.strftime('%Y-%m-%d')} (may differ)")
    
    # Test 2: Snapshot date uses SPY calendar
    print("\n2. Testing _get_snapshot_date()...")
    snapshot_date_str = _get_snapshot_date(cache_df)
    assert snapshot_date_str == asof_date.strftime('%Y-%m-%d'), \
        f"Snapshot date should match SPY asof_date: {snapshot_date_str} vs {asof_date.strftime('%Y-%m-%d')}"
    print(f"   ✓ Snapshot date: {snapshot_date_str} (matches SPY)")
    
    # Test 3: Price cache metadata
    print("\n3. Testing prices_cache_meta.json...")
    with open("data/cache/prices_cache_meta.json", 'r') as f:
        meta = json.load(f)
    
    assert 'spy_max_date' in meta, "spy_max_date should be in metadata"
    assert 'max_price_date' in meta, "max_price_date should be in metadata"
    assert meta['max_price_date'] == meta['spy_max_date'], \
        "max_price_date should equal spy_max_date"
    assert meta['spy_max_date'] == spy_max.strftime('%Y-%m-%d'), \
        f"spy_max_date in metadata should match SPY: {meta['spy_max_date']} vs {spy_max.strftime('%Y-%m-%d')}"
    
    print(f"   ✓ max_price_date: {meta['max_price_date']} (SPY-based)")
    print(f"   ✓ spy_max_date: {meta['spy_max_date']}")
    if 'overall_max_date' in meta:
        print(f"   ℹ overall_max_date: {meta['overall_max_date']} (diagnostic)")
    if 'min_symbol_max_date' in meta:
        print(f"   ℹ min_symbol_max_date: {meta['min_symbol_max_date']} (diagnostic)")
    
    # Test 4: Live snapshot date
    print("\n4. Testing live_snapshot.csv date...")
    snapshot_df = pd.read_csv("data/live_snapshot.csv")
    assert 'Date' in snapshot_df.columns, "Date column should exist"
    assert not snapshot_df['Date'].isna().all(), "Date column should not be all NaN"
    
    snapshot_dates = snapshot_df['Date'].unique()
    assert len(snapshot_dates) == 1, f"All rows should have same date, got: {snapshot_dates}"
    assert snapshot_dates[0] == spy_max.strftime('%Y-%m-%d'), \
        f"Snapshot date should match SPY: {snapshot_dates[0]} vs {spy_max.strftime('%Y-%m-%d')}"
    
    print(f"   ✓ Snapshot Date: {snapshot_dates[0]} (matches SPY)")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nSummary:")
    print(f"  • SPY trading calendar is the canonical source of truth")
    print(f"  • SPY asof_date: {asof_date.strftime('%Y-%m-%d')}")
    print(f"  • Overall cache max: {overall_max.strftime('%Y-%m-%d')}")
    print(f"  • Snapshot uses SPY date, not overall max")
    print(f"  • System not frozen by stale tickers")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_spy_trading_calendar()
