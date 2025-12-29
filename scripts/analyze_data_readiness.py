#!/usr/bin/env python3
"""
Analyze Data Readiness
Analyzes existing data and generates readiness report.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_data_readiness():
    """
    Analyze current data readiness from data/prices.csv.
    """
    print("="*60)
    print("🔍 Data Readiness Analysis")
    print("="*60)
    
    # Check if prices.csv exists
    prices_path = 'data/prices.csv'
    if not os.path.exists(prices_path):
        print(f"\n❌ {prices_path} not found")
        print("\nRun: python scripts/enable_full_data.py")
        return
    
    # Load prices data
    print(f"\n📂 Loading {prices_path}...")
    df = pd.read_csv(prices_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load universe
    universe_path = 'universal_universe.csv'
    if not os.path.exists(universe_path):
        print(f"⚠️  {universe_path} not found")
        return
    
    universe = pd.read_csv(universe_path)
    universe = universe[universe['status'] == 'active']
    all_tickers = universe['ticker'].unique().tolist()
    
    # Analyze coverage
    tickers_in_prices = df['ticker'].unique().tolist()
    missing_tickers = sorted(list(set(all_tickers) - set(tickers_in_prices)))
    
    coverage_pct = (len(tickers_in_prices) / len(all_tickers) * 100) if all_tickers else 0
    
    print("\n" + "="*60)
    print("📊 COVERAGE SUMMARY")
    print("="*60)
    
    print(f"\nTotal tickers expected:   {len(all_tickers)}")
    print(f"Tickers with data:        {len(tickers_in_prices)}")
    print(f"Missing tickers:          {len(missing_tickers)}")
    print(f"Coverage:                 {coverage_pct:.1f}%")
    
    print(f"\nTotal data points:        {len(df):,}")
    
    # Get date range (handle NaT values)
    min_date = df['date'].min()
    max_date = df['date'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        print(f"Date range:               {min_date.date()} to {max_date.date()}")
        print(f"Days of history:          {(max_date - min_date).days}")
    else:
        print(f"Date range:               N/A")
        print(f"Days of history:          N/A")
    
    # Check for stale data
    cutoff_date = datetime.now() - timedelta(days=7)
    stale_tickers = []
    
    for ticker in tickers_in_prices:
        ticker_df = df[df['ticker'] == ticker]
        if not ticker_df.empty:
            latest_date = ticker_df['date'].max()
            # Handle NaT values
            if pd.notna(latest_date) and latest_date < cutoff_date:
                days_old = (datetime.now() - latest_date).days
                stale_tickers.append((ticker, latest_date.date(), days_old))
    
    if stale_tickers:
        print(f"\n⚠️  Stale data detected ({len(stale_tickers)} tickers):")
        for ticker, latest, days in sorted(stale_tickers, key=lambda x: x[2], reverse=True)[:10]:
            print(f"  {ticker}: {latest} ({days} days old)")
        if len(stale_tickers) > 10:
            print(f"  ... and {len(stale_tickers) - 10} more")
    
    # Check for missing tickers
    if missing_tickers:
        print(f"\n❌ Missing tickers ({len(missing_tickers)}):")
        for ticker in missing_tickers[:20]:
            print(f"  - {ticker}")
        if len(missing_tickers) > 20:
            print(f"  ... and {len(missing_tickers) - 20} more")
    
    # Wave-specific analysis
    print("\n" + "="*60)
    print("🌊 WAVE-LEVEL READINESS")
    print("="*60)
    
    wave_cols = [col for col in universe.columns if 'WAVE_' in str(col).upper()]
    
    # Check index_membership for wave assignments
    if 'index_membership' in universe.columns:
        print("\nAnalyzing wave memberships...")
        
        # Extract unique wave names from index_membership
        waves = set()
        for idx_mem in universe['index_membership'].dropna():
            for wave in str(idx_mem).split(','):
                wave = wave.strip()
                if wave.startswith('WAVE_'):
                    waves.add(wave)
        
        waves = sorted(list(waves))
        
        if waves:
            print(f"\nFound {len(waves)} waves:")
            
            wave_readiness = []
            for wave in waves:
                # Get tickers for this wave
                wave_tickers = universe[
                    universe['index_membership'].str.contains(wave, case=False, na=False)
                ]['ticker'].unique().tolist()
                
                # Check how many have data
                wave_tickers_with_data = [t for t in wave_tickers if t in tickers_in_prices]
                
                wave_coverage = (len(wave_tickers_with_data) / len(wave_tickers) * 100) if wave_tickers else 0
                
                wave_readiness.append({
                    'wave': wave.replace('WAVE_', '').replace('_', ' ').title(),
                    'total': len(wave_tickers),
                    'with_data': len(wave_tickers_with_data),
                    'coverage': wave_coverage
                })
            
            # Sort by coverage
            wave_readiness.sort(key=lambda x: x['coverage'])
            
            # Show waves with incomplete data
            incomplete = [w for w in wave_readiness if w['coverage'] < 100]
            if incomplete:
                print(f"\n⚠️  Waves with incomplete data ({len(incomplete)}):")
                for w in incomplete[:10]:
                    status = "🟢" if w['coverage'] >= 80 else "🟡" if w['coverage'] >= 50 else "🔴"
                    print(f"  {status} {w['wave']}: {w['with_data']}/{w['total']} ({w['coverage']:.0f}%)")
            
            # Show fully operational waves
            complete = [w for w in wave_readiness if w['coverage'] == 100]
            if complete:
                print(f"\n✅ Fully operational waves ({len(complete)}):")
                for w in complete[:10]:
                    print(f"  🟢 {w['wave']}: {w['with_data']}/{w['total']} (100%)")
                if len(complete) > 10:
                    print(f"  ... and {len(complete) - 10} more")
    
    print("\n" + "="*60)
    
    # Operational status
    if coverage_pct >= 95:
        print("\n🟢 STATUS: FULLY OPERATIONAL")
    elif coverage_pct >= 80:
        print("\n🟡 STATUS: MOSTLY OPERATIONAL")
    elif coverage_pct >= 50:
        print("\n🟠 STATUS: PARTIALLY OPERATIONAL")
    else:
        print("\n🔴 STATUS: LIMITED OPERATIONAL")
    
    print("="*60)


if __name__ == '__main__':
    analyze_data_readiness()
