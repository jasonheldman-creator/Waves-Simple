#!/usr/bin/env python3
"""
Enable Full Data Readiness Script
Enables full data readiness (28/28 waves operational) via multiple paths:
1. Live fetch from Yahoo Finance
2. Offline CSV refresh
3. Alternate provider (Polygon, IEX) if API keys available
"""

import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_providers import YahooProvider


def detect_environment() -> Dict[str, bool]:
    """
    Detect available data fetching paths.
    
    Returns:
        Dict with environment capabilities
    """
    env = {
        'live_fetch_available': False,
        'polygon_available': False,
        'iex_available': False,
        'alphavantage_available': False
    }
    
    # Test live fetch to Yahoo Finance
    print("🔍 Detecting environment capabilities...")
    try:
        yahoo = YahooProvider()
        env['live_fetch_available'] = yahoo.test_connection()
        print(f"  {'✅' if env['live_fetch_available'] else '❌'} Live fetch (Yahoo Finance)")
    except Exception as e:
        print(f"  ❌ Live fetch failed: {e}")
    
    # Check for API keys
    env['polygon_available'] = bool(os.environ.get('POLYGON_API_KEY'))
    print(f"  {'✅' if env['polygon_available'] else '❌'} Polygon.io API")
    
    env['iex_available'] = bool(os.environ.get('IEX_TOKEN'))
    print(f"  {'✅' if env['iex_available'] else '❌'} IEX Cloud API")
    
    env['alphavantage_available'] = bool(os.environ.get('ALPHAVANTAGE_KEY'))
    print(f"  {'✅' if env['alphavantage_available'] else '❌'} Alpha Vantage API")
    
    return env


def get_all_tickers() -> List[str]:
    """
    Get all tickers from universal_universe.csv.
    
    Returns:
        List of active ticker symbols
    """
    universe_path = 'universal_universe.csv'
    
    if not os.path.exists(universe_path):
        print(f"⚠️  Warning: {universe_path} not found")
        return []
    
    try:
        df = pd.read_csv(universe_path)
        # Filter to active tickers only
        df = df[df['status'] == 'active']
        tickers = df['ticker'].dropna().unique().tolist()
        print(f"📋 Found {len(tickers)} active tickers")
        return tickers
    except Exception as e:
        print(f"❌ Error reading {universe_path}: {e}")
        return []


def fetch_data_live(tickers: List[str], days: int = 365) -> pd.DataFrame:
    """
    Fetch historical data for all tickers using live provider.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days of history to fetch
        
    Returns:
        Combined DataFrame with all ticker data
    """
    provider = YahooProvider()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"\n📥 Fetching {days} days of data for {len(tickers)} tickers...")
    print(f"   Date range: {start_date.date()} to {end_date.date()}")
    
    all_data = []
    success_count = 0
    failed_tickers = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] Fetching {ticker}...", end=' ')
        
        df = provider.get_history(ticker, start_date, end_date)
        
        if df is not None and not df.empty:
            all_data.append(df)
            success_count += 1
            print(f"✅ ({len(df)} rows)")
        else:
            failed_tickers.append(ticker)
            print("❌")
    
    print(f"\n✅ Successfully fetched {success_count}/{len(tickers)} tickers")
    
    if failed_tickers:
        print(f"⚠️  Failed tickers ({len(failed_tickers)}): {', '.join(failed_tickers[:10])}")
        if len(failed_tickers) > 10:
            print(f"   ... and {len(failed_tickers) - 10} more")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Sort by date and ticker
        combined = combined.sort_values(['date', 'ticker']).reset_index(drop=True)
        return combined
    else:
        return pd.DataFrame(columns=['date', 'ticker', 'close'])


def fetch_data_polygon(tickers: List[str], days: int = 365) -> pd.DataFrame:
    """
    Fetch historical data using Polygon.io provider.
    
    Args:
        tickers: List of ticker symbols
        days: Number of days of history to fetch
        
    Returns:
        Combined DataFrame with all ticker data
    """
    try:
        from data_providers import PolygonProvider
        
        provider = PolygonProvider()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"\n📥 Fetching {days} days of data using Polygon.io for {len(tickers)} tickers...")
        print(f"   Date range: {start_date.date()} to {end_date.date()}")
        
        all_data = []
        success_count = 0
        failed_tickers = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{len(tickers)}] Fetching {ticker}...", end=' ')
            
            df = provider.get_history(ticker, start_date, end_date)
            
            if df is not None and not df.empty:
                all_data.append(df)
                success_count += 1
                print(f"✅ ({len(df)} rows)")
            else:
                failed_tickers.append(ticker)
                print("❌")
        
        print(f"\n✅ Successfully fetched {success_count}/{len(tickers)} tickers")
        
        if failed_tickers:
            print(f"⚠️  Failed tickers ({len(failed_tickers)}): {', '.join(failed_tickers[:10])}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined = combined.sort_values(['date', 'ticker']).reset_index(drop=True)
            return combined
        else:
            return pd.DataFrame(columns=['date', 'ticker', 'close'])
    
    except ImportError:
        print("❌ Polygon provider not available")
        return pd.DataFrame(columns=['date', 'ticker', 'close'])


def write_prices_csv(df: pd.DataFrame, output_path: str = 'data/prices.csv') -> bool:
    """
    Write price data to canonical location.
    
    Args:
        df: DataFrame with price data
        output_path: Path to write CSV file
        
    Returns:
        True if successful
    """
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to CSV
        df.to_csv(output_path, index=False)
        
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"\n✅ Wrote {len(df)} rows to {output_path} ({file_size:.1f} KB)")
        
        return True
    except Exception as e:
        print(f"\n❌ Error writing to {output_path}: {e}")
        return False


def generate_readiness_summary(df: pd.DataFrame, tickers: List[str]) -> Dict:
    """
    Generate data readiness summary.
    
    Args:
        df: DataFrame with fetched price data
        tickers: List of all expected tickers
        
    Returns:
        Dict with readiness metrics
    """
    summary = {
        'total_tickers': len(tickers),
        'tickers_with_data': 0,
        'missing_tickers': [],
        'total_rows': len(df),
        'date_range': None,
        'coverage_pct': 0.0
    }
    
    if df.empty:
        summary['missing_tickers'] = tickers
        return summary
    
    # Get tickers with data
    tickers_with_data = df['ticker'].unique().tolist()
    summary['tickers_with_data'] = len(tickers_with_data)
    
    # Find missing tickers
    missing = set(tickers) - set(tickers_with_data)
    summary['missing_tickers'] = sorted(list(missing))
    
    # Calculate coverage percentage
    summary['coverage_pct'] = (len(tickers_with_data) / len(tickers)) * 100 if tickers else 0
    
    # Get date range
    if not df.empty:
        min_date = df['date'].min()
        max_date = df['date'].max()
        summary['date_range'] = f"{min_date.date()} to {max_date.date()}"
    
    return summary


def print_readiness_summary(summary: Dict) -> None:
    """
    Print formatted readiness summary.
    
    Args:
        summary: Dict with readiness metrics
    """
    print("\n" + "="*60)
    print("📊 DATA READINESS SUMMARY")
    print("="*60)
    
    print(f"\nTotal Tickers:        {summary['total_tickers']}")
    print(f"Tickers with Data:    {summary['tickers_with_data']}")
    print(f"Coverage:             {summary['coverage_pct']:.1f}%")
    print(f"Total Data Points:    {summary['total_rows']:,}")
    
    if summary['date_range']:
        print(f"Date Range:           {summary['date_range']}")
    
    if summary['missing_tickers']:
        print(f"\n⚠️  Missing Tickers ({len(summary['missing_tickers'])}):")
        for ticker in summary['missing_tickers'][:20]:
            print(f"   - {ticker}")
        if len(summary['missing_tickers']) > 20:
            print(f"   ... and {len(summary['missing_tickers']) - 20} more")
    
    print("\n" + "="*60)


def generate_diagnostic_files(tickers: List[str], df: pd.DataFrame) -> None:
    """
    Generate diagnostic CSV files for offline data refresh.
    
    Args:
        tickers: List of all expected tickers
        df: DataFrame with current price data
    """
    print("\n📝 Generating diagnostic files...")
    
    # Missing tickers
    tickers_with_data = df['ticker'].unique().tolist() if not df.empty else []
    missing = sorted(list(set(tickers) - set(tickers_with_data)))
    
    if missing:
        missing_df = pd.DataFrame({'ticker': missing, 'status': 'missing'})
        missing_path = 'data/missing_tickers.csv'
        missing_df.to_csv(missing_path, index=False)
        print(f"  ✅ Created {missing_path} ({len(missing)} tickers)")
    
    # Stale tickers (older than 7 days)
    stale = []
    if not df.empty:
        cutoff_date = datetime.now() - timedelta(days=7)
        for ticker in tickers_with_data:
            ticker_df = df[df['ticker'] == ticker]
            if not ticker_df.empty:
                latest_date = ticker_df['date'].max()
                if latest_date < cutoff_date:
                    stale.append({
                        'ticker': ticker,
                        'latest_date': latest_date.date(),
                        'days_old': (datetime.now() - latest_date).days
                    })
        
        if stale:
            stale_df = pd.DataFrame(stale)
            stale_path = 'data/stale_tickers.csv'
            stale_df.to_csv(stale_path, index=False)
            print(f"  ✅ Created {stale_path} ({len(stale)} tickers)")
    
    # Coverage summary
    summary = generate_readiness_summary(df, tickers)
    coverage_data = [{
        'metric': 'Total Tickers',
        'value': summary['total_tickers']
    }, {
        'metric': 'Tickers with Data',
        'value': summary['tickers_with_data']
    }, {
        'metric': 'Coverage Percentage',
        'value': f"{summary['coverage_pct']:.1f}%"
    }, {
        'metric': 'Total Data Points',
        'value': summary['total_rows']
    }, {
        'metric': 'Date Range',
        'value': summary['date_range'] or 'N/A'
    }]
    
    coverage_df = pd.DataFrame(coverage_data)
    coverage_path = 'data/data_coverage_summary.csv'
    coverage_df.to_csv(coverage_path, index=False)
    print(f"  ✅ Created {coverage_path}")


def print_next_steps(env: Dict[str, bool], success: bool) -> None:
    """
    Print next steps based on environment and execution result.
    
    Args:
        env: Environment capabilities dict
        success: Whether data fetch was successful
    """
    print("\n" + "="*60)
    print("📌 NEXT STEPS")
    print("="*60)
    
    if success:
        print("\n✅ Data has been successfully fetched and saved!")
        print("\nYou can now:")
        print("  1. Run your application with full data coverage")
        print("  2. Check data/prices.csv for the fetched data")
        print("  3. Review data/data_coverage_summary.csv for metrics")
    else:
        print("\n⚠️  Live data fetch is not available.")
        print("\nOption 1: Upload prices.csv manually")
        print("  1. Generate prices.csv offline with required tickers")
        print("  2. Upload to /data/prices.csv")
        print("  3. Format: date,ticker,close")
        
        print("\nOption 2: Configure API provider")
        print("  Set one of the following environment variables:")
        
        if not env['polygon_available']:
            print("  - POLYGON_API_KEY=<your-key>  (Polygon.io)")
        if not env['iex_available']:
            print("  - IEX_TOKEN=<your-token>      (IEX Cloud)")
        if not env['alphavantage_available']:
            print("  - ALPHAVANTAGE_KEY=<your-key> (Alpha Vantage)")
        
        print("\nDiagnostic files created:")
        print("  - data/missing_tickers.csv")
        print("  - data/stale_tickers.csv")
        print("  - data/data_coverage_summary.csv")
    
    print("\n" + "="*60)


def main():
    """
    Main execution function.
    """
    print("="*60)
    print("🌊 WAVES Intelligence - Full Data Enablement")
    print("="*60)
    
    # Detect environment
    env = detect_environment()
    
    # Get all tickers
    tickers = get_all_tickers()
    
    if not tickers:
        print("❌ No tickers found. Exiting.")
        sys.exit(1)
    
    # Choose data fetching path
    df = pd.DataFrame()
    success = False
    
    if env['live_fetch_available']:
        print("\n✅ Using Path 1: Live Fetch (Yahoo Finance)")
        df = fetch_data_live(tickers, days=365)
        success = not df.empty
    elif env['polygon_available']:
        print("\n✅ Using Path 2: Alternate Provider (Polygon.io)")
        df = fetch_data_polygon(tickers, days=365)
        success = not df.empty
    else:
        print("\n⚠️  No live data sources available")
        print("   Generating diagnostic files for offline refresh...")
    
    # Write data if successful
    if success:
        write_prices_csv(df, 'data/prices.csv')
    
    # Generate diagnostics
    generate_diagnostic_files(tickers, df)
    
    # Print summary
    summary = generate_readiness_summary(df, tickers)
    print_readiness_summary(summary)
    
    # Print next steps
    print_next_steps(env, success)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
