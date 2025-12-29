#!/usr/bin/env python3
"""
Enable Full Live Data for All 28 Waves

This script attempts to fetch fresh market data for all waves and 
provides actionable recommendations if data fetching fails.

Usage:
    python enable_full_live_data.py [--test-only] [--use-cache]
    
Options:
    --test-only   Test data fetching without actually running the pipeline
    --use-cache   Use offline mode with cached data only
"""

import sys
import os
import subprocess
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_section(title):
    """Print a formatted section."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

def test_yfinance_access():
    """Test if yfinance can fetch data."""
    print_section("Testing yfinance API Access")
    
    try:
        import yfinance as yf
        print("✓ yfinance module imported successfully")
        
        # Try to fetch a simple ticker
        print("Testing data fetch for SPY...")
        data = yf.download('SPY', period='5d', progress=False)
        
        if data.empty:
            print("✗ yfinance returned empty data (API may be blocked)")
            return False
        else:
            print(f"✓ Successfully fetched {len(data)} days of SPY data")
            return True
            
    except ImportError:
        print("✗ yfinance not installed")
        print("  Run: pip install yfinance")
        return False
    except Exception as e:
        print(f"✗ yfinance test failed: {e}")
        return False

def get_missing_tickers():
    """Get list of missing tickers from diagnostics."""
    try:
        from wave_readiness_diagnostics import diagnose_all_waves
        
        print_section("Analyzing Missing Tickers")
        
        diagnostics = diagnose_all_waves()
        
        all_missing = set()
        by_wave = {}
        
        for diag in diagnostics:
            if diag.prices_missing_tickers:
                wave_name = diag.display_name
                missing = diag.prices_missing_tickers
                all_missing.update(missing)
                by_wave[wave_name] = missing
        
        print(f"\nTotal unique missing tickers: {len(all_missing)}")
        print("\nMissing tickers by category:")
        
        crypto_tickers = [t for t in all_missing if '-USD' in t]
        equity_tickers = [t for t in all_missing if '-USD' not in t]
        
        print(f"  Crypto: {len(crypto_tickers)} tickers")
        print(f"  Equity: {len(equity_tickers)} tickers")
        
        print("\nCrypto tickers needed:")
        for ticker in sorted(crypto_tickers)[:10]:
            print(f"  - {ticker}")
        if len(crypto_tickers) > 10:
            print(f"  ... and {len(crypto_tickers) - 10} more")
        
        print("\nEquity tickers needed:")
        for ticker in sorted(equity_tickers)[:10]:
            print(f"  - {ticker}")
        if len(equity_tickers) > 10:
            print(f"  ... and {len(equity_tickers) - 10} more")
        
        return all_missing, by_wave
        
    except Exception as e:
        print(f"Error analyzing missing tickers: {e}")
        return set(), {}

def run_analytics_pipeline(test_only=False):
    """Run the analytics pipeline to fetch fresh data."""
    print_section("Running Analytics Pipeline")
    
    if test_only:
        print("TEST MODE: Would run analytics pipeline with:")
        print("  python analytics_pipeline.py --all --lookback=365")
        return True
    
    print("Fetching fresh data for all 28 waves with 365 days of history...")
    print("This may take 10-15 minutes...\n")
    
    try:
        result = subprocess.run(
            ['python3', 'analytics_pipeline.py', '--all', '--lookback=365'],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n✓ Analytics pipeline completed successfully")
            return True
        else:
            print(f"\n✗ Analytics pipeline failed with exit code {result.returncode}")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✗ Analytics pipeline timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"\n✗ Error running analytics pipeline: {e}")
        return False

def run_offline_loader():
    """Run the offline data loader as fallback."""
    print_section("Running Offline Data Loader (Fallback Mode)")
    
    print("Using cached data from prices.csv...")
    
    try:
        result = subprocess.run(
            ['python3', 'offline_data_loader.py', '--overwrite'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n✓ Offline data loader completed successfully")
            return True
        else:
            print(f"\n✗ Offline data loader failed with exit code {result.returncode}")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"\n✗ Error running offline data loader: {e}")
        return False

def show_final_status():
    """Show final wave readiness status."""
    print_section("Final Wave Readiness Status")
    
    try:
        result = subprocess.run(
            ['python3', 'wave_readiness_diagnostics.py', 'text'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Extract summary section
        output = result.stdout
        if 'SECTION 2:' in output:
            summary_start = output.find('SECTION 2:')
            summary_end = output.find('SECTION 3:')
            if summary_end > summary_start:
                print(output[summary_start:summary_end])
        
        # Show recommendations
        if 'SECTION 5:' in output:
            rec_start = output.find('SECTION 5:')
            print(output[rec_start:])
            
    except Exception as e:
        print(f"Error getting final status: {e}")

def main():
    """Main execution function."""
    test_only = '--test-only' in sys.argv
    use_cache = '--use-cache' in sys.argv
    
    print_header("ENABLE FULL LIVE DATA FOR ALL 28 WAVES")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if test_only:
        print("\n⚠️  TEST MODE - No changes will be made")
    
    # Step 1: Test yfinance access
    yfinance_works = test_yfinance_access()
    
    # Step 2: Analyze missing tickers
    missing_tickers, by_wave = get_missing_tickers()
    
    # Step 3: Choose strategy
    print_section("Recommended Strategy")
    
    if yfinance_works and not use_cache:
        print("\n✓ yfinance is accessible")
        print("Recommended: Run analytics pipeline to fetch fresh live data")
        
        if test_only:
            success = run_analytics_pipeline(test_only=True)
        else:
            proceed = input("\nProceed with analytics pipeline? (y/n): ").lower().strip()
            if proceed == 'y':
                success = run_analytics_pipeline()
            else:
                print("Skipped analytics pipeline")
                success = False
    else:
        print("\n✗ yfinance is not accessible or --use-cache specified")
        print("Recommended: Use offline data loader with cached data")
        print("\nNote: This will only populate waves with tickers in prices.csv")
        print(f"      {len(missing_tickers)} tickers are missing from cache")
        
        if not test_only:
            success = run_offline_loader()
        else:
            print("\nTEST MODE: Would run offline_data_loader.py --overwrite")
            success = True
    
    # Step 4: Show final status
    if not test_only:
        show_final_status()
    
    # Step 5: Provide next steps
    print_section("Next Steps")
    
    if success:
        print("\n✓ Data population completed")
        print("\nTo verify full analytics are available:")
        print("  1. Run: python wave_readiness_diagnostics.py")
        print("  2. Check that 'Full Ready' count is 28 (100%)")
        print("  3. Launch the Streamlit app to see all waves with full analytics")
    else:
        print("\n⚠️  Data population encountered issues")
        print("\nTroubleshooting steps:")
        print("  1. Review ENABLE_FULL_LIVE_DATA_GUIDE.md for detailed instructions")
        print("  2. Check if you need to add missing tickers to prices.csv")
        print("  3. Consider using an alternative data provider")
    
    if missing_tickers:
        print(f"\n⚠️  {len(missing_tickers)} tickers are missing from cache")
        print("    See ENABLE_FULL_LIVE_DATA_GUIDE.md for the complete list")
    
    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
