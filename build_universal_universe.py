#!/usr/bin/env python3
"""
Universal Universe Builder Script

Creates the canonical universal_universe.csv file as the single source of truth
for all tickers across the Waves platform.

This script:
1. Fetches Russell 3000, Russell 2000, and S&P 500 equities
2. Fetches top 100-200 cryptocurrencies by market cap
3. Includes income and defensive ETFs
4. Includes thematic and sector ETFs required for existing Waves
5. Validates each ticker with lightweight price history check
6. Deduplicates tickers across all sources
7. Logs excluded tickers
8. Creates CSV with required metadata columns

Usage:
    python build_universal_universe.py [--validate] [--verbose]
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict
import time

# Try importing required packages
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas/numpy not available. Install with: pip install pandas numpy")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not available. Install with: pip install yfinance")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not available. Install with: pip install requests")


# Configuration
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(REPO_ROOT, "universal_universe.csv")
VALIDATION_LOG = os.path.join(REPO_ROOT, "universe_validation_log.json")
EXCLUDED_TICKERS_LOG = os.path.join(REPO_ROOT, "universe_excluded_tickers.json")

# Validation parameters
VALIDATION_LOOKBACK_DAYS = 30  # Check for price data in last 30 days
MIN_VALID_PRICE_POINTS = 5  # Require at least 5 valid price points
BATCH_SIZE = 50  # Validate tickers in batches to avoid rate limits
BATCH_DELAY_SECONDS = 1  # Delay between batches


def normalize_index_tag(text: str) -> str:
    """
    Normalize text to create consistent index membership tags.
    
    Args:
        text: Text to normalize (e.g., "S&P 500 Wave", "Russell 3000")
    
    Returns:
        Normalized tag (e.g., "SP500_WAVE", "RUSSELL_3000")
    """
    # Remove special characters except alphanumeric, spaces, hyphens
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
    # Replace spaces and hyphens with underscores
    normalized = cleaned.replace(' ', '_').replace('-', '_')
    # Convert to uppercase
    return normalized.upper()


def get_wave_tickers() -> Dict[str, List[str]]:
    """
    Extract all tickers currently used in Wave definitions.
    This ensures all existing Waves remain operational.
    """
    try:
        from waves_engine import WAVE_WEIGHTS
        
        wave_tickers = {}
        for wave_name, holdings in WAVE_WEIGHTS.items():
            tickers = [holding.ticker for holding in holdings]
            wave_tickers[wave_name] = tickers
        
        print(f"✓ Extracted tickers from {len(WAVE_WEIGHTS)} Wave definitions")
        return wave_tickers
    except Exception as e:
        print(f"⚠ Warning: Could not extract Wave tickers: {e}")
        return {}


def get_sp500_tickers() -> List[Dict[str, str]]:
    """
    Fetch S&P 500 constituents from Wikipedia.
    """
    if not HAS_PANDAS or not HAS_REQUESTS:
        print("⚠ Skipping S&P 500: pandas/requests not available")
        return []
    
    try:
        print("Fetching S&P 500 constituents from Wikipedia...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        
        constituents = []
        for _, row in df.iterrows():
            ticker = str(row['Symbol']).strip()
            # Replace dots with hyphens for yfinance compatibility
            ticker = ticker.replace('.', '-')
            
            constituents.append({
                'ticker': ticker,
                'name': str(row['Security']).strip(),
                'sector': str(row['GICS Sector']).strip(),
                'index_membership': 'SP500',
                'asset_class': 'equity'
            })
        
        print(f"✓ Fetched {len(constituents)} S&P 500 constituents")
        return constituents
    except Exception as e:
        print(f"⚠ Error fetching S&P 500: {e}")
        return []


def get_russell_3000_tickers() -> List[Dict[str, str]]:
    """
    Fetch Russell 3000 constituents.
    
    Note: Russell 3000 data is not freely available via API.
    This is a placeholder that could be enhanced with:
    - Cached/static file of Russell 3000 constituents
    - Paid data provider integration
    - Manual upload of constituent list
    """
    print("⚠ Russell 3000 data not available via free API")
    print("   Consider: 1) Upload cached file, 2) Use paid provider, 3) Manual list")
    
    # For now, return empty list
    # In production, you would load from a cached file:
    # russell_file = os.path.join(REPO_ROOT, "data", "russell_3000_constituents.csv")
    # if os.path.exists(russell_file):
    #     df = pd.read_csv(russell_file)
    #     ...
    
    return []


def get_russell_2000_tickers() -> List[Dict[str, str]]:
    """
    Fetch Russell 2000 constituents.
    
    Note: Similar to Russell 3000, this data requires a paid source or cached file.
    """
    print("⚠ Russell 2000 data not available via free API")
    print("   Consider: 1) Upload cached file, 2) Use paid provider, 3) Manual list")
    
    return []


def get_crypto_tickers(limit: int = 200) -> List[Dict[str, str]]:
    """
    Fetch top cryptocurrencies by market cap from CoinGecko API.
    """
    if not HAS_REQUESTS:
        print("⚠ Skipping crypto: requests not available")
        return []
    
    try:
        print(f"Fetching top {limit} cryptocurrencies from CoinGecko...")
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(limit, 250),
            "page": 1,
            "sparkline": False,
            "locale": "en"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        coins = response.json()
        
        constituents = []
        for coin in coins:
            symbol = coin.get('symbol', '').upper()
            # Add -USD suffix for yfinance compatibility
            ticker = f"{symbol}-USD" if not symbol.endswith('-USD') else symbol
            
            # Categorize crypto by market cap
            market_cap = coin.get('market_cap', 0)
            if market_cap > 10_000_000_000:
                cap_bucket = 'large_cap'
            elif market_cap > 1_000_000_000:
                cap_bucket = 'mid_cap'
            else:
                cap_bucket = 'small_cap'
            
            constituents.append({
                'ticker': ticker,
                'name': coin.get('name', ''),
                'sector': categorize_crypto(symbol, coin.get('name', '')),
                'index_membership': 'CRYPTO_TOP200',
                'asset_class': 'crypto',
                'market_cap_bucket': cap_bucket,
                'market_cap': market_cap
            })
        
        print(f"✓ Fetched {len(constituents)} cryptocurrencies")
        return constituents
    except Exception as e:
        print(f"⚠ Error fetching crypto data: {e}")
        return []


def categorize_crypto(symbol: str, name: str) -> str:
    """Categorize cryptocurrency by symbol and name."""
    symbol_upper = symbol.upper()
    name_lower = name.lower()
    
    # Store of Value / Settlement
    if symbol_upper in {"BTC", "BCH", "BSV", "LTC"}:
        return "Store of Value"
    
    # Smart Contract Platforms (Layer 1)
    l1_symbols = {"ETH", "SOL", "ADA", "AVAX", "DOT", "NEAR", "ALGO", "XTZ", "FTM",
                  "EGLD", "ONE", "CSPR", "TON", "TRX", "SUI", "HBAR", "ETC", "KAS", "APT", "ZIL"}
    if symbol_upper in l1_symbols:
        return "Smart Contract Platform"
    
    # Scaling Solutions (Layer 2)
    l2_symbols = {"MATIC", "POL", "ARB", "OP", "MNT", "STX", "LRC", "IMX"}
    if symbol_upper in l2_symbols:
        return "Layer 2 Scaling"
    
    # DeFi
    defi_symbols = {"UNI", "AAVE", "MKR", "SNX", "CRV", "COMP", "SUSHI", "YFI", "BAL",
                    "CAKE", "1INCH", "ALPHA", "DYDX", "GMX"}
    if symbol_upper in defi_symbols or "defi" in name_lower:
        return "DeFi"
    
    # Oracles
    if symbol_upper in {"LINK", "BAND", "TRB", "API3"}:
        return "Oracle"
    
    # Stablecoins
    if symbol_upper in {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "USDD", "FRAX"}:
        return "Stablecoin"
    
    # Exchange Tokens
    if symbol_upper in {"BNB", "CRO", "FTT", "HT", "OKB", "KCS", "LEO"}:
        return "Exchange Token"
    
    # Gaming / Metaverse
    gaming_symbols = {"MANA", "SAND", "AXS", "GALA", "ENJ", "IMX", "THETA", "ROSE",
                      "WAXP", "ALICE", "TLM", "ILV"}
    if symbol_upper in gaming_symbols or "gaming" in name_lower or "metaverse" in name_lower:
        return "Gaming/Metaverse"
    
    # Payments
    payment_symbols = {"XRP", "XLM", "DOGE", "SHIB", "XMR", "DASH", "CELO"}
    if symbol_upper in payment_symbols:
        return "Payment"
    
    # Default
    return "Cryptocurrency"


def get_income_defensive_etfs() -> List[Dict[str, str]]:
    """
    Get list of income and defensive ETFs.
    """
    etfs = [
        # Treasury/Government Bond ETFs
        {"ticker": "AGG", "name": "iShares Core U.S. Aggregate Bond ETF", "sector": "Aggregate Bonds", "asset_class": "fixed_income"},
        {"ticker": "BND", "name": "Vanguard Total Bond Market ETF", "sector": "Aggregate Bonds", "asset_class": "fixed_income"},
        {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "sector": "Long-Term Treasury", "asset_class": "fixed_income"},
        {"ticker": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF", "sector": "Intermediate Treasury", "asset_class": "fixed_income"},
        {"ticker": "SHY", "name": "iShares 1-3 Year Treasury Bond ETF", "sector": "Short-Term Treasury", "asset_class": "fixed_income"},
        {"ticker": "SHV", "name": "iShares Short Treasury Bond ETF", "sector": "Short-Term Treasury", "asset_class": "fixed_income"},
        {"ticker": "GOVT", "name": "iShares U.S. Treasury Bond ETF", "sector": "Treasury", "asset_class": "fixed_income"},
        {"ticker": "BIL", "name": "SPDR Bloomberg 1-3 Month T-Bill ETF", "sector": "T-Bills", "asset_class": "fixed_income"},
        {"ticker": "SGOV", "name": "iShares 0-3 Month Treasury Bond ETF", "sector": "T-Bills", "asset_class": "fixed_income"},
        
        # Corporate Bond ETFs
        {"ticker": "LQD", "name": "iShares iBoxx Investment Grade Corporate Bond ETF", "sector": "Investment Grade Corp", "asset_class": "fixed_income"},
        {"ticker": "VCIT", "name": "Vanguard Intermediate-Term Corporate Bond ETF", "sector": "Investment Grade Corp", "asset_class": "fixed_income"},
        {"ticker": "HYG", "name": "iShares iBoxx High Yield Corporate Bond ETF", "sector": "High Yield Corp", "asset_class": "fixed_income"},
        
        # Municipal Bond ETFs
        {"ticker": "MUB", "name": "iShares National Muni Bond ETF", "sector": "Municipal Bonds", "asset_class": "fixed_income"},
        {"ticker": "HYD", "name": "VanEck High Yield Muni ETF", "sector": "Municipal Bonds", "asset_class": "fixed_income"},
        {"ticker": "SHM", "name": "SPDR Nuveen Bloomberg Short Term Municipal Bond ETF", "sector": "Municipal Bonds", "asset_class": "fixed_income"},
        
        # Dividend ETFs
        {"ticker": "VIG", "name": "Vanguard Dividend Appreciation ETF", "sector": "Dividend", "asset_class": "etf"},
        {"ticker": "SCHD", "name": "Schwab U.S. Dividend Equity ETF", "sector": "Dividend", "asset_class": "etf"},
        {"ticker": "DVY", "name": "iShares Select Dividend ETF", "sector": "Dividend", "asset_class": "etf"},
        {"ticker": "HDV", "name": "iShares Core High Dividend ETF", "sector": "Dividend", "asset_class": "etf"},
        {"ticker": "NOBL", "name": "ProShares S&P 500 Dividend Aristocrats ETF", "sector": "Dividend", "asset_class": "etf"},
        
        # REIT ETFs
        {"ticker": "VNQ", "name": "Vanguard Real Estate ETF", "sector": "Real Estate", "asset_class": "etf"},
        {"ticker": "XLRE", "name": "Real Estate Select Sector SPDR Fund", "sector": "Real Estate", "asset_class": "etf"},
        
        # Gold/Commodities
        {"ticker": "GLD", "name": "SPDR Gold Trust", "sector": "Gold", "asset_class": "commodity"},
        {"ticker": "IAU", "name": "iShares Gold Trust", "sector": "Gold", "asset_class": "commodity"},
    ]
    
    for etf in etfs:
        etf['index_membership'] = 'INCOME_DEFENSIVE'
        if 'market_cap_bucket' not in etf:
            etf['market_cap_bucket'] = 'N/A'
    
    print(f"✓ Added {len(etfs)} income/defensive ETFs")
    return etfs


def get_thematic_sector_etfs() -> List[Dict[str, str]]:
    """
    Get thematic and sector ETFs required for existing Waves.
    """
    etfs = [
        # Broad Market ETFs
        {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "sector": "Broad Market", "asset_class": "etf"},
        {"ticker": "VOO", "name": "Vanguard S&P 500 ETF", "sector": "Broad Market", "asset_class": "etf"},
        {"ticker": "IVV", "name": "iShares Core S&P 500 ETF", "sector": "Broad Market", "asset_class": "etf"},
        {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "sector": "Broad Market", "asset_class": "etf"},
        {"ticker": "QQQ", "name": "Invesco QQQ Trust", "sector": "Technology", "asset_class": "etf"},
        {"ticker": "DIA", "name": "SPDR Dow Jones Industrial Average ETF", "sector": "Broad Market", "asset_class": "etf"},
        
        # Size-based ETFs
        {"ticker": "IWM", "name": "iShares Russell 2000 ETF", "sector": "Small-Cap", "asset_class": "etf"},
        {"ticker": "IWO", "name": "iShares Russell 2000 Growth ETF", "sector": "Small-Cap Growth", "asset_class": "etf"},
        {"ticker": "IJH", "name": "iShares Core S&P Mid-Cap ETF", "sector": "Mid-Cap", "asset_class": "etf"},
        {"ticker": "MDY", "name": "SPDR S&P MidCap 400 ETF", "sector": "Mid-Cap", "asset_class": "etf"},
        {"ticker": "IWP", "name": "iShares Russell Mid-Cap Growth ETF", "sector": "Mid-Cap Growth", "asset_class": "etf"},
        {"ticker": "IWV", "name": "iShares Russell 3000 ETF", "sector": "Broad Market", "asset_class": "etf"},
        
        # Sector ETFs
        {"ticker": "XLK", "name": "Technology Select Sector SPDR Fund", "sector": "Technology", "asset_class": "etf"},
        {"ticker": "XLF", "name": "Financial Select Sector SPDR Fund", "sector": "Financial", "asset_class": "etf"},
        {"ticker": "XLE", "name": "Energy Select Sector SPDR Fund", "sector": "Energy", "asset_class": "etf"},
        {"ticker": "XLV", "name": "Health Care Select Sector SPDR Fund", "sector": "Healthcare", "asset_class": "etf"},
        {"ticker": "XLY", "name": "Consumer Discretionary Select Sector SPDR Fund", "sector": "Consumer", "asset_class": "etf"},
        {"ticker": "XLP", "name": "Consumer Staples Select Sector SPDR Fund", "sector": "Consumer", "asset_class": "etf"},
        {"ticker": "XLI", "name": "Industrial Select Sector SPDR Fund", "sector": "Industrial", "asset_class": "etf"},
        {"ticker": "XLU", "name": "Utilities Select Sector SPDR Fund", "sector": "Utilities", "asset_class": "etf"},
        {"ticker": "XLB", "name": "Materials Select Sector SPDR Fund", "sector": "Materials", "asset_class": "etf"},
        
        # Thematic ETFs
        {"ticker": "ARKK", "name": "ARK Innovation ETF", "sector": "Innovation", "asset_class": "etf"},
        {"ticker": "ICLN", "name": "iShares Global Clean Energy ETF", "sector": "Clean Energy", "asset_class": "etf"},
        {"ticker": "SMH", "name": "VanEck Semiconductor ETF", "sector": "Semiconductors", "asset_class": "etf"},
        {"ticker": "PAVE", "name": "Global X U.S. Infrastructure Development ETF", "sector": "Infrastructure", "asset_class": "etf"},
        {"ticker": "BITO", "name": "ProShares Bitcoin Strategy ETF", "sector": "Cryptocurrency", "asset_class": "etf"},
    ]
    
    for etf in etfs:
        etf['index_membership'] = 'THEMATIC_SECTOR'
        if 'market_cap_bucket' not in etf:
            etf['market_cap_bucket'] = 'N/A'
    
    print(f"✓ Added {len(etfs)} thematic/sector ETFs")
    return etfs


def validate_ticker(ticker: str, verbose: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate ticker by checking if it has recent price data.
    
    Returns:
        (is_valid, error_reason)
    """
    if not HAS_YFINANCE:
        # Skip validation if yfinance not available
        return True, None
    
    try:
        # Fetch recent price history
        end_date = datetime.now()
        start_date = end_date - timedelta(days=VALIDATION_LOOKBACK_DAYS)
        
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            return False, "no_price_data"
        
        if len(hist) < MIN_VALID_PRICE_POINTS:
            return False, f"insufficient_data_points_{len(hist)}"
        
        # Check for valid prices (not all NaN)
        if hist['Close'].isna().all():
            return False, "all_prices_nan"
        
        if verbose:
            print(f"  ✓ {ticker}: {len(hist)} price points")
        
        return True, None
        
    except Exception as e:
        error_msg = str(e).lower()
        if "404" in error_msg or "not found" in error_msg:
            return False, "ticker_not_found"
        elif "delisted" in error_msg:
            return False, "delisted"
        else:
            return False, f"error_{type(e).__name__}"


def validate_tickers_batch(tickers: List[str], verbose: bool = False) -> Dict[str, Tuple[bool, Optional[str]]]:
    """
    Validate a batch of tickers.
    
    Returns:
        Dict mapping ticker -> (is_valid, error_reason)
    """
    results = {}
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{total} tickers validated...")
        
        is_valid, error = validate_ticker(ticker, verbose=verbose)
        results[ticker] = (is_valid, error)
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
    
    return results


def deduplicate_tickers(
    all_assets: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deduplicate tickers across all sources.
    
    Priority rules:
    1. Keep entry with most complete data (most non-empty fields)
    2. Merge index_membership tags from all duplicates
    3. Prefer equity over ETF, ETF over crypto for asset_class (if mixed)
    
    Returns:
        (deduplicated_assets, deduplication_report)
    """
    print("\nDeduplicating tickers...")
    
    ticker_map = defaultdict(list)
    
    # Group by ticker
    for asset in all_assets:
        ticker = asset.get('ticker', '').strip().upper()
        if ticker:
            ticker_map[ticker].append(asset)
    
    deduplicated = []
    report = {
        'total_input': len(all_assets),
        'unique_tickers': len(ticker_map),
        'duplicates_found': 0,
        'duplicate_details': []
    }
    
    for ticker, entries in ticker_map.items():
        if len(entries) == 1:
            deduplicated.append(entries[0])
        else:
            # Handle duplicates
            report['duplicates_found'] += 1
            
            # Merge index memberships
            all_indices = set()
            for entry in entries:
                indices = entry.get('index_membership', '')
                if indices:
                    for idx in indices.split(','):
                        all_indices.add(idx.strip())
            
            # Find most complete entry
            best_entry = max(entries, key=lambda e: sum(1 for v in e.values() if v and str(v).strip()))
            
            # Update with merged indices
            best_entry['index_membership'] = ','.join(sorted(all_indices))
            
            deduplicated.append(best_entry)
            
            report['duplicate_details'].append({
                'ticker': ticker,
                'count': len(entries),
                'merged_indices': list(all_indices),
                'kept_asset_class': best_entry.get('asset_class')
            })
    
    report['total_output'] = len(deduplicated)
    report['duplicates_removed'] = report['total_input'] - report['total_output']
    
    print(f"✓ Deduplication complete:")
    print(f"  Input: {report['total_input']} assets")
    print(f"  Unique: {report['unique_tickers']} tickers")
    print(f"  Duplicates merged: {report['duplicates_removed']}")
    
    return deduplicated, report


def write_universal_universe(
    assets: List[Dict[str, Any]],
    output_path: str,
    validation_results: Optional[Dict[str, Tuple[bool, Optional[str]]]] = None
) -> None:
    """
    Write the universal universe CSV file.
    """
    print(f"\nWriting universal universe to: {output_path}")
    
    # Define required columns
    columns = [
        'ticker',
        'name',
        'asset_class',
        'index_membership',
        'sector',
        'market_cap_bucket',
        'status',
        'validated',
        'validation_error'
    ]
    
    # Prepare data
    for asset in assets:
        # Set default values
        asset.setdefault('market_cap_bucket', 'N/A')
        asset.setdefault('status', 'active')
        
        # Add validation status
        ticker = asset.get('ticker', '')
        if validation_results and ticker in validation_results:
            is_valid, error = validation_results[ticker]
            asset['validated'] = 'yes' if is_valid else 'no'
            asset['validation_error'] = error if error else ''
        else:
            asset['validated'] = 'not_checked'
            asset['validation_error'] = ''
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(assets)
    
    print(f"✓ Wrote {len(assets)} assets to {output_path}")


def save_logs(
    dedup_report: Dict[str, Any],
    validation_results: Dict[str, Tuple[bool, Optional[str]]],
    excluded_tickers: List[Dict[str, str]]
) -> None:
    """
    Save validation and exclusion logs.
    """
    # Validation log
    validation_log = {
        'timestamp': datetime.now().isoformat(),
        'deduplication': dedup_report,
        'validation_summary': {
            'total_validated': len(validation_results),
            'valid': sum(1 for v, _ in validation_results.values() if v),
            'invalid': sum(1 for v, _ in validation_results.values() if not v),
        },
        'validation_details': {
            ticker: {'valid': is_valid, 'error': error}
            for ticker, (is_valid, error) in validation_results.items()
        }
    }
    
    with open(VALIDATION_LOG, 'w', encoding='utf-8') as f:
        json.dump(validation_log, f, indent=2)
    
    print(f"✓ Validation log saved to: {VALIDATION_LOG}")
    
    # Excluded tickers log
    excluded_log = {
        'timestamp': datetime.now().isoformat(),
        'total_excluded': len(excluded_tickers),
        'excluded_tickers': excluded_tickers
    }
    
    with open(EXCLUDED_TICKERS_LOG, 'w', encoding='utf-8') as f:
        json.dump(excluded_log, f, indent=2)
    
    print(f"✓ Excluded tickers log saved to: {EXCLUDED_TICKERS_LOG}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Build universal universe CSV')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate tickers with price history check')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()
    
    print("=" * 70)
    print("UNIVERSAL UNIVERSE BUILDER")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check dependencies
    if not HAS_PANDAS:
        print("ERROR: pandas is required. Install with: pip install pandas")
        return 1
    
    # Collect all assets
    all_assets = []
    
    # 1. Extract tickers from Wave definitions (highest priority)
    print("\n[1/7] Extracting tickers from Wave definitions...")
    wave_tickers = get_wave_tickers()
    for wave_name, tickers in wave_tickers.items():
        for ticker in tickers:
            # Normalize ticker
            ticker = ticker.strip().upper()
            
            # Determine asset class
            if '-USD' in ticker or ticker.startswith('BTC') or ticker.startswith('ETH'):
                asset_class = 'crypto'
                sector = 'Cryptocurrency'
            elif ticker in ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'AGG', 'BND', 'IEF', 'SHY']:
                asset_class = 'etf'
                sector = 'ETF'
            else:
                asset_class = 'equity'
                sector = 'Equity'
            
            all_assets.append({
                'ticker': ticker,
                'name': '',  # Will be filled if available from other sources
                'asset_class': asset_class,
                'index_membership': f'WAVE_{normalize_index_tag(wave_name)}',
                'sector': sector,
                'market_cap_bucket': 'N/A'
            })
    
    print(f"✓ Extracted {len(all_assets)} tickers from Waves")
    
    # 2. S&P 500
    print("\n[2/7] Fetching S&P 500 constituents...")
    sp500 = get_sp500_tickers()
    all_assets.extend(sp500)
    
    # 3. Russell 3000
    print("\n[3/7] Fetching Russell 3000 constituents...")
    russell_3000 = get_russell_3000_tickers()
    all_assets.extend(russell_3000)
    
    # 4. Russell 2000
    print("\n[4/7] Fetching Russell 2000 constituents...")
    russell_2000 = get_russell_2000_tickers()
    all_assets.extend(russell_2000)
    
    # 5. Cryptocurrencies
    print("\n[5/7] Fetching top cryptocurrencies...")
    crypto = get_crypto_tickers(limit=200)
    all_assets.extend(crypto)
    
    # 6. Income/Defensive ETFs
    print("\n[6/7] Adding income and defensive ETFs...")
    income_etfs = get_income_defensive_etfs()
    all_assets.extend(income_etfs)
    
    # 7. Thematic/Sector ETFs
    print("\n[7/7] Adding thematic and sector ETFs...")
    thematic_etfs = get_thematic_sector_etfs()
    all_assets.extend(thematic_etfs)
    
    # Deduplicate
    deduplicated_assets, dedup_report = deduplicate_tickers(all_assets)
    
    # Validate tickers if requested
    validation_results = {}
    excluded_tickers = []
    
    if args.validate and HAS_YFINANCE:
        print("\n" + "=" * 70)
        print("VALIDATING TICKERS")
        print("=" * 70)
        
        all_tickers = [asset['ticker'] for asset in deduplicated_assets]
        total_tickers = len(all_tickers)
        
        print(f"\nValidating {total_tickers} tickers in batches of {BATCH_SIZE}...")
        print(f"Lookback period: {VALIDATION_LOOKBACK_DAYS} days")
        print(f"Minimum price points: {MIN_VALID_PRICE_POINTS}\n")
        
        # Process in batches
        for i in range(0, total_tickers, BATCH_SIZE):
            batch = all_tickers[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"Batch {batch_num}/{total_batches} ({len(batch)} tickers)...")
            batch_results = validate_tickers_batch(batch, verbose=args.verbose)
            validation_results.update(batch_results)
            
            # Delay between batches
            if i + BATCH_SIZE < total_tickers:
                time.sleep(BATCH_DELAY_SECONDS)
        
        # Filter out invalid tickers
        valid_count = sum(1 for v, _ in validation_results.values() if v)
        invalid_count = sum(1 for v, _ in validation_results.values() if not v)
        
        print(f"\n✓ Validation complete:")
        print(f"  Valid: {valid_count}")
        print(f"  Invalid: {invalid_count}")
        
        # Log excluded tickers
        for asset in deduplicated_assets:
            ticker = asset['ticker']
            if ticker in validation_results:
                is_valid, error = validation_results[ticker]
                if not is_valid:
                    excluded_tickers.append({
                        'ticker': ticker,
                        'name': asset.get('name', ''),
                        'reason': error,
                        'asset_class': asset.get('asset_class', ''),
                        'index_membership': asset.get('index_membership', '')
                    })
    
    # Write output
    write_universal_universe(deduplicated_assets, OUTPUT_CSV, validation_results)
    
    # Save logs
    if args.validate:
        save_logs(dedup_report, validation_results, excluded_tickers)
    
    # Summary
    print("\n" + "=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Total assets: {len(deduplicated_assets)}")
    print(f"Validated: {len(validation_results)}")
    print(f"Excluded: {len(excluded_tickers)}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
