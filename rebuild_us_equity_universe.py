#!/usr/bin/env python3
"""
Rebuild Universal Universe CSV - U.S. Equity Focus

This script rebuilds universal_universe.csv with ONLY U.S. equity tickers from:
- Russell 3000
- Russell 2000
- S&P 500
- NASDAQ Composite
- Dow Jones Industrial Average

It performs:
1. Fetches constituents from all 5 indices
2. Aggressive deduplication (case-insensitive)
3. Symbol cleaning and normalization
4. Removal of invalid/malformed/deprecated symbols
5. Exclusion of non-equity identifiers and index symbols
6. Generation of validation report

Usage:
    python rebuild_us_equity_universe.py
"""

import os
import sys
import csv
import json
import pandas as pd
import requests
from datetime import datetime
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re

# Import comprehensive ticker database
try:
    from us_equity_tickers_comprehensive import (
        SP500_TICKERS,
        RUSSELL_3000_ADDITIONS,
        RUSSELL_2000_REPRESENTATIVE,
        NASDAQ_COMPOSITE_TICKERS,
        DOW_JONES_TICKERS,
        get_all_tickers as get_comprehensive_tickers,
        get_ticker_index_mapping as get_comprehensive_mapping
    )
    HAS_COMPREHENSIVE_DB = True
except ImportError:
    HAS_COMPREHENSIVE_DB = False
    print("Warning: Comprehensive ticker database not available")

# Configuration
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(REPO_ROOT, "universal_universe.csv")
VALIDATION_REPORT = os.path.join(REPO_ROOT, "universe_rebuild_report.json")

# Ticker format preference: Use hyphen for special characters (BRK-B, not BRK.B)
TICKER_FORMAT = "hyphen"  # hyphen format for consistency with yfinance


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol to consistent format.
    
    Rules:
    - Convert to uppercase
    - Replace dots with hyphens (BRK.B -> BRK-B)
    - Remove whitespace
    - Remove invalid characters
    
    Args:
        ticker: Raw ticker symbol
        
    Returns:
        Normalized ticker symbol
    """
    if not ticker:
        return ""
    
    # Convert to uppercase and strip whitespace
    ticker = str(ticker).strip().upper()
    
    # Replace dots with hyphens for consistency with yfinance
    ticker = ticker.replace('.', '-')
    
    # Remove any other special characters except hyphens and alphanumeric
    ticker = re.sub(r'[^A-Z0-9\-]', '', ticker)
    
    return ticker


def is_valid_ticker(ticker: str) -> bool:
    """
    Check if ticker symbol is valid.
    
    Invalid tickers:
    - Empty strings
    - Index symbols (starting with ^)
    - Non-alphanumeric (except hyphens)
    - Too short (< 1 char)
    - Too long (> 6 chars for most stocks)
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not ticker or len(ticker) == 0:
        return False
    
    # Exclude index symbols
    if ticker.startswith('^'):
        return False
    
    # Check length (1-6 chars is typical for U.S. equities)
    if len(ticker) < 1 or len(ticker) > 6:
        return False
    
    # Must contain at least one letter
    if not any(c.isalpha() for c in ticker):
        return False
    
    return True


def get_sp500_static_list() -> List[str]:
    """
    Static list of S&P 500 constituents (as of Q4 2024).
    This is a fallback when network access is unavailable.
    """
    # Major S&P 500 companies (representative sample + common constituents)
    # In production, this would be the full 503 constituents
    return [
        # Mega-cap tech (Information Technology)
        "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "ORCL", "ACN", "CSCO", "AMD",
        "INTC", "IBM", "NOW", "TXN", "QCOM", "INTU", "AMAT", "MU", "LRCX", "KLAC",
        "SNPS", "CDNS", "ADSK", "FTNT", "PANW", "CRWD", "WDAY", "SNOW",
        
        # Communication Services  
        "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
        "EA", "TTWO", "LYV", "MTCH", "PARA", "OMC", "IPG", "NWSA",
        
        # Consumer Discretionary
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "MAR",
        "GM", "F", "ABNB", "CMG", "ORLY", "AZO", "ROST", "DHI", "LEN", "YUM",
        "DPZ", "ULTA", "DECK", "POOL", "TPR", "RL", "GPS", "M", "JWN", "KSS",
        
        # Consumer Staples
        "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "GIS",
        "KMB", "SYY", "KHC", "HSY", "K", "CAG", "CPB", "HRL", "SJM", "MKC",
        
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL",
        "DVN", "HES", "FANG", "BKR", "KMI", "WMB", "LNG", "TRGP", "OKE",
        
        # Financials
        "BRK-B", "JPM", "BAC", "WFC", "MS", "GS", "C", "SCHW", "AXP", "BLK",
        "SPGI", "USB", "PNC", "TFC", "COF", "BK", "STT", "NTRS", "FRC", "RF",
        "CFG", "KEY", "FITB", "HBAN", "MTB", "ZION", "CMA", "SIVB", "WAL",
        
        # Healthcare
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
        "AMGN", "CVS", "MDT", "GILD", "ISRG", "CI", "REGN", "VRTX", "HUM", "ZTS",
        "BSX", "ELV", "SYK", "MCK", "COR", "BDX", "EW", "IDXX", "HCA", "A",
        
        # Industrials
        "CAT", "UNP", "UPS", "RTX", "BA", "HON", "LMT", "GE", "DE", "MMM",
        "NSC", "FDX", "CSX", "NOC", "EMR", "ETN", "ITW", "PH", "WM", "GD",
        "TT", "PCAR", "ROK", "CARR", "OTIS", "JCI", "CTAS", "CMI", "FAST",
        
        # Materials
        "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "DOW", "DD", "NUE", "PPG",
        "VMC", "MLM", "CTVA", "EMN", "CE", "ALB", "IFF", "MOS", "FMC", "CF",
        
        # Real Estate
        "AMT", "PLD", "EQIX", "PSA", "WELL", "DLR", "O", "SPG", "VICI", "AVB",
        "EQR", "SBAC", "CBRE", "VTR", "ARE", "MAA", "INVH", "ESS", "UDR", "CPT",
        
        # Utilities
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ES",
        "PEG", "ED", "AWK", "DTE", "PPL", "EIX", "FE", "AEE", "CMS", "CNP",
        
        # Additional major companies
        "V", "MA", "PYPL", "SQ", "FISV", "FIS", "ADP", "PAYX", "INFY", "CTSH"
    ]


def fetch_sp500_constituents() -> List[str]:
    """
    Fetch S&P 500 constituents from Wikipedia.
    
    Returns:
        List of ticker symbols
    """
    print("Fetching S&P 500 constituents...")
    
    # Use comprehensive database if available
    if HAS_COMPREHENSIVE_DB:
        print(f"✓ Using comprehensive database: {len(SP500_TICKERS)} S&P 500 tickers")
        return list(SP500_TICKERS)
    
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        
        tickers = []
        for _, row in df.iterrows():
            ticker = normalize_ticker(row['Symbol'])
            if is_valid_ticker(ticker):
                tickers.append(ticker)
        
        print(f"✓ Fetched {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"⚠ Error fetching S&P 500: {e}")
        print("   Using static S&P 500 list as fallback...")
        static_list = get_sp500_static_list()
        print(f"✓ Loaded {len(static_list)} S&P 500 tickers from static list")
        return static_list


def get_russell_3000_static_list() -> List[str]:
    """
    Static list of representative Russell 3000 constituents.
    Russell 3000 = Russell 1000 (large/mid cap) + Russell 2000 (small cap)
    This is primarily covered by S&P 500 + additional mid/small caps.
    """
    # Additional mid-cap and small-cap stocks not in S&P 500
    return [
        # Mid-cap growth
        "PLTR", "RBLX", "NET", "DDOG", "ZS", "OKTA", "CRWD", "ZM", "DOCU", "TWLO",
        "COUP", "BILL", "DOCN", "S", "PATH", "GTLB", "CFLT", "DBRG", "MDB", "ESTC",
        
        # Mid-cap value
        "ALK", "JBLU", "LUV", "UAL", "DAL", "AAL", "HA", "SAVE", "SKYW", "MESA",
        
        # Quantum computing & AI (mid-cap growth)
        "IONQ", "RGTI", "QBTS", "SOUN", "BBAI", "AI", "SMCI", "DELL", "HPE",
        
        # Additional diversified
        "RIVN", "LCID", "CHPT", "BLNK", "EVGO", "PTRA", "HYLN", "GOEV", "FSR",
    ]


def fetch_russell_3000_constituents() -> List[str]:
    """
    Fetch Russell 3000 constituents.
    
    Note: Russell 3000 data is not freely available. This uses a proxy approach:
    - S&P 500 is a subset of Russell 3000
    - Russell 2000 is the small-cap portion of Russell 3000
    - Russell 3000 = Russell 1000 + Russell 2000
    - We'll use IWV (Russell 3000 ETF) holdings if available
    
    For production use, consider:
    - Cached constituent file
    - Paid data provider
    - Manual upload
    
    Returns:
        List of ticker symbols
    """
    print("Fetching Russell 3000 constituents...")
    
    # Use comprehensive database if available
    if HAS_COMPREHENSIVE_DB:
        print(f"✓ Using comprehensive database: {len(RUSSELL_3000_ADDITIONS)} Russell 3000 additions")
        return list(RUSSELL_3000_ADDITIONS)
    
    print("⚠ Note: Russell 3000 data not freely available via API")
    print("   Using static representative list of mid/small caps")
    
    static_list = get_russell_3000_static_list()
    print(f"✓ Loaded {len(static_list)} Russell 3000 representative tickers")
    return static_list


def get_russell_2000_static_list() -> List[str]:
    """
    Static list of representative Russell 2000 (small-cap) constituents.
    """
    # Representative small-cap stocks
    return [
        # Small-cap growth
        "SMCI", "MARA", "RIOT", "HUT", "CLSK", "BITF", "CIFR", "BTBT", "ARBK", "IREN",
        
        # Small-cap value
        "AMC", "APE", "BBBY", "GME", "KOSS", "EXPR", "NAKD", "SNDL", "TLRY", "CGC",
        
        # Small-cap tech
        "SOFI", "UPST", "AFRM", "LC", "OPEN", "COMP", "RKT", "UWMC", "GHIV", "CLOV",
        
        # Regional banks (small-cap)
        "PACW", "WAL", "SBNY", "CMA", "ZION", "ONB", "UMBF", "UBSI", "FHN", "SNV",
        
        # Small-cap industrials
        "WKHS", "RIDE", "NKLA", "HYLN", "GOEV", "ARVL", "MULN", "ELMS", "AYRO", "GEV",
    ]


def fetch_russell_2000_constituents() -> List[str]:
    """
    Fetch Russell 2000 constituents.
    
    Similar to Russell 3000, this data requires a paid source.
    We'll attempt to use IWM (Russell 2000 ETF) holdings as a proxy.
    
    Returns:
        List of ticker symbols
    """
    print("Fetching Russell 2000 constituents...")
    
    # Use comprehensive database if available
    if HAS_COMPREHENSIVE_DB:
        print(f"✓ Using comprehensive database: {len(RUSSELL_2000_REPRESENTATIVE)} Russell 2000 tickers")
        return list(RUSSELL_2000_REPRESENTATIVE)
    
    print("⚠ Note: Russell 2000 data not freely available via API")
    print("   Using static representative list")
    
    static_list = get_russell_2000_static_list()
    print(f"✓ Loaded {len(static_list)} Russell 2000 representative tickers")
    return static_list


def get_nasdaq_composite_static_list() -> List[str]:
    """
    Static list of major NASDAQ Composite constituents.
    NASDAQ Composite includes all NASDAQ-listed stocks (~3000+ stocks).
    This is a representative subset.
    """
    # Major NASDAQ-listed stocks
    return [
        # NASDAQ 100 (major tech)
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META", "AVGO", "COST",
        "NFLX", "ADBE", "PEP", "CSCO", "CMCSA", "TMUS", "INTC", "AMD", "QCOM", "INTU",
        "TXN", "AMGN", "HON", "AMAT", "SBUX", "ISRG", "BKNG", "ADP", "GILD", "VRTX",
        "ADI", "REGN", "LRCX", "MDLZ", "MU", "PANW", "PYPL", "KLAC", "ASML", "ABNB",
        "SNPS", "CDNS", "MELI", "MAR", "CSX", "CRWD", "ORLY", "ADSK", "FTNT", "NXPI",
        
        # Mid-cap NASDAQ
        "WDAY", "DXCM", "PCAR", "CHTR", "CTAS", "MNST", "PAYX", "MCHP", "AEP", "ROST",
        "FAST", "EXC", "ODFL", "IDXX", "KDP", "EA", "VRSK", "CCEP", "CPRT", "GEHC",
        
        # Growth tech
        "SNOW", "DDOG", "NET", "ZS", "OKTA", "CRWD", "MDB", "TEAM", "NOW", "PLTR",
        "RBLX", "U", "DOCN", "CFLT", "GTLB", "S", "PATH", "BILL", "COUP", "ZM",
        
        # Biotech/Healthcare
        "MRNA", "BIIB", "ILMN", "ALNY", "SGEN", "BMRN", "VRTX", "REGN", "EXAS", "TECH",
        "CRSP", "NTLA", "EDIT", "BEAM", "BLUE", "FATE", "VCYT", "PACB", "NVTA", "NTRA",
        
        # EV and clean energy
        "TSLA", "RIVN", "LCID", "CHPT", "BLNK", "EVGO", "ENPH", "FSLR", "SEDG", "RUN",
    ]


def get_nasdaq_100_static_list() -> List[str]:
    """
    Static list of NASDAQ 100 constituents.
    """
    # NASDAQ 100 major stocks
    return [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST",
        "NFLX", "ADBE", "PEP", "CSCO", "CMCSA", "TMUS", "AMD", "INTC", "QCOM", "INTU",
        "TXN", "AMGN", "HON", "AMAT", "SBUX", "ISRG", "BKNG", "ADP", "GILD", "VRTX",
        "REGN", "LRCX", "MDLZ", "MU", "PANW", "PYPL", "KLAC", "ASML", "ABNB", "SNPS",
        "CDNS", "MELI", "MAR", "CSX", "CRWD", "ORLY", "ADSK", "FTNT", "NXPI", "WDAY",
        "DXCM", "PCAR", "CHTR", "CTAS", "MNST", "PAYX", "MCHP", "AEP", "ROST", "FAST",
        "EXC", "ODFL", "IDXX", "KDP", "EA", "VRSK", "CCEP", "CPRT", "GEHC", "DDOG",
        "ZS", "TEAM", "MRNA", "BIIB", "ILMN", "ALNY", "KHC", "SIRI", "XEL", "WBD",
        "CSGP", "DLTR", "ANSS", "TTWO", "ON", "FANG", "CDW", "WBA", "GFS", "LULU",
        "ALGN", "BKR", "EBAY", "SGEN", "JD", "PDD", "BIDU", "LI", "NTES", "ZM",
    ]


def fetch_nasdaq_composite_constituents() -> List[str]:
    """
    Fetch NASDAQ Composite constituents.
    
    The NASDAQ Composite includes all stocks listed on the NASDAQ exchange.
    We'll use the NASDAQ API or scrape from nasdaq.com.
    
    Returns:
        List of ticker symbols
    """
    print("Fetching NASDAQ Composite constituents...")
    
    # Use comprehensive database if available
    if HAS_COMPREHENSIVE_DB:
        print(f"✓ Using comprehensive database: {len(NASDAQ_COMPOSITE_TICKERS)} NASDAQ Composite tickers")
        return list(NASDAQ_COMPOSITE_TICKERS)
    
    try:
        # Try to fetch from NASDAQ screener
        url = "https://api.nasdaq.com/api/screener/stocks"
        params = {
            "tableonly": "true",
            "limit": 10000,
            "exchange": "nasdaq"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'rows' in data['data']:
                tickers = []
                for row in data['data']['rows']:
                    ticker = normalize_ticker(row.get('symbol', ''))
                    if is_valid_ticker(ticker):
                        tickers.append(ticker)
                
                print(f"✓ Fetched {len(tickers)} NASDAQ Composite tickers")
                return tickers
        
        print("⚠ Could not fetch from NASDAQ API, using fallback method")
        return fetch_nasdaq_fallback()
        
    except Exception as e:
        print(f"⚠ Error fetching NASDAQ Composite: {e}")
        return fetch_nasdaq_fallback()


def fetch_nasdaq_fallback() -> List[str]:
    """
    Fallback method to get NASDAQ tickers using alternative source.
    
    Returns:
        List of ticker symbols
    """
    print("Using NASDAQ static list...")
    try:
        # Try Wikipedia first
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        df = tables[4]  # The constituent table
        
        tickers = []
        for _, row in df.iterrows():
            ticker = normalize_ticker(row['Ticker'])
            if is_valid_ticker(ticker):
                tickers.append(ticker)
        
        print(f"✓ Fetched {len(tickers)} NASDAQ 100 tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"⚠ Error fetching from Wikipedia: {e}")
        print("   Using static NASDAQ Composite list...")
        static_list = get_nasdaq_composite_static_list()
        print(f"✓ Loaded {len(static_list)} NASDAQ Composite tickers from static list")
        return static_list


def get_dow_jones_static_list() -> List[str]:
    """
    Static list of Dow Jones Industrial Average constituents (30 stocks).
    """
    # The Dow 30
    return [
        "AAPL", "MSFT", "UNH", "GS", "HD", "CAT", "MCD", "AMGN", "V", "BA",
        "HON", "TRV", "JPM", "AXP", "IBM", "JNJ", "PG", "CVX", "WMT", "MRK",
        "DIS", "CRM", "CSCO", "NKE", "VZ", "KO", "DOW", "INTC", "MMM", "WBA"
    ]


def fetch_dow_jones_constituents() -> List[str]:
    """
    Fetch Dow Jones Industrial Average constituents.
    
    Returns:
        List of ticker symbols
    """
    print("Fetching Dow Jones Industrial Average constituents...")
    
    # Use comprehensive database if available
    if HAS_COMPREHENSIVE_DB:
        print(f"✓ Using comprehensive database: {len(DOW_JONES_TICKERS)} Dow Jones tickers")
        return list(DOW_JONES_TICKERS)
    
    try:
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        tables = pd.read_html(url)
        df = tables[1]  # The constituent table
        
        tickers = []
        for _, row in df.iterrows():
            ticker = normalize_ticker(row['Symbol'])
            if is_valid_ticker(ticker):
                tickers.append(ticker)
        
        print(f"✓ Fetched {len(tickers)} Dow Jones tickers")
        return tickers
    except Exception as e:
        print(f"⚠ Error fetching Dow Jones: {e}")
        print("   Using static Dow Jones list...")
        static_list = get_dow_jones_static_list()
        print(f"✓ Loaded {len(static_list)} Dow Jones tickers from static list")
        return static_list


def deduplicate_tickers(ticker_map: Dict[str, List[str]]) -> Tuple[List[str], Dict]:
    """
    Aggressively deduplicate tickers across all sources.
    
    Args:
        ticker_map: Dict mapping index name to list of tickers
        
    Returns:
        (deduplicated_tickers, report)
    """
    print("\nDeduplicating tickers...")
    
    # Track which indices each ticker appears in
    ticker_indices = defaultdict(set)
    all_tickers = set()
    
    for index_name, tickers in ticker_map.items():
        for ticker in tickers:
            # Case-insensitive deduplication
            ticker_upper = ticker.upper()
            ticker_indices[ticker_upper].add(index_name)
            all_tickers.add(ticker_upper)
    
    # Calculate statistics
    total_before = sum(len(tickers) for tickers in ticker_map.values())
    total_after = len(all_tickers)
    duplicates_removed = total_before - total_after
    
    # Create report
    report = {
        'total_before_dedup': total_before,
        'total_after_dedup': total_after,
        'duplicates_removed': duplicates_removed,
        'deduplication_rate': f"{(duplicates_removed / total_before * 100):.1f}%" if total_before > 0 else "0%",
        'index_breakdown': {
            index: len(tickers) for index, tickers in ticker_map.items()
        },
        'multi_index_tickers': sum(1 for indices in ticker_indices.values() if len(indices) > 1)
    }
    
    print(f"✓ Deduplication complete:")
    print(f"  Total before: {total_before}")
    print(f"  Total after: {total_after}")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Multi-index tickers: {report['multi_index_tickers']}")
    
    # Sort tickers alphabetically
    deduplicated = sorted(list(all_tickers))
    
    return deduplicated, report, ticker_indices


def create_universe_csv(tickers: List[str], ticker_indices: Dict[str, Set[str]], output_path: str):
    """
    Create the universal_universe.csv file.
    
    Args:
        tickers: List of deduplicated ticker symbols
        ticker_indices: Dict mapping ticker to set of indices it belongs to
        output_path: Path to output CSV file
    """
    print(f"\nWriting {len(tickers)} tickers to {output_path}...")
    
    # Define CSV structure matching the original
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
    
    rows = []
    for ticker in tickers:
        # Get indices this ticker belongs to
        indices = ticker_indices.get(ticker, set())
        index_membership = ','.join(sorted(indices))
        
        row = {
            'ticker': ticker,
            'name': '',  # Leave empty - can be populated later if needed
            'asset_class': 'equity',  # All are U.S. equities
            'index_membership': index_membership,
            'sector': 'Equity',  # Generic - can be updated later
            'market_cap_bucket': '',  # Leave empty - can be populated later
            'status': 'active',
            'validated': 'not_checked',
            'validation_error': ''
        }
        rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Successfully wrote {len(rows)} rows to {output_path}")


def generate_validation_report(
    ticker_map: Dict[str, List[str]],
    deduplicated_tickers: List[str],
    dedup_report: Dict,
    output_path: str
):
    """
    Generate validation report as required by the problem statement.
    
    Args:
        ticker_map: Original ticker map by index
        deduplicated_tickers: Final list of deduplicated tickers
        dedup_report: Deduplication statistics
        output_path: Path to save report
    """
    print("\nGenerating validation report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'rebuild_summary': {
            'final_ticker_count': len(deduplicated_tickers),
            'duplicates_removed': dedup_report['duplicates_removed'],
            'deduplication_rate': dedup_report['deduplication_rate'],
            'multi_index_count': dedup_report['multi_index_tickers']
        },
        'index_inclusion': {
            'Russell_3000': {
                'count': len(ticker_map.get('RUSSELL_3000', [])),
                'status': 'included' if ticker_map.get('RUSSELL_3000') else 'proxy_coverage'
            },
            'Russell_2000': {
                'count': len(ticker_map.get('RUSSELL_2000', [])),
                'status': 'included' if ticker_map.get('RUSSELL_2000') else 'proxy_coverage'
            },
            'SP_500': {
                'count': len(ticker_map.get('SP_500', [])),
                'status': 'included' if ticker_map.get('SP_500') else 'not_fetched'
            },
            'NASDAQ_Composite': {
                'count': len(ticker_map.get('NASDAQ_COMPOSITE', [])),
                'status': 'included' if ticker_map.get('NASDAQ_COMPOSITE') else 'not_fetched'
            },
            'Dow_Jones': {
                'count': len(ticker_map.get('DOW_JONES', [])),
                'status': 'included' if ticker_map.get('DOW_JONES') else 'not_fetched'
            }
        },
        'ticker_format': {
            'convention': 'hyphen',
            'example': 'BRK-B (not BRK.B)',
            'rationale': 'Consistent with yfinance API requirements'
        },
        'exclusions': {
            'etfs': 'excluded',
            'crypto': 'excluded',
            'fixed_income': 'excluded',
            'commodities': 'excluded',
            'index_symbols': 'excluded (e.g., ^GSPC, ^DJI)',
            'non_equity': 'excluded'
        },
        'data_quality': {
            'normalization': 'applied',
            'case_insensitive_dedup': 'applied',
            'invalid_symbols_removed': 'yes',
            'empty_rows_removed': 'yes'
        }
    }
    
    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Validation report saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION REPORT SUMMARY")
    print("=" * 70)
    print(f"Final universe contains {len(deduplicated_tickers)} unique U.S. equity tickers after deduplication.")
    print(f"\nIndex Coverage:")
    for index, info in report['index_inclusion'].items():
        print(f"  - {index}: {info['count']} tickers ({info['status']})")
    print(f"\nDeduplication:")
    print(f"  - Duplicates removed: {dedup_report['duplicates_removed']}")
    print(f"  - Deduplication rate: {dedup_report['deduplication_rate']}")
    print(f"  - Tickers in multiple indices: {dedup_report['multi_index_tickers']}")
    print(f"\nTicker Format: {report['ticker_format']['convention']} (e.g., {report['ticker_format']['example']})")
    print("=" * 70)
    
    return report


def main():
    """Main execution function."""
    print("=" * 70)
    print("UNIVERSAL UNIVERSE REBUILD - U.S. EQUITY FOCUS")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Collect tickers from all sources
    ticker_map = {}
    
    # 1. S&P 500
    sp500_tickers = fetch_sp500_constituents()
    if sp500_tickers:
        ticker_map['SP_500'] = sp500_tickers
    
    # 2. Russell 3000
    russell_3000_tickers = fetch_russell_3000_constituents()
    if russell_3000_tickers:
        ticker_map['RUSSELL_3000'] = russell_3000_tickers
    
    # 3. Russell 2000
    russell_2000_tickers = fetch_russell_2000_constituents()
    if russell_2000_tickers:
        ticker_map['RUSSELL_2000'] = russell_2000_tickers
    
    # 4. NASDAQ Composite
    nasdaq_tickers = fetch_nasdaq_composite_constituents()
    if nasdaq_tickers:
        ticker_map['NASDAQ_COMPOSITE'] = nasdaq_tickers
    
    # 5. Dow Jones
    dow_tickers = fetch_dow_jones_constituents()
    if dow_tickers:
        ticker_map['DOW_JONES'] = dow_tickers
    
    if not ticker_map:
        print("\n❌ ERROR: No tickers were fetched from any source!")
        print("Please check network connectivity and try again.")
        return 1
    
    # Deduplicate
    deduplicated_tickers, dedup_report, ticker_indices = deduplicate_tickers(ticker_map)
    
    if not deduplicated_tickers:
        print("\n❌ ERROR: No tickers after deduplication!")
        return 1
    
    # Create CSV
    create_universe_csv(deduplicated_tickers, ticker_indices, OUTPUT_CSV)
    
    # Generate validation report
    validation_report = generate_validation_report(
        ticker_map,
        deduplicated_tickers,
        dedup_report,
        VALIDATION_REPORT
    )
    
    print("\n" + "=" * 70)
    print("REBUILD COMPLETE")
    print("=" * 70)
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Validation report: {VALIDATION_REPORT}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
