#!/usr/bin/env python3
"""
Master Universe Builder Script

This script builds a comprehensive master universe CSV file containing:
1. Russell 3000 constituents
2. Top ~50 ETFs
3. Top 100-200 cryptocurrencies

The script applies deduplication logic and generates a report.
"""

import json
import csv
import os
import sys
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

# Try importing requests, but make it optional for initial development
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: 'requests' library not found. API calls will be disabled.")


# Configuration
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_STOCK_SHEET = os.path.join(REPO_ROOT, "Master_Stock_Sheet.csv")
OUTPUT_CSV = os.path.join(REPO_ROOT, "data", "master_universe.csv")
DEDUPE_REPORT = os.path.join(REPO_ROOT, "data_tools", "master_universe_dedupe_report.json")

# Top ETFs list (curated)
TOP_ETFS = [
    {"Ticker": "SPY", "Company": "SPDR S&P 500 ETF Trust", "Sector": "Broad Market ETF"},
    {"Ticker": "VOO", "Company": "Vanguard S&P 500 ETF", "Sector": "Broad Market ETF"},
    {"Ticker": "IVV", "Company": "iShares Core S&P 500 ETF", "Sector": "Broad Market ETF"},
    {"Ticker": "VTI", "Company": "Vanguard Total Stock Market ETF", "Sector": "Broad Market ETF"},
    {"Ticker": "QQQ", "Company": "Invesco QQQ Trust", "Sector": "Technology ETF"},
    {"Ticker": "VEA", "Company": "Vanguard FTSE Developed Markets ETF", "Sector": "International ETF"},
    {"Ticker": "IEFA", "Company": "iShares Core MSCI EAFE ETF", "Sector": "International ETF"},
    {"Ticker": "AGG", "Company": "iShares Core U.S. Aggregate Bond ETF", "Sector": "Bond ETF"},
    {"Ticker": "VWO", "Company": "Vanguard FTSE Emerging Markets ETF", "Sector": "International ETF"},
    {"Ticker": "BND", "Company": "Vanguard Total Bond Market ETF", "Sector": "Bond ETF"},
    {"Ticker": "VUG", "Company": "Vanguard Growth ETF", "Sector": "Growth ETF"},
    {"Ticker": "VO", "Company": "Vanguard Mid-Cap ETF", "Sector": "Mid-Cap ETF"},
    {"Ticker": "IEMG", "Company": "iShares Core MSCI Emerging Markets ETF", "Sector": "International ETF"},
    {"Ticker": "VTV", "Company": "Vanguard Value ETF", "Sector": "Value ETF"},
    {"Ticker": "IJH", "Company": "iShares Core S&P Mid-Cap ETF", "Sector": "Mid-Cap ETF"},
    {"Ticker": "GLD", "Company": "SPDR Gold Trust", "Sector": "Commodity ETF"},
    {"Ticker": "VIG", "Company": "Vanguard Dividend Appreciation ETF", "Sector": "Dividend ETF"},
    {"Ticker": "SCHF", "Company": "Schwab International Equity ETF", "Sector": "International ETF"},
    {"Ticker": "VB", "Company": "Vanguard Small-Cap ETF", "Sector": "Small-Cap ETF"},
    {"Ticker": "EFA", "Company": "iShares MSCI EAFE ETF", "Sector": "International ETF"},
    {"Ticker": "IJR", "Company": "iShares Core S&P Small-Cap ETF", "Sector": "Small-Cap ETF"},
    {"Ticker": "VCIT", "Company": "Vanguard Intermediate-Term Corporate Bond ETF", "Sector": "Bond ETF"},
    {"Ticker": "VNQ", "Company": "Vanguard Real Estate ETF", "Sector": "Real Estate ETF"},
    {"Ticker": "XLK", "Company": "Technology Select Sector SPDR Fund", "Sector": "Technology ETF"},
    {"Ticker": "XLF", "Company": "Financial Select Sector SPDR Fund", "Sector": "Financial ETF"},
    {"Ticker": "XLE", "Company": "Energy Select Sector SPDR Fund", "Sector": "Energy ETF"},
    {"Ticker": "XLV", "Company": "Health Care Select Sector SPDR Fund", "Sector": "Healthcare ETF"},
    {"Ticker": "XLY", "Company": "Consumer Discretionary Select Sector SPDR Fund", "Sector": "Consumer ETF"},
    {"Ticker": "XLP", "Company": "Consumer Staples Select Sector SPDR Fund", "Sector": "Consumer ETF"},
    {"Ticker": "XLI", "Company": "Industrial Select Sector SPDR Fund", "Sector": "Industrial ETF"},
    {"Ticker": "XLU", "Company": "Utilities Select Sector SPDR Fund", "Sector": "Utilities ETF"},
    {"Ticker": "XLRE", "Company": "Real Estate Select Sector SPDR Fund", "Sector": "Real Estate ETF"},
    {"Ticker": "XLB", "Company": "Materials Select Sector SPDR Fund", "Sector": "Materials ETF"},
    {"Ticker": "TLT", "Company": "iShares 20+ Year Treasury Bond ETF", "Sector": "Bond ETF"},
    {"Ticker": "IWM", "Company": "iShares Russell 2000 ETF", "Sector": "Small-Cap ETF"},
    {"Ticker": "IWF", "Company": "iShares Russell 1000 Growth ETF", "Sector": "Growth ETF"},
    {"Ticker": "IWD", "Company": "iShares Russell 1000 Value ETF", "Sector": "Value ETF"},
    {"Ticker": "SHY", "Company": "iShares 1-3 Year Treasury Bond ETF", "Sector": "Bond ETF"},
    {"Ticker": "EEM", "Company": "iShares MSCI Emerging Markets ETF", "Sector": "International ETF"},
    {"Ticker": "LQD", "Company": "iShares iBoxx $ Investment Grade Corporate Bond ETF", "Sector": "Bond ETF"},
    {"Ticker": "HYG", "Company": "iShares iBoxx $ High Yield Corporate Bond ETF", "Sector": "Bond ETF"},
    {"Ticker": "MUB", "Company": "iShares National Muni Bond ETF", "Sector": "Municipal Bond ETF"},
    {"Ticker": "SHV", "Company": "iShares Short Treasury Bond ETF", "Sector": "Bond ETF"},
    {"Ticker": "GOVT", "Company": "iShares U.S. Treasury Bond ETF", "Sector": "Bond ETF"},
    {"Ticker": "USMV", "Company": "iShares MSCI USA Min Vol Factor ETF", "Sector": "Low Volatility ETF"},
    {"Ticker": "SCHD", "Company": "Schwab U.S. Dividend Equity ETF", "Sector": "Dividend ETF"},
    {"Ticker": "DIA", "Company": "SPDR Dow Jones Industrial Average ETF Trust", "Sector": "Broad Market ETF"},
    {"Ticker": "RSP", "Company": "Invesco S&P 500 Equal Weight ETF", "Sector": "Broad Market ETF"},
    {"Ticker": "ARKK", "Company": "ARK Innovation ETF", "Sector": "Innovation ETF"},
    {"Ticker": "BITO", "Company": "ProShares Bitcoin Strategy ETF", "Sector": "Cryptocurrency ETF"},
]


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol for comparison."""
    if not symbol:
        return ""
    return symbol.strip().upper()


def normalize_asset_class(sector: str) -> str:
    """
    Normalize asset class/sector for deduplication.
    Maps various sector descriptions to standardized asset classes.
    """
    if not sector:
        return "Unknown"
    
    sector_lower = sector.lower()
    
    # Cryptocurrency categories
    crypto_keywords = ["cryptocurrency", "crypto", "store of value", "settlement", "smart contract", "layer 1", "layer 2",
                       "scaling", "defi", "exchange", "oracle", "gaming", "metaverse",
                       "payments", "remittance", "yield", "staking", "stablecoin", "nft"]
    if any(kw in sector_lower for kw in crypto_keywords):
        return "Cryptocurrency"
    
    # ETF categories
    if "etf" in sector_lower:
        return "ETF"
    
    # Stock/Equity categories (default)
    return "Equity"


def get_russell_3000_data() -> List[Dict[str, Any]]:
    """
    Fetch Russell 3000 constituents.
    
    For now, returns empty list. In production, this could:
    1. Call a free API (if available)
    2. Load from a cached/static file
    3. Use a paid data provider
    """
    print("Note: Russell 3000 data fetching not yet implemented.")
    print("      This would require a data provider or cached file.")
    return []


def get_crypto_data(limit: int = 200) -> List[Dict[str, Any]]:
    """
    Fetch top cryptocurrencies from CoinGecko API.
    
    Args:
        limit: Number of top cryptocurrencies to fetch (100-200)
    
    Returns:
        List of cryptocurrency data dictionaries
    """
    if not HAS_REQUESTS:
        print("Warning: Cannot fetch crypto data without 'requests' library.")
        return []
    
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": min(limit, 250),  # CoinGecko max is 250
            "page": 1,
            "sparkline": False,
            "locale": "en"
        }
        
        print(f"Fetching top {limit} cryptocurrencies from CoinGecko...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        coins = response.json()
        
        crypto_list = []
        for coin in coins:
            # Determine crypto style/category
            crypto_style = categorize_crypto(coin.get('symbol', '').upper(), coin.get('name', ''))
            
            crypto_list.append({
                "Ticker": coin.get('symbol', '').upper(),
                "Company": coin.get('name', ''),
                "Weight": 0.0,  # To be calculated later if needed
                "Sector": crypto_style,
                "MarketValue": coin.get('market_cap', 0),
                "Price": coin.get('current_price', 0)
            })
        
        print(f"Successfully fetched {len(crypto_list)} cryptocurrencies.")
        return crypto_list
        
    except Exception as e:
        print(f"Error fetching crypto data: {e}")
        return []


def categorize_crypto(symbol: str, name: str) -> str:
    """
    Categorize cryptocurrency by symbol and name.
    Returns crypto_style categorization.
    """
    symbol_upper = symbol.upper()
    name_lower = name.lower()
    
    # Store of Value / Settlement
    if symbol_upper in ["BTC", "BCH", "BSV", "LTC"]:
        return "Store of Value / Settlement"
    
    # Smart Contract Platforms (Layer 1)
    l1_symbols = ["ETH", "SOL", "ADA", "AVAX", "DOT", "NEAR", "ALGO", "XTZ", "FTM",
                  "EGLD", "ONE", "CSPR", "TON", "TRX", "SUI", "HBAR", "ETC", "KAS", "APT", "ZIL"]
    if symbol_upper in l1_symbols or "platform" in name_lower or "blockchain" in name_lower:
        return "Smart Contract Platforms (Layer 1)"
    
    # Scaling Solutions (Layer 2)
    l2_symbols = ["MATIC", "POL", "ARB", "OP", "MNT", "STX", "LRC", "IMX"]
    if symbol_upper in l2_symbols or "layer 2" in name_lower or "rollup" in name_lower:
        return "Scaling Solutions (Layer 2)"
    
    # DeFi
    defi_symbols = ["UNI", "AAVE", "MKR", "SNX", "CRV", "COMP", "SUSHI", "YFI", "BAL",
                    "CAKE", "1INCH", "ALPHA", "DYDX", "GMX"]
    if symbol_upper in defi_symbols or "defi" in name_lower or "swap" in name_lower:
        return "DeFi / DEX"
    
    # Oracles
    if symbol_upper in ["LINK", "BAND", "TRB", "API3"]:
        return "Oracles / Data"
    
    # Stablecoins
    if symbol_upper in ["USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "USDD", "FRAX"]:
        return "Stablecoin Infrastructure"
    
    # Exchange Tokens
    if symbol_upper in ["BNB", "CRO", "FTT", "HT", "OKB", "KCS", "LEO"]:
        return "Exchange Tokens"
    
    # Gaming / Metaverse
    gaming_symbols = ["MANA", "SAND", "AXS", "GALA", "ENJ", "IMX", "THETA", "ROSE",
                      "WAXP", "ALICE", "TLM", "ILV", "NFTX", "RARI", "WHALE"]
    if symbol_upper in gaming_symbols or "gaming" in name_lower or "metaverse" in name_lower or "nft" in name_lower:
        return "Gaming / Metaverse"
    
    # Payments / Remittance
    payment_symbols = ["XRP", "XLM", "DOGE", "SHIB", "XMR", "DASH", "CELO", "AMP", "ACH", "CRPT", "COTI", "REQ", "MTL"]
    if symbol_upper in payment_symbols or "payment" in name_lower:
        return "Payments / Remittance"
    
    # Yield/Staking
    if symbol_upper in ["LDO", "RPL"] or "staking" in name_lower or "liquid" in name_lower:
        return "Yield/Staking"
    
    # Default to general cryptocurrency
    return "Cryptocurrency"


def load_existing_master_stock_sheet() -> List[Dict[str, Any]]:
    """Load existing Master_Stock_Sheet.csv for backward compatibility."""
    if not os.path.exists(MASTER_STOCK_SHEET):
        print(f"Warning: {MASTER_STOCK_SHEET} not found.")
        return []
    
    try:
        data = []
        with open(MASTER_STOCK_SHEET, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        print(f"Loaded {len(data)} rows from existing Master_Stock_Sheet.csv")
        return data
    except Exception as e:
        print(f"Error loading Master_Stock_Sheet.csv: {e}")
        return []


def deduplicate_data(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deduplicate data based on normalized asset_class + symbol.
    
    Collision rules (in priority order):
    1. Keep entry with more complete data (more non-empty fields)
    2. Keep entry with higher market value
    3. Keep first entry encountered
    
    Returns:
        Tuple of (deduplicated_data, report_dict)
    """
    print("\nDeduplicating data...")
    
    # Track for reporting
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_count": len(data),
        "collisions": [],
        "rules_applied": {
            "completeness": 0,
            "market_value": 0,
            "first_encountered": 0
        }
    }
    
    # Map: (normalized_asset_class, normalized_symbol) -> dict
    unique_map = {}
    
    for row in data:
        symbol = row.get("Ticker", "")
        sector = row.get("Sector", "")
        
        norm_symbol = normalize_symbol(symbol)
        norm_asset_class = normalize_asset_class(sector)
        
        if not norm_symbol:
            continue  # Skip rows without symbol
        
        key = (norm_asset_class, norm_symbol)
        
        if key not in unique_map:
            unique_map[key] = row
        else:
            # Collision detected - apply rules
            existing = unique_map[key]
            
            # Count non-empty fields
            existing_completeness = sum(1 for v in existing.values() if v and str(v).strip())
            new_completeness = sum(1 for v in row.values() if v and str(v).strip())
            
            decision_reason = ""
            
            if new_completeness > existing_completeness:
                unique_map[key] = row
                report["rules_applied"]["completeness"] += 1
                decision_reason = f"New entry more complete ({new_completeness} vs {existing_completeness} fields)"
            elif new_completeness == existing_completeness:
                # Compare market values
                try:
                    existing_mv = float(existing.get("MarketValue", 0) or 0)
                    new_mv = float(row.get("MarketValue", 0) or 0)
                    
                    if new_mv > existing_mv:
                        unique_map[key] = row
                        report["rules_applied"]["market_value"] += 1
                        decision_reason = f"New entry has higher market value ({new_mv} vs {existing_mv})"
                    else:
                        report["rules_applied"]["first_encountered"] += 1
                        decision_reason = f"Kept first entry (market value tie or existing higher)"
                except (ValueError, TypeError):
                    report["rules_applied"]["first_encountered"] += 1
                    decision_reason = "Kept first entry (market value comparison failed)"
            else:
                report["rules_applied"]["completeness"] += 1
                decision_reason = f"Existing entry more complete ({existing_completeness} vs {new_completeness} fields)"
            
            # Record collision
            report["collisions"].append({
                "asset_class": norm_asset_class,
                "symbol": norm_symbol,
                "existing_company": existing.get("Company", ""),
                "new_company": row.get("Company", ""),
                "decision": decision_reason
            })
    
    deduplicated = list(unique_map.values())
    report["output_count"] = len(deduplicated)
    report["duplicates_removed"] = report["input_count"] - report["output_count"]
    
    print(f"Input rows: {report['input_count']}")
    print(f"Output rows: {report['output_count']}")
    print(f"Duplicates removed: {report['duplicates_removed']}")
    print(f"Collisions detected: {len(report['collisions'])}")
    
    return deduplicated, report


def write_csv(data: List[Dict[str, Any]], output_path: str):
    """Write data to CSV file."""
    if not data:
        print(f"Warning: No data to write to {output_path}")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Clean up data - ensure all rows have required fields
    for row in data:
        # Ensure Ticker is present
        if not row.get("Ticker", "").strip():
            continue  # Skip rows without ticker
        
        # Ensure Sector is present (required field)
        if not row.get("Sector", "").strip():
            # Infer from other fields or set default
            row["Sector"] = "Equity"  # Default to Equity for stocks
    
    # Get all unique field names (preserve column order)
    fieldnames = ["Ticker", "Company", "Weight", "Sector", "MarketValue", "Price"]
    
    # Add any additional fields from data
    for row in data:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\nWrote {len(data)} rows to {output_path}")


def write_report(report: Dict[str, Any], report_path: str):
    """Write deduplication report to JSON file."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"Wrote deduplication report to {report_path}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Master Universe Builder")
    print("=" * 70)
    
    # Collect all data sources
    all_data = []
    
    # 1. Load existing Master_Stock_Sheet.csv
    print("\n[1/4] Loading existing master stock sheet...")
    existing_data = load_existing_master_stock_sheet()
    all_data.extend(existing_data)
    
    # 2. Add Russell 3000 constituents
    print("\n[2/4] Fetching Russell 3000 constituents...")
    russell_data = get_russell_3000_data()
    all_data.extend(russell_data)
    
    # 3. Add top ETFs
    print("\n[3/4] Adding top ETFs...")
    all_data.extend(TOP_ETFS)
    print(f"Added {len(TOP_ETFS)} ETFs")
    
    # 4. Add cryptocurrencies
    print("\n[4/4] Fetching cryptocurrency data...")
    crypto_data = get_crypto_data(limit=200)
    all_data.extend(crypto_data)
    
    # Deduplicate
    deduplicated_data, report = deduplicate_data(all_data)
    
    # Write outputs
    write_csv(deduplicated_data, OUTPUT_CSV)
    write_report(report, DEDUPE_REPORT)
    
    print("\n" + "=" * 70)
    print("Build Complete!")
    print("=" * 70)
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"Dedupe Report: {DEDUPE_REPORT}")
    print(f"Total unique assets: {len(deduplicated_data)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
