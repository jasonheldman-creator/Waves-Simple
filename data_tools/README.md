# Data Tools - Master Universe Builder

This directory contains tools for building and maintaining the master universe of tradable assets.

## Overview

The master universe CSV (`/data/master_universe.csv`) is a comprehensive list of all assets tracked by the Waves system, including:
- **Equities**: Stocks from major indices (including Russell 3000 constituents)
- **ETFs**: Top ~50 exchange-traded funds across various categories
- **Cryptocurrencies**: Top 100-200 cryptocurrencies by market capitalization

## Files

- **`build_master_universe.py`**: Main script to regenerate the master universe CSV
- **`master_universe_dedupe_report.json`**: JSON report of deduplication process (generated)
- **`test_master_universe.py`**: Validation tests for the master universe

## Usage

### Building/Updating the Master Universe

To regenerate the master universe CSV:

```bash
python data_tools/build_master_universe.py
```

This will:
1. Load existing data from `Master_Stock_Sheet.csv` (for backward compatibility)
2. Fetch Russell 3000 constituents (if available)
3. Add a curated list of top ETFs
4. Fetch top 100-200 cryptocurrencies from CoinGecko API
5. Deduplicate entries based on asset class and symbol
6. Generate `/data/master_universe.csv`
7. Generate `/data_tools/master_universe_dedupe_report.json`

### Running Tests

To validate the master universe:

```bash
python data_tools/test_master_universe.py
```

This will verify:
- Required fields are present
- Row counts fall within expected ranges
- No duplicates exist
- All data types are valid

## Data Sources

### 1. Existing Master Stock Sheet
- **Source**: `Master_Stock_Sheet.csv` (existing file in repository)
- **Purpose**: Backward compatibility with existing data
- **Update Frequency**: Manual updates historically

### 2. Russell 3000 Constituents
- **Source**: Free/public APIs or cached data files
- **Purpose**: Comprehensive coverage of U.S. equities
- **Update Frequency**: Quarterly (following Russell reconstitution)
- **Note**: Currently not implemented; requires data provider or cached file

### 3. Top ETFs
- **Source**: Hardcoded curated list in `build_master_universe.py`
- **Purpose**: Coverage of major ETF categories
- **Count**: ~50 ETFs
- **Update Frequency**: Manual updates as needed
- **Categories Included**:
  - Broad Market ETFs (SPY, VOO, VTI, etc.)
  - Sector ETFs (XLK, XLF, XLE, etc.)
  - International ETFs (VEA, IEFA, VWO, etc.)
  - Bond ETFs (AGG, BND, TLT, etc.)
  - Specialized ETFs (ARKK, BITO, etc.)

### 4. Cryptocurrencies
- **Source**: CoinGecko API (https://api.coingecko.com/api/v3/coins/markets)
- **Purpose**: Top cryptocurrencies by market capitalization
- **Count**: 100-200 (configurable)
- **Update Frequency**: Can be updated daily/weekly as needed
- **Categorization**: Automatic categorization by crypto style:
  - Store of Value / Settlement (BTC, BCH, etc.)
  - Smart Contract Platforms Layer 1 (ETH, SOL, ADA, etc.)
  - Scaling Solutions Layer 2 (MATIC, ARB, OP, etc.)
  - DeFi / DEX (UNI, AAVE, etc.)
  - Oracles / Data (LINK, BAND, etc.)
  - Stablecoin Infrastructure (USDC, USDT, DAI, etc.)
  - Exchange Tokens (BNB, CRO, etc.)
  - Gaming / Metaverse (MANA, SAND, AXS, etc.)
  - Payments / Remittance (XRP, XLM, DOGE, etc.)
  - Yield/Staking (LDO, RPL, etc.)

## Deduplication Logic

The script applies intelligent deduplication to handle cases where the same asset appears in multiple sources.

### Normalization
- **Symbol**: Uppercase, trimmed
- **Asset Class**: Mapped to standardized categories (Equity, ETF, Cryptocurrency, etc.)

### Collision Detection
Assets are considered duplicates if they share the same:
- **Normalized Asset Class** AND
- **Normalized Symbol**

### Resolution Rules (Priority Order)

When duplicates are detected, the script applies these rules in order:

1. **Completeness**: Keep the entry with more non-empty fields
   - More complete data is preferred for better system functionality

2. **Market Value**: If completeness is equal, keep the entry with higher market value
   - Ensures we use the most significant/liquid version of the asset

3. **First Encountered**: If both rules tie, keep the first entry
   - Deterministic behavior for reproducibility

### Deduplication Report

The deduplication process generates `master_universe_dedupe_report.json` containing:
- **Timestamp**: When the build was run
- **Input/Output Counts**: Number of rows before and after deduplication
- **Collisions**: List of all duplicate detections with:
  - Asset class and symbol
  - Company names of both entries
  - Decision reason (which rule was applied)
- **Rules Applied**: Count of how many times each rule was used

Example report structure:
```json
{
  "timestamp": "2024-12-24T10:30:00.000000",
  "input_count": 4500,
  "output_count": 4200,
  "duplicates_removed": 300,
  "collisions": [
    {
      "asset_class": "Cryptocurrency",
      "symbol": "BTC",
      "existing_company": "Bitcoin",
      "new_company": "Bitcoin",
      "decision": "New entry more complete (6 vs 4 fields)"
    }
  ],
  "rules_applied": {
    "completeness": 150,
    "market_value": 100,
    "first_encountered": 50
  }
}
```

## Validation

The CI pipeline automatically validates the master universe after each build:

### Automated Checks
1. **Required Fields**: Ensures all rows have Ticker, Company, and Sector
2. **Row Count Ranges**: Verifies total count is within expected ranges (3000-5000)
3. **Zero Duplicates**: Confirms no duplicates exist after deduplication
4. **Data Types**: Validates numeric fields (Weight, MarketValue, Price)

### Manual Verification
Before committing changes, review:
- The deduplication report for unexpected collisions
- Row counts by asset class (Equity, ETF, Cryptocurrency)
- Any warnings or errors from the build script

## Maintenance

### Adding New ETFs
1. Edit `build_master_universe.py`
2. Add new ETF to the `TOP_ETFS` list
3. Run the build script
4. Verify in the output and commit

### Updating Crypto Count
1. Edit `build_master_universe.py`
2. Modify the `limit` parameter in the main() function's crypto_data call
3. Run the build script

### Adding Russell 3000 Data
Currently not implemented. To add:
1. Obtain Russell 3000 constituent data (CSV file or API)
2. Implement `get_russell_3000_data()` function
3. Test and verify deduplication works correctly

## Dependencies

Required Python packages:
- `requests>=2.31.0` - For CoinGecko API calls
- `csv` (built-in) - For CSV reading/writing
- `json` (built-in) - For report generation

## Troubleshooting

### "requests library not found"
Install dependencies: `pip install -r requirements.txt`

### "Error fetching crypto data"
- Check internet connectivity
- Verify CoinGecko API is accessible
- Check rate limits (CoinGecko free tier: 10-50 calls/minute)

### "Master_Stock_Sheet.csv not found"
- This is a warning, not an error
- The script will continue with other data sources
- For backward compatibility, ensure this file exists

### Unexpected duplicate count
- Review the deduplication report JSON
- Check if data sources have overlapping symbols
- Verify normalization logic is correct

## Future Enhancements

Potential improvements:
1. Implement Russell 3000 data fetching
2. Add data caching to reduce API calls
3. Support for additional asset classes (futures, options, etc.)
4. Automated scheduling for regular updates
5. Data quality scoring and validation
6. Historical versioning of master universe snapshots
