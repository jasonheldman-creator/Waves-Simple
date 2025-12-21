# WAVES Intelligence™ Crypto Asset Expansion

## Overview

This document describes the expansion of the WAVES Intelligence™ master asset list (`Master_Stock_Sheet.csv`) to include **149 cryptocurrency assets**, bringing the total investable universe to 169 assets (20 original equities + 149 crypto assets).

## Purpose

Expand the WAVES Intelligence™ master asset list CSV file to include the top ~100–200 cryptocurrencies. This update ensures that native crypto assets are treated as first-class investable instruments with proper sector classification, allowing for:

- Deeper allocation insights
- Broader governance application in Crypto Waves
- Cross-asset regime analysis
- Native attribution and WaveScore computation

## Implementation Details

### Total Assets Added: 149 Cryptocurrencies

The expansion includes cryptocurrencies selected based on:
- Market capitalization (top ~150 by market cap as of December 2025)
- Liquidity and trading volume
- Sector representation across the crypto ecosystem
- Exclusion of: microcaps, illiquid tokens, deprecated assets, and stablecoins (except infrastructure-related)

### Data Structure

Each cryptocurrency asset includes the following attributes in the CSV:
- **Ticker**: Crypto symbol (e.g., BTC, ETH, SOL)
- **Company**: Full name (e.g., Bitcoin, Ethereum, Solana)
- **Weight**: Allocation weight based on market capitalization
- **Sector**: Classification based on the sector framework (see below)
- **MarketValue**: Current market capitalization in USD
- **Price**: Current price in USD

### Sector Classification Framework

The crypto assets are classified into **10 distinct sectors** following institutional crypto classification standards:

#### 1. Store of Value / Settlement (3 assets)
Purpose: Digital gold, settlement layer, store of value
- **Examples**: BTC, BCH, BSV
- **Characteristics**: Proven security, limited supply, established networks

#### 2. Smart Contract Platforms (Layer 1) (19 assets)
Purpose: Base layer blockchains with smart contract functionality
- **Examples**: ETH, SOL, ADA, AVAX, NEAR, TON, APT
- **Characteristics**: Programmable, support dApps, native consensus mechanisms

#### 3. Scaling Solutions (Layer 2) (13 assets)
Purpose: Enhance Layer 1 scalability, reduce costs, increase throughput
- **Examples**: ARB, OP, POL, MNT, STX, IMX
- **Characteristics**: Optimistic or zero-knowledge rollups, off-chain computation

#### 4. Decentralized Finance (DeFi) (22 assets)
Purpose: Peer-to-peer financial services, DEXs, lending, derivatives
- **Examples**: UNI, AAVE, CAKE, CRV, INJ, GMX, COMP
- **Characteristics**: Permissionless, composable, automated market makers

#### 5. Infrastructure (36 assets)
Purpose: Oracles, data feeds, interoperability, storage, tooling
- **Examples**: LINK, DOT, BNB, FIL, GRT, ATOM, VET
- **Characteristics**: Enable ecosystem growth, cross-chain communication, essential services

#### 6. AI / Compute / Data (11 assets)
Purpose: Decentralized AI, computing resources, data verification
- **Examples**: TAO, RENDER, FET, ICP, OCEAN, AGIX
- **Characteristics**: Machine learning, GPU rendering, decentralized compute networks

#### 7. Gaming / Metaverse (28 assets)
Purpose: Play-to-earn, NFTs, virtual worlds, digital property
- **Examples**: MANA, SAND, AXS, GALA, IMX, ENJ
- **Characteristics**: In-game economies, NFT creation, metaverse platforms

#### 8. Payments / Remittance (14 assets)
Purpose: Medium of exchange, cross-border payments, digital cash
- **Examples**: XRP, DOGE, XLM, LTC, XMR, DASH
- **Characteristics**: Fast settlement, low fees, privacy features

#### 9. Yield/Staking (2 assets)
Purpose: Income generation, liquid staking derivatives
- **Examples**: LDO, RPL
- **Characteristics**: Staking rewards, yield optimization

#### 10. Stablecoin Infrastructure (1 asset)
Purpose: Stablecoin-related infrastructure (not stablecoins themselves)
- **Examples**: USDC (infrastructure token)
- **Note**: Pure stablecoins excluded per requirements

## Sector Distribution

```
Infrastructure                         36 assets
Gaming / Metaverse                     28 assets
Decentralized Finance (DeFi)           22 assets
Smart Contract Platforms (Layer 1)     19 assets
Payments / Remittance                  14 assets
Scaling Solutions (Layer 2)            13 assets
AI / Compute / Data                    11 assets
Store of Value / Settlement             3 assets
Yield/Staking                           2 assets
Stablecoin Infrastructure               1 asset
```

## Top 10 Crypto Assets by Weight

1. **BTC** - Bitcoin (Store of Value)
2. **ETH** - Ethereum (Smart Contract Platform L1)
3. **BNB** - Binance Coin (Infrastructure)
4. **XRP** - XRP (Payments)
5. **SOL** - Solana (Smart Contract Platform L1)
6. **USDC** - USD Coin (Stablecoin Infrastructure)
7. **TRX** - TRON (Smart Contract Platform L1)
8. **DOGE** - Dogecoin (Payments)
9. **ADA** - Cardano (Smart Contract Platform L1)
10. **BCH** - Bitcoin Cash (Store of Value)

## Design Principles

### No Engine Changes
- **Zero modifications** to engine logic, benchmarks, or core algorithms
- Crypto assets integrated purely through data expansion
- Attribution, WaveScore, and governance systems naturally accommodate new instruments

### Backward Compatibility
- Existing equity assets remain unchanged
- Load universe functionality (`load_universe.py`) works seamlessly
- Same CSV structure maintained: `Ticker, Company, Weight, Sector, MarketValue, Price`

### Asset Class Identification
- Crypto assets distinguished by their sector classifications
- All crypto sectors are unique and non-overlapping with equity sectors
- Easy filtering: `df[df['Sector'].str.contains('Layer|DeFi|Gaming|AI|Infrastructure')]`

## Cross-Asset Analysis Support

The expanded universe enables:

1. **Allocation Insights**: Compare allocation across traditional equities vs. crypto sectors
2. **Regime Analysis**: Consistent cross-asset regime detection
3. **Attribution**: Native crypto attribution alongside equity attribution
4. **Governance**: Full governance layer support for crypto waves
5. **WaveScore**: Native WaveScore computation for crypto assets

## Validation

All changes validated through:
- ✅ Data integrity checks (no NaN values in critical columns)
- ✅ Load universe compatibility
- ✅ Sector classification completeness
- ✅ Weight normalization
- ✅ 149 crypto assets successfully loaded

## Usage Example

```python
from load_universe import load_universe

# Load full universe (equities + crypto)
df = load_universe()

# Filter crypto assets
crypto_sectors = [
    'Store of Value / Settlement',
    'Smart Contract Platforms (Layer 1)',
    'Scaling Solutions (Layer 2)',
    'Decentralized Finance (DeFi)',
    'Infrastructure',
    'AI / Compute / Data',
    'Gaming / Metaverse',
    'Payments / Remittance',
    'Yield/Staking',
    'Stablecoin Infrastructure'
]
crypto_df = df[df['Sector'].isin(crypto_sectors)]

# Analyze by sector
print(crypto_df['Sector'].value_counts())

# Get specific assets
btc = df[df['Ticker'] == 'BTC']
eth = df[df['Ticker'] == 'ETH']
```

## Future Considerations

- Market cap and price data should be updated regularly
- Weights may need periodic rebalancing
- New crypto sectors may emerge (e.g., zkVM, social protocols)
- Asset additions/removals should follow liquidity and market cap criteria

## References

- Market data sourced from CoinMarketCap, CoinGecko (December 2025)
- Sector framework based on:
  - Grayscale Crypto Classification Framework
  - 21Shares Global Crypto Classification Standard
  - MarketVector Digital Asset Classification

---

**Last Updated**: December 2025  
**Total Assets**: 169 (20 equities + 149 crypto)  
**Status**: ✅ Production Ready
