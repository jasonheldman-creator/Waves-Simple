#!/usr/bin/env python3
"""
Merge crypto universe expansion into universal_universe.csv

This script:
1. Loads the current universal_universe.csv
2. Loads the crypto_universe_expansion.csv
3. Removes existing crypto entries from expansion (avoids duplicates)
4. Appends new crypto entries to universal universe
5. Saves the updated universal_universe.csv
"""

import pandas as pd
import os

def merge_crypto_expansion():
    """Merge crypto expansion into universal universe."""
    
    # File paths
    universe_file = 'universal_universe.csv'
    expansion_file = 'crypto_universe_expansion.csv'
    backup_file = 'universal_universe.csv.backup'
    
    # Load current universe
    print(f"Loading {universe_file}...")
    universe_df = pd.read_csv(universe_file)
    print(f"  Current total: {len(universe_df)} assets")
    print(f"  Current crypto: {len(universe_df[universe_df['asset_class'] == 'crypto'])} assets")
    
    # Backup current universe
    print(f"\nCreating backup: {backup_file}")
    universe_df.to_csv(backup_file, index=False)
    
    # Load expansion
    print(f"\nLoading {expansion_file}...")
    expansion_df = pd.read_csv(expansion_file)
    print(f"  Expansion crypto: {len(expansion_df)} assets")
    
    # Get existing crypto tickers (case insensitive)
    existing_crypto_tickers = set(
        universe_df[universe_df['asset_class'] == 'crypto']['ticker'].str.upper().tolist()
    )
    
    # Filter expansion to only new tickers
    new_crypto_mask = ~expansion_df['ticker'].str.upper().isin(existing_crypto_tickers)
    new_crypto_df = expansion_df[new_crypto_mask].copy()
    
    print(f"\nFiltering expansion:")
    print(f"  Duplicate tickers removed: {len(expansion_df) - len(new_crypto_df)}")
    print(f"  New crypto tickers to add: {len(new_crypto_df)}")
    
    # Remove old crypto entries from universe (we'll add them back from expansion)
    # This ensures we have the latest metadata for existing crypto assets
    non_crypto_df = universe_df[universe_df['asset_class'] != 'crypto'].copy()
    
    # Get updated entries for existing crypto from expansion
    existing_crypto_mask = expansion_df['ticker'].str.upper().isin(existing_crypto_tickers)
    updated_crypto_df = expansion_df[existing_crypto_mask].copy()
    
    print(f"\nUpdating existing crypto entries: {len(updated_crypto_df)}")
    
    # Combine: non-crypto assets + all crypto (updated + new)
    merged_df = pd.concat([
        non_crypto_df,
        updated_crypto_df,
        new_crypto_df
    ], ignore_index=True)
    
    # Sort by asset class, then ticker
    merged_df = merged_df.sort_values(['asset_class', 'ticker']).reset_index(drop=True)
    
    print(f"\nMerged universe:")
    print(f"  Total assets: {len(merged_df)}")
    print(f"  Crypto assets: {len(merged_df[merged_df['asset_class'] == 'crypto'])}")
    print(f"  Asset class breakdown:")
    for asset_class, count in merged_df['asset_class'].value_counts().items():
        print(f"    {asset_class}: {count}")
    
    # Save merged universe
    print(f"\nSaving updated {universe_file}...")
    merged_df.to_csv(universe_file, index=False)
    
    print("\n✓ Crypto universe expansion complete!")
    print(f"  Added {len(new_crypto_df)} new crypto assets")
    print(f"  Updated {len(updated_crypto_df)} existing crypto assets")
    print(f"  Total crypto assets now: {len(merged_df[merged_df['asset_class'] == 'crypto'])}")
    
    return merged_df


if __name__ == '__main__':
    merge_crypto_expansion()
