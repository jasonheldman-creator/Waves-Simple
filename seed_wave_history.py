#!/usr/bin/env python3
"""
Seed Wave History Script - Automated Initial Data Seeding

This script generates synthetic historical data for waves that don't yet have
real data in wave_history.csv. Key features:

1. Idempotent: Can be run multiple times without creating duplicates
2. Marks synthetic data: Adds 'is_synthetic' column to track placeholder rows
3. Phase 6 ready: Generates data compatible with all analytics components
4. Minimal footprint: Small, realistic-looking returns for initialization

Usage:
    python seed_wave_history.py [--days DAYS] [--start-date YYYY-MM-DD]
    
    --days: Number of days of history to generate (default: 90)
    --start-date: Start date for synthetic data (default: 90 days before today)
    --dry-run: Show what would be done without modifying the file
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import List, Set, Tuple

import numpy as np
import pandas as pd

# Import wave registry
from waves_engine import (
    get_all_wave_ids,
    get_display_name_from_wave_id,
    WAVE_ID_REGISTRY
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Seed synthetic historical data for waves without real data'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Number of days of history to generate (default: 90)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for synthetic data (YYYY-MM-DD format). Default: DAYS before today'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without modifying the file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='wave_history.csv',
        help='Output CSV file path (default: wave_history.csv)'
    )
    return parser.parse_args()


def load_existing_wave_history(csv_path: str) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Load existing wave_history.csv and return DataFrame and set of wave_ids with data.
    
    Returns:
        Tuple of (DataFrame, Set of wave_ids that already have data)
    """
    if not os.path.exists(csv_path):
        # Create empty DataFrame with expected schema
        df = pd.DataFrame(columns=[
            'wave_id', 'display_name', 'date', 
            'portfolio_return', 'benchmark_return', 'is_synthetic'
        ])
        return df, set()
    
    df = pd.read_csv(csv_path)
    
    # Add is_synthetic column if it doesn't exist
    if 'is_synthetic' not in df.columns:
        df['is_synthetic'] = False
    
    # Get set of wave_ids that already have data
    existing_wave_ids = set(df['wave_id'].unique())
    
    return df, existing_wave_ids


def get_waves_needing_seeding(existing_wave_ids: Set[str]) -> List[str]:
    """
    Get list of wave_ids that need synthetic data seeding.
    
    Args:
        existing_wave_ids: Set of wave_ids that already have data
        
    Returns:
        List of wave_ids that need seeding
    """
    all_wave_ids = set(get_all_wave_ids())
    missing_wave_ids = all_wave_ids - existing_wave_ids
    return sorted(missing_wave_ids)


def generate_synthetic_returns(n_days: int, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic portfolio and benchmark returns.
    
    Creates realistic-looking daily returns with:
    - Small daily volatility (1-2% typical moves)
    - Slight positive drift (mimics long-term equity growth)
    - Some correlation between portfolio and benchmark
    
    Args:
        n_days: Number of days to generate
        seed: Random seed for reproducibility (uses hash of wave_id if None)
        
    Returns:
        Tuple of (portfolio_returns, benchmark_returns) as numpy arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Parameters for synthetic returns
    daily_vol = 0.012  # ~1.2% daily volatility (realistic for equities)
    daily_drift = 0.0003  # ~7.5% annualized return (252 trading days)
    correlation = 0.65  # Moderate correlation with benchmark
    
    # Generate correlated random returns
    # Portfolio returns
    portfolio_returns = np.random.normal(daily_drift, daily_vol, n_days)
    
    # Benchmark returns (correlated with portfolio)
    independent_noise = np.random.normal(0, daily_vol * 0.8, n_days)
    benchmark_returns = (correlation * portfolio_returns + 
                        (1 - correlation) * independent_noise)
    
    return portfolio_returns, benchmark_returns


def generate_business_days(start_date: datetime, n_days: int) -> List[datetime]:
    """
    Generate list of business days (Monday-Friday).
    
    Args:
        start_date: Starting date
        n_days: Number of business days to generate
        
    Returns:
        List of datetime objects for business days
    """
    business_days = []
    current_date = start_date
    
    while len(business_days) < n_days:
        # Skip weekends (5 = Saturday, 6 = Sunday)
        if current_date.weekday() < 5:
            business_days.append(current_date)
        current_date += timedelta(days=1)
    
    return business_days


def seed_wave_data(
    wave_id: str,
    start_date: datetime,
    n_days: int
) -> pd.DataFrame:
    """
    Generate synthetic historical data for a single wave.
    
    Args:
        wave_id: Canonical wave identifier
        start_date: Starting date for synthetic data
        n_days: Number of days to generate
        
    Returns:
        DataFrame with synthetic data for this wave
    """
    display_name = get_display_name_from_wave_id(wave_id)
    if not display_name:
        print(f"Warning: No display name found for wave_id: {wave_id}")
        display_name = wave_id.replace('_', ' ').title()
    
    # Generate business days
    dates = generate_business_days(start_date, n_days)
    
    # Generate synthetic returns (use hash of wave_id as seed for consistency)
    seed = hash(wave_id) % (2**31)
    portfolio_returns, benchmark_returns = generate_synthetic_returns(n_days, seed)
    
    # Create DataFrame
    rows = []
    for i, date in enumerate(dates):
        rows.append({
            'wave_id': wave_id,
            'display_name': display_name,
            'date': date.strftime('%Y-%m-%d'),
            'portfolio_return': portfolio_returns[i],
            'benchmark_return': benchmark_returns[i],
            'is_synthetic': True
        })
    
    return pd.DataFrame(rows)


def seed_all_missing_waves(
    existing_df: pd.DataFrame,
    start_date: datetime,
    n_days: int,
    dry_run: bool = False
) -> pd.DataFrame:
    """
    Seed synthetic data for all waves that don't have real data.
    
    Args:
        existing_df: Existing wave_history DataFrame
        start_date: Starting date for synthetic data
        n_days: Number of days to generate
        dry_run: If True, only show what would be done
        
    Returns:
        Combined DataFrame with existing and new synthetic data
    """
    existing_wave_ids = set(existing_df['wave_id'].unique()) if not existing_df.empty else set()
    waves_to_seed = get_waves_needing_seeding(existing_wave_ids)
    
    if not waves_to_seed:
        print("✅ All waves already have data. No seeding needed.")
        return existing_df
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Seeding {len(waves_to_seed)} waves:")
    print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to "
          f"{(start_date + timedelta(days=n_days*1.5)).strftime('%Y-%m-%d')} "
          f"({n_days} business days)")
    print()
    
    new_dfs = []
    for wave_id in waves_to_seed:
        display_name = get_display_name_from_wave_id(wave_id)
        print(f"  • {wave_id} ({display_name})")
        
        if not dry_run:
            wave_df = seed_wave_data(wave_id, start_date, n_days)
            new_dfs.append(wave_df)
    
    if dry_run:
        print(f"\n🔍 Dry run complete. No files modified.")
        return existing_df
    
    # Combine existing and new data
    if new_dfs:
        combined_df = pd.concat([existing_df] + new_dfs, ignore_index=True)
        
        # Sort by wave_id and date
        combined_df = combined_df.sort_values(['wave_id', 'date']).reset_index(drop=True)
        
        return combined_df
    
    return existing_df


def save_wave_history(df: pd.DataFrame, csv_path: str, create_backup: bool = True):
    """
    Save wave_history DataFrame to CSV, optionally creating a backup.
    
    Args:
        df: DataFrame to save
        csv_path: Output CSV file path
        create_backup: If True, create a backup of existing file
    """
    if create_backup and os.path.exists(csv_path):
        backup_path = f"{csv_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(csv_path, backup_path)
        print(f"\n✅ Created backup: {backup_path}")
    
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df)} rows to {csv_path}")
    
    # Print summary statistics
    if 'is_synthetic' in df.columns:
        n_synthetic = df['is_synthetic'].sum()
        n_real = len(df) - n_synthetic
        n_synthetic_waves = df[df['is_synthetic']]['wave_id'].nunique()
        n_real_waves = df[~df['is_synthetic']]['wave_id'].nunique() if n_real > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"  Real data: {n_real:,} rows across {n_real_waves} waves")
        print(f"  Synthetic data: {n_synthetic:,} rows across {n_synthetic_waves} waves")
        print(f"  Total: {len(df):,} rows across {df['wave_id'].nunique()} waves")


def main():
    """Main entry point for the seeding script."""
    args = parse_args()
    
    print("=" * 70)
    print("Wave History Seeding Script")
    print("=" * 70)
    
    # Determine start date
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print(f"❌ Error: Invalid date format '{args.start_date}'. Use YYYY-MM-DD")
            return 1
    else:
        # Default: start DAYS before today
        start_date = datetime.now() - timedelta(days=args.days * 1.5)  # Extra days for weekends
    
    # Load existing data
    print(f"\n📂 Loading existing data from {args.output}...")
    existing_df, existing_wave_ids = load_existing_wave_history(args.output)
    print(f"  Found {len(existing_df)} existing rows for {len(existing_wave_ids)} waves")
    
    # Seed missing waves
    combined_df = seed_all_missing_waves(
        existing_df,
        start_date,
        args.days,
        dry_run=args.dry_run
    )
    
    # Save results
    if not args.dry_run and not combined_df.equals(existing_df):
        save_wave_history(combined_df, args.output, create_backup=True)
        print(f"\n✅ Seeding complete!")
    elif args.dry_run:
        print(f"\n✅ Dry run complete. Run without --dry-run to apply changes.")
    else:
        print(f"\n✅ No changes needed.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
