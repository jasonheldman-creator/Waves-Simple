#!/usr/bin/env python3
"""
Wave History Seeding Script

Creates synthetic wave history data for bootstrap purposes.
Designed to be idempotent and deterministic.

Usage:
    python scripts/seed_wave_history.py --days 60 --mode Standard
    python scripts/seed_wave_history.py --days 90 --mode "Private Logic"
    python scripts/seed_wave_history.py --help
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import waves_engine
sys.path.insert(0, str(Path(__file__).parent.parent))

from waves_engine import get_all_wave_ids, get_display_name_from_wave_id


# Configuration constants
SEED_VERSION = "seed_v1"
DEFAULT_DAYS = 60
DEFAULT_MODE = "Standard"
SKIP_THRESHOLD = 30  # Skip seeding if wave_id already has >= this many rows
INITIAL_NAV = 100.0
BUSINESS_DAYS_PER_WEEK = 5
RNG_SEED = 42  # For deterministic data generation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Seed wave_history.csv with synthetic data for bootstrap purposes"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Number of business days to generate (default: {DEFAULT_DAYS})"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help=f"Wave mode (default: {DEFAULT_MODE})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: wave_history.csv in repo root)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate data even if wave_id has existing rows"
    )
    return parser.parse_args()


def get_business_dates(num_days):
    """
    Generate list of business dates (approximately).
    
    Args:
        num_days: Number of business days to generate
        
    Returns:
        List of datetime.date objects in ascending order
    """
    dates = []
    current_date = datetime.now().date()
    days_added = 0
    offset = 0
    
    while days_added < num_days:
        candidate = current_date - timedelta(days=offset)
        # Simple business day approximation (skip weekends)
        if candidate.weekday() < 5:  # Monday=0, Friday=4
            dates.append(candidate)
            days_added += 1
        offset += 1
    
    # Return in ascending order (oldest first)
    return sorted(dates)


def generate_synthetic_returns(num_days, wave_id, rng):
    """
    Generate synthetic returns with realistic noise.
    
    Args:
        num_days: Number of days to generate
        wave_id: Wave identifier for seeding variation
        rng: Random number generator
        
    Returns:
        Tuple of (wave_returns, bm_returns) as numpy arrays
    """
    # Use wave_id hash for reproducible but varied returns per wave
    wave_seed = hash(wave_id) % (2**31)
    rng_wave = np.random.RandomState(wave_seed)
    
    # Generate benchmark returns first
    # Approximate market: ~10% annual return, ~16% annual volatility
    # Daily: ~0.04% mean, ~1% std dev
    bm_mean = 0.0004
    bm_std = 0.01
    bm_returns = rng_wave.normal(bm_mean, bm_std, num_days)
    
    # Generate wave returns with small alpha
    # Wave should be similar to benchmark with small tracking error
    # Alpha should be close to 0 to avoid large synthetic alpha values
    alpha_mean = 0.00  # No systematic alpha
    alpha_std = 0.003  # Small tracking error (~5% annual)
    
    alpha_noise = rng_wave.normal(alpha_mean, alpha_std, num_days)
    wave_returns = bm_returns + alpha_noise
    
    return wave_returns, bm_returns


def generate_wave_data(wave_id, display_name, mode, dates, rng):
    """
    Generate synthetic data for a single wave.
    
    Args:
        wave_id: Wave identifier
        display_name: Human-readable wave name
        mode: Wave mode (e.g., "Standard", "Private Logic")
        dates: List of dates to generate data for
        rng: Random number generator
        
    Returns:
        List of dictionaries with wave data
    """
    num_days = len(dates)
    wave_returns, bm_returns = generate_synthetic_returns(num_days, wave_id, rng)
    
    # Calculate NAVs starting from INITIAL_NAV
    wave_nav = INITIAL_NAV
    bm_nav = INITIAL_NAV
    
    rows = []
    for i, date in enumerate(dates):
        # Apply returns to NAVs
        wave_nav *= (1 + wave_returns[i])
        bm_nav *= (1 + bm_returns[i])
        
        # Calculate alpha as difference in returns
        alpha = wave_returns[i] - bm_returns[i]
        
        rows.append({
            'date': date.isoformat(),
            'wave_id': wave_id,
            'display_name': display_name,
            'mode': mode,
            'wave_nav': round(wave_nav, 4),
            'bm_nav': round(bm_nav, 4),
            'wave_return': round(wave_returns[i], 8),
            'bm_return': round(bm_returns[i], 8),
            'alpha': round(alpha, 8),
            'is_synthetic': True,
            'seed_version': SEED_VERSION,
            # Add legacy columns for backward compatibility
            'portfolio_return': round(wave_returns[i], 8),
            'benchmark_return': round(bm_returns[i], 8),
        })
    
    return rows


def load_existing_history(file_path):
    """
    Load existing wave_history.csv if it exists.
    
    Args:
        file_path: Path to wave_history.csv
        
    Returns:
        DataFrame or None if file doesn't exist or is empty
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing file: {e}")
        return None


def get_existing_row_counts(df):
    """
    Count existing rows per wave_id.
    
    Args:
        df: Existing wave history DataFrame
        
    Returns:
        Dictionary mapping wave_id to row count
    """
    if df is None:
        return {}
    
    if 'wave_id' not in df.columns:
        return {}
    
    return df['wave_id'].value_counts().to_dict()


def seed_wave_history(days, mode, output_path=None, force=False):
    """
    Main seeding function.
    
    Args:
        days: Number of business days to generate
        mode: Wave mode
        output_path: Output file path (default: wave_history.csv)
        force: Force regenerate even if rows exist
        
    Returns:
        Dictionary with seeding statistics
    """
    # Determine output path
    if output_path is None:
        repo_root = Path(__file__).parent.parent
        output_path = repo_root / "wave_history.csv"
    else:
        output_path = Path(output_path)
    
    print(f"🌊 Wave History Seeding Script")
    print(f"=" * 60)
    print(f"Parameters:")
    print(f"  Days: {days}")
    print(f"  Mode: {mode}")
    print(f"  Output: {output_path}")
    print(f"  Force: {force}")
    print(f"  Seed version: {SEED_VERSION}")
    print(f"=" * 60)
    
    # Load wave registry
    print("\n📋 Loading wave registry...")
    wave_ids = get_all_wave_ids()
    print(f"   Found {len(wave_ids)} wave_ids")
    
    # Load existing data
    print(f"\n📂 Checking existing data...")
    existing_df = load_existing_history(output_path)
    existing_counts = get_existing_row_counts(existing_df)
    
    if existing_df is not None:
        print(f"   Existing file has {len(existing_df)} rows")
        print(f"   Covering {len(existing_counts)} unique wave_ids")
    else:
        print(f"   No existing file found")
    
    # Generate dates
    print(f"\n📅 Generating {days} business dates...")
    dates = get_business_dates(days)
    print(f"   Date range: {dates[0]} to {dates[-1]}")
    
    # Initialize RNG for deterministic data
    rng = np.random.RandomState(RNG_SEED)
    
    # Generate data for each wave
    print(f"\n🔄 Generating synthetic data...")
    all_rows = []
    waves_seeded = 0
    waves_skipped = 0
    
    for wave_id in wave_ids:
        display_name = get_display_name_from_wave_id(wave_id)
        
        # Check if we should skip this wave
        existing_count = existing_counts.get(wave_id, 0)
        if not force and existing_count >= SKIP_THRESHOLD:
            print(f"   ⏭️  Skipping {wave_id}: already has {existing_count} rows (>= {SKIP_THRESHOLD})")
            waves_skipped += 1
            continue
        
        print(f"   ✨ Generating {wave_id} ({display_name})")
        wave_data = generate_wave_data(wave_id, display_name, mode, dates, rng)
        all_rows.extend(wave_data)
        waves_seeded += 1
    
    # Create DataFrame from new data
    new_df = pd.DataFrame(all_rows)
    
    # Merge with existing data
    print(f"\n🔗 Merging with existing data...")
    if existing_df is not None:
        # Create unique key for deduplication
        key_cols = ['date', 'wave_id', 'mode']
        
        # Add mode column to existing data if missing
        if 'mode' not in existing_df.columns:
            existing_df['mode'] = 'Standard'
        
        # Combine dataframes
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates, keeping first occurrence (existing data takes precedence)
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=key_cols, keep='first')
        after_dedup = len(combined_df)
        duplicates_removed = before_dedup - after_dedup
        
        print(f"   Removed {duplicates_removed} duplicate rows")
        final_df = combined_df
    else:
        final_df = new_df
    
    # Sort by wave_id and date
    final_df = final_df.sort_values(['wave_id', 'date'])
    
    # Write output
    print(f"\n💾 Writing to {output_path}...")
    final_df.to_csv(output_path, index=False)
    
    # Summary statistics
    print(f"\n✅ Seeding complete!")
    print(f"=" * 60)
    print(f"Summary:")
    print(f"  Total rows in output: {len(final_df)}")
    print(f"  Waves seeded: {waves_seeded}")
    print(f"  Waves skipped: {waves_skipped}")
    print(f"  New rows added: {len(new_df)}")
    print(f"  Unique wave_ids: {final_df['wave_id'].nunique()}")
    print(f"=" * 60)
    
    return {
        'total_rows': len(final_df),
        'waves_seeded': waves_seeded,
        'waves_skipped': waves_skipped,
        'new_rows': len(new_df),
        'unique_waves': final_df['wave_id'].nunique(),
    }


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        stats = seed_wave_history(
            days=args.days,
            mode=args.mode,
            output_path=args.output,
            force=args.force
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
