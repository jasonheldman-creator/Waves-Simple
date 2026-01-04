#!/usr/bin/env python3
"""
Migration Script: Add wave_id column to wave_history.csv

This script:
1. Reads existing wave_history.csv
2. Adds a wave_id column based on the display_name -> wave_id mapping
3. Preserves all numerical data
4. Backs up the original file
5. Writes the updated CSV with wave_id as the primary identifier

Usage:
    python migrate_wave_history_to_wave_id.py
"""

import pandas as pd
import os
import shutil
from datetime import datetime
from waves_engine import get_wave_id_from_display_name, WAVE_ID_REGISTRY

def migrate_wave_history():
    """Migrate wave_history.csv to include wave_id column."""
    
    csv_path = "wave_history.csv"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found")
        return False
    
    # Create backup
    backup_path = f"{csv_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(csv_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    
    # Read existing CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows from {csv_path}")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    # Check required columns
    if 'wave' not in df.columns:
        print("❌ Error: 'wave' column not found in CSV")
        return False
    
    # Check if wave_id column already exists
    if 'wave_id' in df.columns:
        print("⚠️  wave_id column already exists in CSV")
        print("   Updating wave_id values based on current mapping...")
    
    # Map display_name to wave_id
    print("\nMapping display names to wave_ids...")
    df['wave_id'] = df['wave'].apply(get_wave_id_from_display_name)
    
    # Check for unmapped waves
    unmapped = df[df['wave_id'].isna()]['wave'].unique()
    if len(unmapped) > 0:
        print(f"\n⚠️  Warning: {len(unmapped)} wave(s) could not be mapped to wave_id:")
        for wave_name in unmapped:
            print(f"   - '{wave_name}'")
        print("\n   These rows will be excluded from the migrated file.")
        print("   Please add mappings to WAVE_ID_REGISTRY in waves_engine.py if needed.")
        
        # Filter out unmapped rows
        original_count = len(df)
        df = df[df['wave_id'].notna()]
        print(f"\n   Removed {original_count - len(df)} rows with unmapped waves")
    
    # Show mapping summary
    wave_id_counts = df['wave_id'].value_counts()
    print(f"\n✓ Successfully mapped {len(wave_id_counts)} unique wave_ids:")
    for wave_id, count in wave_id_counts.head(10).items():
        print(f"   - {wave_id}: {count} rows")
    if len(wave_id_counts) > 10:
        print(f"   ... and {len(wave_id_counts) - 10} more")
    
    # Reorder columns to have wave_id first, then display_name (wave), then other columns
    cols = df.columns.tolist()
    # Remove wave_id and wave from their current positions
    cols = [c for c in cols if c not in ['wave_id', 'wave']]
    # Put wave_id first, then wave (as display_name), then rest
    new_cols = ['wave_id', 'wave'] + cols
    df = df[new_cols]
    
    # Rename 'wave' to 'display_name' for clarity
    df.rename(columns={'wave': 'display_name'}, inplace=True)
    
    # Save migrated CSV
    try:
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Successfully migrated {csv_path}")
        print(f"   Total rows: {len(df)}")
        print(f"   Columns: {', '.join(df.columns.tolist())}")
    except Exception as e:
        print(f"\n❌ Error writing CSV: {e}")
        print(f"   Restoring from backup...")
        shutil.copy2(backup_path, csv_path)
        return False
    
    # Verify data integrity
    print("\nVerifying data integrity...")
    try:
        df_verify = pd.read_csv(csv_path)
        if len(df_verify) == len(df):
            print("✓ Row count matches")
        else:
            print(f"⚠️  Row count mismatch: {len(df)} -> {len(df_verify)}")
        
        # Check numerical columns are preserved
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) > 0:
            print(f"✓ {len(numerical_cols)} numerical columns preserved")
    except Exception as e:
        print(f"⚠️  Verification error: {e}")
    
    print("\n✅ Migration complete!")
    print(f"   Backup saved as: {backup_path}")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Wave History Migration Script")
    print("Adding wave_id column to wave_history.csv")
    print("=" * 60)
    print()
    
    success = migrate_wave_history()
    
    if success:
        print("\n✅ Migration completed successfully")
        exit(0)
    else:
        print("\n❌ Migration failed")
        exit(1)
