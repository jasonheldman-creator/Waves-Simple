"""
Test snapshot schema and column normalization.
"""

import pandas as pd
import numpy as np
from analytics_truth import generate_live_snapshot_csv


def test_snapshot_schema_consistency():
    """Test that snapshot has consistent schema with required columns."""
    
    # Load existing snapshot if it exists
    try:
        df = pd.read_csv('data/live_snapshot.csv')
        
        # Check required columns exist
        required_columns = [
            'wave_id', 'wave',
            'return_1d', 'return_30d', 'return_60d', 'return_365d',
            'alpha_1d', 'alpha_30d', 'alpha_60d', 'alpha_365d'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Check that all column names are lowercase
        non_lowercase = [col for col in df.columns if col != col.lower()]
        if non_lowercase:
            print(f"❌ Non-lowercase column names: {non_lowercase}")
            return False
        
        # Check that wave_id column has no nulls or blanks
        if df['wave_id'].isna().any():
            print(f"❌ wave_id column contains null values")
            return False
        
        # Check for blank wave_ids
        blank_count = df['wave_id'].astype(str).str.strip().eq('').sum()
        if blank_count > 0:
            print(f"❌ wave_id column contains {blank_count} blank values")
            return False
        
        print(f"✅ Snapshot schema validation passed")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns)}")
        print(f"   - Unique wave_ids: {df['wave_id'].nunique()}")
        
        return True
        
    except FileNotFoundError:
        print("⚠️ data/live_snapshot.csv not found - skipping test")
        return True
    except Exception as e:
        print(f"❌ Error testing snapshot schema: {e}")
        return False


if __name__ == '__main__':
    success = test_snapshot_schema_consistency()
    exit(0 if success else 1)
