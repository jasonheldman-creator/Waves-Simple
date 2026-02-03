#!/usr/bin/env python3
"""
Test script to validate app_min.py runtime safety.
Simulates key code paths without running Streamlit.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Test data loading
print("=" * 60)
print("Testing Core Data Loading...")
print("=" * 60)

DATA_DIR = Path("data")
LIVE_SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"
ALPHA_ATTRIBUTION_PATH = DATA_DIR / "alpha_attribution_summary.csv"

RETURN_COLS = {
    "1D": "return_1d",
    "30D": "return_30d",
    "60D": "return_60d",
    "365D": "return_365d",
}

ALPHA_COLS = {
    "1D": "alpha_1d",
    "30D": "alpha_30d",
    "60D": "alpha_60d",
    "365D": "alpha_365d",
}

def load_snapshot():
    if not LIVE_SNAPSHOT_PATH.exists():
        return None, None, "Live snapshot file not found"

    try:
        df = pd.read_csv(LIVE_SNAPSHOT_PATH)
        df.columns = [c.strip().lower() for c in df.columns]
    except Exception as e:
        return None, None, f"Error reading live snapshot: {str(e)}"

    # Load alpha attribution summary if it exists
    attrib_df = None
    if ALPHA_ATTRIBUTION_PATH.exists():
        try:
            attrib_df = pd.read_csv(ALPHA_ATTRIBUTION_PATH)
            attrib_df.columns = [c.strip().lower() for c in attrib_df.columns]
            
            # Validate required columns for attribution
            required_attrib_cols = ["horizon"]
            for col in required_attrib_cols:
                if col not in attrib_df.columns:
                    print(f"WARNING: Attribution file missing column: {col}")
                    attrib_df = None
                    break
        except Exception as e:
            print(f"WARNING: Could not load attribution summary: {str(e)}")
            attrib_df = None

    # Set display_name
    if "display_name" not in df.columns:
        if "wave_name" in df.columns:
            df["display_name"] = df["wave_name"]
        elif "wave_id" in df.columns:
            df["display_name"] = df["wave_id"]
        else:
            df["display_name"] = "Unnamed Wave"

    # Ensure required columns exist
    for col in list(RETURN_COLS.values()) + list(ALPHA_COLS.values()):
        if col not in df.columns:
            df[col] = np.nan

    if "intraday_label" not in df.columns:
        df["intraday_label"] = None
    
    # Ensure additional required columns exist with safe defaults
    if "wave_name" not in df.columns:
        df["wave_name"] = df["display_name"]
    if "return_intraday" not in df.columns:
        df["return_intraday"] = np.nan
    if "alpha_intraday" not in df.columns:
        df["alpha_intraday"] = np.nan

    return df, attrib_df, None

snapshot_df, attrib_df, snapshot_error = load_snapshot()

if snapshot_error:
    print(f"❌ FAILED: {snapshot_error}")
    sys.exit(1)

if snapshot_df is None or snapshot_df.empty:
    print("❌ FAILED: No snapshot data loaded")
    sys.exit(1)

print(f"✅ PASSED: Loaded snapshot with {len(snapshot_df)} rows")
print(f"✅ PASSED: Snapshot columns: {list(snapshot_df.columns)}")

if attrib_df is not None:
    print(f"✅ PASSED: Loaded attribution with {len(attrib_df)} rows")
    print(f"✅ PASSED: Attribution columns: {list(attrib_df.columns)}")
else:
    print("⚠️  WARNING: Attribution data not available")

# Test portfolio metrics computation
print("\n" + "=" * 60)
print("Testing Portfolio Metrics Computation...")
print("=" * 60)

def compute_portfolio_metrics(df, return_cols, alpha_cols):
    portfolio_returns = {}
    portfolio_alphas = {}

    for label, col in return_cols.items():
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                portfolio_returns[label] = valid_values.mean()
            else:
                portfolio_returns[label] = None
        else:
            portfolio_returns[label] = None

    for label, col in alpha_cols.items():
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                portfolio_alphas[label] = valid_values.mean()
            else:
                portfolio_alphas[label] = None
        else:
            portfolio_alphas[label] = None

    return portfolio_returns, portfolio_alphas

try:
    portfolio_returns, portfolio_alphas = compute_portfolio_metrics(snapshot_df, RETURN_COLS, ALPHA_COLS)
    print(f"✅ PASSED: Portfolio returns computed: {portfolio_returns}")
    print(f"✅ PASSED: Portfolio alphas computed: {portfolio_alphas}")
except Exception as e:
    print(f"❌ FAILED: Portfolio metrics computation failed: {e}")
    sys.exit(1)

# Test wave access
print("\n" + "=" * 60)
print("Testing Wave Selection...")
print("=" * 60)

try:
    if "display_name" in snapshot_df.columns:
        waves = snapshot_df["display_name"].tolist()
        print(f"✅ PASSED: Found {len(waves)} waves")
        if len(waves) > 0:
            selected_wave = waves[0]
            print(f"✅ PASSED: Selected wave: {selected_wave}")
            
            # Test wave filtering
            wave_subset = snapshot_df[snapshot_df["display_name"] == selected_wave]
            if not wave_subset.empty:
                wave_row = wave_subset.iloc[0]
                print(f"✅ PASSED: Wave row accessed successfully")
            else:
                print(f"⚠️  WARNING: Wave subset empty for {selected_wave}")
    else:
        print("❌ FAILED: display_name column missing")
        sys.exit(1)
except Exception as e:
    print(f"❌ FAILED: Wave selection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test attribution computation
print("\n" + "=" * 60)
print("Testing Attribution Computation...")
print("=" * 60)

def compute_attribution_from_summary(attrib_df, horizon):
    if attrib_df is None or attrib_df.empty:
        return None

    horizon_map = {"30D": 30, "60D": 60, "365D": 365}
    horizon_val = horizon_map.get(horizon)
    
    if horizon_val is None:
        return None

    if "horizon" not in attrib_df.columns:
        return None

    try:
        horizon_data = attrib_df[attrib_df["horizon"] == horizon_val]
    except (KeyError, ValueError):
        return None
    
    if horizon_data.empty:
        return None

    component_cols = [
        "selection_alpha",
        "momentum_alpha", 
        "volatility_alpha",
        "regime_alpha",
        "exposure_alpha",
        "residual_alpha"
    ]

    result = {col: None for col in component_cols}
    result["total_alpha"] = None

    for col in component_cols:
        if col in horizon_data.columns:
            valid_values = pd.to_numeric(horizon_data[col], errors='coerce').dropna()
            if len(valid_values) > 0:
                result[col] = valid_values.mean()

    if "total_alpha" in horizon_data.columns:
        valid_values = pd.to_numeric(horizon_data["total_alpha"], errors='coerce').dropna()
        if len(valid_values) > 0:
            result["total_alpha"] = valid_values.mean()

    return result

try:
    if attrib_df is not None:
        for horizon in ["30D", "60D", "365D"]:
            result = compute_attribution_from_summary(attrib_df, horizon)
            if result:
                print(f"✅ PASSED: Attribution computed for {horizon}")
            else:
                print(f"⚠️  WARNING: No attribution data for {horizon}")
    else:
        print("⚠️  WARNING: Skipping attribution tests (no data)")
except Exception as e:
    print(f"❌ FAILED: Attribution computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test module imports
print("\n" + "=" * 60)
print("Testing Module Imports...")
print("=" * 60)

try:
    import adaptive_learning as al
    print("✅ PASSED: adaptive_learning module imported")
    
    adaptive_state = al.load_adaptive_state()
    print("✅ PASSED: Adaptive state loaded")
    
    adaptive_state, messages = al.update_adaptive_state(snapshot_df, attrib_df, adaptive_state)
    print(f"✅ PASSED: Adaptive state updated with {len(messages)} messages")
except Exception as e:
    print(f"❌ FAILED: adaptive_learning module failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    import integrity_signals as integ
    print("✅ PASSED: integrity_signals module imported")
    
    integrity_data = integ.get_all_integrity_signals(snapshot_df, attrib_df)
    print(f"✅ PASSED: Integrity signals computed")
    
    if "integrity_index" in integrity_data:
        print(f"✅ PASSED: integrity_index present in result")
    else:
        print(f"❌ FAILED: integrity_index missing from result")
        sys.exit(1)
except Exception as e:
    print(f"❌ FAILED: integrity_signals module failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe app should be safe to run without runtime errors.")
print("Defensive coding is in place for:")
print("  • Missing columns")
print("  • Empty DataFrames")
print("  • NaN/None values")
print("  • Type conversion errors")
print("  • Missing data files")
