"""
Validation Script for 9 Equity Waves Implementation

This script validates that the 9 equity-only waves meet all requirements:
1. Wave registry entries with required fields
2. Dedicated weights files exist
3. Weights sum to 1.0 (with tolerance)
4. All waves are active and categorized as Equity
5. All tickers are unique within each wave
6. Referenced files exist

Exit code 0 if all validations pass, 1 otherwise (for CI integration).
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Tuple

# The 9 equity waves to validate
EQUITY_WAVES = [
    "clean_transit_infrastructure_wave",
    "demas_fund_wave",
    "ev_infrastructure_wave",
    "future_power_energy_wave",
    "infinity_multi_asset_growth_wave",
    "next_gen_compute_semis_wave",
    "quantum_computing_wave",
    "small_to_mid_cap_growth_wave",
    "us_megacap_core_wave"
]

# Expected display names
EXPECTED_DISPLAY_NAMES = {
    "clean_transit_infrastructure_wave": "Clean Transit-Infrastructure Wave",
    "demas_fund_wave": "Demas Fund Wave",
    "ev_infrastructure_wave": "EV & Infrastructure Wave",
    "future_power_energy_wave": "Future Power & Energy Wave",
    "infinity_multi_asset_growth_wave": "Infinity Multi-Asset Growth Wave",
    "next_gen_compute_semis_wave": "Next-Gen Compute & Semis Wave",
    "quantum_computing_wave": "Quantum Computing Wave",
    "small_to_mid_cap_growth_wave": "Small to Mid Cap Growth Wave",
    "us_megacap_core_wave": "US MegaCap Core Wave",
}

# Weight tolerance
WEIGHT_TOLERANCE = 0.01

def validate_wave_registry() -> Tuple[bool, List[str], List[str]]:
    """
    Validate wave registry entries for all 9 equity waves.
    
    Returns:
        Tuple of (success, errors, warnings)
    """
    errors = []
    warnings = []
    
    registry_path = "data/wave_registry.csv"
    
    if not os.path.exists(registry_path):
        errors.append(f"Wave registry not found: {registry_path}")
        return False, errors, warnings
    
    try:
        registry = pd.read_csv(registry_path)
    except Exception as e:
        errors.append(f"Failed to read wave registry: {e}")
        return False, errors, warnings
    
    # Required columns
    required_cols = ['wave_id', 'wave_name', 'category', 'active', 'benchmark_spec']
    missing_cols = [col for col in required_cols if col not in registry.columns]
    if missing_cols:
        errors.append(f"Registry missing required columns: {missing_cols}")
        return False, errors, warnings
    
    # Check each equity wave
    for wave_id in EQUITY_WAVES:
        wave_row = registry[registry['wave_id'] == wave_id]
        
        if wave_row.empty:
            errors.append(f"Wave not found in registry: {wave_id}")
            continue
        
        wave_row = wave_row.iloc[0]
        
        # Check display name
        expected_name = EXPECTED_DISPLAY_NAMES[wave_id]
        actual_name = wave_row['wave_name']
        if actual_name != expected_name:
            errors.append(f"{wave_id}: Expected display_name '{expected_name}', got '{actual_name}'")
        
        # Check category is Equity (allow equity_growth, equity_income, etc.)
        category = wave_row['category']
        if not (category == 'Equity' or str(category).startswith('equity')):
            errors.append(f"{wave_id}: Expected category 'Equity' or 'equity_*', got '{category}'")
        
        # Check is_active
        is_active = wave_row['active']
        if not is_active:
            errors.append(f"{wave_id}: Wave is not active (active={is_active})")
        
        # Check benchmark_spec exists
        benchmark = wave_row['benchmark_spec']
        if pd.isna(benchmark) or str(benchmark).strip() == '':
            errors.append(f"{wave_id}: Missing benchmark_spec")
    
    success = len(errors) == 0
    return success, errors, warnings


def validate_positions_files() -> Tuple[bool, List[str], List[str]]:
    """
    Validate positions files exist and weights sum to 1.0.
    
    Returns:
        Tuple of (success, errors, warnings)
    """
    errors = []
    warnings = []
    
    for wave_id in EQUITY_WAVES:
        positions_path = f"data/waves/{wave_id}/positions.csv"
        
        if not os.path.exists(positions_path):
            errors.append(f"{wave_id}: Positions file not found: {positions_path}")
            continue
        
        try:
            positions = pd.read_csv(positions_path)
        except Exception as e:
            errors.append(f"{wave_id}: Failed to read positions file: {e}")
            continue
        
        # Check required columns
        required_cols = ['ticker', 'weight']
        missing_cols = [col for col in required_cols if col not in positions.columns]
        if missing_cols:
            errors.append(f"{wave_id}: Positions file missing columns: {missing_cols}")
            continue
        
        # Check for duplicate tickers
        tickers = positions['ticker'].tolist()
        unique_tickers = set(tickers)
        if len(tickers) != len(unique_tickers):
            duplicates = [t for t in unique_tickers if tickers.count(t) > 1]
            errors.append(f"{wave_id}: Duplicate tickers found: {duplicates}")
        
        # Check weights sum to 1.0
        total_weight = positions['weight'].sum()
        if abs(total_weight - 1.0) > WEIGHT_TOLERANCE:
            errors.append(
                f"{wave_id}: Weights sum to {total_weight:.4f}, expected 1.0 "
                f"(tolerance {WEIGHT_TOLERANCE})"
            )
        
        # Check all weights are positive
        negative_weights = positions[positions['weight'] < 0]
        if not negative_weights.empty:
            errors.append(f"{wave_id}: Found negative weights")
        
        # Info: count tickers
        if len(errors) == 0:
            print(f"✓ {wave_id}: {len(positions)} tickers, weight sum = {total_weight:.4f}")
    
    success = len(errors) == 0
    return success, errors, warnings


def collect_all_tickers() -> Dict[str, List[str]]:
    """
    Collect all tickers from the 9 equity waves.
    
    Returns:
        Dictionary mapping wave_id to list of tickers
    """
    all_tickers = {}
    
    for wave_id in EQUITY_WAVES:
        positions_path = f"data/waves/{wave_id}/positions.csv"
        
        if os.path.exists(positions_path):
            try:
                positions = pd.read_csv(positions_path)
                tickers = positions['ticker'].tolist()
                all_tickers[wave_id] = tickers
            except Exception:
                all_tickers[wave_id] = []
        else:
            all_tickers[wave_id] = []
    
    return all_tickers


def main():
    """Main validation entry point."""
    print("=" * 80)
    print("EQUITY WAVES VALIDATION")
    print("=" * 80)
    print()
    
    print(f"Validating {len(EQUITY_WAVES)} equity waves:")
    for wave_id in EQUITY_WAVES:
        print(f"  - {wave_id}")
    print()
    
    all_errors = []
    all_warnings = []
    
    # Validate registry
    print("1. Validating wave registry entries...")
    success, errors, warnings = validate_wave_registry()
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    if success:
        print("   ✓ All waves found in registry with required fields")
    else:
        print(f"   ✗ Registry validation failed ({len(errors)} errors)")
    print()
    
    # Validate positions files
    print("2. Validating positions files and weights...")
    success, errors, warnings = validate_positions_files()
    all_errors.extend(errors)
    all_warnings.extend(warnings)
    if success:
        print("   ✓ All positions files valid")
    else:
        print(f"   ✗ Positions validation failed ({len(errors)} errors)")
    print()
    
    # Collect tickers
    print("3. Collecting tickers from all waves...")
    wave_tickers = collect_all_tickers()
    all_unique_tickers = set()
    for wave_id, tickers in wave_tickers.items():
        all_unique_tickers.update(tickers)
    print(f"   Total unique tickers across all equity waves: {len(all_unique_tickers)}")
    print()
    
    # Print summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if all_errors:
        print(f"\n❌ ERRORS ({len(all_errors)}):")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
    
    if all_warnings:
        print(f"\n⚠️  WARNINGS ({len(all_warnings)}):")
        for i, warning in enumerate(all_warnings, 1):
            print(f"  {i}. {warning}")
    
    if not all_errors and not all_warnings:
        print("\n✅ ALL VALIDATIONS PASSED")
        print(f"\nAll {len(EQUITY_WAVES)} equity waves are properly configured:")
        print(f"  - Registry entries: ✓")
        print(f"  - Positions files: ✓")
        print(f"  - Weights sum to 1.0: ✓")
        print(f"  - Total unique tickers: {len(all_unique_tickers)}")
        return 0
    elif all_errors:
        print(f"\n❌ VALIDATION FAILED with {len(all_errors)} error(s)")
        return 1
    else:
        print(f"\n✅ VALIDATION PASSED with {len(all_warnings)} warning(s)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
