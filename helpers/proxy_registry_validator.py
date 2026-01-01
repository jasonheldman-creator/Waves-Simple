"""
helpers/proxy_registry_validator.py

Proxy Registry Validation Module

This module provides validation functions for the Wave Proxy Registry,
which is the single source of truth for Plan B proxy analytics.

Functions:
- load_proxy_registry(): Load and parse the proxy registry CSV
- validate_proxy_registry(): Validate registry integrity and return report
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Registry file path
PROXY_REGISTRY_PATH = "config/wave_proxy_registry.csv"

# Expected fields
REQUIRED_FIELDS = [
    'wave_id',
    'display_name', 
    'category',
    'primary_proxy_ticker',
    'secondary_proxy_ticker',
    'benchmark_ticker',
    'enabled'
]

# Expected wave count
EXPECTED_WAVE_COUNT = 28


def load_proxy_registry(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the proxy registry CSV file.
    
    Args:
        path: Optional custom path to registry file
        
    Returns:
        DataFrame with registry data, or empty DataFrame if file not found
    """
    registry_path = path or PROXY_REGISTRY_PATH
    
    if not os.path.exists(registry_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(registry_path)
        return df
    except Exception as e:
        print(f"Error loading proxy registry: {e}")
        return pd.DataFrame()


def validate_proxy_registry(path: Optional[str] = None, strict: bool = False) -> Dict:
    """
    Validate the proxy registry CSV file.
    
    Checks:
    1. File exists
    2. Has all required fields
    3. Has exactly 28 enabled rows
    4. All wave_ids are unique
    5. All primary_proxy_tickers are valid (non-empty)
    6. All benchmark_tickers are valid (non-empty)
    
    Args:
        path: Optional custom path to registry file
        strict: If True, fail on any validation error. If False, allow degraded mode.
        
    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'report': str,
            'warnings': list,
            'errors': list,
            'enabled_count': int,
            'total_count': int,
            'degraded_mode': bool
        }
    """
    registry_path = path or PROXY_REGISTRY_PATH
    
    warnings = []
    errors = []
    
    # Check file exists
    if not os.path.exists(registry_path):
        errors.append(f"Registry file not found: {registry_path}")
        return {
            'valid': False,
            'report': f"❌ VALIDATION FAILED: Registry file not found at {registry_path}",
            'warnings': warnings,
            'errors': errors,
            'enabled_count': 0,
            'total_count': 0,
            'degraded_mode': not strict
        }
    
    # Load registry
    df = load_proxy_registry(registry_path)
    
    if df.empty:
        errors.append("Registry file is empty or could not be parsed")
        return {
            'valid': False,
            'report': "❌ VALIDATION FAILED: Registry file is empty or invalid",
            'warnings': warnings,
            'errors': errors,
            'enabled_count': 0,
            'total_count': 0,
            'degraded_mode': not strict
        }
    
    # Check required fields
    missing_fields = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Convert 'enabled' to boolean if string
    if 'enabled' in df.columns:
        if df['enabled'].dtype == 'object':
            df['enabled'] = df['enabled'].str.lower().isin(['true', '1', 'yes'])
    
    # Count enabled waves
    enabled_df = df[df['enabled'] == True] if 'enabled' in df.columns else df
    enabled_count = len(enabled_df)
    total_count = len(df)
    
    # Check enabled count
    if enabled_count != EXPECTED_WAVE_COUNT:
        errors.append(f"Expected {EXPECTED_WAVE_COUNT} enabled waves, found {enabled_count}")
    
    # Check for unique wave_ids
    if 'wave_id' in df.columns:
        duplicate_wave_ids = df[df.duplicated(subset=['wave_id'], keep=False)]['wave_id'].tolist()
        if duplicate_wave_ids:
            errors.append(f"Duplicate wave_ids found: {', '.join(set(duplicate_wave_ids))}")
    else:
        errors.append("wave_id field is missing")
    
    # Check for valid primary_proxy_ticker
    if 'primary_proxy_ticker' in df.columns:
        invalid_primary = enabled_df[enabled_df['primary_proxy_ticker'].isna() | (enabled_df['primary_proxy_ticker'] == '')]
        if len(invalid_primary) > 0:
            errors.append(f"{len(invalid_primary)} waves have invalid primary_proxy_ticker")
    else:
        errors.append("primary_proxy_ticker field is missing")
    
    # Check for valid benchmark_ticker
    if 'benchmark_ticker' in df.columns:
        invalid_benchmark = enabled_df[enabled_df['benchmark_ticker'].isna() | (enabled_df['benchmark_ticker'] == '')]
        if len(invalid_benchmark) > 0:
            errors.append(f"{len(invalid_benchmark)} waves have invalid benchmark_ticker")
    else:
        errors.append("benchmark_ticker field is missing")
    
    # Check for secondary_proxy_ticker (warning if missing, not error)
    if 'secondary_proxy_ticker' in df.columns:
        missing_secondary = enabled_df[enabled_df['secondary_proxy_ticker'].isna() | (enabled_df['secondary_proxy_ticker'] == '')]
        if len(missing_secondary) > 0:
            warnings.append(f"{len(missing_secondary)} waves have no secondary_proxy_ticker (degraded fallback only)")
    
    # Determine if validation passed
    is_valid = len(errors) == 0
    degraded_mode = not is_valid and not strict
    
    # Build report
    report_lines = []
    
    if is_valid:
        report_lines.append("✅ PROXY REGISTRY VALIDATION PASSED")
        report_lines.append(f"   • Enabled waves: {enabled_count}/{EXPECTED_WAVE_COUNT}")
        report_lines.append(f"   • Total waves: {total_count}")
        report_lines.append(f"   • All wave_ids unique: Yes")
        report_lines.append(f"   • All primary proxies valid: Yes")
        report_lines.append(f"   • All benchmarks valid: Yes")
    else:
        if degraded_mode:
            report_lines.append("⚠️ PROXY REGISTRY VALIDATION FAILED (DEGRADED MODE)")
        else:
            report_lines.append("❌ PROXY REGISTRY VALIDATION FAILED (STRICT MODE)")
        
        report_lines.append(f"   • Enabled waves: {enabled_count}/{EXPECTED_WAVE_COUNT}")
        report_lines.append(f"   • Total waves: {total_count}")
        
        if errors:
            report_lines.append("\n   ERRORS:")
            for error in errors:
                report_lines.append(f"   • {error}")
    
    if warnings:
        report_lines.append("\n   WARNINGS:")
        for warning in warnings:
            report_lines.append(f"   • {warning}")
    
    report = "\n".join(report_lines)
    
    return {
        'valid': is_valid,
        'report': report,
        'warnings': warnings,
        'errors': errors,
        'enabled_count': enabled_count,
        'total_count': total_count,
        'degraded_mode': degraded_mode
    }


def get_enabled_proxy_waves(path: Optional[str] = None) -> List[Dict]:
    """
    Get list of enabled waves from the proxy registry.
    
    Args:
        path: Optional custom path to registry file
        
    Returns:
        List of dictionaries, one per enabled wave
    """
    df = load_proxy_registry(path)
    
    if df.empty:
        return []
    
    # Convert 'enabled' to boolean if string
    if 'enabled' in df.columns:
        if df['enabled'].dtype == 'object':
            df['enabled'] = df['enabled'].str.lower().isin(['true', '1', 'yes'])
    
    # Filter to enabled waves
    enabled_df = df[df['enabled'] == True] if 'enabled' in df.columns else df
    
    # Convert to list of dicts
    return enabled_df.to_dict('records')


def get_proxy_tickers_for_wave(wave_id: str, path: Optional[str] = None) -> Dict:
    """
    Get proxy ticker configuration for a specific wave.
    
    Args:
        wave_id: Wave identifier
        path: Optional custom path to registry file
        
    Returns:
        Dictionary with proxy configuration, or None if wave not found
    """
    df = load_proxy_registry(path)
    
    if df.empty:
        return None
    
    # Find wave
    wave_row = df[df['wave_id'] == wave_id]
    
    if len(wave_row) == 0:
        return None
    
    # Return first match as dict
    return wave_row.iloc[0].to_dict()
