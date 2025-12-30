"""
truth_frame_helpers.py

Helper functions for consuming TruthFrame data in the UI.

These helpers make it easy to extract specific metrics from the TruthFrame
without recomputing them locally. All UI components should use these helpers
instead of computing returns, alphas, or exposures directly.

Usage:
    from truth_frame_helpers import (
        get_wave_returns,
        get_wave_alphas,
        get_wave_exposure,
        format_return_display,
        format_alpha_display,
        get_wave_metric
    )
    
    # Get returns for a wave
    returns = get_wave_returns(truth_df, "sp500_wave")
    # Returns: {'1d': 0.0123, '30d': 0.0456, '60d': 0.0789, '365d': 0.15}
    
    # Format for display
    display_text = format_return_display(returns['1d'])
    # Returns: "+1.23%"
    
    # Get specific metric
    exposure = get_wave_metric(truth_df, "sp500_wave", "exposure_pct")
    # Returns: 0.95
"""

from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np


# Readiness status icon mapping
READINESS_ICONS = {
    'full': '🟢',
    'partial': '🟡',
    'operational': '🟠',
    'unavailable': '🔴',
}


# Column mapping from TruthFrame to legacy snapshot format
# Used for backward compatibility during migration
TRUTHFRAME_TO_SNAPSHOT_COLUMNS = {
    'wave_id': 'Wave_ID',
    'display_name': 'Wave',
    'mode': 'Mode',
    'readiness_status': 'Data_Regime_Tag',
    'coverage_pct': 'Coverage_Score',
    'return_1d': 'Return_1D',
    'return_30d': 'Return_30D',
    'return_60d': 'Return_60D',
    'return_365d': 'Return_365D',
    'alpha_1d': 'Alpha_1D',
    'alpha_30d': 'Alpha_30D',
    'alpha_60d': 'Alpha_60D',
    'alpha_365d': 'Alpha_365D',
    'benchmark_return_1d': 'Benchmark_Return_1D',
    'benchmark_return_30d': 'Benchmark_Return_30D',
    'benchmark_return_60d': 'Benchmark_Return_60D',
    'benchmark_return_365d': 'Benchmark_Return_365D',
    'exposure_pct': 'Exposure',
    'cash_pct': 'CashPercent',
    'beta_real': 'Beta_Real',
    'beta_target': 'Beta_Target',
    'beta_drift': 'Beta_Drift',
    'turnover_est': 'Turnover_Est',
    'drawdown_60d': 'MaxDD',
    'alert_badges': 'Flags',
    'last_snapshot_ts': 'Date',
}


def convert_truthframe_to_snapshot_format(truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TruthFrame to legacy snapshot format for backward compatibility.
    
    This is a temporary helper for tabs that haven't been fully migrated yet.
    
    Args:
        truth_df: TruthFrame DataFrame
        
    Returns:
        DataFrame in legacy snapshot format
    """
    if truth_df is None or truth_df.empty:
        return pd.DataFrame()
    
    return truth_df.rename(columns=TRUTHFRAME_TO_SNAPSHOT_COLUMNS)


def get_wave_metric(
    truth_df: pd.DataFrame,
    wave_id: str,
    metric_name: str,
    default: Any = None
) -> Any:
    """
    Get a specific metric for a wave from TruthFrame.
    
    Args:
        truth_df: TruthFrame DataFrame
        wave_id: Wave identifier
        metric_name: Name of metric column
        default: Default value if metric not available
        
    Returns:
        Metric value or default
    """
    if truth_df is None or truth_df.empty:
        return default
    
    wave_data = truth_df[truth_df['wave_id'] == wave_id]
    
    if wave_data.empty:
        return default
    
    if metric_name not in wave_data.columns:
        return default
    
    value = wave_data.iloc[0][metric_name]
    
    if pd.isna(value):
        return default
    
    return value


def get_wave_returns(
    truth_df: pd.DataFrame,
    wave_id: str
) -> Dict[str, Optional[float]]:
    """
    Get all return timeframes for a wave.
    
    Args:
        truth_df: TruthFrame DataFrame
        wave_id: Wave identifier
        
    Returns:
        Dictionary with keys: '1d', '30d', '60d', '365d'
    """
    return {
        '1d': get_wave_metric(truth_df, wave_id, 'return_1d'),
        '30d': get_wave_metric(truth_df, wave_id, 'return_30d'),
        '60d': get_wave_metric(truth_df, wave_id, 'return_60d'),
        '365d': get_wave_metric(truth_df, wave_id, 'return_365d'),
    }


def get_wave_alphas(
    truth_df: pd.DataFrame,
    wave_id: str
) -> Dict[str, Optional[float]]:
    """
    Get all alpha timeframes for a wave.
    
    Args:
        truth_df: TruthFrame DataFrame
        wave_id: Wave identifier
        
    Returns:
        Dictionary with keys: '1d', '30d', '60d', '365d'
    """
    return {
        '1d': get_wave_metric(truth_df, wave_id, 'alpha_1d'),
        '30d': get_wave_metric(truth_df, wave_id, 'alpha_30d'),
        '60d': get_wave_metric(truth_df, wave_id, 'alpha_60d'),
        '365d': get_wave_metric(truth_df, wave_id, 'alpha_365d'),
    }


def get_wave_benchmark_returns(
    truth_df: pd.DataFrame,
    wave_id: str
) -> Dict[str, Optional[float]]:
    """
    Get all benchmark return timeframes for a wave.
    
    Args:
        truth_df: TruthFrame DataFrame
        wave_id: Wave identifier
        
    Returns:
        Dictionary with keys: '1d', '30d', '60d', '365d'
    """
    return {
        '1d': get_wave_metric(truth_df, wave_id, 'benchmark_return_1d'),
        '30d': get_wave_metric(truth_df, wave_id, 'benchmark_return_30d'),
        '60d': get_wave_metric(truth_df, wave_id, 'benchmark_return_60d'),
        '365d': get_wave_metric(truth_df, wave_id, 'benchmark_return_365d'),
    }


def get_wave_exposure(
    truth_df: pd.DataFrame,
    wave_id: str
) -> Dict[str, Optional[float]]:
    """
    Get exposure and cash percentage for a wave.
    
    Args:
        truth_df: TruthFrame DataFrame
        wave_id: Wave identifier
        
    Returns:
        Dictionary with keys: 'exposure_pct', 'cash_pct'
    """
    return {
        'exposure_pct': get_wave_metric(truth_df, wave_id, 'exposure_pct'),
        'cash_pct': get_wave_metric(truth_df, wave_id, 'cash_pct'),
    }


def get_wave_beta_metrics(
    truth_df: pd.DataFrame,
    wave_id: str
) -> Dict[str, Optional[float]]:
    """
    Get beta-related metrics for a wave.
    
    Args:
        truth_df: TruthFrame DataFrame
        wave_id: Wave identifier
        
    Returns:
        Dictionary with keys: 'beta_real', 'beta_target', 'beta_drift'
    """
    return {
        'beta_real': get_wave_metric(truth_df, wave_id, 'beta_real'),
        'beta_target': get_wave_metric(truth_df, wave_id, 'beta_target'),
        'beta_drift': get_wave_metric(truth_df, wave_id, 'beta_drift'),
    }


def get_wave_risk_metrics(
    truth_df: pd.DataFrame,
    wave_id: str
) -> Dict[str, Optional[float]]:
    """
    Get risk-related metrics for a wave.
    
    Args:
        truth_df: TruthFrame DataFrame
        wave_id: Wave identifier
        
    Returns:
        Dictionary with keys: 'turnover_est', 'drawdown_60d'
    """
    return {
        'turnover_est': get_wave_metric(truth_df, wave_id, 'turnover_est'),
        'drawdown_60d': get_wave_metric(truth_df, wave_id, 'drawdown_60d'),
    }


def get_wave_readiness(
    truth_df: pd.DataFrame,
    wave_id: str
) -> Dict[str, Any]:
    """
    Get readiness and coverage information for a wave.
    
    Args:
        truth_df: TruthFrame DataFrame
        wave_id: Wave identifier
        
    Returns:
        Dictionary with readiness info
    """
    return {
        'readiness_status': get_wave_metric(truth_df, wave_id, 'readiness_status', 'unavailable'),
        'coverage_pct': get_wave_metric(truth_df, wave_id, 'coverage_pct', 0.0),
        'data_regime_tag': get_wave_metric(truth_df, wave_id, 'data_regime_tag', 'UNAVAILABLE'),
        'alert_badges': get_wave_metric(truth_df, wave_id, 'alert_badges', ''),
    }


def format_return_display(
    return_value: Optional[float],
    decimal_places: int = 2,
    show_sign: bool = True
) -> str:
    """
    Format return value for display.
    
    Args:
        return_value: Return as decimal (e.g., 0.0123 for 1.23%)
        decimal_places: Number of decimal places
        show_sign: Show + sign for positive values
        
    Returns:
        Formatted string (e.g., "+1.23%", "-0.56%", "N/A")
    """
    if return_value is None or (isinstance(return_value, float) and np.isnan(return_value)):
        return "N/A"
    
    pct_value = return_value * 100
    sign = "+" if pct_value >= 0 and show_sign else ""
    
    return f"{sign}{pct_value:.{decimal_places}f}%"


def format_alpha_display(
    alpha_value: Optional[float],
    decimal_places: int = 2,
    show_sign: bool = True
) -> str:
    """
    Format alpha value for display.
    
    Same as format_return_display but kept separate for semantic clarity.
    
    Args:
        alpha_value: Alpha as decimal
        decimal_places: Number of decimal places
        show_sign: Show + sign for positive values
        
    Returns:
        Formatted string
    """
    return format_return_display(alpha_value, decimal_places, show_sign)


def format_percentage_display(
    value: Optional[float],
    decimal_places: int = 1,
    show_sign: bool = False
) -> str:
    """
    Format percentage value for display (already in percentage form).
    
    Args:
        value: Percentage value (e.g., 95.0 for 95%)
        decimal_places: Number of decimal places
        show_sign: Show + sign for positive values
        
    Returns:
        Formatted string (e.g., "95.0%", "N/A")
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    
    sign = "+" if value >= 0 and show_sign else ""
    
    return f"{sign}{value:.{decimal_places}f}%"


def format_exposure_display(
    exposure_value: Optional[float],
    as_percentage: bool = True
) -> str:
    """
    Format exposure value for display.
    
    Args:
        exposure_value: Exposure as decimal (e.g., 0.95 for 95%)
        as_percentage: Convert to percentage
        
    Returns:
        Formatted string
    """
    if exposure_value is None or (isinstance(exposure_value, float) and np.isnan(exposure_value)):
        return "N/A"
    
    if as_percentage:
        return f"{exposure_value * 100:.1f}%"
    else:
        return f"{exposure_value:.2f}x"


def format_beta_display(
    beta_value: Optional[float],
    decimal_places: int = 2
) -> str:
    """
    Format beta value for display.
    
    Args:
        beta_value: Beta value
        decimal_places: Number of decimal places
        
    Returns:
        Formatted string
    """
    if beta_value is None or (isinstance(beta_value, float) and np.isnan(beta_value)):
        return "N/A"
    
    return f"{beta_value:.{decimal_places}f}"


def get_wave_summary(
    truth_df: pd.DataFrame,
    wave_id: str
) -> Dict[str, Any]:
    """
    Get comprehensive summary for a wave.
    
    This is a convenience function that gets all common metrics in one call.
    
    Args:
        truth_df: TruthFrame DataFrame
        wave_id: Wave identifier
        
    Returns:
        Dictionary with all wave metrics
    """
    if truth_df is None or truth_df.empty:
        return {}
    
    wave_data = truth_df[truth_df['wave_id'] == wave_id]
    
    if wave_data.empty:
        return {}
    
    # Return the entire row as a dictionary
    return wave_data.iloc[0].to_dict()


def get_top_performers(
    truth_df: pd.DataFrame,
    metric: str = 'return_30d',
    n: int = 5,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Get top N waves by a specific metric.
    
    Args:
        truth_df: TruthFrame DataFrame
        metric: Metric to sort by
        n: Number of top waves to return
        ascending: Sort ascending (True) or descending (False)
        
    Returns:
        DataFrame with top N waves
    """
    if truth_df is None or truth_df.empty:
        return pd.DataFrame()
    
    if metric not in truth_df.columns:
        return pd.DataFrame()
    
    # Filter out NaN values
    valid_df = truth_df[truth_df[metric].notna()].copy()
    
    # Sort and get top N
    sorted_df = valid_df.sort_values(metric, ascending=ascending)
    
    return sorted_df.head(n)


def get_waves_by_readiness(
    truth_df: pd.DataFrame,
    readiness_status: str
) -> pd.DataFrame:
    """
    Get all waves with a specific readiness status.
    
    Args:
        truth_df: TruthFrame DataFrame
        readiness_status: Readiness status to filter by
        
    Returns:
        Filtered DataFrame
    """
    if truth_df is None or truth_df.empty:
        return pd.DataFrame()
    
    return truth_df[truth_df['readiness_status'] == readiness_status]


def get_readiness_summary(
    truth_df: pd.DataFrame
) -> Dict[str, int]:
    """
    Get summary of waves by readiness status.
    
    Args:
        truth_df: TruthFrame DataFrame
        
    Returns:
        Dictionary with counts by status
    """
    if truth_df is None or truth_df.empty:
        return {
            'full': 0,
            'partial': 0,
            'operational': 0,
            'unavailable': 0,
            'total': 0
        }
    
    # Normalize status to lowercase for consistent counting
    normalized_status = truth_df['readiness_status'].str.lower()
    status_counts = normalized_status.value_counts().to_dict()
    
    return {
        'full': status_counts.get('full', 0),
        'partial': status_counts.get('partial', 0),
        'operational': status_counts.get('operational', 0),
        'unavailable': status_counts.get('unavailable', 0),
        'total': len(truth_df)
    }


# Readiness status icon mapping
READINESS_ICONS = {
    'full': '🟢',
    'partial': '🟡',
    'operational': '🟠',
    'unavailable': '🔴',
}


def format_readiness_badge(
    readiness_status: str,
    coverage_pct: Optional[float] = None
) -> str:
    """
    Format readiness status as a badge with icon.
    
    Args:
        readiness_status: Readiness status
        coverage_pct: Optional coverage percentage
        
    Returns:
        Formatted badge string
    """
    # Normalize to lowercase for lookup
    status_lower = str(readiness_status).lower()
    
    icon = READINESS_ICONS.get(status_lower, '⚪')
    status_text = str(readiness_status).title()
    
    if coverage_pct is not None and not np.isnan(coverage_pct):
        return f"{icon} {status_text} ({coverage_pct:.0f}%)"
    else:
        return f"{icon} {status_text}"


def create_returns_dataframe(
    truth_df: pd.DataFrame,
    timeframes: List[str] = ['1d', '30d', '60d', '365d']
) -> pd.DataFrame:
    """
    Create a formatted DataFrame with returns for all waves.
    
    Args:
        truth_df: TruthFrame DataFrame
        timeframes: List of timeframes to include
        
    Returns:
        DataFrame with wave names and formatted returns
    """
    if truth_df is None or truth_df.empty:
        return pd.DataFrame()
    
    result_df = truth_df[['wave_id', 'display_name']].copy()
    
    for tf in timeframes:
        col_name = f'return_{tf}'
        if col_name in truth_df.columns:
            result_df[f'Return {tf.upper()}'] = truth_df[col_name].apply(
                lambda x: format_return_display(x)
            )
    
    return result_df


def create_alpha_dataframe(
    truth_df: pd.DataFrame,
    timeframes: List[str] = ['1d', '30d', '60d', '365d']
) -> pd.DataFrame:
    """
    Create a formatted DataFrame with alphas for all waves.
    
    Args:
        truth_df: TruthFrame DataFrame
        timeframes: List of timeframes to include
        
    Returns:
        DataFrame with wave names and formatted alphas
    """
    if truth_df is None or truth_df.empty:
        return pd.DataFrame()
    
    result_df = truth_df[['wave_id', 'display_name']].copy()
    
    for tf in timeframes:
        col_name = f'alpha_{tf}'
        if col_name in truth_df.columns:
            result_df[f'Alpha {tf.upper()}'] = truth_df[col_name].apply(
                lambda x: format_alpha_display(x)
            )
    
    return result_df
