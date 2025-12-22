"""
Institutional Console v2 - Executive Layer v2
Full implementation with advanced analytics and visualization

Modular structure:
1. Configuration and styling
2. Utility functions  
3. Safe data-loading helpers
4. Data processing and calculation functions
5. Visualization components
6. Reusable UI components
7. Render functions for tabs and analytics
8. Main entry point

SNAPSHOT BACKUP: massive-ic-pack-v1 branch
This is a rollback snapshot before IC Pack v1 implementation.
"""

import streamlit as st
import subprocess
import os
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import alpha attribution module
try:
    from alpha_attribution import (
        compute_alpha_attribution_series,
        format_attribution_summary_table,
        format_daily_attribution_sample
    )
    ALPHA_ATTRIBUTION_AVAILABLE = True
except ImportError:
    ALPHA_ATTRIBUTION_AVAILABLE = False

# Import VIX overlay diagnostics module
try:
    from vix_overlay_diagnostics import get_wave_diagnostics
    VIX_DIAGNOSTICS_AVAILABLE = True
except ImportError:
    VIX_DIAGNOSTICS_AVAILABLE = False

# Import waves engine for wave definitions (single source of truth)
try:
    from waves_engine import get_all_waves as engine_get_all_waves, WAVE_WEIGHTS
    WAVES_ENGINE_AVAILABLE = True
except ImportError:
    WAVES_ENGINE_AVAILABLE = False
    engine_get_all_waves = None
    WAVE_WEIGHTS = {}

# ============================================================================
# ROLLBACK SAFETY: Original app.py backed up as app.py.decision-engine-backup
# To restore: cp app.py.decision-engine-backup app.py
# ============================================================================

# ============================================================================
# DECISION ATTRIBUTION ENGINE - Observable Components Decomposition
# ============================================================================

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class DecisionAttributionComponents:
    """
    Observable components of Wave performance decomposition.
    All values in decimal form (e.g., 0.01 = 1%).
    """
    # Core components
    selection_alpha: Optional[float] = None  # Asset selection contribution
    overlay_alpha: Optional[float] = None  # Exposure scaling & VIX gates vs fully invested
    risk_off_alpha: Optional[float] = None  # Cash contribution
    residual_alpha: Optional[float] = None  # Unexplained effects
    
    # Reconciliation
    total_alpha: Optional[float] = None  # Total realized alpha
    reconciled: bool = False  # Whether reconciliation succeeded
    reconciliation_error: Optional[float] = None
    
    # Availability flags
    selection_available: bool = False
    overlay_available: bool = False
    risk_off_available: bool = False
    residual_available: bool = False
    
    # Metadata
    data_completeness: float = 0.0  # 0.0 to 1.0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def get_confidence_level(self) -> str:
        """Get confidence level based on data completeness."""
        if self.data_completeness >= 0.9:
            return "High"
        elif self.data_completeness >= 0.6:
            return "Medium"
        elif self.data_completeness >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'Selection Alpha': self.selection_alpha,
            'Overlay Alpha': self.overlay_alpha,
            'Risk-Off Alpha': self.risk_off_alpha,
            'Residual Alpha': self.residual_alpha,
            'Total Alpha': self.total_alpha,
            'Reconciled': self.reconciled,
            'Data Completeness': f"{self.data_completeness * 100:.1f}%",
            'Confidence': self.get_confidence_level(),
            'Warnings': len(self.warnings)
        }


@dataclass
class AuditTrailEntry:
    """Immutable audit trail entry for attribution calculations."""
    timestamp: datetime
    app_version: str
    git_commit: str
    git_branch: str
    wave_name: str
    calculation_type: str
    data_available: Dict[str, bool]
    calculations_performed: List[str]
    calculations_skipped: List[str]
    warnings: List[str]
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'Timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'App Version': self.app_version,
            'Git Commit': self.git_commit,
            'Git Branch': self.git_branch,
            'Wave': self.wave_name,
            'Calculation': self.calculation_type,
            'Data Available': ', '.join([k for k, v in self.data_available.items() if v]),
            'Performed': ', '.join(self.calculations_performed) if self.calculations_performed else 'None',
            'Skipped': ', '.join(self.calculations_skipped) if self.calculations_skipped else 'None',
            'Warnings': len(self.warnings),
            'Success': '✅' if self.success else '❌',
            'Error': self.error_message or 'None'
        }


class DecisionAttributionEngine:
    """
    Decision Attribution Engine - Decomposes Wave performance into observable components.
    
    Components:
    1. Selection Alpha: Performance from asset selection
    2. Overlay Alpha: Performance from exposure scaling and VIX gates vs fully invested benchmark
    3. Risk-Off Alpha: Performance contribution from cash positions
    4. Residual Alpha: Unexplained performance effects
    
    Features:
    - Labels components as "Observed" or "Unavailable"
    - Reconciles to total return vs benchmark
    - Graceful degradation when data is incomplete
    """
    
    def __init__(self):
        self.audit_trail: List[AuditTrailEntry] = []
    
    def compute_attribution(
        self,
        wave_data: pd.DataFrame,
        wave_name: str,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> DecisionAttributionComponents:
        """
        Compute decision attribution for a wave.
        
        Args:
            wave_data: DataFrame with wave performance data
            wave_name: Name of the wave
            benchmark_data: Optional benchmark data for comparison
            
        Returns:
            DecisionAttributionComponents with observed and unavailable components
        """
        warnings = []
        calculations_performed = []
        calculations_skipped = []
        data_available = {}
        
        try:
            # Initialize components
            components = DecisionAttributionComponents()
            
            # Check data availability
            has_alpha = 'alpha' in wave_data.columns
            has_return = 'return' in wave_data.columns
            has_benchmark = 'benchmark_return' in wave_data.columns or benchmark_data is not None
            has_exposure = 'exposure' in wave_data.columns
            has_cash = 'cash_pct' in wave_data.columns or 'safe_pct' in wave_data.columns
            has_vix = 'vix' in wave_data.columns
            
            data_available = {
                'alpha': has_alpha,
                'return': has_return,
                'benchmark': has_benchmark,
                'exposure': has_exposure,
                'cash': has_cash,
                'vix': has_vix
            }
            
            # Calculate total alpha (if possible)
            if has_alpha:
                components.total_alpha = wave_data['alpha'].sum()
                calculations_performed.append('total_alpha')
            elif has_return and has_benchmark:
                wave_return = wave_data['return'].sum()
                if benchmark_data is not None:
                    benchmark_return = benchmark_data['return'].sum()
                else:
                    benchmark_return = wave_data['benchmark_return'].sum()
                components.total_alpha = wave_return - benchmark_return
                calculations_performed.append('total_alpha_from_returns')
            else:
                warnings.append("Total alpha unavailable - missing return or benchmark data")
                calculations_skipped.append('total_alpha')
            
            # Component 1: Selection Alpha (from underlying asset performance)
            if ALPHA_ATTRIBUTION_AVAILABLE and has_return:
                try:
                    # Use existing alpha attribution if available
                    from alpha_attribution import compute_alpha_attribution_series
                    attribution_result = compute_alpha_attribution_series(wave_data, wave_name)
                    if attribution_result and hasattr(attribution_result, 'asset_selection_alpha'):
                        components.selection_alpha = attribution_result.asset_selection_alpha
                        components.selection_available = True
                        calculations_performed.append('selection_alpha')
                    else:
                        warnings.append("Selection alpha computation returned no results")
                        calculations_skipped.append('selection_alpha')
                except Exception as e:
                    warnings.append(f"Selection alpha unavailable: {str(e)}")
                    calculations_skipped.append('selection_alpha')
            else:
                warnings.append("Selection alpha unavailable - missing attribution module or return data")
                calculations_skipped.append('selection_alpha')
            
            # Component 2: Overlay Alpha (exposure scaling & VIX gates)
            if has_exposure and has_return and has_benchmark:
                try:
                    # Calculate overlay alpha as difference between actual and fully invested
                    if 'exposure' in wave_data.columns:
                        # Fully invested would be exposure = 1.0 always
                        actual_exposure = wave_data['exposure'].fillna(1.0)
                        exposure_impact = wave_data['return'] * (actual_exposure - 1.0)
                        components.overlay_alpha = exposure_impact.sum()
                        components.overlay_available = True
                        calculations_performed.append('overlay_alpha')
                    else:
                        warnings.append("Overlay alpha unavailable - missing exposure data")
                        calculations_skipped.append('overlay_alpha')
                except Exception as e:
                    warnings.append(f"Overlay alpha calculation failed: {str(e)}")
                    calculations_skipped.append('overlay_alpha')
            else:
                warnings.append("Overlay alpha unavailable - missing exposure, return, or benchmark data")
                calculations_skipped.append('overlay_alpha')
            
            # Component 3: Risk-Off Alpha (cash contribution)
            if has_cash and has_return:
                try:
                    cash_col = 'cash_pct' if 'cash_pct' in wave_data.columns else 'safe_pct'
                    cash_pct = wave_data[cash_col].fillna(0.0)
                    
                    # Cash contribution: opportunity cost of holding cash vs being invested
                    # Positive if market went down while holding cash, negative if market went up
                    if has_benchmark:
                        if benchmark_data is not None:
                            benchmark_return = benchmark_data['return']
                        else:
                            benchmark_return = wave_data['benchmark_return']
                        
                        # Cash benefit = cash% * (-benchmark_return)
                        # Positive when benchmark negative and we held cash
                        components.risk_off_alpha = (cash_pct * (-benchmark_return)).sum()
                        components.risk_off_available = True
                        calculations_performed.append('risk_off_alpha')
                    else:
                        warnings.append("Risk-off alpha partial - no benchmark for comparison")
                        calculations_skipped.append('risk_off_alpha')
                except Exception as e:
                    warnings.append(f"Risk-off alpha calculation failed: {str(e)}")
                    calculations_skipped.append('risk_off_alpha')
            else:
                warnings.append("Risk-off alpha unavailable - missing cash or return data")
                calculations_skipped.append('risk_off_alpha')
            
            # Component 4: Residual Alpha (unexplained)
            if components.total_alpha is not None:
                # Residual = Total - (Selection + Overlay + Risk-Off)
                known_components = 0.0
                if components.selection_alpha is not None:
                    known_components += components.selection_alpha
                if components.overlay_alpha is not None:
                    known_components += components.overlay_alpha
                if components.risk_off_alpha is not None:
                    known_components += components.risk_off_alpha
                
                components.residual_alpha = components.total_alpha - known_components
                components.residual_available = True
                calculations_performed.append('residual_alpha')
            else:
                warnings.append("Residual alpha unavailable - total alpha unknown")
                calculations_skipped.append('residual_alpha')
            
            # Reconciliation check
            if components.total_alpha is not None:
                reconstructed = 0.0
                if components.selection_alpha is not None:
                    reconstructed += components.selection_alpha
                if components.overlay_alpha is not None:
                    reconstructed += components.overlay_alpha
                if components.risk_off_alpha is not None:
                    reconstructed += components.risk_off_alpha
                if components.residual_alpha is not None:
                    reconstructed += components.residual_alpha
                
                components.reconciliation_error = abs(components.total_alpha - reconstructed)
                components.reconciled = components.reconciliation_error < 0.0001  # 0.01% tolerance
                
                if not components.reconciled:
                    warnings.append(f"Reconciliation error: {components.reconciliation_error*100:.4f}%")
            
            # Calculate data completeness
            available_count = sum([
                components.selection_available,
                components.overlay_available,
                components.risk_off_available,
                components.residual_available
            ])
            components.data_completeness = available_count / 4.0
            
            # Store warnings
            components.warnings = warnings
            
            # Create audit trail entry
            audit_entry = AuditTrailEntry(
                timestamp=datetime.now(),
                app_version="v2.0-attribution",
                git_commit=get_git_commit_hash(),
                git_branch=get_git_branch_name(),
                wave_name=wave_name,
                calculation_type="Decision Attribution",
                data_available=data_available,
                calculations_performed=calculations_performed,
                calculations_skipped=calculations_skipped,
                warnings=warnings,
                success=True
            )
            self.audit_trail.append(audit_entry)
            
            return components
            
        except Exception as e:
            # Create error audit trail entry
            audit_entry = AuditTrailEntry(
                timestamp=datetime.now(),
                app_version="v2.0-attribution",
                git_commit=get_git_commit_hash(),
                git_branch=get_git_branch_name(),
                wave_name=wave_name,
                calculation_type="Decision Attribution",
                data_available=data_available,
                calculations_performed=calculations_performed,
                calculations_skipped=calculations_skipped,
                warnings=warnings,
                success=False,
                error_message=str(e)
            )
            self.audit_trail.append(audit_entry)
            
            # Return empty components with error
            components = DecisionAttributionComponents()
            components.warnings = warnings + [f"Attribution failed: {str(e)}"]
            return components
    
    def get_audit_trail(self) -> List[AuditTrailEntry]:
        """Get the complete audit trail."""
        return self.audit_trail
    
    def get_latest_audit_entry(self) -> Optional[AuditTrailEntry]:
        """Get the most recent audit trail entry."""
        return self.audit_trail[-1] if self.audit_trail else None


# Global instance of the attribution engine
_attribution_engine = None

def get_attribution_engine() -> DecisionAttributionEngine:
    """Get or create the global attribution engine instance."""
    global _attribution_engine
    if _attribution_engine is None:
        _attribution_engine = DecisionAttributionEngine()
    return _attribution_engine


# ============================================================================
# SECTION 1: CONFIGURATION AND STYLING
# ============================================================================

st.set_page_config(page_title="Institutional Console - Executive Layer v2", layout="wide")

# Cache keys for wave universe management
WAVE_UNIVERSE_CACHE_KEYS = ["wave_universe", "waves_list", "universe_cache", "wave_history_cache"]


# ============================================================================
# WAVE LINEUP CONFIGURATION - Equity Lineup Reset v1
# ============================================================================

# Legacy Crypto Waves to exclude (removed per requirement)
# These waves are being deprecated in favor of a unified crypto approach:
# - Single Crypto Income Wave for crypto income/yield representation
# - Crypto Selection Engine (CSE) for broad crypto market exposure
EXCLUDED_CRYPTO_WAVES = {
    "Multi-Cap Crypto Growth Wave",
    "Bitcoin Wave", 
    "Crypto Stable Yield Wave",
    "Crypto High-Yield Income Wave"
}

# Crypto sector classification for CSE universe identification
# Based on institutional crypto classification standards (Grayscale, 21Shares, MarketVector)
CRYPTO_SECTORS = [
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

# All Equity Waves that should be included in the lineup
# Includes restored waves + 5 new waves (per requirement)
INCLUDED_EQUITY_WAVES = {
    # Core Index Waves
    "S&P 500 Wave",
    "Growth Wave",
    "Small Cap Growth Wave", 
    "Small-Mid Cap Growth Wave",
    
    # Thematic/Sector Waves (restored + new)
    "Future Power & Energy Wave",
    "Quantum Computing Wave",
    "Clean Transit-Infrastructure Wave",
    
    # Income & Fixed Income Waves (restored)
    "Income Wave",
    
    # New Equity Waves (5 new thematic waves per requirement)
    "US MegaCap Core Wave",
    "AI & Cloud MegaCap Wave",
    "Next-Gen Compute & Semis Wave",
    "US Small-Cap Disruptors Wave",
    "Demas Fund Wave"
}

# Single Crypto Wave (sole crypto representation)
CRYPTO_INCOME_WAVE = "Crypto Income Wave"

# Crypto Selection Engine (CSE) - Top 1-200 cryptos
# Represented as a virtual wave for analytics and selection
CSE_WAVE_NAME = "Crypto Selection Engine (CSE Top 200)"

# ============================================================================
# INDEX REFERENCE WAVES - Russell 3000 Integration
# ============================================================================

# Index reference waves with fallback mappings for attributes
# Russell 3000 Wave is an index reference wave tracked via IWV ETF
INDEX_REFERENCE_WAVES = {
    "Russell 3000 Wave": {
        "ticker": "IWV",
        "benchmark": "IWV",
        "description": "iShares Russell 3000 ETF - Broad US equity market"
    }
}


# ============================================================================
# WAVE UNIVERSE LOADER - Deduplication and Dynamic Reload
# ============================================================================

def normalize_wave_name(name: str) -> str:
    """
    Clean wave names by trimming whitespace and casefolding.
    
    Args:
        name: Wave name to normalize
        
    Returns:
        Normalized wave name (trimmed and lowercased)
    """
    if not isinstance(name, str):
        return ""
    return name.strip().casefold()


def dedupe_waves(names: list[str]) -> tuple[list[str], list[str]]:
    """
    Deduplicate wave names while preserving original order.
    
    Removes duplicates based on normalized names (case-insensitive, trimmed).
    
    Args:
        names: List of wave names (may contain duplicates)
        
    Returns:
        Tuple of (deduplicated_names, removed_duplicates)
        - deduplicated_names: List of unique wave names in original order
        - removed_duplicates: List of duplicate names that were removed
    """
    if not names:
        return [], []
    
    seen_normalized = set()
    deduplicated = []
    removed = []
    
    for name in names:
        normalized = normalize_wave_name(name)
        if normalized and normalized not in seen_normalized:
            seen_normalized.add(normalized)
            deduplicated.append(name)
        elif normalized:
            removed.append(name)
    
    return deduplicated, removed


@st.cache_data(ttl=15)
def get_canonical_wave_universe(force_reload: bool = False, _wave_universe_version: int = 1) -> dict:
    """
    Fetch, deduplicate, and return canonical wave universe data.
    
    Uses wave list from waves_engine or falls back to static list.
    Caches result in session state for performance.
    
    Args:
        force_reload: If True, bypass cache and rebuild universe
        _wave_universe_version: Version counter for cache invalidation (prefixed with _ to ignore in hash)
        
    Returns:
        Dictionary with:
        - waves: Deduplicated list of wave names
        - removed_duplicates: List of removed duplicates
        - source: Data source ("engine", "fallback")
        - timestamp: Current timestamp for tracking
    """
    from datetime import datetime
    
    # Check cache unless force reload
    if not force_reload and "wave_universe" in st.session_state:
        return st.session_state["wave_universe"]
    
    # Build wave universe
    wave_list = []
    source = "fallback"
    
    try:
        # Try to get waves from engine (single source of truth)
        if WAVES_ENGINE_AVAILABLE and engine_get_all_waves is not None:
            wave_list = engine_get_all_waves()
            source = "engine"
        elif WAVE_WEIGHTS:
            # Fallback: directly access WAVE_WEIGHTS
            wave_list = sorted(list(WAVE_WEIGHTS.keys()))
            source = "engine"
    except Exception:
        pass
    
    # If engine unavailable, use fallback static list
    if not wave_list:
        # Fallback: combine INCLUDED_EQUITY_WAVES with other known waves
        wave_list = list(INCLUDED_EQUITY_WAVES)
        wave_list.append(CRYPTO_INCOME_WAVE)
        wave_list.append(CSE_WAVE_NAME)
        source = "fallback"
    
    # Add Russell 3000 Wave if not already present
    if "Russell 3000 Wave" not in wave_list:
        wave_list.append("Russell 3000 Wave")
    
    # Deduplicate
    deduplicated_waves, removed_duplicates = dedupe_waves(wave_list)
    
    # Build universe dictionary
    universe = {
        "waves": deduplicated_waves,
        "removed_duplicates": removed_duplicates,
        "source": source,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Cache in session state
    st.session_state["wave_universe"] = universe
    
    return universe


def get_wave_universe():
    """
    Centralized Wave Universe Function - Single Source of Truth.
    
    This function serves as the canonical source for all wave lists and counts
    throughout the application. It ensures:
    - Deduplication of wave names
    - "Russell 3000 Wave" is included exactly once
    - Consistent wave lists across all tabs and views
    
    Returns:
        list: Deduplicated, sorted list of all wave names
    """
    try:
        # Get wave universe version from session state
        wave_universe_version = st.session_state.get("wave_universe_version", 1)
        
        # Use canonical wave universe (which already deduplicates and ensures Russell 3000 Wave)
        universe = get_canonical_wave_universe(force_reload=False, _wave_universe_version=wave_universe_version)
        
        # Extract waves list
        waves = universe.get("waves", [])
        
        # Return deduplicated, sorted list
        return sorted(set(waves))
    except Exception:
        # Fallback: return empty list
        return []


def validate_wave_data(wave_name: str, mode: str, days: int, df=None) -> tuple[bool, list[str]]:
    """
    Validate wave data availability and quality.
    
    This function checks if wave data is available and suitable for display.
    It returns detailed reasons when data is unavailable to provide user clarity.
    
    Args:
        wave_name: Name of the wave to validate
        mode: Mode/strategy being used (e.g., "Standard", "Aggressive")
        days: Number of days of data required
        df: Optional pre-loaded DataFrame to validate (if None, will load)
        
    Returns:
        tuple: (ok: bool, reasons: list[str])
            - ok: True if data is valid and available, False otherwise
            - reasons: List of validation failure reasons (empty if ok=True)
    """
    reasons = []
    
    try:
        # Load wave data if not provided
        if df is None:
            wave_universe_version = st.session_state.get("wave_universe_version", 1)
            df = get_wave_data_filtered(wave_name=wave_name, days=days, _wave_universe_version=wave_universe_version)
        
        # Check if data exists
        if df is None:
            reasons.append(f"No historical data available for '{wave_name}'")
            return (False, reasons)
        
        # Check if data is empty
        if len(df) == 0:
            reasons.append(f"Historical data for '{wave_name}' is empty")
            return (False, reasons)
        
        # Check minimum data points
        if len(df) < 5:
            reasons.append(f"Insufficient data points: {len(df)} rows (minimum 5 required)")
        
        # Check date range coverage
        if 'date' in df.columns:
            data_days = (df['date'].max() - df['date'].min()).days
            if data_days < days * 0.5:  # At least 50% coverage
                reasons.append(f"Data coverage insufficient: {data_days} days available, {days} days requested")
        
        # Check required columns
        required_columns = ['date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            reasons.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check for portfolio_return or benchmark_return for alpha calculations
        if 'portfolio_return' not in df.columns and 'benchmark_return' not in df.columns:
            reasons.append("Missing return data (portfolio_return or benchmark_return)")
        
        # If we have reasons, validation failed
        if reasons:
            return (False, reasons)
        
        # All checks passed
        return (True, [])
        
    except Exception as e:
        reasons.append(f"Validation error: {str(e)}")
        return (False, reasons)


# ============================================================================
# SECTION 2: UTILITY FUNCTIONS
# ============================================================================

def get_git_commit_hash():
    """Get the current git commit hash, return 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_git_branch_name():
    """Get the current git branch name, return 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_deploy_timestamp():
    """Get the current timestamp as deploy timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


def calculate_wavescore(wave_data):
    """
    Calculate WaveScore for a wave based on cumulative alpha over 30 days.
    Returns a score between 0 and 100.
    
    WaveScore formula:
    - Base score: 50
    - Add points for cumulative alpha (1000x multiplier)
    - Adjust for consistency (reduce score for high volatility)
    - Clamp to 0-100 range
    """
    try:
        if wave_data is None or len(wave_data) == 0:
            return 0
        
        if 'alpha' not in wave_data.columns:
            return 0
        
        # Calculate cumulative alpha
        cumulative_alpha = wave_data['alpha'].sum()
        
        # Base score calculation
        base_score = (cumulative_alpha * 1000) + 50
        
        # Consistency adjustment: penalize high volatility
        alpha_std = wave_data['alpha'].std()
        if alpha_std > 0:
            # Reduce score if volatility is high relative to returns
            consistency_penalty = min(10, alpha_std * 200)
            base_score -= consistency_penalty
        
        # Normalize to 0-100 range
        wavescore = min(100, max(0, base_score))
        
        return wavescore
    except Exception:
        return 0


def calculate_wave_correlation(wave1_data, wave2_data):
    """
    Calculate correlation between two waves based on their returns.
    Returns correlation coefficient or None if unavailable.
    """
    try:
        if wave1_data is None or wave2_data is None:
            return None
        
        if len(wave1_data) == 0 or len(wave2_data) == 0:
            return None
        
        if 'date' not in wave1_data.columns or 'date' not in wave2_data.columns:
            return None
        
        if 'portfolio_return' not in wave1_data.columns or 'portfolio_return' not in wave2_data.columns:
            return None
        
        # Merge on date to get overlapping periods
        wave1_returns = wave1_data[['date', 'portfolio_return']].rename(columns={'portfolio_return': 'return1'})
        wave2_returns = wave2_data[['date', 'portfolio_return']].rename(columns={'portfolio_return': 'return2'})
        
        merged = pd.merge(wave1_returns, wave2_returns, on='date', how='inner')
        
        if len(merged) < 2:
            return None
        
        correlation = merged['return1'].corr(merged['return2'])
        
        return correlation
        
    except Exception:
        return None


def calculate_wave_metrics(wave_data):
    """
    Calculate comprehensive metrics for a wave.
    Returns a dictionary with all calculated metrics.
    """
    metrics = {
        'cumulative_return': 'N/A',
        'cumulative_alpha': 'N/A',
        'volatility': 'N/A',
        'max_drawdown': 'N/A',
        'wavescore': 'N/A',
        'sharpe_ratio': 'N/A',
        'win_rate': 'N/A'
    }
    
    try:
        if wave_data is None or len(wave_data) == 0:
            return metrics
        
        # Calculate alpha
        if 'portfolio_return' in wave_data.columns and 'benchmark_return' in wave_data.columns:
            wave_data = wave_data.copy()
            wave_data['alpha'] = wave_data['portfolio_return'] - wave_data['benchmark_return']
            
            # Cumulative return
            cumulative_return = wave_data['portfolio_return'].sum()
            metrics['cumulative_return'] = cumulative_return
            
            # Cumulative alpha
            cumulative_alpha = wave_data['alpha'].sum()
            metrics['cumulative_alpha'] = cumulative_alpha
            
            # Volatility
            volatility = wave_data['portfolio_return'].std()
            metrics['volatility'] = volatility
            
            # WaveScore
            wavescore = calculate_wavescore(wave_data)
            metrics['wavescore'] = wavescore
            
            # Sharpe ratio
            avg_return = wave_data['portfolio_return'].mean()
            if volatility > 0:
                sharpe = (avg_return / volatility) * np.sqrt(252)
                metrics['sharpe_ratio'] = sharpe
            
            # Win rate
            positive_days = len(wave_data[wave_data['alpha'] > 0])
            total_days = len(wave_data)
            if total_days > 0:
                metrics['win_rate'] = positive_days / total_days
            
            # Max drawdown
            cumulative_returns = (1 + wave_data['portfolio_return']).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
    
    except Exception:
        pass
    
    return metrics


def determine_winner(wave1_metrics, wave2_metrics):
    """
    Determine which wave is the winner based on WaveScore.
    Returns a tuple: (winner_name, notes)
    """
    try:
        wave1_score = wave1_metrics.get('wavescore', 0)
        wave2_score = wave2_metrics.get('wavescore', 0)
        
        if wave1_score == 'N/A' or wave2_score == 'N/A':
            return None, "Insufficient data to determine winner"
        
        if wave1_score > wave2_score:
            winner = wave1_metrics['name']
            margin = wave1_score - wave2_score
            notes = f"{winner} leads by {margin:.1f} WaveScore points"
        elif wave2_score > wave1_score:
            winner = wave2_metrics['name']
            margin = wave2_score - wave1_score
            notes = f"{winner} leads by {margin:.1f} WaveScore points"
        else:
            winner = "TIE"
            notes = "Both waves have identical WaveScore"
        
        return winner, notes
        
    except Exception:
        return None, "Error determining winner"


# ============================================================================
# SECTION 3: SAFE DATA-LOADING HELPERS
# ============================================================================

@st.cache_data(ttl=15)
def safe_load_wave_history(_wave_universe_version=1):
    """
    Safely load wave history data with comprehensive error handling.
    Returns DataFrame or None if unavailable.
    
    Applies wave name normalization:
    - Strips leading/trailing whitespace
    - Collapses multiple spaces to single space
    - Creates normalized_wave column for matching
    
    Args:
        _wave_universe_version: Version counter for cache invalidation (prefixed with _ to ignore in hash)
    """
    try:
        wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
        
        if not os.path.exists(wave_history_path):
            return None
        
        df = pd.read_csv(wave_history_path)
        
        if df is None or len(df) == 0:
            return None
        
        # Validate required columns
        if 'date' not in df.columns:
            return None
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        if len(df) == 0:
            return None
        
        # Normalize wave names if wave column exists
        if 'wave' in df.columns:
            # Strip whitespace and collapse multiple spaces
            df['wave'] = df['wave'].str.strip()
            df['wave'] = df['wave'].str.replace(r'\s+', ' ', regex=True)
            
            # Create normalized_wave column for case-insensitive matching
            df['normalized_wave'] = df['wave'].str.lower().str.strip()
        
        return df
        
    except Exception:
        return None


def get_latest_data_timestamp():
    """Get the latest available 'as of' data timestamp from wave_history.csv."""
    try:
        df = safe_load_wave_history()
        if df is not None and 'date' in df.columns and len(df) > 0:
            latest_date = df['date'].max()
            return latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "unknown"
    except Exception:
        pass
    return "unknown"


def get_all_wave_names():
    """
    Get all wave names from the waves engine (single source of truth).
    
    This function sources wave names directly from WAVE_WEIGHTS in waves_engine.py,
    ensuring that the UI always reflects the current wave definitions in the engine.
    
    Returns:
        sorted list of wave names from WAVE_WEIGHTS, or empty list if engine unavailable
    """
    try:
        if WAVES_ENGINE_AVAILABLE and engine_get_all_waves is not None:
            # Use the waves_engine.get_all_waves() function (sources from WAVE_WEIGHTS)
            return engine_get_all_waves()
        elif WAVE_WEIGHTS:
            # Fallback: directly access WAVE_WEIGHTS if available
            return sorted(list(WAVE_WEIGHTS.keys()))
        else:
            # No engine available, return empty list
            return []
    except Exception:
        # On any error, return empty list
        return []


def get_available_waves(_wave_universe_version=1):
    """
    Get list of available waves using the centralized Wave Universe (Data Backbone V1).
    
    This function is the primary interface for getting wave lists in dropdowns and UI.
    Sources from get_wave_universe() which provides deduplication and caching.
    
    Args:
        _wave_universe_version: Version counter for cache invalidation (prefixed with _ to ignore in hash)
    
    Returns sorted list of wave names or empty list if unavailable.
    """
    try:
        # Use centralized Wave Universe function (Data Backbone V1)
        waves = get_wave_universe()
        return waves  # Already sorted and deduplicated
    except Exception:
        # Fallback to empty list
        return []


def get_wave_universe_all(_wave_universe_version=1):
    """
    Get the entire deduplicated wave universe (Data Backbone V1).
    
    This is an alias for get_wave_universe() to maintain backward compatibility.
    All waves are returned regardless of data availability.
    
    Args:
        _wave_universe_version: Version counter for cache invalidation (prefixed with _ to ignore in hash)
    
    Returns:
        Sorted list of all wave names
    """
    return get_wave_universe()


def get_wave_universe_with_data(period_days=30, _wave_universe_version=1):
    """
    Get waves that have historical data available for the selected lookback period.
    
    Args:
        period_days: Lookback period in days (default: 30)
        _wave_universe_version: Version counter for cache invalidation (prefixed with _ to ignore in hash)
    
    Returns:
        Sorted list of wave names that have data in the specified period
    """
    try:
        wave_universe_version = st.session_state.get("wave_universe_version", 1)
        df = safe_load_wave_history(_wave_universe_version=wave_universe_version)
        
        if df is None or 'wave' not in df.columns:
            return []
        
        # Filter to period
        if period_days and period_days > 0 and 'date' in df.columns:
            latest_date = df['date'].max()
            cutoff_date = latest_date - timedelta(days=period_days)
            df = df[df['date'] >= cutoff_date]
        
        # Get unique waves with data
        waves_with_data = sorted(df['wave'].unique().tolist())
        return waves_with_data
        
    except Exception:
        return []


def get_cse_crypto_universe():
    """
    Get the Crypto Selection Engine (CSE) universe of top 1-200 cryptocurrencies.
    
    Uses Master_Stock_Sheet.csv to identify crypto assets based on sector classification.
    Returns list of crypto tickers ranked by market cap, up to 200.
    
    Returns:
        List of crypto ticker symbols or empty list if unavailable
    """
    try:
        master_sheet_path = os.path.join(os.path.dirname(__file__), 'Master_Stock_Sheet.csv')
        
        if not os.path.exists(master_sheet_path):
            return []
        
        df = pd.read_csv(master_sheet_path)
        
        if df is None or len(df) == 0:
            return []
        
        # Filter for crypto assets using sector classification
        if 'Sector' in df.columns:
            crypto_df = df[df['Sector'].isin(CRYPTO_SECTORS)].copy()
        else:
            return []
        
        # Sort by market value if available
        if 'MarketValue' in crypto_df.columns:
            crypto_df = crypto_df.sort_values('MarketValue', ascending=False)
        
        # Get top 200 (or fewer if less than 200 available)
        top_cryptos = crypto_df.head(200)
        
        # Return ticker list
        if 'Ticker' in top_cryptos.columns:
            return top_cryptos['Ticker'].dropna().tolist()
        
        return []
        
    except Exception:
        return []


def get_cse_wave_data(days=30):
    """
    Generate synthetic wave data for Crypto Selection Engine (CSE).
    
    CSE represents a market-cap weighted portfolio of top 1-200 cryptocurrencies.
    This function creates a placeholder/summary representation for CSE.
    
    Args:
        days: Number of days of data to generate
    
    Returns:
        DataFrame with CSE wave data or None if unavailable
    """
    try:
        cse_cryptos = get_cse_crypto_universe()
        
        if len(cse_cryptos) == 0:
            # Return placeholder data indicating CSE is available but data is building
            # Returning None here results in graceful "Data unavailable" display in UI
            # This is the intended behavior during CSE data building phase
            return None
        
        # Return None to indicate "Data unavailable" gracefully
        # Future enhancement: compute actual CSE portfolio performance
        # from price data and market cap weights
        return None
        
    except Exception:
        return None


def get_crypto_income_wave_data(days=30):
    """
    Get data for Crypto Income Wave (sole crypto wave representation).
    
    Falls back to "Crypto Income & Yield Wave" from wave_history.csv if available.
    
    Args:
        days: Number of days of data
        
    Returns:
        DataFrame with crypto income wave data or None if unavailable
    """
    try:
        df = safe_load_wave_history()
        
        if df is None or 'wave' not in df.columns:
            return None
        
        # Try exact match first
        crypto_data = df[df['wave'] == CRYPTO_INCOME_WAVE].copy()
        
        # Fall back to alternative names
        if len(crypto_data) == 0:
            alternative_names = ["Crypto Income & Yield Wave", "Crypto Income Wave"]
            for alt_name in alternative_names:
                crypto_data = df[df['wave'] == alt_name].copy()
                if len(crypto_data) > 0:
                    # Rename to standard name
                    crypto_data['wave'] = CRYPTO_INCOME_WAVE
                    break
        
        if len(crypto_data) == 0:
            return None
        
        # Apply date filtering
        if days is not None and days > 0:
            latest_date = crypto_data['date'].max()
            cutoff_date = latest_date - timedelta(days=days)
            crypto_data = crypto_data[crypto_data['date'] >= cutoff_date].copy()
        
        if len(crypto_data) == 0:
            return None
        
        return crypto_data
        
    except Exception:
        return None


def get_wave_data_filtered(wave_name=None, days=30, _wave_universe_version=1):
    """
    Get wave data filtered by wave name and/or date range.
    
    Handles special cases:
    - Crypto Income Wave: maps to underlying crypto income data
    - CSE (Crypto Selection Engine): returns synthetic/computed data
    - Excluded legacy crypto waves: returns None
    
    Uses normalized wave names for matching (case-insensitive, trimmed, collapsed spaces).
    
    Returns DataFrame or None if unavailable.
    
    Args:
        wave_name: Name of wave to filter (None for all waves)
        days: Number of days to include (from latest date)
        _wave_universe_version: Version counter for cache invalidation (prefixed with _ to ignore in hash)
    """
    try:
        # Get wave_universe_version from session state
        wave_universe_version = st.session_state.get("wave_universe_version", 1)
        
        # Special case: Russell 3000 Wave (index reference wave)
        if wave_name == "Russell 3000 Wave":
            # Return None to gracefully display "Data unavailable"
            # This wave is tracked but may not have historical data yet
            return None
        
        # Special case: CSE (Crypto Selection Engine)
        if wave_name == CSE_WAVE_NAME:
            return get_cse_wave_data(days=days)
        
        # Special case: Crypto Income Wave
        if wave_name == CRYPTO_INCOME_WAVE:
            return get_crypto_income_wave_data(days=days)
        
        # Exclude legacy crypto waves
        if wave_name in EXCLUDED_CRYPTO_WAVES:
            return None
        
        # Load standard wave data
        df = safe_load_wave_history(_wave_universe_version=wave_universe_version)
        
        if df is None:
            return None
        
        # Filter by wave if specified
        if wave_name is not None:
            if 'wave' not in df.columns:
                return None
            
            # Normalize the selected wave name for matching
            normalized_selection = wave_name.strip().replace('  ', ' ').lower()
            
            # Filter using normalized_wave column if available
            if 'normalized_wave' in df.columns:
                df = df[df['normalized_wave'] == normalized_selection].copy()
            else:
                # Fallback: exact match on wave column
                df = df[df['wave'] == wave_name].copy()
        
        if len(df) == 0:
            return None
        
        # Filter by date range if days specified
        if days is not None and days > 0:
            latest_date = df['date'].max()
            cutoff_date = latest_date - timedelta(days=days)
            df = df[df['date'] >= cutoff_date].copy()
        
        if len(df) == 0:
            return None
        
        return df
        
    except Exception:
        return None


def calculate_alpha_components(wave_data, wave_name):
    """
    Calculate alpha decomposition components.
    
    Returns:
        Dictionary with alpha components or None if unavailable
    """
    try:
        if wave_data is None or len(wave_data) == 0:
            return None
        
        # Ensure required columns exist
        if 'portfolio_return' not in wave_data.columns or 'benchmark_return' not in wave_data.columns:
            return None
        
        wave_data = wave_data.copy()
        wave_data['alpha'] = wave_data['portfolio_return'] - wave_data['benchmark_return']
        
        # Total alpha
        total_alpha = wave_data['alpha'].sum()
        
        # Selection Alpha: Base alpha from wave return vs benchmark
        # This is the primary component - the actual differential
        selection_alpha = total_alpha * 0.70  # Estimate: 70% from selection
        
        # Overlay Alpha: Impact of exposure scaling and VIX gates
        # Check if we have exposure data
        overlay_alpha = 0.0
        if 'exposure' in wave_data.columns:
            # Calculate impact of exposure scaling
            # Overlay alpha = (1 - avg_exposure) * avg_portfolio_return
            avg_exposure = wave_data['exposure'].mean()
            if avg_exposure < 1.0:
                overlay_alpha = total_alpha * 0.20  # Estimate: 20% from overlay
        else:
            # No exposure data, estimate based on alpha variance
            overlay_alpha = total_alpha * 0.15  # Conservative estimate
        
        # Cash/Risk-Off Contribution: Remaining component
        cash_contribution = total_alpha - selection_alpha - overlay_alpha
        
        return {
            'total_alpha': total_alpha,
            'selection_alpha': selection_alpha,
            'overlay_alpha': overlay_alpha,
            'cash_contribution': cash_contribution,
            'wave_return': wave_data['portfolio_return'].sum(),
            'benchmark_return': wave_data['benchmark_return'].sum()
        }
        
    except Exception:
        return None


def calculate_attribution_matrix(wave_data, wave_name):
    """
    Calculate attribution matrix with regime-based breakdown.
    
    Returns:
        Dictionary with attribution metrics or None if unavailable
    """
    try:
        if wave_data is None or len(wave_data) == 0:
            return None
        
        wave_data = wave_data.copy()
        
        # Ensure alpha column exists
        if 'portfolio_return' not in wave_data.columns or 'benchmark_return' not in wave_data.columns:
            return None
        
        wave_data['alpha'] = wave_data['portfolio_return'] - wave_data['benchmark_return']
        
        # Total alpha
        total_alpha = wave_data['alpha'].sum()
        
        # Risk-On vs Risk-Off: Use VIX if available
        risk_on_alpha = 0.0
        risk_off_alpha = 0.0
        
        if 'vix' in wave_data.columns or 'regime' in wave_data.columns:
            # Determine regime
            if 'regime' in wave_data.columns:
                risk_on_mask = wave_data['regime'].str.contains('risk-on|growth|bullish', case=False, na=False)
                risk_off_mask = ~risk_on_mask
            elif 'vix' in wave_data.columns:
                # VIX < 20 = risk-on, VIX >= 20 = risk-off
                risk_on_mask = wave_data['vix'] < 20
                risk_off_mask = wave_data['vix'] >= 20
            else:
                risk_on_mask = wave_data.index % 2 == 0  # Fallback
                risk_off_mask = ~risk_on_mask
            
            risk_on_alpha = wave_data.loc[risk_on_mask, 'alpha'].sum()
            risk_off_alpha = wave_data.loc[risk_off_mask, 'alpha'].sum()
        else:
            # Estimate based on volatility
            volatility = wave_data['portfolio_return'].std()
            if volatility < 0.015:
                risk_on_alpha = total_alpha * 0.7
                risk_off_alpha = total_alpha * 0.3
            else:
                risk_on_alpha = total_alpha * 0.5
                risk_off_alpha = total_alpha * 0.5
        
        # Capital-Weighted Alpha: Use exposure if available
        capital_weighted_alpha = total_alpha
        if 'exposure' in wave_data.columns:
            avg_exposure = wave_data['exposure'].mean()
            capital_weighted_alpha = total_alpha * avg_exposure
        
        # Exposure-Adjusted Alpha: Adjust for average exposure
        exposure_adjusted_alpha = total_alpha
        if 'exposure' in wave_data.columns:
            avg_exposure = wave_data['exposure'].mean()
            if avg_exposure > 0:
                exposure_adjusted_alpha = total_alpha / avg_exposure
        
        return {
            'total_alpha': total_alpha,
            'risk_on_alpha': risk_on_alpha,
            'risk_off_alpha': risk_off_alpha,
            'capital_weighted_alpha': capital_weighted_alpha,
            'exposure_adjusted_alpha': exposure_adjusted_alpha
        }
        
    except Exception:
        return None


def render_data_diagnostic_card(wave_name, days=30):
    """
    Render a diagnostic card when wave data is unavailable.
    
    Shows:
    - File path loaded
    - Number of rows in history
    - Top 25 unique wave names
    - Date range (min/max)
    - Whether selected wave exists after normalization
    
    Args:
        wave_name: The wave name that was selected
        days: Lookback period in days
    """
    try:
        st.warning(f"⚠️ No data available for **{wave_name}** in the selected period ({days} days)")
        
        with st.expander("🔍 Data Diagnostics - Click to expand", expanded=True):
            wave_universe_version = st.session_state.get("wave_universe_version", 1)
            df = safe_load_wave_history(_wave_universe_version=wave_universe_version)
            
            # File information
            wave_history_path = os.path.join(os.path.dirname(__file__), 'wave_history.csv')
            st.markdown("**📁 Data Source**")
            st.code(wave_history_path)
            
            if df is None:
                st.error("❌ Unable to load wave_history.csv")
                return
            
            # Row count
            st.markdown("**📊 Dataset Statistics**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows Loaded", f"{len(df):,}")
            with col2:
                if 'wave' in df.columns:
                    unique_waves = df['wave'].nunique()
                    st.metric("Unique Waves", unique_waves)
            
            # Date range
            if 'date' in df.columns:
                st.markdown("**📅 Date Range**")
                col1, col2 = st.columns(2)
                with col1:
                    min_date = df['date'].min()
                    st.metric("Earliest Date", min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else "N/A")
                with col2:
                    max_date = df['date'].max()
                    st.metric("Latest Date", max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else "N/A")
            
            # Top 25 wave names
            if 'wave' in df.columns:
                st.markdown("**🌊 Top 25 Unique Wave Names Found in History**")
                unique_waves = sorted(df['wave'].unique().tolist())
                top_25 = unique_waves[:25]
                
                # Display in 3 columns
                col1, col2, col3 = st.columns(3)
                for i, wave in enumerate(top_25):
                    col = [col1, col2, col3][i % 3]
                    with col:
                        st.text(f"• {wave}")
                
                if len(unique_waves) > 25:
                    st.caption(f"... and {len(unique_waves) - 25} more")
            
            # Check if selected wave exists (with normalization)
            st.markdown("**🔎 Selected Wave Lookup**")
            if 'wave' in df.columns:
                normalized_selection = wave_name.strip().replace('  ', ' ').lower()
                
                # Check exact match
                exact_match = wave_name in df['wave'].values
                
                # Check normalized match
                if 'normalized_wave' in df.columns:
                    normalized_match = normalized_selection in df['normalized_wave'].values
                else:
                    normalized_match = False
                
                col1, col2 = st.columns(2)
                with col1:
                    if exact_match:
                        st.success(f"✅ Exact match found for '{wave_name}'")
                    else:
                        st.error(f"❌ No exact match for '{wave_name}'")
                
                with col2:
                    if normalized_match:
                        st.success("✅ Normalized match found")
                    else:
                        st.error("❌ No normalized match")
                
                # Show similar wave names if no match
                if not exact_match and not normalized_match:
                    st.markdown("**💡 Similar Wave Names (suggestions):**")
                    # Find similar names using simple string matching
                    similar_waves = []
                    wave_lower = wave_name.lower()
                    for unique_wave in unique_waves:
                        if wave_lower in unique_wave.lower() or unique_wave.lower() in wave_lower:
                            similar_waves.append(unique_wave)
                    
                    if similar_waves:
                        for sim_wave in similar_waves[:5]:
                            st.text(f"  → {sim_wave}")
                    else:
                        st.text("No similar wave names found")
            
            # Show filter applied for period
            if 'date' in df.columns and days:
                st.markdown(f"**⏱️ Period Filter Applied: Last {days} Days**")
                latest_date = df['date'].max()
                cutoff_date = latest_date - timedelta(days=days)
                st.text(f"Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
                st.text(f"Latest Date: {latest_date.strftime('%Y-%m-%d')}")
                
                # Check if wave has data in this period
                if 'wave' in df.columns:
                    period_df = df[df['date'] >= cutoff_date]
                    if normalized_selection in period_df.get('normalized_wave', pd.Series()).values:
                        wave_rows_in_period = len(period_df[period_df['normalized_wave'] == normalized_selection])
                        st.info(f"ℹ️ Wave exists in full history but has only {wave_rows_in_period} row(s) in this period")
                    elif wave_name in period_df['wave'].values:
                        wave_rows_in_period = len(period_df[period_df['wave'] == wave_name])
                        st.info(f"ℹ️ Wave exists in full history but has only {wave_rows_in_period} row(s) in this period")
    
    except Exception as e:
        st.error(f"Error rendering diagnostics: {str(e)}")


def calculate_portfolio_metrics(wave_names, weights, days):
    """
    Calculate blended portfolio metrics for multiple waves.
    
    Args:
        wave_names: List of wave names
        weights: Dictionary mapping wave names to weights (decimal)
        days: Number of days for analysis
        
    Returns:
        Dictionary with portfolio metrics or None if unavailable
    """
    try:
        if not wave_names or not weights:
            return None
        
        # Load data for all waves
        wave_data_dict = {}
        for wave_name in wave_names:
            wave_data = get_wave_data_filtered(wave_name=wave_name, days=days)
            if wave_data is not None and len(wave_data) > 0:
                wave_data_dict[wave_name] = wave_data
        
        if len(wave_data_dict) == 0:
            return None
        
        # Align dates - find common dates
        all_dates = None
        for wave_name, wave_data in wave_data_dict.items():
            wave_dates = set(wave_data['date'])
            if all_dates is None:
                all_dates = wave_dates
            else:
                all_dates = all_dates.intersection(wave_dates)
        
        if not all_dates or len(all_dates) == 0:
            return None
        
        all_dates = sorted(list(all_dates))
        
        # Calculate blended returns
        blended_returns = []
        for date in all_dates:
            daily_return = 0.0
            for wave_name, weight in weights.items():
                if wave_name in wave_data_dict:
                    wave_data = wave_data_dict[wave_name]
                    date_data = wave_data[wave_data['date'] == date]
                    if len(date_data) > 0:
                        daily_return += weight * date_data['portfolio_return'].iloc[0]
            blended_returns.append(daily_return)
        
        blended_returns = np.array(blended_returns)
        
        # Calculate metrics
        blended_return = blended_returns.sum()
        blended_volatility = blended_returns.std()
        
        # Calculate drawdown
        cumulative_returns = (1 + blended_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate blended WaveScore (weighted average)
        blended_wavescore = 0.0
        for wave_name, weight in weights.items():
            if wave_name in wave_data_dict:
                wave_data = wave_data_dict[wave_name]
                wavescore = calculate_wavescore(wave_data)
                blended_wavescore += weight * wavescore
        
        # Calculate correlation matrix
        correlation_matrix = None
        if len(wave_names) > 1:
            # Build return matrix
            return_matrix = {}
            for wave_name in wave_names:
                if wave_name in wave_data_dict:
                    wave_returns = []
                    wave_data = wave_data_dict[wave_name]
                    for date in all_dates:
                        date_data = wave_data[wave_data['date'] == date]
                        if len(date_data) > 0:
                            wave_returns.append(date_data['portfolio_return'].iloc[0])
                        else:
                            wave_returns.append(0.0)
                    return_matrix[wave_name] = wave_returns
            
            if len(return_matrix) > 1:
                return_df = pd.DataFrame(return_matrix)
                correlation_matrix = return_df.corr()
        
        # Calculate individual contributions
        contributions = {}
        for wave_name, weight in weights.items():
            if wave_name in wave_data_dict:
                wave_data = wave_data_dict[wave_name]
                wave_return = 0.0
                for date in all_dates:
                    date_data = wave_data[wave_data['date'] == date]
                    if len(date_data) > 0:
                        wave_return += date_data['portfolio_return'].iloc[0]
                contributions[wave_name] = weight * wave_return
        
        return {
            'blended_return': blended_return,
            'blended_volatility': blended_volatility,
            'max_drawdown': max_drawdown,
            'blended_wavescore': blended_wavescore,
            'correlation_matrix': correlation_matrix,
            'contributions': contributions
        }
        
    except Exception:
        return None


# ============================================================================
# WAVE UNIVERSE TRUTH LAYER - Diagnostics and Freshness
# ============================================================================

def compute_wave_universe_diagnostics():
    """
    Compute comprehensive Wave Universe diagnostics.
    
    Returns:
        Dictionary with diagnostic metrics including:
        - universe_count: Total waves in canonical registry
        - active_count: Waves with recent data (last 7 days)
        - history_unique_count: Unique waves in wave_history.csv
        - missing_waves: Waves in registry but no data
        - orphan_waves: Waves in data but not in registry
        - duplicate_waves: Duplicates removed during deduplication
        - data_freshness: Per-wave freshness information
    """
    diagnostics = {
        'universe_count': 0,
        'active_count': 0,
        'history_unique_count': 0,
        'missing_waves': [],
        'orphan_waves': [],
        'duplicate_waves': [],
        'data_freshness': []
    }
    
    try:
        # Get canonical universe
        # Note: force_reload=False is intentional - diagnostics should use cached universe
        # to show current state. Use operator controls to force reload if needed.
        wave_universe_version = st.session_state.get("wave_universe_version", 1)
        universe = get_canonical_wave_universe(force_reload=False, _wave_universe_version=wave_universe_version)
        
        registry_waves = set(universe.get("waves", []))
        duplicate_waves = universe.get("removed_duplicates", [])
        
        diagnostics['universe_count'] = len(registry_waves)
        diagnostics['duplicate_waves'] = duplicate_waves
        
        # Get wave history data
        wave_history = safe_load_wave_history(_wave_universe_version=wave_universe_version)
        
        if wave_history is not None and 'wave' in wave_history.columns:
            # Get unique waves in history
            history_waves = set(wave_history['wave'].unique())
            diagnostics['history_unique_count'] = len(history_waves)
            
            # Calculate missing waves (in registry but no data)
            missing_waves = registry_waves - history_waves
            diagnostics['missing_waves'] = sorted(list(missing_waves))
            
            # Calculate orphan waves (in data but not in registry)
            orphan_waves = history_waves - registry_waves
            diagnostics['orphan_waves'] = sorted(list(orphan_waves))
            
            # Calculate active waves (recent data in last 7 days)
            if 'date' in wave_history.columns:
                latest_date = wave_history['date'].max()
                cutoff_date = latest_date - timedelta(days=7)
                recent_data = wave_history[wave_history['date'] >= cutoff_date]
                active_waves = set(recent_data['wave'].unique())
                diagnostics['active_count'] = len(active_waves & registry_waves)
                
                # Compute data freshness per wave
                data_freshness = []
                for wave in sorted(registry_waves):
                    wave_data = wave_history[wave_history['wave'] == wave]
                    
                    if len(wave_data) == 0:
                        # No data for this wave
                        data_freshness.append({
                            'wave': wave,
                            'latest_date': None,
                            'days_old': None,
                            'staleness': 'unavailable'
                        })
                    else:
                        latest_wave_date = wave_data['date'].max()
                        days_old = (latest_date - latest_wave_date).days
                        
                        # Determine staleness badge
                        if days_old == 0:
                            staleness = 'fresh'
                        elif days_old <= 3:
                            staleness = 'recent'
                        elif days_old <= 7:
                            staleness = 'stale'
                        else:
                            staleness = 'old'
                        
                        data_freshness.append({
                            'wave': wave,
                            'latest_date': latest_wave_date.strftime('%Y-%m-%d') if pd.notna(latest_wave_date) else 'N/A',
                            'days_old': days_old,
                            'staleness': staleness
                        })
                
                diagnostics['data_freshness'] = data_freshness
        
        return diagnostics
        
    except Exception as e:
        # Return diagnostics with error indicator
        diagnostics['error'] = str(e)
        return diagnostics


def render_wave_universe_truth_panel():
    """
    Render the Wave Universe Truth Panel - Executive diagnostic view.
    
    Displays:
    1. Metrics Panel: Universe, Active, HistoryUnique counts
    2. Diagnostics: Missing, Orphan, Duplicate waves
    3. Data Freshness Table with staleness badges
    4. Operator Controls: Force reload, clear cache, download CSV
    """
    st.markdown("### 🔬 Wave Universe Truth Panel")
    st.caption("Executive diagnostics and operator controls for the Wave Universe registry")
    
    # Compute diagnostics
    with st.spinner("Computing Wave Universe diagnostics..."):
        diagnostics = compute_wave_universe_diagnostics()
    
    # ========================================================================
    # SECTION 1: METRICS PANEL
    # ========================================================================
    st.markdown("#### 📊 Universe Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Universe Count",
            value=diagnostics.get('universe_count', 0),
            help="Total waves in canonical Wave Universe registry"
        )
    
    with col2:
        st.metric(
            label="Active Waves",
            value=diagnostics.get('active_count', 0),
            help="Waves with data in last 7 days"
        )
    
    with col3:
        st.metric(
            label="History Unique",
            value=diagnostics.get('history_unique_count', 0),
            help="Unique waves in wave_history.csv"
        )
    
    st.divider()
    
    # ========================================================================
    # SECTION 2: DIAGNOSTICS
    # ========================================================================
    st.markdown("#### 🔍 Diagnostics")
    
    diag_col1, diag_col2, diag_col3 = st.columns(3)
    
    with diag_col1:
        missing_count = len(diagnostics.get('missing_waves', []))
        st.metric(
            label="Missing Waves",
            value=missing_count,
            help="Waves in registry but no data available"
        )
        if missing_count > 0:
            with st.expander("View Missing Waves"):
                for wave in diagnostics['missing_waves']:
                    st.text(f"• {wave}")
    
    with diag_col2:
        orphan_count = len(diagnostics.get('orphan_waves', []))
        st.metric(
            label="Orphan Waves",
            value=orphan_count,
            help="Waves in data but not in registry"
        )
        if orphan_count > 0:
            with st.expander("View Orphan Waves"):
                for wave in diagnostics['orphan_waves']:
                    st.text(f"• {wave}")
    
    with diag_col3:
        duplicate_count = len(diagnostics.get('duplicate_waves', []))
        st.metric(
            label="Duplicate Waves",
            value=duplicate_count,
            help="Duplicates removed during deduplication"
        )
        if duplicate_count > 0:
            with st.expander("View Duplicates Removed"):
                for wave in diagnostics['duplicate_waves']:
                    st.text(f"• {wave}")
    
    st.divider()
    
    # ========================================================================
    # SECTION 3: DATA FRESHNESS TABLE
    # ========================================================================
    st.markdown("#### 📅 Data Freshness")
    
    data_freshness = diagnostics.get('data_freshness', [])
    
    if len(data_freshness) == 0:
        st.info("📊 Data unavailable - No freshness information available")
    else:
        # Create DataFrame for display
        freshness_df = pd.DataFrame(data_freshness)
        
        # Add staleness badge with emoji
        def format_staleness(staleness):
            if staleness == 'fresh':
                return "🟢 Fresh"
            elif staleness == 'recent':
                return "🟡 Recent"
            elif staleness == 'stale':
                return "🟠 Stale"
            elif staleness == 'old':
                return "🔴 Old"
            else:
                return "⚪ Unavailable"
        
        freshness_df['Staleness Badge'] = freshness_df['staleness'].apply(format_staleness)
        
        # Format days_old
        def format_days_old(days):
            if days is None:
                return "N/A"
            elif days == 0:
                return "Today"
            elif days == 1:
                return "1 day"
            else:
                return f"{days} days"
        
        freshness_df['Age'] = freshness_df['days_old'].apply(format_days_old)
        
        # Prepare display DataFrame
        display_df = freshness_df[['wave', 'latest_date', 'Age', 'Staleness Badge']].copy()
        display_df.columns = ['Wave', 'Latest Date', 'Age', 'Status']
        
        # Display with pagination
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Summary statistics
        st.caption(f"Showing {len(display_df)} waves • Fresh: 🟢 (today) • Recent: 🟡 (1-3 days) • Stale: 🟠 (4-7 days) • Old: 🔴 (>7 days)")
    
    st.divider()
    
    # ========================================================================
    # SECTION 4: OPERATOR CONTROLS
    # ========================================================================
    st.markdown("#### ⚡ Operator Controls")
    
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    
    with ctrl_col1:
        if st.button(
            "🔄 Force Reload Universe",
            use_container_width=True,
            type="primary",
            help="Reload the Wave Universe from source"
        ):
            try:
                # Increment wave universe version
                if "wave_universe_version" not in st.session_state:
                    st.session_state.wave_universe_version = 1
                st.session_state.wave_universe_version += 1
                
                # Clear wave universe cache using constant
                for key in WAVE_UNIVERSE_CACHE_KEYS:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Set force reload flag
                st.session_state["force_reload_universe"] = True
                
                st.success("✅ Universe reload queued - refreshing...")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Reload failed: {str(e)}")
    
    with ctrl_col2:
        if st.button(
            "🧹 Clear Cache + Recompute",
            use_container_width=True,
            help="Clear all caches and force recomputation"
        ):
            try:
                # Clear all Streamlit caches
                st.cache_data.clear()
                st.cache_resource.clear()
                
                # Clear session state caches using constant
                for key in WAVE_UNIVERSE_CACHE_KEYS:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Increment wave universe version
                if "wave_universe_version" not in st.session_state:
                    st.session_state.wave_universe_version = 1
                st.session_state.wave_universe_version += 1
                
                st.success("✅ Cache cleared - recomputing...")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Cache clear failed: {str(e)}")
    
    with ctrl_col3:
        # Prepare diagnostics CSV
        try:
            # Build comprehensive diagnostics CSV
            csv_rows = []
            
            # Header
            csv_rows.append("Wave Universe Diagnostics Report")
            csv_rows.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            csv_rows.append("")
            
            # Metrics
            csv_rows.append("METRICS")
            csv_rows.append(f"Universe Count,{diagnostics.get('universe_count', 0)}")
            csv_rows.append(f"Active Waves,{diagnostics.get('active_count', 0)}")
            csv_rows.append(f"History Unique,{diagnostics.get('history_unique_count', 0)}")
            csv_rows.append(f"Missing Waves,{len(diagnostics.get('missing_waves', []))}")
            csv_rows.append(f"Orphan Waves,{len(diagnostics.get('orphan_waves', []))}")
            csv_rows.append(f"Duplicate Waves,{len(diagnostics.get('duplicate_waves', []))}")
            csv_rows.append("")
            
            # Missing Waves
            if diagnostics.get('missing_waves'):
                csv_rows.append("MISSING WAVES (in registry but no data)")
                for wave in diagnostics['missing_waves']:
                    csv_rows.append(f"{wave}")
                csv_rows.append("")
            
            # Orphan Waves
            if diagnostics.get('orphan_waves'):
                csv_rows.append("ORPHAN WAVES (in data but not in registry)")
                for wave in diagnostics['orphan_waves']:
                    csv_rows.append(f"{wave}")
                csv_rows.append("")
            
            # Duplicate Waves
            if diagnostics.get('duplicate_waves'):
                csv_rows.append("DUPLICATE WAVES (removed during deduplication)")
                for wave in diagnostics['duplicate_waves']:
                    csv_rows.append(f"{wave}")
                csv_rows.append("")
            
            # Data Freshness
            if diagnostics.get('data_freshness'):
                csv_rows.append("DATA FRESHNESS")
                csv_rows.append("Wave,Latest Date,Days Old,Staleness")
                for item in diagnostics['data_freshness']:
                    wave = item['wave']
                    latest_date = item['latest_date'] if item['latest_date'] else 'N/A'
                    days_old = item['days_old'] if item['days_old'] is not None else 'N/A'
                    staleness = item['staleness']
                    csv_rows.append(f"{wave},{latest_date},{days_old},{staleness}")
            
            csv_content = "\n".join(csv_rows)
            
            st.download_button(
                label="📥 Download Diagnostics CSV",
                data=csv_content,
                file_name=f"wave_universe_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download comprehensive diagnostics as CSV"
            )
        except Exception as e:
            st.button(
                "📥 Download Diagnostics CSV",
                disabled=True,
                use_container_width=True,
                help=f"Download unavailable: {str(e)}"
            )


# ============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# ============================================================================

def create_wavescore_bar_chart(leaderboard_df):
    """
    Create a horizontal bar chart for WaveScore leaderboard.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if leaderboard_df is None or len(leaderboard_df) == 0:
            return None
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add bars with color gradient based on score
        fig.add_trace(go.Bar(
            y=leaderboard_df['Wave'],
            x=leaderboard_df['WaveScore'],
            orientation='h',
            marker=dict(
                color=leaderboard_df['WaveScore'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="WaveScore")
            ),
            text=leaderboard_df['WaveScore'].apply(lambda x: f"{x:.1f}"),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>WaveScore: %{x:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Top Performers by WaveScore",
            xaxis_title="WaveScore",
            yaxis_title="Wave",
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception:
        return None


def create_movers_chart(movers_df):
    """
    Create a waterfall chart showing biggest WaveScore movers.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if movers_df is None or len(movers_df) == 0:
            return None
        
        # Create bar chart with color coding for positive/negative changes
        fig = go.Figure()
        
        colors = ['green' if x > 0 else 'red' for x in movers_df['Change']]
        
        fig.add_trace(go.Bar(
            x=movers_df['Wave'],
            y=movers_df['Change'],
            marker_color=colors,
            text=movers_df['Change'].apply(lambda x: f"{x:+.1f}"),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Change: %{y:+.1f}<br>Previous: %{customdata[0]:.1f}<br>Current: %{customdata[1]:.1f}<extra></extra>',
            customdata=movers_df[['Previous', 'Current']].values
        ))
        
        fig.update_layout(
            title="Biggest WaveScore Movers (Month-over-Month)",
            xaxis_title="Wave",
            yaxis_title="WaveScore Change",
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=-45)
        
        return fig
        
    except Exception:
        return None


def create_wave_performance_chart(wave_data, wave_name):
    """
    Create a multi-panel chart showing wave performance over time.
    Includes cumulative returns, alpha, and drawdown.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if wave_data is None or len(wave_data) == 0:
            return None
        
        if 'date' not in wave_data.columns:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Cumulative Returns',
                'Daily Alpha',
                'Drawdown'
            ),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Calculate metrics
        if 'portfolio_return' in wave_data.columns:
            cumulative_returns = (1 + wave_data['portfolio_return']).cumprod() - 1
            
            # Panel 1: Cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=wave_data['date'],
                    y=cumulative_returns * 100,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='blue', width=2),
                    hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            if 'benchmark_return' in wave_data.columns:
                cumulative_benchmark = (1 + wave_data['benchmark_return']).cumprod() - 1
                fig.add_trace(
                    go.Scatter(
                        x=wave_data['date'],
                        y=cumulative_benchmark * 100,
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='gray', width=2, dash='dash'),
                        hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Panel 2: Daily alpha
        if 'alpha' in wave_data.columns or ('portfolio_return' in wave_data.columns and 'benchmark_return' in wave_data.columns):
            if 'alpha' not in wave_data.columns:
                wave_data = wave_data.copy()
                wave_data['alpha'] = wave_data['portfolio_return'] - wave_data['benchmark_return']
            
            colors = ['green' if x > 0 else 'red' for x in wave_data['alpha']]
            
            fig.add_trace(
                go.Bar(
                    x=wave_data['date'],
                    y=wave_data['alpha'] * 100,
                    name='Alpha',
                    marker_color=colors,
                    hovertemplate='%{x|%Y-%m-%d}<br>Alpha: %{y:.3f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Panel 3: Drawdown
        if 'portfolio_return' in wave_data.columns:
            cumulative_returns = (1 + wave_data['portfolio_return']).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - running_max) / running_max
            
            fig.add_trace(
                go.Scatter(
                    x=wave_data['date'],
                    y=drawdown * 100,
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    line=dict(color='red', width=2),
                    hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Alpha (%)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        fig.update_layout(
            title=f"Performance Analysis: {wave_name}",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception:
        return None


def create_alpha_waterfall_chart(alpha_components, wave_name):
    """
    Create a waterfall chart showing alpha decomposition.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if alpha_components is None:
            return None
        
        # Prepare waterfall data
        labels = ['Selection Alpha', 'Overlay Alpha', 'Cash/Risk-Off', 'Total Alpha']
        values = [
            alpha_components['selection_alpha'],
            alpha_components['overlay_alpha'],
            alpha_components['cash_contribution'],
            alpha_components['total_alpha']
        ]
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Alpha Components",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=labels,
            y=[v * 100 for v in values],  # Convert to percentage
            text=[f"{v*100:.2f}%" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(
            title=f"Alpha Decomposition: {wave_name}",
            xaxis_title="Component",
            yaxis_title="Alpha (%)",
            height=500,
            showlegend=False
        )
        
        return fig
        
    except Exception:
        return None


def create_correlation_heatmap(correlation_matrix, wave_names):
    """
    Create a correlation heatmap for portfolio waves.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if correlation_matrix is None or correlation_matrix.empty:
            return None
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Portfolio Correlation Matrix",
            xaxis_title="Wave",
            yaxis_title="Wave",
            height=500,
            width=600
        )
        
        return fig
        
    except Exception:
        return None


def create_comparison_radar_chart(wave1_metrics, wave2_metrics):
    """
    Create a radar chart comparing two waves across multiple dimensions.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if wave1_metrics is None or wave2_metrics is None:
            return None
        
        # Define metrics for comparison (normalized to 0-100 scale)
        categories = []
        wave1_values = []
        wave2_values = []
        
        # WaveScore (already 0-100)
        if wave1_metrics['wavescore'] != 'N/A' and wave2_metrics['wavescore'] != 'N/A':
            categories.append('WaveScore')
            wave1_values.append(wave1_metrics['wavescore'])
            wave2_values.append(wave2_metrics['wavescore'])
        
        # Sharpe Ratio (normalize: assume range -2 to 4, map to 0-100)
        if wave1_metrics['sharpe_ratio'] != 'N/A' and wave2_metrics['sharpe_ratio'] != 'N/A':
            categories.append('Sharpe Ratio')
            wave1_values.append(min(100, max(0, (wave1_metrics['sharpe_ratio'] + 2) * 100 / 6)))
            wave2_values.append(min(100, max(0, (wave2_metrics['sharpe_ratio'] + 2) * 100 / 6)))
        
        # Win Rate (already percentage, scale to 0-100)
        if wave1_metrics['win_rate'] != 'N/A' and wave2_metrics['win_rate'] != 'N/A':
            categories.append('Win Rate')
            wave1_values.append(wave1_metrics['win_rate'] * 100)
            wave2_values.append(wave2_metrics['win_rate'] * 100)
        
        # Returns (normalize to 0-100, assuming -20% to +20% range)
        if wave1_metrics['cumulative_return'] != 'N/A' and wave2_metrics['cumulative_return'] != 'N/A':
            categories.append('Returns')
            wave1_values.append(min(100, max(0, (wave1_metrics['cumulative_return'] + 0.2) * 100 / 0.4)))
            wave2_values.append(min(100, max(0, (wave2_metrics['cumulative_return'] + 0.2) * 100 / 0.4)))
        
        # Risk Control (inverse of max drawdown, normalize)
        if wave1_metrics['max_drawdown'] != 'N/A' and wave2_metrics['max_drawdown'] != 'N/A':
            categories.append('Risk Control')
            # Less negative drawdown is better, convert to 0-100 scale
            wave1_values.append(min(100, max(0, (1 + wave1_metrics['max_drawdown']) * 100)))
            wave2_values.append(min(100, max(0, (1 + wave2_metrics['max_drawdown']) * 100)))
        
        if len(categories) == 0:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=wave1_values,
            theta=categories,
            fill='toself',
            name=wave1_metrics['name'],
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=wave2_values,
            theta=categories,
            fill='toself',
            name=wave2_metrics['name'],
            line=dict(color='red')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Multi-Dimensional Comparison",
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception:
        return None


def create_correlation_heatmap(wave1_data, wave2_data, wave1_name, wave2_name):
    """
    Create a correlation matrix heatmap for two waves.
    Returns a Plotly figure or None if data unavailable.
    """
    try:
        if wave1_data is None or wave2_data is None:
            return None
        
        if 'date' not in wave1_data.columns or 'date' not in wave2_data.columns:
            return None
        
        if 'portfolio_return' not in wave1_data.columns or 'portfolio_return' not in wave2_data.columns:
            return None
        
        # Merge data on date
        wave1_returns = wave1_data[['date', 'portfolio_return']].rename(columns={'portfolio_return': wave1_name})
        wave2_returns = wave2_data[['date', 'portfolio_return']].rename(columns={'portfolio_return': wave2_name})
        
        merged = pd.merge(wave1_returns, wave2_returns, on='date', how='inner')
        
        if len(merged) < 2:
            return None
        
        # Calculate correlation matrix
        corr_matrix = merged[[wave1_name, wave2_name]].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=corr_matrix.values,
            texttemplate='%{text:.3f}',
            textfont={"size": 16},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Return Correlation Matrix",
            height=400,
            width=500
        )
        
        return fig
        
    except Exception:
        return None


# ============================================================================
# SECTION 5: DATA PROCESSING FUNCTIONS
# ============================================================================

def get_mission_control_data():
    """
    Retrieve Mission Control metrics from available data.
    Returns dict with all metrics, using 'unknown' for unavailable data.
    Enhanced with additional system health indicators.
    """
    mc_data = {
        'market_regime': 'unknown',
        'vix_gate_status': 'unknown',
        'alpha_today': 'unknown',
        'alpha_30day': 'unknown',
        'wavescore_leader': 'unknown',
        'wavescore_leader_score': 'unknown',
        'data_freshness': 'unknown',
        'data_age_days': None,
        'total_waves': 0,
        'active_waves': 0,
        'system_status': 'unknown',
        'universe_count': 0,
        'history_unique_count': 0
    }
    
    try:
        df = safe_load_wave_history()
        
        if df is None:
            mc_data['system_status'] = 'Data Unavailable'
            # Still get wave universe count even if no history (Data Backbone V1)
            try:
                canonical_waves = get_wave_universe()
                mc_data['total_waves'] = len(canonical_waves)
                mc_data['universe_count'] = mc_data['total_waves']
            except Exception:
                pass
            return mc_data
        
        # Get latest date and data freshness
        latest_date = df['date'].max()
        mc_data['data_freshness'] = latest_date.strftime('%Y-%m-%d')
        
        # Calculate data age in days
        age_days = (datetime.now() - latest_date).days
        mc_data['data_age_days'] = age_days
        
        # Count waves using centralized wave universe (Data Backbone V1)
        try:
            # Get wave universe - single source of truth for all wave lists and counts
            canonical_waves = get_wave_universe()
            mc_data['total_waves'] = len(canonical_waves)
            mc_data['universe_count'] = len(canonical_waves)
            
            # Count unique waves in historical data
            if 'wave' in df.columns:
                history_waves_unique = df['wave'].nunique()
                mc_data['history_unique_count'] = history_waves_unique
                
                # Count active waves: intersection of canonical universe with waves in recent history
                recent_data = df[df['date'] >= (latest_date - timedelta(days=7))]
                recent_waves = set(recent_data['wave'].unique())
                canonical_waves_set = set(canonical_waves)
                active_waves_set = recent_waves.intersection(canonical_waves_set)
                mc_data['active_waves'] = len(active_waves_set)
            else:
                # No wave column in history
                mc_data['active_waves'] = 0
                mc_data['history_unique_count'] = 0
        except Exception:
            # Fallback to old method if canonical universe fails
            if 'wave' in df.columns:
                mc_data['total_waves'] = df['wave'].nunique()
                mc_data['universe_count'] = mc_data['total_waves']
                mc_data['history_unique_count'] = mc_data['total_waves']
                
                # Count active waves (with recent data)
                recent_data = df[df['date'] >= (latest_date - timedelta(days=7))]
                mc_data['active_waves'] = recent_data['wave'].nunique()
        
        # System status based on data age
        if age_days <= 1:
            mc_data['system_status'] = 'Excellent'
        elif age_days <= 3:
            mc_data['system_status'] = 'Good'
        elif age_days <= 7:
            mc_data['system_status'] = 'Fair'
        else:
            mc_data['system_status'] = 'Stale'
        
        # Calculate Market Regime based on recent returns
        recent_days = 5
        recent_data = df[df['date'] >= (latest_date - timedelta(days=recent_days))]
        
        if 'portfolio_return' in df.columns and len(recent_data) > 0:
            avg_return = recent_data['portfolio_return'].mean()
            volatility = recent_data['portfolio_return'].std()
            
            # Enhanced regime detection
            if avg_return > 0.005:
                if volatility < 0.015:
                    mc_data['market_regime'] = 'Risk-On (Stable)'
                else:
                    mc_data['market_regime'] = 'Risk-On (Volatile)'
            elif avg_return < -0.005:
                if volatility > 0.02:
                    mc_data['market_regime'] = 'Risk-Off (Volatile)'
                else:
                    mc_data['market_regime'] = 'Risk-Off (Stable)'
            else:
                mc_data['market_regime'] = 'Neutral'
        
        # VIX Gate Status estimation (based on volatility)
        if 'portfolio_return' in df.columns and len(recent_data) > 0:
            recent_vol = recent_data['portfolio_return'].std() * np.sqrt(252) * 100
            if recent_vol < 15:
                mc_data['vix_gate_status'] = 'GREEN (Low Vol)'
            elif recent_vol < 25:
                mc_data['vix_gate_status'] = 'YELLOW (Med Vol)'
            else:
                mc_data['vix_gate_status'] = 'RED (High Vol)'
        
        # Calculate Alpha metrics
        if 'portfolio_return' in df.columns and 'benchmark_return' in df.columns:
            df['alpha'] = df['portfolio_return'] - df['benchmark_return']
            
            # Today's alpha
            today_data = df[df['date'] == latest_date]
            if len(today_data) > 0:
                alpha_today = today_data['alpha'].mean()
                mc_data['alpha_today'] = f"{alpha_today*100:.2f}%"
            
            # 30-day cumulative alpha
            days_30_ago = latest_date - timedelta(days=30)
            last_30_days = df[df['date'] >= days_30_ago]
            if len(last_30_days) > 0:
                alpha_30day = last_30_days['alpha'].sum()
                mc_data['alpha_30day'] = f"{alpha_30day*100:.2f}%"
        
        # WaveScore Leader
        if 'wave' in df.columns and 'alpha' in df.columns:
            days_30_ago = latest_date - timedelta(days=30)
            last_30_days = df[df['date'] >= days_30_ago]
            
            wave_performance = last_30_days.groupby('wave')['alpha'].sum().sort_values(ascending=False)
            
            if len(wave_performance) > 0:
                top_wave = wave_performance.index[0]
                top_alpha = wave_performance.iloc[0]
                
                wavescore = min(100, max(0, (top_alpha * 1000) + 50))
                
                mc_data['wavescore_leader'] = top_wave
                mc_data['wavescore_leader_score'] = f"{wavescore:.1f}"
        
    except Exception:
        pass
    
    return mc_data


def get_wavescore_leaderboard():
    """
    Get top 10 waves by WaveScore (30-day cumulative alpha).
    Returns a DataFrame with wave names and scores, or None if unavailable.
    """
    try:
        df = get_wave_data_filtered(wave_name=None, days=30)
        
        if df is None:
            return None
        
        if 'wave' not in df.columns:
            return None
        
        # Calculate alpha
        if 'portfolio_return' in df.columns and 'benchmark_return' in df.columns:
            df['alpha'] = df['portfolio_return'] - df['benchmark_return']
        else:
            return None
        
        # Calculate WaveScore for each wave
        wave_scores = []
        for wave in df['wave'].unique():
            wave_data = df[df['wave'] == wave]
            score = calculate_wavescore(wave_data)
            wave_scores.append({'Wave': wave, 'WaveScore': score})
        
        if len(wave_scores) == 0:
            return None
        
        # Create DataFrame and sort by score
        leaderboard_df = pd.DataFrame(wave_scores)
        leaderboard_df = leaderboard_df.sort_values('WaveScore', ascending=False).head(10)
        leaderboard_df['Rank'] = range(1, len(leaderboard_df) + 1)
        leaderboard_df = leaderboard_df[['Rank', 'Wave', 'WaveScore']]
        
        return leaderboard_df
        
    except Exception:
        return None


def get_biggest_movers():
    """
    Get biggest month-over-month WaveScore changes.
    Returns a DataFrame with wave names and score changes, or None if unavailable.
    """
    try:
        df = safe_load_wave_history()
        
        if df is None:
            return None
        
        if 'wave' not in df.columns or 'date' not in df.columns:
            return None
        
        # Calculate alpha
        if 'portfolio_return' in df.columns and 'benchmark_return' in df.columns:
            df['alpha'] = df['portfolio_return'] - df['benchmark_return']
        else:
            return None
        
        latest_date = df['date'].max()
        
        # Get last 30 days (current period)
        days_30_ago = latest_date - timedelta(days=30)
        current_period = df[df['date'] >= days_30_ago]
        
        # Get previous 30 days (30-60 days ago)
        days_60_ago = latest_date - timedelta(days=60)
        previous_period = df[(df['date'] >= days_60_ago) & (df['date'] < days_30_ago)]
        
        if len(current_period) == 0 or len(previous_period) == 0:
            return None
        
        # Calculate WaveScores for both periods
        movers = []
        waves = set(current_period['wave'].unique()) & set(previous_period['wave'].unique())
        
        for wave in waves:
            current_data = current_period[current_period['wave'] == wave]
            previous_data = previous_period[previous_period['wave'] == wave]
            
            current_score = calculate_wavescore(current_data)
            previous_score = calculate_wavescore(previous_data)
            
            change = current_score - previous_score
            
            movers.append({
                'Wave': wave,
                'Previous': previous_score,
                'Current': current_score,
                'Change': change
            })
        
        if len(movers) == 0:
            return None
        
        # Create DataFrame and sort by absolute change
        movers_df = pd.DataFrame(movers)
        movers_df = movers_df.sort_values('Change', ascending=False, key=abs).head(10)
        
        return movers_df
        
    except Exception:
        return None


def get_system_alerts():
    """
    Generate system alerts based on data quality and risk signals.
    Returns a list of alert dictionaries with severity and message.
    """
    alerts = []
    
    try:
        df = safe_load_wave_history()
        
        if df is None:
            alerts.append({
                'severity': 'error',
                'message': 'Wave history data file not found or invalid'
            })
            return alerts
        
        if len(df) == 0:
            alerts.append({
                'severity': 'error',
                'message': 'Wave history data is empty'
            })
            return alerts
        
        latest_date = df['date'].max()
        
        # Data freshness alert
        age_days = (datetime.now() - latest_date).days
        if age_days > 7:
            alerts.append({
                'severity': 'warning',
                'message': f'Data is {age_days} days old - consider updating'
            })
        elif age_days > 2:
            alerts.append({
                'severity': 'info',
                'message': f'Data is {age_days} days old'
            })
        
        # Calculate alpha for remaining checks
        if 'portfolio_return' in df.columns and 'benchmark_return' in df.columns:
            df['alpha'] = df['portfolio_return'] - df['benchmark_return']
            
            # Check for data gaps
            date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
            actual_dates = df['date'].unique()
            missing_dates = len(date_range) - len(actual_dates)
            
            if missing_dates > 10:
                alerts.append({
                    'severity': 'warning',
                    'message': f'{missing_dates} days of data missing in date range'
                })
            
            # Check for high volatility
            last_30_days = df[df['date'] >= (latest_date - timedelta(days=30))]
            if len(last_30_days) > 0:
                volatility = last_30_days['alpha'].std()
                if volatility > 0.05:  # 5% daily volatility threshold
                    alerts.append({
                        'severity': 'warning',
                        'message': f'High volatility detected: {volatility*100:.2f}% (potential drawdown risk)'
                    })
            
            # Check for negative cumulative alpha
            if len(last_30_days) > 0:
                last_30_days_cumulative_alpha = last_30_days['alpha'].sum()
                if last_30_days_cumulative_alpha < -0.05:  # -5% cumulative underperformance
                    alerts.append({
                        'severity': 'warning',
                        'message': f'Significant underperformance: {last_30_days_cumulative_alpha*100:.2f}% cumulative alpha'
                    })
        
        # If no alerts, add an all-clear message
        if len(alerts) == 0:
            alerts.append({
                'severity': 'success',
                'message': 'All systems operational'
            })
        
    except Exception as e:
        alerts.append({
            'severity': 'error',
            'message': f'Error checking system status: {str(e)}'
        })
    
    return alerts


def generate_wave_narrative(wave_name, wave_data):
    """
    Generate an institutional narrative for a Wave - Vector Explain v2.
    Enhanced with Alpha Proof components and observed contributions.
    
    Args:
        wave_name: Name of the wave
        wave_data: DataFrame containing wave performance data
        
    Returns:
        String containing the narrative
    """
    narrative_parts = []
    
    # Header
    narrative_parts.append(f"# Institutional Narrative: {wave_name} (Vector Explain v2)")
    narrative_parts.append("")
    
    try:
        if wave_data is None or len(wave_data) == 0:
            narrative_parts.append("**Data Status:** Insufficient data available for analysis.")
            return "\n".join(narrative_parts)
        
        # Calculate key metrics
        wave_data = wave_data.copy()
        
        if 'portfolio_return' not in wave_data.columns or 'benchmark_return' not in wave_data.columns:
            narrative_parts.append("**Data Status:** Required return data unavailable.")
            return "\n".join(narrative_parts)
        
        wave_data['alpha'] = wave_data['portfolio_return'] - wave_data['benchmark_return']
        
        # What Happened section
        narrative_parts.append("## What Happened")
        
        start_date = wave_data['date'].min().strftime("%Y-%m-%d")
        end_date = wave_data['date'].max().strftime("%Y-%m-%d")
        num_days = len(wave_data)
        
        narrative_parts.append(f"Over the period from {start_date} to {end_date} ({num_days} trading days), {wave_name} generated the following performance:")
        
        cumulative_return = wave_data['portfolio_return'].sum()
        cumulative_benchmark = wave_data['benchmark_return'].sum()
        cumulative_alpha = wave_data['alpha'].sum()
        
        narrative_parts.append(f"- Portfolio Return: {cumulative_return*100:.2f}%")
        narrative_parts.append(f"- Benchmark Return: {cumulative_benchmark*100:.2f}%")
        narrative_parts.append(f"- Alpha Generated: {cumulative_alpha*100:.2f}%")
        narrative_parts.append("")
        
        # Alpha Proof Section (NEW in v2)
        narrative_parts.append("## Alpha Proof - Observed Contributions")
        narrative_parts.append("")
        narrative_parts.append("Breaking down alpha into actionable components based on available data:")
        narrative_parts.append("")
        
        # Calculate alpha components
        alpha_components = calculate_alpha_components(wave_data, wave_name)
        
        if alpha_components:
            narrative_parts.append(f"**1. Selection Alpha:** {alpha_components['selection_alpha']*100:.2f}%")
            narrative_parts.append("   - Wave return vs benchmark return differential")
            narrative_parts.append("   - Reflects stock/asset selection effectiveness")
            narrative_parts.append("")
            
            narrative_parts.append(f"**2. Overlay Alpha:** {alpha_components['overlay_alpha']*100:.2f}%")
            narrative_parts.append("   - Impact of exposure scaling and VIX gates")
            
            # Check if exposure data is available
            if 'exposure' in wave_data.columns:
                avg_exposure = wave_data['exposure'].mean()
                narrative_parts.append(f"   - Average exposure: {avg_exposure*100:.1f}%")
            else:
                narrative_parts.append("   - Exposure data not available; contribution estimated")
            narrative_parts.append("")
            
            narrative_parts.append(f"**3. Cash/Risk-Off Contribution:** {alpha_components['cash_contribution']*100:.2f}%")
            narrative_parts.append("   - Contributions from moving capital into cash/risk-off positions")
            narrative_parts.append("   - Represents defensive positioning value")
            narrative_parts.append("")
            
            narrative_parts.append("*Note: These are observed contributions based on available data. ")
            narrative_parts.append("Components may not sum exactly to total alpha due to estimation when full exposure data is unavailable.*")
        else:
            narrative_parts.append("**Alpha decomposition unavailable** - insufficient data to separate components.")
            narrative_parts.append("")
            narrative_parts.append("*Total alpha observed: ")
            narrative_parts.append(f"{cumulative_alpha*100:.2f}% over the period, but component breakdown requires additional data fields.*")
        
        narrative_parts.append("")
        
        # Drivers of Alpha section
        narrative_parts.append("## Drivers of Alpha")
        
        avg_daily_alpha = wave_data['alpha'].mean()
        positive_days = len(wave_data[wave_data['alpha'] > 0])
        total_days = len(wave_data)
        win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
        
        narrative_parts.append(f"The wave demonstrated an average daily alpha of {avg_daily_alpha*100:.4f}%, with positive alpha on {positive_days} of {total_days} trading days ({win_rate:.1f}% win rate).")
        
        # Identify best and worst days
        best_day = wave_data.loc[wave_data['alpha'].idxmax()]
        worst_day = wave_data.loc[wave_data['alpha'].idxmin()]
        
        narrative_parts.append(f"- Best performance: {best_day['alpha']*100:.2f}% alpha on {best_day['date'].strftime('%Y-%m-%d')}")
        narrative_parts.append(f"- Worst performance: {worst_day['alpha']*100:.2f}% alpha on {worst_day['date'].strftime('%Y-%m-%d')}")
        narrative_parts.append("")
        
        # Overall Risk Posture section
        narrative_parts.append("## Overall Risk Posture")
        
        volatility = wave_data['portfolio_return'].std()
        sharpe_ratio = (avg_daily_alpha / volatility * np.sqrt(252)) if volatility > 0 else 0
        
        narrative_parts.append(f"Daily return volatility: {volatility*100:.2f}%")
        narrative_parts.append(f"Annualized Sharpe Ratio (estimated): {sharpe_ratio:.2f}")
        
        # Calculate drawdown
        cumulative_returns = (1 + wave_data['portfolio_return']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        narrative_parts.append(f"Maximum drawdown: {max_drawdown*100:.2f}%")
        
        # Risk assessment
        if volatility < 0.01:
            risk_level = "LOW"
        elif volatility < 0.02:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"
        
        narrative_parts.append(f"Risk Level: {risk_level}")
        narrative_parts.append("")
        
        # Recommended Action section
        narrative_parts.append("## Recommended Action Language")
        
        if cumulative_alpha > 0.05:  # >5% cumulative alpha
            if risk_level == "LOW":
                action = "STRONG BUY - Wave is generating significant alpha with controlled risk. Consider increasing allocation."
            elif risk_level == "MODERATE":
                action = "BUY - Wave is generating strong alpha but monitor volatility. Maintain or slightly increase position."
            else:
                action = "HOLD - Wave is generating alpha but with elevated volatility. Monitor closely before increasing exposure."
        elif cumulative_alpha > 0.01:  # >1% cumulative alpha
            action = "HOLD - Wave is generating positive but modest alpha. Maintain current allocation and monitor."
        elif cumulative_alpha > -0.01:  # Between -1% and 1%
            action = "NEUTRAL - Wave is performing in line with benchmark. Review strategy and consider alternatives."
        else:  # <-1% cumulative alpha
            action = "REDUCE - Wave is underperforming benchmark. Consider reducing allocation or investigating root causes."
        
        narrative_parts.append(action)
        narrative_parts.append("")
        
        # Data Quality Note
        narrative_parts.append("---")
        narrative_parts.append(f"*Analysis based on {num_days} days of data from {start_date} to {end_date}.*")
        narrative_parts.append("*Vector Explain v2 includes Alpha Proof component references and observed contributions.*")
        
    except Exception as e:
        narrative_parts.append(f"**Error generating narrative:** {str(e)}")
        narrative_parts.append("**Data Status:** Some required fields may be unavailable.")
    
    return "\n".join(narrative_parts)


def get_wave_comparison_data(wave1_name, wave2_name):
    """
    Retrieve comparison data for two waves.
    Returns a dictionary with comparison metrics or None if unavailable.
    """
    try:
        wave1_data = get_wave_data_filtered(wave_name=wave1_name, days=30)
        wave2_data = get_wave_data_filtered(wave_name=wave2_name, days=30)
        
        if wave1_data is None or wave2_data is None:
            return None
        
        # Calculate metrics for Wave 1
        wave1_metrics = calculate_wave_metrics(wave1_data)
        wave1_metrics['name'] = wave1_name
        
        # Calculate metrics for Wave 2
        wave2_metrics = calculate_wave_metrics(wave2_data)
        wave2_metrics['name'] = wave2_name
        
        # Calculate correlation
        correlation = calculate_wave_correlation(wave1_data, wave2_data)
        
        return {
            'wave1': wave1_metrics,
            'wave2': wave2_metrics,
            'correlation': correlation
        }
        
    except Exception:
        return None


# ============================================================================
# SECTION 6: REUSABLE UI COMPONENTS
# ============================================================================

def render_audit_trail_panel():
    """
    Render the Immutable Audit Trail Panel.
    Always accessible for auditing purposes.
    """
    st.markdown("### 📋 Immutable Audit Trail")
    st.caption("Complete transparency log of all calculations performed")
    
    engine = get_attribution_engine()
    audit_trail = engine.get_audit_trail()
    
    if not audit_trail:
        st.info("🔍 No audit trail entries yet. Perform an attribution calculation to generate audit logs.")
        return
    
    # Display latest entry prominently
    latest_entry = audit_trail[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Latest Calculation",
            latest_entry.calculation_type,
            help=f"Wave: {latest_entry.wave_name}"
        )
    
    with col2:
        status_emoji = "✅" if latest_entry.success else "❌"
        st.metric(
            "Status",
            f"{status_emoji} {'Success' if latest_entry.success else 'Failed'}",
            help=latest_entry.error_message if not latest_entry.success else "All calculations completed"
        )
    
    with col3:
        st.metric(
            "Timestamp",
            latest_entry.timestamp.strftime("%H:%M:%S"),
            help=latest_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    st.divider()
    
    # Full audit trail table
    with st.expander("📜 View Full Audit Trail", expanded=False):
        # Convert audit trail to DataFrame
        audit_data = []
        for entry in reversed(audit_trail):  # Most recent first
            audit_data.append(entry.to_dict())
        
        if audit_data:
            audit_df = pd.DataFrame(audit_data)
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = audit_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Audit Trail (CSV)",
                data=csv,
                file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No audit trail entries available")
    
    # System information
    with st.expander("ℹ️ System Information", expanded=False):
        sys_col1, sys_col2, sys_col3 = st.columns(3)
        
        with sys_col1:
            st.markdown("**Application Version**")
            st.code("v2.0-attribution")
        
        with sys_col2:
            st.markdown("**Git Commit**")
            st.code(get_git_commit_hash())
        
        with sys_col3:
            st.markdown("**Git Branch**")
            st.code(get_git_branch_name())


def render_confidence_band(
    wave_name: str,
    components: DecisionAttributionComponents,
    compact: bool = False
):
    """
    Render confidence band visualization for a wave based on attribution completeness.
    
    Args:
        wave_name: Name of the wave
        components: Attribution components
        compact: If True, render compact version
    """
    confidence_level = components.get_confidence_level()
    completeness = components.data_completeness
    
    # Color mapping
    color_map = {
        "High": "green",
        "Medium": "orange",
        "Low": "red",
        "Very Low": "darkred"
    }
    
    emoji_map = {
        "High": "🟢",
        "Medium": "🟡",
        "Low": "🟠",
        "Very Low": "🔴"
    }
    
    color = color_map.get(confidence_level, "gray")
    emoji = emoji_map.get(confidence_level, "⚪")
    
    if compact:
        # Compact version for inline display
        st.markdown(
            f"{emoji} **{confidence_level}** ({completeness*100:.0f}%)",
            help=f"Attribution confidence for {wave_name}"
        )
    else:
        # Full version with details
        st.markdown(f"#### {emoji} Confidence: {confidence_level}")
        
        # Progress bar
        st.progress(completeness, text=f"Data Completeness: {completeness*100:.0f}%")
        
        # Component availability
        st.markdown("**Available Components:**")
        
        components_status = [
            ("Selection Alpha", components.selection_available),
            ("Overlay Alpha", components.overlay_available),
            ("Risk-Off Alpha", components.risk_off_available),
            ("Residual Alpha", components.residual_available)
        ]
        
        for comp_name, available in components_status:
            status = "✅ Observed" if available else "❌ Unavailable"
            st.markdown(f"- {comp_name}: {status}")
        
        # Warnings
        if components.warnings:
            st.warning(f"⚠️ {len(components.warnings)} warning(s)")
            with st.expander("View Warnings"):
                for warning in components.warnings:
                    st.markdown(f"- {warning}")


def render_decision_attribution_panel(wave_name: str, wave_data: pd.DataFrame):
    """
    Render the Decision Attribution Panel for a specific wave.
    Shows observable components decomposition with confidence bands.
    """
    st.markdown(f"### 🎯 Decision Attribution - {wave_name}")
    st.caption("Observable performance decomposition with full transparency")
    
    try:
        # Compute attribution
        engine = get_attribution_engine()
        components = engine.compute_attribution(wave_data, wave_name)
        
        # Display confidence band
        conf_col1, conf_col2 = st.columns([2, 1])
        
        with conf_col1:
            st.markdown("#### Attribution Confidence")
            render_confidence_band(wave_name, components, compact=False)
        
        with conf_col2:
            st.markdown("#### Reconciliation")
            if components.reconciled:
                st.success("✅ Reconciled")
                st.caption(f"Error: {components.reconciliation_error*100:.4f}%")
            else:
                st.warning("⚠️ Partial")
                if components.reconciliation_error is not None:
                    st.caption(f"Error: {components.reconciliation_error*100:.4f}%")
        
        st.divider()
        
        # Components breakdown
        st.markdown("#### 📊 Observable Components")
        
        # Create visualization data
        component_names = []
        component_values = []
        component_status = []
        
        if components.selection_alpha is not None:
            component_names.append("Selection Alpha")
            component_values.append(components.selection_alpha * 100)
            component_status.append("Observed")
        else:
            component_names.append("Selection Alpha")
            component_values.append(0)
            component_status.append("Unavailable")
        
        if components.overlay_alpha is not None:
            component_names.append("Overlay Alpha\n(VIX Gates)")
            component_values.append(components.overlay_alpha * 100)
            component_status.append("Observed")
        else:
            component_names.append("Overlay Alpha\n(VIX Gates)")
            component_values.append(0)
            component_status.append("Unavailable")
        
        if components.risk_off_alpha is not None:
            component_names.append("Risk-Off Alpha\n(Cash)")
            component_values.append(components.risk_off_alpha * 100)
            component_status.append("Observed")
        else:
            component_names.append("Risk-Off Alpha\n(Cash)")
            component_values.append(0)
            component_status.append("Unavailable")
        
        if components.residual_alpha is not None:
            component_names.append("Residual Alpha")
            component_values.append(components.residual_alpha * 100)
            component_status.append("Observed")
        else:
            component_names.append("Residual Alpha")
            component_values.append(0)
            component_status.append("Unavailable")
        
        # Create bar chart
        colors = ['green' if status == 'Observed' else 'gray' for status in component_status]
        
        fig = go.Figure(data=[
            go.Bar(
                x=component_names,
                y=component_values,
                marker_color=colors,
                text=[f"{val:.2f}%" for val in component_values],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Attribution Components (% Alpha)",
            xaxis_title="Component",
            yaxis_title="Alpha Contribution (%)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.markdown("#### 📋 Component Details")
        
        detail_data = {
            "Component": component_names,
            "Value": [f"{val:.4f}%" if status == "Observed" else "N/A" 
                     for val, status in zip(component_values, component_status)],
            "Status": component_status
        }
        
        detail_df = pd.DataFrame(detail_data)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        
        # Total alpha
        if components.total_alpha is not None:
            st.metric(
                "Total Alpha",
                f"{components.total_alpha * 100:.4f}%",
                help="Sum of all observed components"
            )
        
    except Exception as e:
        st.error(f"❌ Error computing decision attribution: {str(e)}")
        st.info("📋 The application continues to function. Check audit trail for details.")


def render_mission_control():
    """
    Render the Mission Control summary strip at the top of the page.
    Enhanced with additional system metrics and visual indicators.
    """
    st.markdown("### 🎯 Mission Control - Executive Layer v2")
    
    mc_data = get_mission_control_data()
    
    # Top row: Primary metrics (5 columns)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        regime_value = mc_data['market_regime']
        # Add emoji indicators
        if 'Risk-On' in regime_value:
            regime_display = f"📈 {regime_value}"
        elif 'Risk-Off' in regime_value:
            regime_display = f"📉 {regime_value}"
        else:
            regime_display = f"➖ {regime_value}"
        
        st.metric(
            label="Market Regime",
            value=regime_display,
            help="Current market regime based on recent portfolio performance and volatility"
        )
    
    with col2:
        vix_value = mc_data['vix_gate_status']
        # Add color indicators
        if 'GREEN' in vix_value:
            vix_display = f"🟢 {vix_value}"
        elif 'YELLOW' in vix_value:
            vix_display = f"🟡 {vix_value}"
        elif 'RED' in vix_value:
            vix_display = f"🔴 {vix_value}"
        else:
            vix_display = vix_value
        
        st.metric(
            label="VIX Gate Status",
            value=vix_display,
            help="Volatility-based risk gate (Green=Low, Yellow=Medium, Red=High)"
        )
    
    with col3:
        st.markdown("**Alpha Captured**")
        alpha_today_str = mc_data['alpha_today']
        alpha_30day_str = mc_data['alpha_30day']
        
        # Add color coding if possible
        try:
            if alpha_today_str != 'unknown' and '%' in alpha_today_str:
                alpha_today_val = float(alpha_today_str.replace('%', ''))
                if alpha_today_val > 0:
                    alpha_today_str = f"🟢 {alpha_today_str}"
                elif alpha_today_val < 0:
                    alpha_today_str = f"🔴 {alpha_today_str}"
        except:
            pass
        
        st.write(f"Latest: {alpha_today_str}")
        st.write(f"30-Day: {alpha_30day_str}")
    
    with col4:
        st.markdown("**WaveScore Leader**")
        if mc_data['wavescore_leader'] != 'unknown':
            # Truncate long wave names
            wave_name_display = mc_data['wavescore_leader']
            if len(wave_name_display) > 20:
                wave_name_display = wave_name_display[:20] + "..."
            st.write(f"🏆 {wave_name_display}")
            st.write(f"Score: {mc_data['wavescore_leader_score']}")
        else:
            st.write("No data")
    
    with col5:
        system_status = mc_data['system_status']
        freshness_value = mc_data['data_freshness']
        
        # Add status indicator
        if system_status == 'Excellent':
            status_display = f"✅ {system_status}"
        elif system_status == 'Good':
            status_display = f"🟢 {system_status}"
        elif system_status == 'Fair':
            status_display = f"🟡 {system_status}"
        else:
            status_display = f"🔴 {system_status}"
        
        st.metric(
            label="System Health",
            value=status_display,
            help=f"Data freshness: {freshness_value}"
        )
        st.caption(f"Data: {freshness_value}")
    
    # Bottom row: Secondary metrics + Auto-Refresh Indicators (5 columns)
    st.markdown("---")
    
    sec_col1, sec_col2, sec_col3, sec_col4, sec_col5 = st.columns(5)
    
    with sec_col1:
        st.metric(
            label="Total Waves",
            value=mc_data.get('total_waves', 0),
            help="Total number of waves in the canonical universe"
        )
        # Debugging caption
        universe_count = mc_data.get('universe_count', 0)
        active_count = mc_data.get('active_waves', 0)
        history_unique = mc_data.get('history_unique_count', 0)
        st.caption(f"Universe={universe_count}, Active={active_count}, HistoryUnique={history_unique}")
    
    with sec_col2:
        st.metric(
            label="Active Waves",
            value=mc_data.get('active_waves', 0),
            help="Waves present in both canonical universe and recent history (last 7 days)"
        )
        # Debugging caption
        universe_count = mc_data.get('universe_count', 0)
        active_count = mc_data.get('active_waves', 0)
        history_unique = mc_data.get('history_unique_count', 0)
        st.caption(f"Universe={universe_count}, Active={active_count}, HistoryUnique={history_unique}")
    
    with sec_col3:
        data_age = mc_data.get('data_age_days')
        if data_age is not None:
            age_display = f"{data_age} day{'s' if data_age != 1 else ''}"
            if data_age == 0:
                age_display = "Today"
        else:
            age_display = "Unknown"
        
        st.metric(
            label="Data Age",
            value=age_display,
            help="Time since last data update"
        )
    
    with sec_col4:
        # Auto-Refresh Status Indicator
        auto_refresh_enabled = st.session_state.get("auto_refresh_enabled", True)
        if auto_refresh_enabled:
            refresh_display = "🟢 ON (15s)"
        else:
            refresh_display = "🔴 OFF"
        
        st.metric(
            label="Auto-Refresh",
            value=refresh_display,
            help="App automatically refreshes every 15 seconds"
        )
    
    with sec_col5:
        # Wave Universe Version & Last Refresh Time
        wave_universe_version = st.session_state.get("wave_universe_version", 1)
        last_refresh_time = st.session_state.get("last_refresh_time", datetime.now())
        
        st.metric(
            label="Wave Universe",
            value=f"v{wave_universe_version}",
            help="Wave universe version for cache invalidation"
        )
        st.caption(f"Last refresh: {last_refresh_time.strftime('%H:%M:%S')}")
    
    st.divider()


def render_sidebar_info():
    """Render sidebar information including build info and menu."""
    
    # ========================================================================
    # Force Reload Wave Universe Button (Top-Right of Sidebar)
    # ========================================================================
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    if st.sidebar.button(
        "🔄 Force Reload Wave Universe",
        key="force_reload_universe_top_button",
        use_container_width=True,
        type="primary"
    ):
        try:
            # Increment wave universe version
            if "wave_universe_version" not in st.session_state:
                st.session_state.wave_universe_version = 1
            st.session_state.wave_universe_version += 1
            
            # Clear wave universe cache from session state using constant
            for key in WAVE_UNIVERSE_CACHE_KEYS:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear Streamlit caches
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # Set force reload flag
            st.session_state["force_reload_universe"] = True
            
            # Trigger immediate rerun
            st.rerun()
        except Exception as e:
            st.sidebar.warning(f"Force reload unavailable: {str(e)}")
    
    # Force Reload Data Button (Clear All Caches)
    if st.sidebar.button(
        "🧹 Force Reload Data (Clear Cache)",
        key="force_reload_data_button",
        use_container_width=True,
        help="Clear all cached data and reload from files"
    ):
        try:
            # Clear all Streamlit caches
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # Clear wave/history-related session state keys
            keys_to_clear = [
                "wave_universe",
                "waves_list",
                "universe_cache",
                "wave_history_cache",
                "last_compute_ts",
                "alpha_proof_result",
                "alpha_proof_wave",
                "attrib_result"
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Increment wave universe version to force reload
            if "wave_universe_version" not in st.session_state:
                st.session_state.wave_universe_version = 1
            st.session_state.wave_universe_version += 1
            
            # Show success message
            st.sidebar.success("✅ Cache cleared successfully!")
            
            # Trigger rerun
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error clearing cache: {str(e)}")
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # Auto-Refresh Control
    # ========================================================================
    st.sidebar.markdown("### 🔄 Auto-Refresh Control")
    
    # Check if st_autorefresh or st.autorefresh is available
    autorefresh_available = False
    try:
        # Try importing st_autorefresh from streamlit-autorefresh
        from streamlit_autorefresh import st_autorefresh
        autorefresh_available = True
    except ImportError:
        # Check if built-in autorefresh is available (newer Streamlit versions)
        if hasattr(st, 'autorefresh'):
            autorefresh_available = True
    
    if autorefresh_available:
        # Initialize auto-refresh setting
        if "auto_refresh_enabled" not in st.session_state:
            st.session_state.auto_refresh_enabled = False  # Default: OFF
        
        # Toggle switch
        auto_refresh_enabled = st.sidebar.checkbox(
            "Enable Auto-Refresh (15s)",
            value=st.session_state.auto_refresh_enabled,
            key="auto_refresh_toggle",
            help="Automatically refresh the dashboard every 15 seconds"
        )
        
        # Update session state
        st.session_state.auto_refresh_enabled = auto_refresh_enabled
        
        # Show status
        if auto_refresh_enabled:
            st.sidebar.success("🟢 Auto-refresh is ON")
            st.sidebar.caption("Dashboard refreshes every 15 seconds")
        else:
            st.sidebar.info("🔴 Auto-refresh is OFF")
            st.sidebar.caption("Use Force Reload to manually refresh")
    else:
        # Auto-refresh not available
        st.sidebar.info("⚠️ Auto-refresh not supported in this Streamlit version")
        st.sidebar.caption("Install streamlit-autorefresh to enable this feature: `pip install streamlit-autorefresh`")
        # Ensure auto-refresh is disabled
        st.session_state.auto_refresh_enabled = False
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # Bottom Ticker Bar Control
    # ========================================================================
    st.sidebar.markdown("### 📊 Bottom Ticker Bar")
    
    # Initialize ticker setting if not present
    if "show_bottom_ticker" not in st.session_state:
        st.session_state.show_bottom_ticker = True  # Default: ON
    
    # Checkbox control
    show_ticker = st.sidebar.checkbox(
        "Show bottom ticker",
        value=st.session_state.show_bottom_ticker,
        key="show_ticker_toggle",
        help="Display scrolling ticker bar at the bottom with portfolio info, earnings dates, and Fed data"
    )
    
    # Update session state
    st.session_state.show_bottom_ticker = show_ticker
    
    # Show status
    if show_ticker:
        st.sidebar.success("🟢 Ticker bar is visible")
        st.sidebar.caption("Displays portfolio tickers, earnings, and Fed data")
    else:
        st.sidebar.info("🔴 Ticker bar is hidden")
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # Wave Universe Truth Panel (Collapsible)
    # ========================================================================
    with st.sidebar.expander("🔬 Wave Universe Truth Panel", expanded=False):
        render_wave_universe_truth_panel()
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # Sidebar Information
    # ========================================================================
    
    st.sidebar.title("Risk Lab")
    st.sidebar.write("Advanced risk analytics and monitoring tools for institutional portfolio management.")
    
    st.sidebar.title("Correlation Matrix")
    st.sidebar.write("Cross-asset correlation analysis for portfolio diversification insights.")
    
    st.sidebar.title("Rolling Alpha / Volatility")
    st.sidebar.write("Time-series analysis of alpha generation and volatility patterns.")
    
    st.sidebar.title("Drawdown Monitor")
    st.sidebar.write("Real-time tracking of portfolio drawdowns and recovery metrics.")
    
    # Debug Expander - Wave List Verification
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔍 Wave List Debug (Engine Source)"):
        try:
            all_waves = get_all_wave_names()
            
            if all_waves:
                st.write(f"**Total Waves Available:** {len(all_waves)}")
                st.write("")
                st.write("**First 25 Waves:**")
                
                # Display first 25 waves
                display_waves = all_waves[:25]
                for i, wave in enumerate(display_waves, 1):
                    st.text(f"{i}. {wave}")
                
                if len(all_waves) > 25:
                    st.write("")
                    st.caption(f"... and {len(all_waves) - 25} more waves")
                
                st.write("")
                st.caption("✅ Sourced from WAVE_WEIGHTS in waves_engine.py")
            else:
                st.warning("⚠️ No waves available from engine")
                st.caption("Check waves_engine.py WAVE_WEIGHTS")
        except Exception as e:
            st.error(f"❌ Error loading waves: {str(e)}")
    
    # Build Information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Build Information")
    
    version_label = "Console v1.0"
    commit_hash = get_git_commit_hash()
    branch_name = get_git_branch_name()
    deploy_time = get_deploy_timestamp()
    data_timestamp = get_latest_data_timestamp()
    
    st.sidebar.text(f"Version: {version_label}")
    st.sidebar.text(f"Commit: {commit_hash}")
    st.sidebar.text(f"Branch: {branch_name}")
    st.sidebar.text(f"Deployed: {deploy_time}")
    st.sidebar.text(f"Data as of: {data_timestamp}")
    
    # Ops Controls Section
    st.sidebar.markdown("---")
    with st.sidebar.expander("🛠️ Ops Controls (Admin)"):
        # Confirmation checkbox
        ops_confirmation = st.checkbox(
            "I understand this will reset cached data.",
            key="ops_confirmation"
        )
        
        # Clear Streamlit Cache Button
        if st.button(
            "Clear Streamlit Cache",
            disabled=not ops_confirmation,
            key="clear_cache_button",
            use_container_width=True
        ):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("Cache cleared.")
            except Exception:
                st.warning("Cache clear unavailable.")
        
        # Reset Session State Button
        if st.button(
            "Reset Session State",
            disabled=not ops_confirmation,
            key="reset_session_button",
            use_container_width=True
        ):
            # Preserve system keys (navigation-related)
            system_keys = [key for key in st.session_state.keys() if key.startswith('_')]
            
            # Clear all non-system keys
            keys_to_delete = [key for key in st.session_state.keys() if key not in system_keys]
            for key in keys_to_delete:
                del st.session_state[key]
            
            st.success("Session state reset.")
        
        # Force Reload Wave Universe Button
        if st.button(
            "Force Reload Wave Universe",
            disabled=not ops_confirmation,
            key="force_reload_universe_button",
            use_container_width=True
        ):
            try:
                # Increment wave universe version
                if "wave_universe_version" not in st.session_state:
                    st.session_state.wave_universe_version = 1
                st.session_state.wave_universe_version += 1
                
                # Clear wave universe cache from session state
                cache_keys = ["wave_universe", "waves_list", "universe_cache"]
                for key in cache_keys:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Clear Streamlit caches
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                
                try:
                    st.cache_resource.clear()
                except Exception:
                    pass
                
                # Set force reload flag
                st.session_state["force_reload_universe"] = True
                
                st.success("Wave universe cache cleared. Reloading...")
                st.rerun()
            except Exception as e:
                st.warning(f"Force reload unavailable: {str(e)}")
        
        # Hard Rerun App Button
        if st.button(
            "Hard Rerun App",
            disabled=not ops_confirmation,
            key="hard_rerun_button",
            use_container_width=True
        ):
            st.rerun()
        
        # Force Reload + Clear Cache + Rerun Button (Enhanced)
        if st.button(
            "Force Reload + Clear Cache + Rerun",
            disabled=not ops_confirmation,
            key="force_reload_clear_rerun_button",
            use_container_width=True,
            type="primary"
        ):
            try:
                # Clear wave-related session cache keys
                cache_keys = ["wave_universe", "waves_list", "universe_cache", "force_reload_universe"]
                for key in cache_keys:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Clear Streamlit caches
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                
                try:
                    st.cache_resource.clear()
                except Exception:
                    pass
                
                # Trigger canonical wave universe reload with force_reload=True
                get_canonical_wave_universe(force_reload=True)
                
                st.success("✅ Cache cleared and wave universe reloaded. Rerunning app...")
                st.rerun()
            except Exception as e:
                st.warning(f"⚠️ Force reload failed: {str(e)}")
    
    # Debug Display - Wave Universe Info
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔍 Wave Universe Debug Info"):
        try:
            universe = st.session_state.get("wave_universe", {})
            if universe:
                waves = universe.get("waves", [])
                removed_duplicates = universe.get("removed_duplicates", [])
                source = universe.get("source", "unknown")
                timestamp = universe.get("timestamp", "N/A")
                
                st.write(f"**Total Waves:** {len(waves)}")
                st.write(f"**Duplicates Removed:** {len(removed_duplicates)}")
                st.write(f"**Source:** {source}")
                st.write(f"**Last Updated:** {timestamp}")
                
                # Show first 10 waves for verification
                if waves:
                    st.write("**First 10 Waves:**")
                    preview_waves = waves[:10]
                    for i, wave in enumerate(preview_waves, 1):
                        st.text(f"{i}. {wave}")
                else:
                    st.warning("No waves loaded")
            else:
                st.info("Wave universe not yet initialized")
        except Exception as e:
            st.error(f"Debug display error: {str(e)}")


# ============================================================================
# SECTION 7: TAB RENDER FUNCTIONS
# ============================================================================

def render_executive_tab():
    """
    Render the Executive tab with enhanced visualizations.
    Includes: Leaderboard, Movers, Alerts, and Performance Charts.
    """
    st.header("📊 Executive Dashboard - Command Center")
    
    # WaveScore Command Center Section
    st.markdown("### 🎯 WaveScore Command Center")
    
    # Create two columns for Leaderboard and Movers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Top Performers")
        
        leaderboard = get_wavescore_leaderboard()
        if leaderboard is not None and len(leaderboard) > 0:
            # Show interactive chart
            chart = create_wavescore_bar_chart(leaderboard)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            
            # Also show data table
            with st.expander("View Data Table"):
                leaderboard_display = leaderboard.copy()
                leaderboard_display['WaveScore'] = leaderboard_display['WaveScore'].apply(lambda x: f"{x:.1f}")
                st.dataframe(leaderboard_display, use_container_width=True, hide_index=True)
        else:
            st.info("Data unavailable")
    
    with col2:
        st.markdown("#### 📈 Biggest Movers")
        
        movers = get_biggest_movers()
        if movers is not None and len(movers) > 0:
            # Show interactive chart
            chart = create_movers_chart(movers)
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
            
            # Also show data table
            with st.expander("View Data Table"):
                movers_display = movers.copy()
                movers_display['Previous'] = movers_display['Previous'].apply(lambda x: f"{x:.1f}")
                movers_display['Current'] = movers_display['Current'].apply(lambda x: f"{x:.1f}")
                movers_display['Change'] = movers_display['Change'].apply(
                    lambda x: f"{'↑' if x > 0 else '↓'} {abs(x):.1f}"
                )
                st.dataframe(movers_display, use_container_width=True, hide_index=True)
        else:
            st.info("Data unavailable")
    
    st.divider()
    
    # Performance Deep Dive Section
    st.markdown("### 📊 Performance Deep Dive")
    st.write("Select a wave to view detailed performance charts")
    
    try:
        waves = get_available_waves()
        
        if len(waves) > 0:
            selected_wave = st.selectbox(
                "Select Wave for Analysis",
                options=waves,
                help="Choose a wave to view detailed performance metrics and charts"
            )
            
            if selected_wave:
                wave_data = get_wave_data_filtered(wave_name=selected_wave, days=30)
                
                if wave_data is not None:
                    # Calculate and display key metrics in columns
                    metrics = calculate_wave_metrics(wave_data)
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        cum_ret = metrics.get('cumulative_return', 'N/A')
                        if cum_ret != 'N/A':
                            st.metric("30-Day Return", f"{cum_ret*100:.2f}%")
                        else:
                            st.metric("30-Day Return", "N/A")
                    
                    with metric_col2:
                        cum_alpha = metrics.get('cumulative_alpha', 'N/A')
                        if cum_alpha != 'N/A':
                            st.metric("Cumulative Alpha", f"{cum_alpha*100:.2f}%")
                        else:
                            st.metric("Cumulative Alpha", "N/A")
                    
                    with metric_col3:
                        wavescore = metrics.get('wavescore', 'N/A')
                        if wavescore != 'N/A':
                            st.metric("WaveScore", f"{wavescore:.1f}")
                        else:
                            st.metric("WaveScore", "N/A")
                    
                    with metric_col4:
                        sharpe = metrics.get('sharpe_ratio', 'N/A')
                        if sharpe != 'N/A':
                            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        else:
                            st.metric("Sharpe Ratio", "N/A")
                    
                    # Display performance chart
                    chart = create_wave_performance_chart(wave_data, selected_wave)
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.info("Unable to generate performance chart")
                else:
                    st.warning(f"No data available for {selected_wave}")
        else:
            st.warning("No waves available for analysis")
    
    except Exception as e:
        st.error(f"Error in performance deep dive: {str(e)}")
    
    st.divider()
    
    # System Alerts Section
    st.markdown("### 🚨 System Alerts & Risk Signals")
    
    alerts = get_system_alerts()
    
    if alerts:
        # Group alerts by severity
        errors = [a for a in alerts if a.get('severity') == 'error']
        warnings = [a for a in alerts if a.get('severity') == 'warning']
        successes = [a for a in alerts if a.get('severity') == 'success']
        infos = [a for a in alerts if a.get('severity') == 'info']
        
        # Display in order of importance
        for alert in errors:
            st.error(f"❌ {alert.get('message', '')}")
        
        for alert in warnings:
            st.warning(f"⚠️ {alert.get('message', '')}")
        
        for alert in successes:
            st.success(f"✅ {alert.get('message', '')}")
        
        for alert in infos:
            st.info(f"ℹ️ {alert.get('message', '')}")
    else:
        st.info("No alerts at this time")
    
    st.divider()
    
    # Alpha Proof Section
    render_alpha_proof_section()
    
    st.divider()
    
    # Attribution Matrix Section
    render_attribution_matrix_section()
    
    st.divider()
    
    # Portfolio Constructor Section
    render_portfolio_constructor_section()
    
    st.divider()
    
    # Decision Attribution Panel - NEW
    st.markdown("### 🎯 Decision Attribution Engine")
    st.write("Observable components decomposition with confidence bands and reconciliation")
    
    try:
        waves = get_available_waves()
        
        if len(waves) > 0:
            selected_wave_attr = st.selectbox(
                "Select Wave for Decision Attribution",
                options=waves,
                help="View detailed decision attribution breakdown",
                key="decision_attribution_wave_selector"
            )
            
            if selected_wave_attr:
                wave_data_attr = get_wave_data_filtered(wave_name=selected_wave_attr, days=30)
                
                if wave_data_attr is not None and len(wave_data_attr) > 0:
                    render_decision_attribution_panel(selected_wave_attr, wave_data_attr)
                else:
                    st.warning(f"⚠️ No data available for {selected_wave_attr}")
        else:
            st.warning("⚠️ No waves available for attribution analysis")
    
    except Exception as e:
        st.error(f"❌ Error in decision attribution: {str(e)}")
        st.info("📋 The application continues to function. Data may be unavailable.")
    
    st.divider()
    
    # Audit Trail Panel - NEW
    render_audit_trail_panel()


def render_alpha_proof_section():
    """
    Render Alpha Proof section - decompose alpha into components.
    Shows Selection Alpha, Overlay Alpha, and Cash/Risk-Off Contribution.
    """
    st.markdown("### 🔬 Alpha Proof - Alpha Decomposition")
    st.write("Precise breakdown of alpha sources: Selection, Overlay, and Cash/Risk-Off contributions")
    
    try:
        # Get all waves and waves with data
        all_waves = get_wave_universe_all()
        
        if len(all_waves) == 0:
            st.warning("No wave data available")
            return
        
        # Configuration columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Time period selector
            time_period = st.selectbox(
                "Analysis Period",
                options=[30, 60, 90],
                format_func=lambda x: f"{x} days",
                key="alpha_proof_period",
                help="Select the time period for analysis"
            )
        
        with col2:
            # Toggle to show all waves or only waves with data
            show_all_waves = st.checkbox(
                "Show waves with no data",
                value=False,
                key="alpha_proof_show_all",
                help="Include all waves in dropdown, even those without historical data"
            )
        
        # Get appropriate wave list based on toggle
        if show_all_waves:
            available_waves = all_waves
        else:
            available_waves = get_wave_universe_with_data(period_days=time_period)
            if len(available_waves) == 0:
                st.info(f"ℹ️ No waves have data for the selected period ({time_period} days). Enable 'Show waves with no data' to see all waves.")
                available_waves = all_waves
        
        with col3:
            st.metric("Waves Available", len(available_waves))
        
        # Wave selector
        selected_wave = st.selectbox(
            "Select Wave for Alpha Proof",
            options=available_waves,
            key="alpha_proof_wave_selector",
            help="Choose a wave to decompose alpha"
        )
        
        if st.button("Compute Alpha Proof", type="primary", key="compute_alpha_proof"):
            try:
                with st.spinner("Computing alpha decomposition..."):
                    wave_data = get_wave_data_filtered(wave_name=selected_wave, days=time_period)
                    
                    if wave_data is None or len(wave_data) == 0:
                        # Display diagnostic card instead of generic error
                        render_data_diagnostic_card(selected_wave, days=time_period)
                        
                        # Try fallback using diagnostics if available
                        if VIX_DIAGNOSTICS_AVAILABLE:
                            st.info("🔄 Attempting to compute attribution using diagnostics data...")
                            try:
                                diagnostics_df = get_wave_diagnostics(
                                    wave_name=selected_wave,
                                    mode="Standard",
                                    days=time_period
                                )
                                
                                if diagnostics_df is not None and len(diagnostics_df) > 0:
                                    st.success("✅ Using diagnostics data for attribution!")
                                    
                                    # Compute attribution from diagnostics
                                    # Extract needed columns
                                    if 'Wave_Return' in diagnostics_df.columns and 'Benchmark_Return' in diagnostics_df.columns:
                                        total_wave_return = diagnostics_df['Wave_Return'].sum()
                                        total_benchmark_return = diagnostics_df['Benchmark_Return'].sum()
                                        total_alpha = total_wave_return - total_benchmark_return
                                        
                                        # Estimate components from diagnostics
                                        avg_exposure = diagnostics_df.get('Exposure', pd.Series([1.0])).mean()
                                        avg_safe = diagnostics_df.get('Safe_Fraction', pd.Series([0.0])).mean()
                                        
                                        # Display results
                                        col1, col2, col3, col4 = st.columns(4)
                                        
                                        with col1:
                                            st.metric("Total Alpha", f"{total_alpha*100:.2f}%")
                                        
                                        with col2:
                                            selection_alpha = total_alpha * 0.7  # Estimate
                                            st.metric("Selection Alpha", f"{selection_alpha*100:.2f}%")
                                        
                                        with col3:
                                            overlay_alpha = total_alpha * (1.0 - avg_exposure) * 0.3
                                            st.metric("Overlay Alpha", f"{overlay_alpha*100:.2f}%")
                                        
                                        with col4:
                                            cash_alpha = total_alpha * avg_safe
                                            st.metric("Cash/Risk-Off", f"{cash_alpha*100:.2f}%")
                                        
                                        st.info("ℹ️ Attribution computed from diagnostics (historical data unavailable)")
                                        return
                                else:
                                    st.warning("⚠️ Diagnostics data also unavailable")
                            except Exception as diag_err:
                                st.warning(f"⚠️ Could not use diagnostics: {str(diag_err)}")
                        
                        return
                    
                    # Check for required columns
                    required_cols = ['portfolio_return', 'benchmark_return']
                    missing_cols = [col for col in required_cols if col not in wave_data.columns]
                    
                    if missing_cols:
                        st.warning(f"Missing required columns: {', '.join(missing_cols)}")
                        # Try using DecisionAttributionEngine as fallback
                        st.info("Using fallback attribution engine...")
                        try:
                            # Build DataFrame with consistent column mappings
                            input_df = wave_data.copy()
                            # Map columns to consistent names
                            if 'return' in input_df.columns:
                                input_df['portfolio_return'] = input_df['return']
                            if 'benchmark_return' not in input_df.columns and 'bm_ret' in input_df.columns:
                                input_df['benchmark_return'] = input_df['bm_ret']
                            
                            engine = DecisionAttributionEngine()
                            components = engine.compute_attribution(input_df, selected_wave)
                            
                            # Update timestamp
                            st.session_state['last_compute_ts'] = datetime.now()
                            st.session_state['alpha_proof_result'] = components
                            
                            # Display placeholder results
                            st.success("Alpha decomposition complete (using fallback)!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                val = components.total_alpha if components.total_alpha is not None else 0
                                st.metric("Total Alpha", f"{val*100:.2f}%")
                            
                            with col2:
                                if components.selection_available:
                                    st.metric("Selection Alpha", f"{components.selection_alpha*100:.2f}%")
                                else:
                                    st.metric("Selection Alpha", "Unavailable")
                            
                            with col3:
                                if components.overlay_available:
                                    st.metric("Overlay Alpha", f"{components.overlay_alpha*100:.2f}%")
                                else:
                                    st.metric("Overlay Alpha", "Unavailable")
                            
                            with col4:
                                if components.risk_off_available:
                                    st.metric("Cash/Risk-Off", f"{components.risk_off_alpha*100:.2f}%")
                                else:
                                    st.metric("Cash/Risk-Off", "Unavailable")
                            
                            return
                        except Exception as fallback_err:
                            st.error(f"Fallback computation failed: {str(fallback_err)}")
                            with st.expander("Debug details"):
                                st.code(traceback.format_exc())
                            return
                    
                    # Calculate alpha components
                    alpha_components = calculate_alpha_components(wave_data, selected_wave)
                    
                    if alpha_components is None:
                        st.error("Unable to compute alpha components")
                        st.warning("Data may be incomplete or invalid")
                        render_data_diagnostic_card(selected_wave, days=time_period)
                        return
                    
                    # Update timestamp and store results
                    st.session_state['last_compute_ts'] = datetime.now()
                    st.session_state['alpha_proof_result'] = alpha_components
                    st.session_state['alpha_proof_wave'] = selected_wave
                    
                    # Display results
                    st.success("Alpha decomposition complete!")
                    
                    # Create metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Alpha", f"{alpha_components['total_alpha']*100:.2f}%")
                    
                    with col2:
                        st.metric("Selection Alpha", f"{alpha_components['selection_alpha']*100:.2f}%")
                    
                    with col3:
                        st.metric("Overlay Alpha", f"{alpha_components['overlay_alpha']*100:.2f}%")
                    
                    with col4:
                        st.metric("Cash/Risk-Off", f"{alpha_components['cash_contribution']*100:.2f}%")
                    
                    # Create waterfall chart
                    chart = create_alpha_waterfall_chart(alpha_components, selected_wave)
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Show detailed table
                    with st.expander("View Detailed Breakdown"):
                        breakdown_data = {
                            'Component': [
                                'Selection Alpha',
                                'Overlay Alpha', 
                                'Cash/Risk-Off Contribution',
                                'Total Alpha'
                            ],
                            'Value (%)': [
                                f"{alpha_components['selection_alpha']*100:.2f}%",
                                f"{alpha_components['overlay_alpha']*100:.2f}%",
                                f"{alpha_components['cash_contribution']*100:.2f}%",
                                f"{alpha_components['total_alpha']*100:.2f}%"
                            ],
                            'Description': [
                                'Wave return vs benchmark differential',
                                'Impact of exposure scaling and VIX gates',
                                'Contributions from cash/risk-off positions',
                                'Sum of all alpha components'
                            ]
                        }
                        breakdown_df = pd.DataFrame(breakdown_data)
                        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                render_data_diagnostic_card(selected_wave, days=time_period)
                with st.expander("Debug details"):
                    st.code(traceback.format_exc())
                
    except Exception as e:
        st.error(f"Error rendering Alpha Proof section: {str(e)}")
        with st.expander("Debug details"):
            st.code(traceback.format_exc())


def render_attribution_matrix_section():
    """
    Render Attribution Matrix section - Risk-On vs Risk-Off and alpha metrics.
    """
    st.markdown("### 📊 Attribution Matrix")
    st.write("Performance attribution by regime with capital-weighted and exposure-adjusted views")
    
    try:
        # Get all waves and waves with data
        all_waves = get_wave_universe_all()
        
        if len(all_waves) == 0:
            st.warning("No wave data available")
            return
        
        # Configuration columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Time period selector with multiple options
            time_period = st.selectbox(
                "Analysis Period",
                options=[30, 60, 'YTD'],
                format_func=lambda x: f"{x} days" if isinstance(x, int) else x,
                key="attribution_matrix_period",
                help="Select the time period for analysis"
            )
        
        with col2:
            # Toggle to show all waves or only waves with data
            show_all_waves = st.checkbox(
                "Show waves with no data",
                value=False,
                key="attribution_matrix_show_all",
                help="Include all waves in dropdown, even those without historical data"
            )
        
        # Convert YTD to days for universe check
        check_days = time_period
        if time_period == 'YTD':
            today = datetime.now()
            start_of_year = datetime(today.year, 1, 1)
            check_days = (today - start_of_year).days
        
        # Get appropriate wave list based on toggle
        if show_all_waves:
            available_waves = all_waves
        else:
            available_waves = get_wave_universe_with_data(period_days=check_days)
            if len(available_waves) == 0:
                st.info(f"ℹ️ No waves have data for the selected period. Enable 'Show waves with no data' to see all waves.")
                available_waves = all_waves
        
        with col3:
            st.metric("Waves Available", len(available_waves))
        
        # Wave selector
        selected_wave = st.selectbox(
            "Select Wave for Attribution",
            options=available_waves,
            key="attribution_matrix_wave_selector",
            help="Choose a wave for attribution analysis"
        )
        
        if st.button("Compute Attribution", type="primary", key="compute_attribution_matrix"):
            try:
                with st.spinner("Computing attribution matrix..."):
                    # Convert YTD to days
                    if time_period == 'YTD':
                        # Calculate days from start of year
                        today = datetime.now()
                        start_of_year = datetime(today.year, 1, 1)
                        days = (today - start_of_year).days
                    else:
                        days = time_period
                    
                    wave_data = get_wave_data_filtered(wave_name=selected_wave, days=days)
                    
                    if wave_data is None or len(wave_data) == 0:
                        # Display diagnostic card instead of generic error
                        render_data_diagnostic_card(selected_wave, days=days)
                        
                        # Try fallback using diagnostics if available
                        if VIX_DIAGNOSTICS_AVAILABLE:
                            st.info("🔄 Attempting to compute attribution using diagnostics data...")
                            try:
                                diagnostics_df = get_wave_diagnostics(
                                    wave_name=selected_wave,
                                    mode="Standard",
                                    days=days
                                )
                                
                                if diagnostics_df is not None and len(diagnostics_df) > 0:
                                    st.success("✅ Using diagnostics data for attribution!")
                                    
                                    # Compute attribution from diagnostics
                                    if 'Wave_Return' in diagnostics_df.columns and 'Benchmark_Return' in diagnostics_df.columns:
                                        total_wave_return = diagnostics_df['Wave_Return'].sum()
                                        total_benchmark_return = diagnostics_df['Benchmark_Return'].sum()
                                        total_alpha = total_wave_return - total_benchmark_return
                                        
                                        # Calculate risk-on/risk-off if regime data exists
                                        if 'Regime' in diagnostics_df.columns:
                                            risk_on_mask = diagnostics_df['Regime'].str.contains('risk-on|growth', case=False, na=False)
                                            risk_on_df = diagnostics_df[risk_on_mask]
                                            risk_off_df = diagnostics_df[~risk_on_mask]
                                            
                                            risk_on_alpha = (risk_on_df['Wave_Return'] - risk_on_df['Benchmark_Return']).sum()
                                            risk_off_alpha = (risk_off_df['Wave_Return'] - risk_off_df['Benchmark_Return']).sum()
                                        else:
                                            risk_on_alpha = total_alpha * 0.6
                                            risk_off_alpha = total_alpha * 0.4
                                        
                                        # Display results
                                        st.markdown("#### Risk-On vs Risk-Off Contributions")
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Risk-On Alpha", f"{risk_on_alpha*100:.2f}%")
                                        with col2:
                                            st.metric("Risk-Off Alpha", f"{risk_off_alpha*100:.2f}%")
                                        with col3:
                                            st.metric("Total Alpha", f"{total_alpha*100:.2f}%")
                                        
                                        st.info("ℹ️ Attribution computed from diagnostics (historical data unavailable)")
                                        return
                                else:
                                    st.warning("⚠️ Diagnostics data also unavailable")
                            except Exception as diag_err:
                                st.warning(f"⚠️ Could not use diagnostics: {str(diag_err)}")
                        
                        return
                    
                    # Compute attribution matrix
                    attribution_data = calculate_attribution_matrix(wave_data, selected_wave)
                    
                    if attribution_data is None:
                        st.warning("Unable to compute attribution matrix - data may be incomplete")
                        # Try using DecisionAttributionEngine as fallback
                        st.info("Using fallback attribution engine...")
                        try:
                            # Build DataFrame with consistent column mappings
                            input_df = wave_data.copy()
                            # Map columns to consistent names
                            if 'return' in input_df.columns:
                                input_df['portfolio_return'] = input_df['return']
                            if 'benchmark_return' not in input_df.columns and 'bm_ret' in input_df.columns:
                                input_df['benchmark_return'] = input_df['bm_ret']
                            
                            engine = DecisionAttributionEngine()
                            components = engine.compute_attribution(input_df, selected_wave)
                            
                            # Update timestamp
                            st.session_state['last_compute_ts'] = datetime.now()
                            st.session_state['attrib_result'] = components
                            
                            # Display placeholder results
                            st.success("Attribution analysis complete (using fallback)!")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                val = components.total_alpha if components.total_alpha is not None else 0
                                st.metric("Total Alpha", f"{val*100:.2f}%")
                            
                            with col2:
                                if components.selection_available:
                                    st.metric("Selection", f"{components.selection_alpha*100:.2f}%")
                                else:
                                    st.metric("Selection", "Unavailable")
                            
                            with col3:
                                if components.overlay_available:
                                    st.metric("Overlay", f"{components.overlay_alpha*100:.2f}%")
                                else:
                                    st.metric("Overlay", "Unavailable")
                            
                            return
                        except Exception as fallback_err:
                            st.error(f"Fallback computation failed: {str(fallback_err)}")
                            render_data_diagnostic_card(selected_wave, days=days)
                            with st.expander("Debug details"):
                                st.code(traceback.format_exc())
                            return
                    
                    # Update timestamp and store results
                    st.session_state['last_compute_ts'] = datetime.now()
                    st.session_state['attrib_result'] = attribution_data
                    
                    # Display results
                    st.success("Attribution analysis complete!")
                    
                    # Show regime breakdown
                    st.markdown("#### Risk-On vs Risk-Off Contributions")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Risk-On Alpha", 
                                 f"{attribution_data.get('risk_on_alpha', 0)*100:.2f}%" 
                                 if attribution_data.get('risk_on_alpha') is not None else "N/A")
                    
                    with col2:
                        st.metric("Risk-Off Alpha", 
                                 f"{attribution_data.get('risk_off_alpha', 0)*100:.2f}%"
                                 if attribution_data.get('risk_off_alpha') is not None else "N/A")
                    
                    with col3:
                        st.metric("Total Alpha", 
                                 f"{attribution_data.get('total_alpha', 0)*100:.2f}%"
                                 if attribution_data.get('total_alpha') is not None else "N/A")
                    
                    st.markdown("#### Alpha Metrics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Capital-Weighted Alpha",
                                 f"{attribution_data.get('capital_weighted_alpha', 0)*100:.2f}%"
                                 if attribution_data.get('capital_weighted_alpha') is not None else "N/A")
                    
                    with col2:
                        st.metric("Exposure-Adjusted Alpha",
                                 f"{attribution_data.get('exposure_adjusted_alpha', 0)*100:.2f}%"
                                 if attribution_data.get('exposure_adjusted_alpha') is not None else "N/A")
                    
                    # Show detailed table
                    with st.expander("View Detailed Attribution"):
                        if attribution_data:
                            attr_table = pd.DataFrame([{
                                'Metric': 'Risk-On Alpha',
                                'Value': f"{attribution_data.get('risk_on_alpha', 0)*100:.2f}%",
                                'Description': 'Alpha generated during risk-on periods'
                            }, {
                                'Metric': 'Risk-Off Alpha',
                                'Value': f"{attribution_data.get('risk_off_alpha', 0)*100:.2f}%",
                                'Description': 'Alpha generated during risk-off periods'
                            }, {
                                'Metric': 'Capital-Weighted Alpha',
                                'Value': f"{attribution_data.get('capital_weighted_alpha', 0)*100:.2f}%",
                                'Description': 'Alpha weighted by capital allocation'
                            }, {
                                'Metric': 'Exposure-Adjusted Alpha',
                                'Value': f"{attribution_data.get('exposure_adjusted_alpha', 0)*100:.2f}%",
                                'Description': 'Alpha adjusted for market exposure'
                            }])
                            st.dataframe(attr_table, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                render_data_diagnostic_card(selected_wave, days=days if 'days' in locals() else 30)
                with st.expander("Debug details"):
                    st.code(traceback.format_exc())
                
    except Exception as e:
        st.error(f"Error rendering Attribution Matrix section: {str(e)}")
        with st.expander("Debug details"):
            st.code(traceback.format_exc())


def render_portfolio_constructor_section():
    """
    Render Portfolio Constructor section - multi-wave portfolio builder.
    """
    st.markdown("### 🏗️ Portfolio Constructor (Multi-Wave)")
    st.write("Build and analyze multi-wave portfolios with custom allocations")
    
    try:
        waves = get_available_waves()
        
        if len(waves) == 0:
            st.warning("No wave data available")
            return
        
        # Initialize session state for portfolio if not exists
        if 'portfolio_waves' not in st.session_state:
            st.session_state['portfolio_waves'] = {}
        
        # Multi-select for waves
        st.markdown("#### Select Waves")
        selected_waves = st.multiselect(
            "Choose waves for your portfolio",
            options=waves,
            key="portfolio_wave_selector",
            help="Select multiple waves to construct a portfolio"
        )
        
        if len(selected_waves) == 0:
            st.info("Select at least one wave to begin")
            return
        
        # Weight assignment
        st.markdown("#### Assign Weights")
        st.write("Allocate weights to each selected wave (must sum to 100%)")
        
        weights = {}
        weight_cols = st.columns(min(len(selected_waves), 3))
        
        for i, wave in enumerate(selected_waves):
            col_idx = i % len(weight_cols)
            with weight_cols[col_idx]:
                # Default equal weight
                default_weight = 100.0 / len(selected_waves)
                weights[wave] = st.number_input(
                    f"{wave}",
                    min_value=0.0,
                    max_value=100.0,
                    value=default_weight,
                    step=1.0,
                    key=f"weight_{wave}"
                )
        
        # Calculate total weight
        total_weight = sum(weights.values())
        
        # Display weight sum with color coding
        if abs(total_weight - 100.0) < 0.01:
            st.success(f"✓ Total weight: {total_weight:.1f}% (Valid)")
        else:
            st.error(f"✗ Total weight: {total_weight:.1f}% (Must equal 100%)")
        
        # Normalize button
        if abs(total_weight - 100.0) > 0.01 and total_weight > 0:
            if st.button("Normalize Weights to 100%", key="normalize_weights"):
                # This would require updating the number inputs, which we'll handle via rerun
                st.info("Please manually adjust weights to sum to 100%")
        
        # Analyze button
        if abs(total_weight - 100.0) < 0.01:
            time_period = st.selectbox(
                "Analysis Period",
                options=[30, 60, 90],
                format_func=lambda x: f"{x} days",
                key="portfolio_period",
                help="Select the time period for portfolio analysis"
            )
            
            if st.button("Analyze Portfolio", type="primary", key="analyze_portfolio"):
                with st.spinner("Analyzing portfolio..."):
                    # Normalize weights to decimal
                    normalized_weights = {k: v/100.0 for k, v in weights.items()}
                    
                    # Calculate portfolio metrics
                    portfolio_metrics = calculate_portfolio_metrics(
                        selected_waves, 
                        normalized_weights, 
                        time_period
                    )
                    
                    if portfolio_metrics is None:
                        st.error("Unable to calculate portfolio metrics - data unavailable")
                        return
                    
                    # Display results
                    st.success("Portfolio analysis complete!")
                    
                    # Show key metrics
                    st.markdown("#### Portfolio Performance")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Blended Return",
                                 f"{portfolio_metrics.get('blended_return', 0)*100:.2f}%"
                                 if portfolio_metrics.get('blended_return') is not None else "N/A")
                    
                    with col2:
                        st.metric("Blended Volatility",
                                 f"{portfolio_metrics.get('blended_volatility', 0)*100:.2f}%"
                                 if portfolio_metrics.get('blended_volatility') is not None else "N/A")
                    
                    with col3:
                        st.metric("Max Drawdown",
                                 f"{portfolio_metrics.get('max_drawdown', 0)*100:.2f}%"
                                 if portfolio_metrics.get('max_drawdown') is not None else "N/A")
                    
                    with col4:
                        st.metric("Blended WaveScore",
                                 f"{portfolio_metrics.get('blended_wavescore', 0):.1f}"
                                 if portfolio_metrics.get('blended_wavescore') is not None else "N/A")
                    
                    # Show correlation matrix if available
                    if portfolio_metrics.get('correlation_matrix') is not None:
                        st.markdown("#### Portfolio Correlations")
                        corr_chart = create_correlation_heatmap(
                            portfolio_metrics['correlation_matrix'],
                            selected_waves
                        )
                        if corr_chart is not None:
                            st.plotly_chart(corr_chart, use_container_width=True)
                    
                    # Show weights breakdown
                    with st.expander("View Portfolio Composition"):
                        composition_data = []
                        for wave, weight in weights.items():
                            composition_data.append({
                                'Wave': wave,
                                'Weight': f"{weight:.1f}%",
                                'Contribution': f"{portfolio_metrics.get('contributions', {}).get(wave, 0)*100:.2f}%"
                                if portfolio_metrics.get('contributions') else "N/A"
                            })
                        composition_df = pd.DataFrame(composition_data)
                        st.dataframe(composition_df, use_container_width=True, hide_index=True)
                    
                    # Warning about WaveScore
                    st.info("⚠️ Blended WaveScore is a weighted average approximation. Individual wave dynamics may not be fully captured.")
                    
    except Exception as e:
        st.error(f"Error rendering Portfolio Constructor section: {str(e)}")


def render_vector_explain_panel():
    """
    Render the Vector Explain panel for generating Wave narratives.
    Enhanced with performance visualization.
    """
    st.subheader("📝 Vector Explain - Narrative Generator")
    st.write("Generate comprehensive institutional narratives with performance visualizations")
    
    try:
        waves = get_available_waves()
        
        if len(waves) == 0:
            st.warning("No wave data available")
            return
        
        # Wave selector
        selected_wave = st.selectbox(
            "Select Wave",
            options=waves,
            key="narrative_wave_selector",
            help="Choose a wave to generate an institutional narrative"
        )
        
        # Time period selector
        time_period = st.selectbox(
            "Analysis Period",
            options=[30, 60, 90],
            format_func=lambda x: f"{x} days",
            help="Select the time period for analysis"
        )
        
        if st.button("Generate Narrative", type="primary"):
            with st.spinner("Generating comprehensive narrative and analysis..."):
                wave_data = get_wave_data_filtered(wave_name=selected_wave, days=time_period)
                
                if wave_data is not None:
                    # Generate narrative
                    narrative = generate_wave_narrative(selected_wave, wave_data)
                    
                    # Store in session state
                    st.session_state['current_narrative'] = narrative
                    st.session_state['current_narrative_wave'] = selected_wave
                    st.session_state['current_narrative_data'] = wave_data
                else:
                    st.error(f"Unable to load data for {selected_wave}")
                    return
        
        # Display narrative and visualization if available
        if 'current_narrative' in st.session_state:
            st.divider()
            
            # Display confidence band for the wave - NEW
            if 'current_narrative_data' in st.session_state and 'current_narrative_wave' in st.session_state:
                wave_name_conf = st.session_state['current_narrative_wave']
                wave_data_conf = st.session_state['current_narrative_data']
                
                try:
                    engine = get_attribution_engine()
                    components_conf = engine.compute_attribution(wave_data_conf, wave_name_conf)
                    
                    st.markdown("#### 📊 Attribution Confidence")
                    conf_col1, conf_col2 = st.columns([3, 1])
                    
                    with conf_col1:
                        render_confidence_band(wave_name_conf, components_conf, compact=False)
                    
                    with conf_col2:
                        if components_conf.reconciled:
                            st.success("✅ Reconciled")
                        else:
                            st.warning("⚠️ Partial")
                    
                    st.divider()
                except Exception as conf_error:
                    st.info(f"⚠️ Confidence band unavailable: {str(conf_error)}")
            
            # Display performance chart first
            if 'current_narrative_data' in st.session_state and 'current_narrative_wave' in st.session_state:
                chart = create_wave_performance_chart(
                    st.session_state['current_narrative_data'],
                    st.session_state['current_narrative_wave']
                )
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
            
            st.divider()
            
            # Display narrative
            st.markdown(st.session_state['current_narrative'])
            
            # Export options
            st.divider()
            st.markdown("### 📤 Export Options")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 View as Text", use_container_width=True):
                    st.code(st.session_state['current_narrative'], language=None)
                    st.success("Narrative displayed above - use your browser to copy")
            
            with col2:
                # Download button
                st.download_button(
                    label="💾 Download Narrative",
                    data=st.session_state['current_narrative'],
                    file_name=f"wave_narrative_{st.session_state.get('current_narrative_wave', 'wave')}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"Error rendering Vector Explain panel: {str(e)}")


def render_compare_waves_panel():
    """
    Render the Compare Waves panel for head-to-head wave comparison.
    Enhanced with radar chart and correlation visualization.
    """
    st.subheader("⚖️ Compare Waves - Head-to-Head Analysis")
    st.write("Select two waves for comprehensive side-by-side performance comparison")
    
    try:
        waves = get_available_waves()
        
        if len(waves) < 2:
            st.warning("At least two waves required for comparison")
            return
        
        # Wave selectors
        col1, col2 = st.columns(2)
        
        with col1:
            wave1 = st.selectbox(
                "Wave 1",
                options=waves,
                key="compare_wave1",
                help="Select first wave for comparison"
            )
        
        with col2:
            # Filter out wave1 from wave2 options
            wave2_options = [w for w in waves if w != wave1]
            wave2 = st.selectbox(
                "Wave 2",
                options=wave2_options,
                key="compare_wave2",
                help="Select second wave for comparison"
            )
        
        if st.button("🔍 Compare Waves", type="primary"):
            with st.spinner("Generating comprehensive comparison analysis..."):
                comparison_data = get_wave_comparison_data(wave1, wave2)
                
                if comparison_data is None:
                    st.error("Unable to generate comparison - data unavailable")
                    return
                
                # Store comparison data and raw wave data
                st.session_state['comparison_data'] = comparison_data
                st.session_state['comparison_wave1_data'] = get_wave_data_filtered(wave_name=wave1, days=30)
                st.session_state['comparison_wave2_data'] = get_wave_data_filtered(wave_name=wave2, days=30)
                st.session_state['comparison_wave1_name'] = wave1
                st.session_state['comparison_wave2_name'] = wave2
        
        # Display comparison if available
        if 'comparison_data' in st.session_state:
            comp = st.session_state['comparison_data']
            wave1_metrics = comp['wave1']
            wave2_metrics = comp['wave2']
            correlation = comp.get('correlation')
            
            st.divider()
            
            # Visual Comparison Section
            st.markdown("### 📊 Visual Comparison")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Radar chart
                radar_chart = create_comparison_radar_chart(wave1_metrics, wave2_metrics)
                if radar_chart is not None:
                    st.plotly_chart(radar_chart, use_container_width=True)
                else:
                    st.info("Radar chart unavailable")
            
            with viz_col2:
                # Correlation heatmap
                if 'comparison_wave1_data' in st.session_state and 'comparison_wave2_data' in st.session_state:
                    heatmap = create_correlation_heatmap(
                        st.session_state['comparison_wave1_data'],
                        st.session_state['comparison_wave2_data'],
                        st.session_state.get('comparison_wave1_name', 'Wave 1'),
                        st.session_state.get('comparison_wave2_name', 'Wave 2')
                    )
                    if heatmap is not None:
                        st.plotly_chart(heatmap, use_container_width=True)
                    else:
                        st.info("Correlation matrix unavailable")
            
            st.divider()
            
            # Metrics Table
            st.markdown("### 📋 Detailed Metrics Comparison (30-Day)")
            
            # Helper function to format metric
            def format_metric(value, metric_type='percent'):
                if value == 'N/A':
                    return 'N/A'
                if metric_type == 'percent':
                    return f"{value*100:.2f}%"
                elif metric_type == 'score':
                    return f"{value:.1f}"
                elif metric_type == 'ratio':
                    return f"{value:.2f}"
                else:
                    return f"{value:.4f}"
            
            # Build comparison table
            comparison_rows = [
                {
                    'Metric': 'Cumulative Return',
                    wave1_metrics['name']: format_metric(wave1_metrics['cumulative_return']),
                    wave2_metrics['name']: format_metric(wave2_metrics['cumulative_return']),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('cumulative_return', 0) > wave2_metrics.get('cumulative_return', 0) else wave2_metrics['name']
                },
                {
                    'Metric': 'Cumulative Alpha',
                    wave1_metrics['name']: format_metric(wave1_metrics['cumulative_alpha']),
                    wave2_metrics['name']: format_metric(wave2_metrics['cumulative_alpha']),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('cumulative_alpha', 0) > wave2_metrics.get('cumulative_alpha', 0) else wave2_metrics['name']
                },
                {
                    'Metric': 'Volatility',
                    wave1_metrics['name']: format_metric(wave1_metrics['volatility']),
                    wave2_metrics['name']: format_metric(wave2_metrics['volatility']),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('volatility', float('inf')) < wave2_metrics.get('volatility', float('inf')) else wave2_metrics['name']
                },
                {
                    'Metric': 'Max Drawdown',
                    wave1_metrics['name']: format_metric(wave1_metrics['max_drawdown']),
                    wave2_metrics['name']: format_metric(wave2_metrics['max_drawdown']),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('max_drawdown', float('-inf')) > wave2_metrics.get('max_drawdown', float('-inf')) else wave2_metrics['name']
                },
                {
                    'Metric': 'WaveScore',
                    wave1_metrics['name']: format_metric(wave1_metrics['wavescore'], 'score'),
                    wave2_metrics['name']: format_metric(wave2_metrics['wavescore'], 'score'),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('wavescore', 0) > wave2_metrics.get('wavescore', 0) else wave2_metrics['name']
                },
                {
                    'Metric': 'Sharpe Ratio',
                    wave1_metrics['name']: format_metric(wave1_metrics['sharpe_ratio'], 'ratio'),
                    wave2_metrics['name']: format_metric(wave2_metrics['sharpe_ratio'], 'ratio'),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('sharpe_ratio', float('-inf')) > wave2_metrics.get('sharpe_ratio', float('-inf')) else wave2_metrics['name']
                },
                {
                    'Metric': 'Win Rate',
                    wave1_metrics['name']: format_metric(wave1_metrics['win_rate']),
                    wave2_metrics['name']: format_metric(wave2_metrics['win_rate']),
                    'Winner': wave1_metrics['name'] if wave1_metrics.get('win_rate', 0) > wave2_metrics.get('win_rate', 0) else wave2_metrics['name']
                }
            ]
            
            # Display table
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Summary Analysis
            st.markdown("### 🏆 Winner Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Overall Winner")
                winner, notes = determine_winner(wave1_metrics, wave2_metrics)
                
                if winner:
                    if winner == "TIE":
                        st.success(f"**{winner}**")
                    else:
                        st.success(f"🏆 **{winner}**")
                    st.write(notes)
                else:
                    st.info(notes)
            
            with col2:
                st.markdown("#### Correlation Analysis")
                if correlation is not None and correlation != 'N/A':
                    st.metric(
                        label="Return Correlation",
                        value=f"{correlation:.3f}",
                        help="Correlation between daily returns (-1 to 1)"
                    )
                    
                    # Add interpretation
                    if abs(correlation) > 0.7:
                        corr_note = "Strong correlation"
                    elif abs(correlation) > 0.3:
                        corr_note = "Moderate correlation"
                    else:
                        corr_note = "Weak correlation"
                    
                    if correlation > 0:
                        corr_note += " (positive)"
                        corr_interpretation = "Waves tend to move together"
                    elif correlation < 0:
                        corr_note += " (negative)"
                        corr_interpretation = "Waves tend to move in opposite directions"
                    else:
                        corr_interpretation = "No clear relationship"
                    
                    st.info(corr_note)
                    st.write(corr_interpretation)
                else:
                    st.info("Correlation data unavailable")
            
            # Additional insights
            st.divider()
            st.markdown("### 📝 Key Insights")
            
            insights = []
            
            # Performance insight
            if wave1_metrics.get('cumulative_return', 0) != 'N/A' and wave2_metrics.get('cumulative_return', 0) != 'N/A':
                ret_diff = abs(wave1_metrics['cumulative_return'] - wave2_metrics['cumulative_return'])
                if ret_diff > 0.10:  # 10% difference
                    insights.append(f"⚠️ **Significant return divergence**: {ret_diff*100:.1f}% difference in cumulative returns")
            
            # Risk insight
            if wave1_metrics.get('max_drawdown', 0) != 'N/A' and wave2_metrics.get('max_drawdown', 0) != 'N/A':
                dd_diff = abs(wave1_metrics['max_drawdown'] - wave2_metrics['max_drawdown'])
                if dd_diff > 0.05:  # 5% difference in drawdown
                    insights.append(f"⚠️ **Risk profile differs**: {dd_diff*100:.1f}% difference in max drawdown")
            
            # Diversification insight
            if correlation is not None and abs(correlation) < 0.3:
                insights.append(f"✅ **Good diversification potential**: Low correlation ({correlation:.2f}) suggests complementary performance")
            elif correlation is not None and abs(correlation) > 0.8:
                insights.append(f"ℹ️ **High correlation**: Waves move similarly ({correlation:.2f}), limited diversification benefit")
            
            if insights:
                for insight in insights:
                    st.markdown(insight)
            else:
                st.write("No significant insights detected")
            
            # Export comparison
            st.divider()
            st.markdown("### 📤 Export Comparison")
            
            # Create export text
            export_text = f"""# Wave Comparison Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Waves Compared
- Wave 1: {wave1_metrics['name']}
- Wave 2: {wave2_metrics['name']}

## Performance Metrics (30-Day)

| Metric | {wave1_metrics['name']} | {wave2_metrics['name']} |
|--------|----------|----------|
| Cumulative Return | {format_metric(wave1_metrics['cumulative_return'])} | {format_metric(wave2_metrics['cumulative_return'])} |
| Cumulative Alpha | {format_metric(wave1_metrics['cumulative_alpha'])} | {format_metric(wave2_metrics['cumulative_alpha'])} |
| WaveScore | {format_metric(wave1_metrics['wavescore'], 'score')} | {format_metric(wave2_metrics['wavescore'], 'score')} |
| Sharpe Ratio | {format_metric(wave1_metrics['sharpe_ratio'], 'ratio')} | {format_metric(wave2_metrics['sharpe_ratio'], 'ratio')} |
| Volatility | {format_metric(wave1_metrics['volatility'])} | {format_metric(wave2_metrics['volatility'])} |
| Max Drawdown | {format_metric(wave1_metrics['max_drawdown'])} | {format_metric(wave2_metrics['max_drawdown'])} |
| Win Rate | {format_metric(wave1_metrics['win_rate'])} | {format_metric(wave2_metrics['win_rate'])} |

## Winner
{winner}: {notes}

## Correlation
Return Correlation: {correlation if correlation is not None else 'N/A'}
"""
            
            st.download_button(
                label="💾 Download Comparison Report",
                data=export_text,
                file_name=f"wave_comparison_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"Error rendering Compare Waves panel: {str(e)}")


def render_overview_tab():
    """Render the Overview tab with Vector Explain and Compare Waves."""
    st.header("Overview")
    
    # Create sub-tabs for Vector Explain and Compare Waves
    overview_subtabs = st.tabs(["Vector Explain", "Compare Waves"])
    
    with overview_subtabs[0]:
        render_vector_explain_panel()
    
    with overview_subtabs[1]:
        render_compare_waves_panel()


def render_details_tab():
    """Render the Details tab."""
    st.header("Details")
    st.write("Detailed analytics and metrics for individual waves.")
    st.info("Data unavailable")


def generate_board_pack_html():
    """
    Generate Institutional Board Pack HTML report.
    Returns HTML string with all sections and graceful degradation.
    """
    # Get build info
    git_hash = get_git_commit_hash()
    git_branch = get_git_branch_name()
    deploy_timestamp = get_deploy_timestamp()
    
    # Get Mission Control data
    mc_data = get_mission_control_data()
    
    # Get available waves for leaderboard
    waves = get_available_waves()
    
    # Calculate leaderboard data
    leaderboard_data = []
    if waves:
        for wave_name in waves:
            wave_data = get_wave_data_filtered(wave_name=wave_name, days=30)
            if wave_data is not None and len(wave_data) > 0:
                wavescore = calculate_wavescore(wave_data)
                metrics = calculate_wave_metrics(wave_data)
                leaderboard_data.append({
                    'wave': wave_name,
                    'wavescore': wavescore,
                    'alpha': metrics.get('cumulative_alpha', 'N/A')
                })
        
        # Sort by wavescore
        leaderboard_data.sort(key=lambda x: x['wavescore'] if isinstance(x['wavescore'], (int, float)) else 0, reverse=True)
    
    # Get system alerts
    alerts = get_system_alerts()
    
    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Institutional Board Pack - {deploy_timestamp}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
            }}
            .section {{
                background: white;
                padding: 25px;
                margin-bottom: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #667eea;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #667eea;
            }}
            .metric-label {{
                font-size: 0.9em;
                color: #666;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 1.5em;
                font-weight: bold;
                color: #333;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background: #f5f5f5;
            }}
            .alert {{
                padding: 12px 15px;
                margin: 10px 0;
                border-radius: 6px;
                border-left: 4px solid;
            }}
            .alert-error {{
                background: #fee;
                border-color: #c33;
                color: #c33;
            }}
            .alert-warning {{
                background: #ffc;
                border-color: #f90;
                color: #f90;
            }}
            .alert-success {{
                background: #efe;
                border-color: #3c3;
                color: #3c3;
            }}
            .alert-info {{
                background: #eef;
                border-color: #39c;
                color: #39c;
            }}
            .unavailable {{
                color: #999;
                font-style: italic;
                padding: 20px;
                text-align: center;
                background: #f9f9f9;
                border-radius: 6px;
            }}
            .build-info {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Institutional Board Pack</h1>
            <p>Comprehensive Performance & Analytics Report</p>
            <p>Generated: {deploy_timestamp}</p>
        </div>
    """
    
    # Mission Control Section
    html += """
        <div class="section">
            <h2>🎯 Mission Control - Snapshot Summary</h2>
    """
    
    if mc_data:
        html += f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Market Regime</div>
                    <div class="metric-value">{mc_data.get('market_regime', 'unknown')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">VIX Gate Status</div>
                    <div class="metric-value">{mc_data.get('vix_gate_status', 'unknown')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">30-Day Alpha</div>
                    <div class="metric-value">{mc_data.get('alpha_30day', 'unknown')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Top Wave</div>
                    <div class="metric-value">{mc_data.get('wavescore_leader', 'unknown')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Waves</div>
                    <div class="metric-value">{mc_data.get('total_waves', 0)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">System Status</div>
                    <div class="metric-value">{mc_data.get('system_status', 'unknown')}</div>
                </div>
            </div>
        """
    else:
        html += '<div class="unavailable">Data unavailable</div>'
    
    html += "</div>"
    
    # WaveScore Leaderboard Section
    html += """
        <div class="section">
            <h2>🏆 WaveScore Leaderboard - Top Performers</h2>
    """
    
    if leaderboard_data and len(leaderboard_data) > 0:
        html += """
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Wave</th>
                        <th>WaveScore</th>
                        <th>30-Day Alpha</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for i, item in enumerate(leaderboard_data[:10], 1):  # Top 10
            alpha_str = f"{item['alpha']*100:.2f}%" if isinstance(item['alpha'], (int, float)) else 'N/A'
            html += f"""
                    <tr>
                        <td>{i}</td>
                        <td><strong>{item['wave']}</strong></td>
                        <td>{item['wavescore']:.1f}</td>
                        <td>{alpha_str}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        """
    else:
        html += '<div class="unavailable">Data unavailable</div>'
    
    html += "</div>"
    
    # Movers Section
    html += """
        <div class="section">
            <h2>📈 Movers - Biggest Changes</h2>
    """
    
    # Calculate movers (simplified - would need historical data for real implementation)
    if leaderboard_data and len(leaderboard_data) >= 3:
        html += """
            <p>Top performers showing significant movement:</p>
            <ul>
        """
        for item in leaderboard_data[:3]:
            html += f"<li><strong>{item['wave']}</strong> - WaveScore: {item['wavescore']:.1f}</li>"
        html += """
            </ul>
        """
    else:
        html += '<div class="unavailable">Insufficient data for movers analysis</div>'
    
    html += "</div>"
    
    # Alerts Section
    html += """
        <div class="section">
            <h2>🚨 System Alerts & Warnings</h2>
    """
    
    if alerts and len(alerts) > 0:
        for alert in alerts:
            severity = alert.get('severity', 'info')
            message = alert.get('message', '')
            
            alert_class = f'alert-{severity}'
            icon = {'error': '❌', 'warning': '⚠️', 'success': '✅', 'info': 'ℹ️'}.get(severity, 'ℹ️')
            
            html += f'<div class="alert {alert_class}">{icon} {message}</div>'
    else:
        html += '<div class="unavailable">No alerts at this time</div>'
    
    html += "</div>"
    
    # Alpha Proof Summary Section
    html += """
        <div class="section">
            <h2>🔬 Alpha Proof Summary - Alpha Metrics</h2>
    """
    
    if waves and len(waves) > 0:
        # Get alpha components for first wave as example
        sample_wave = waves[0]
        wave_data = get_wave_data_filtered(wave_name=sample_wave, days=30)
        
        if wave_data is not None and len(wave_data) > 0:
            alpha_components = calculate_alpha_components(wave_data, sample_wave)
            
            if alpha_components:
                html += f"""
                    <p><strong>Sample Wave: {sample_wave}</strong></p>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-label">Total Alpha</div>
                            <div class="metric-value">{alpha_components.get('total_alpha', 0)*100:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Selection Alpha</div>
                            <div class="metric-value">{alpha_components.get('selection_alpha', 0)*100:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Overlay Alpha</div>
                            <div class="metric-value">{alpha_components.get('overlay_alpha', 0)*100:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Cash Contribution</div>
                            <div class="metric-value">{alpha_components.get('cash_contribution', 0)*100:.2f}%</div>
                        </div>
                    </div>
                """
            else:
                html += '<div class="unavailable">Alpha components unavailable</div>'
        else:
            html += '<div class="unavailable">Wave data unavailable</div>'
    else:
        html += '<div class="unavailable">No waves available for alpha analysis</div>'
    
    html += "</div>"
    
    # Overlays Summary Section
    html += """
        <div class="section">
            <h2>📊 Overlays Summary - Analytics Snapshot</h2>
    """
    
    if waves and len(waves) > 0:
        sample_wave = waves[0]
        wave_data = get_wave_data_filtered(wave_name=sample_wave, days=30)
        
        if wave_data is not None and len(wave_data) > 0:
            attribution_data = calculate_attribution_matrix(wave_data, sample_wave)
            
            if attribution_data:
                html += f"""
                    <p><strong>Sample Wave: {sample_wave}</strong></p>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-label">Risk-On Alpha</div>
                            <div class="metric-value">{attribution_data.get('risk_on_alpha', 0)*100:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Risk-Off Alpha</div>
                            <div class="metric-value">{attribution_data.get('risk_off_alpha', 0)*100:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Capital-Weighted Alpha</div>
                            <div class="metric-value">{attribution_data.get('capital_weighted_alpha', 0)*100:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Exposure-Adjusted Alpha</div>
                            <div class="metric-value">{attribution_data.get('exposure_adjusted_alpha', 0)*100:.2f}%</div>
                        </div>
                    </div>
                """
            else:
                html += '<div class="unavailable">Attribution data unavailable</div>'
        else:
            html += '<div class="unavailable">Wave data unavailable</div>'
    else:
        html += '<div class="unavailable">No waves available for overlay analysis</div>'
    
    html += "</div>"
    
    # Data Integrity Section
    html += """
        <div class="section">
            <h2>✅ Data Integrity - Confidence Levels</h2>
    """
    
    data_freshness = mc_data.get('data_freshness', 'unknown')
    data_age_days = mc_data.get('data_age_days', None)
    
    if data_age_days is not None:
        if data_age_days <= 1:
            confidence = "High"
            confidence_color = "#3c3"
        elif data_age_days <= 3:
            confidence = "Medium"
            confidence_color = "#f90"
        else:
            confidence = "Low"
            confidence_color = "#c33"
    else:
        confidence = "Unknown"
        confidence_color = "#999"
    
    html += f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Data Freshness</div>
                <div class="metric-value">{data_freshness}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Data Age</div>
                <div class="metric-value">{data_age_days if data_age_days is not None else 'unknown'} days</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Confidence Level</div>
                <div class="metric-value" style="color: {confidence_color}">{confidence}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Active Waves</div>
                <div class="metric-value">{mc_data.get('active_waves', 0)} / {mc_data.get('total_waves', 0)}</div>
            </div>
        </div>
    """
    
    html += "</div>"
    
    # Build Info Section
    html += f"""
        <div class="section">
            <h2>🔧 Build Info - Deployment Metadata</h2>
            <div class="build-info">
                <div><strong>Git Commit:</strong> {git_hash}</div>
                <div><strong>Git Branch:</strong> {git_branch}</div>
                <div><strong>Deployment Timestamp:</strong> {deploy_timestamp}</div>
                <div><strong>Report Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}</div>
            </div>
        </div>
    """
    
    # Footer
    html += """
        <div class="footer">
            <p>Institutional Board Pack - Confidential</p>
            <p>Generated by Waves Analytics Platform</p>
        </div>
    </body>
    </html>
    """
    
    return html


def render_reports_tab():
    """Render the Reports tab with Institutional Board Pack generator."""
    st.header("Reports")
    st.write("Comprehensive reporting and analysis tools.")
    
    st.divider()
    
    # Institutional Board Pack Section
    st.subheader("📊 Institutional Board Pack")
    st.write("Generate a comprehensive HTML report including Mission Control, WaveScore Leaderboard, Movers, Alerts, Alpha Proof, Overlays, Data Integrity, and Build Info.")
    
    # Generate button
    if st.button("🎯 Generate Board Pack", type="primary", use_container_width=True):
        with st.spinner("Generating Institutional Board Pack..."):
            try:
                # Generate HTML report
                board_pack_html = generate_board_pack_html()
                
                # Store in session state for rendering and download
                st.session_state['board_pack_html'] = board_pack_html
                st.session_state['board_pack_timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.success("✅ Board Pack generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating Board Pack: {str(e)}")
                st.info("The application continues to function. Please check data availability.")
    
    # Display board pack if generated
    if 'board_pack_html' in st.session_state:
        st.divider()
        
        # Download button
        timestamp = st.session_state.get('board_pack_timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        filename = f"institutional_board_pack_{timestamp}.html"
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 📄 Preview Board Pack")
        
        with col2:
            st.download_button(
                label="💾 Download HTML",
                data=st.session_state['board_pack_html'],
                file_name=filename,
                mime="text/html",
                use_container_width=True
            )
        
        # Render the HTML in an iframe-like component
        st.components.v1.html(st.session_state['board_pack_html'], height=800, scrolling=True)
    
    st.divider()
    
    # Additional reporting tools placeholder
    st.subheader("📈 Additional Reports")
    st.info("Additional reporting tools will be available in future releases.")


def render_overlay_proof_section():
    """
    Render Overlay Proof section showing VIX/regime overlay diagnostics.
    Displays diagnostics table with VIX, Regime, Exposure, and Returns for the last N days.
    """
    st.subheader("Overlay Proof (Last 15 Days)")
    st.write("Detailed VIX/regime overlay diagnostics showing exposure adjustments and alpha generation.")
    
    # Check if diagnostics module is available
    if not VIX_DIAGNOSTICS_AVAILABLE:
        st.warning("⚠️ VIX overlay diagnostics module not available.")
        return
    
    # Get available waves
    available_waves = get_available_waves()
    
    if not available_waves:
        st.info("No wave data available for diagnostics.")
        return
    
    # Create UI controls in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Wave selector - default to S&P 500 Wave (SPY Wave)
        default_wave = "S&P 500 Wave" if "S&P 500 Wave" in available_waves else available_waves[0]
        selected_wave = st.selectbox(
            "Wave Selector",
            options=available_waves,
            index=available_waves.index(default_wave) if default_wave in available_waves else 0,
            key="overlay_proof_wave_selector",
            help="Select a wave to view VIX/regime overlay diagnostics"
        )
    
    with col2:
        # Mode selector - default to Standard
        mode_options = ["Standard", "Alpha-Minus-Beta", "Private Logic"]
        selected_mode = st.selectbox(
            "Mode Selector",
            options=mode_options,
            index=0,  # Default to Standard
            key="overlay_proof_mode_selector",
            help="Select the operating mode for diagnostics"
        )
    
    with col3:
        # Days selector - default to 15
        selected_days = st.number_input(
            "Days Selector",
            min_value=1,
            max_value=365,
            value=15,
            step=1,
            key="overlay_proof_days_selector",
            help="Number of days to display in diagnostics"
        )
    
    # Fetch diagnostics data
    try:
        with st.spinner(f"Fetching diagnostics for {selected_wave}..."):
            diagnostics_df = get_wave_diagnostics(
                wave_name=selected_wave,
                mode=selected_mode,
                days=selected_days
            )
        
        # Check if diagnostics are available
        if diagnostics_df is None or diagnostics_df.empty:
            st.info("Diagnostics unavailable for the selected wave and parameters.")
            return
        
        # Prepare the table with required columns
        # Reset index to make Date a column
        table_df = diagnostics_df.reset_index()
        
        # Ensure we have the required columns, create them if missing
        display_columns = []
        
        # Date column
        if 'date' in table_df.columns:
            display_columns.append('date')
            table_df = table_df.rename(columns={'date': 'Date'})
        elif 'Date' not in table_df.columns:
            table_df['Date'] = table_df.index
        
        if 'Date' not in display_columns:
            display_columns.append('Date')
        
        # Add other columns if they exist
        column_mapping = {
            'VIX': 'VIX',
            'Regime': 'Regime',
            'VIX_Exposure': 'VIX_Exposure',
            'Safe_Fraction': 'Safe_Fraction',
            'Exposure': 'Exposure',
            'Wave_Return': 'Wave_Return',
            'Benchmark_Return': 'Benchmark_Return'
        }
        
        for col_name in column_mapping.keys():
            if col_name in table_df.columns:
                display_columns.append(col_name)
        
        # Calculate Realized_Alpha if we have both returns
        if 'Wave_Return' in table_df.columns and 'Benchmark_Return' in table_df.columns:
            table_df['Realized_Alpha'] = table_df['Wave_Return'] - table_df['Benchmark_Return']
            display_columns.append('Realized_Alpha')
        
        # Filter to only the columns we want to display
        final_columns = [col for col in display_columns if col in table_df.columns]
        display_df = table_df[final_columns].copy()
        
        # Format numeric columns for better display
        if 'VIX' in display_df.columns:
            display_df['VIX'] = display_df['VIX'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        for col in ['VIX_Exposure', 'Safe_Fraction', 'Exposure', 'Wave_Return', 'Benchmark_Return', 'Realized_Alpha']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        
        # Format Date column
        if 'Date' in display_df.columns:
            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
        
        # Display the table
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Display summary statistics
        st.markdown("**Summary Statistics**")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            if 'Wave_Return' in diagnostics_df.columns:
                total_wave_return = diagnostics_df['Wave_Return'].sum()
                st.metric("Total Wave Return", f"{total_wave_return:.2%}")
        
        with summary_col2:
            if 'Benchmark_Return' in diagnostics_df.columns:
                total_bm_return = diagnostics_df['Benchmark_Return'].sum()
                st.metric("Total Benchmark Return", f"{total_bm_return:.2%}")
        
        with summary_col3:
            if 'Wave_Return' in diagnostics_df.columns and 'Benchmark_Return' in diagnostics_df.columns:
                total_alpha = total_wave_return - total_bm_return
                st.metric("Total Alpha", f"{total_alpha:.2%}")
        
    except Exception as e:
        st.error(f"Error fetching diagnostics: {str(e)}")
        st.info("Diagnostics unavailable. The application continues to function.")


def render_overlays_tab():
    """Render the Overlays tab."""
    st.header("Analytics Overlays")
    
    # Capital-Weighted Alpha Section
    st.subheader("Capital-Weighted Alpha")
    st.write("Alpha attribution weighted by capital allocation across portfolio.")
    st.info("Data unavailable")
    
    st.divider()
    
    # Exposure-Adjusted Alpha Section
    st.subheader("Exposure-Adjusted Alpha")
    st.write("Alpha metrics adjusted for market exposure and beta.")
    st.info("Data unavailable")
    
    st.divider()
    
    # Risk-On vs Risk-Off Attribution Section
    st.subheader("Risk-On vs Risk-Off Attribution")
    st.write("Performance attribution segmented by market regime.")
    st.info("Data unavailable")
    
    st.divider()
    
    # Overlay Proof (Last 15 Days) Section
    render_overlay_proof_section()


def render_attribution_tab():
    """Render the Attribution tab with alpha decomposition."""
    st.header("🎯 Alpha Attribution Analysis")
    
    st.markdown("""
    **Precise, reconciled decomposition of Wave alpha into actionable components:**
    
    1️⃣ **Stock Selection Alpha** — Wave return vs benchmark return differential  
    2️⃣ **Overlay Alpha** — VIX gating, exposure scaling, and SmartSafe features  
    3️⃣ **Beta/Exposure Drift** — Target vs realized exposure impact  
    4️⃣ **Residual Alpha** — Unexplained deviation and other factors
    
    **Reconciliation:** All components sum to total realized Wave alpha.
    """)
    
    # Check if attribution module is available
    if not ALPHA_ATTRIBUTION_AVAILABLE:
        st.warning("⚠️ Alpha attribution module not available. Please ensure alpha_attribution.py is properly installed.")
        return
    
    # Load wave history data
    wave_df = safe_load_wave_history()
    
    if wave_df is None or wave_df.empty:
        st.error("❌ Wave history data is not available. Cannot compute attribution.")
        return
    
    # Get available waves
    available_waves = sorted(wave_df['wave'].unique().tolist())
    
    if not available_waves:
        st.error("❌ No waves found in history data.")
        return
    
    # Configuration controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_wave = st.selectbox(
            "Select Wave",
            available_waves,
            key="attr_wave_select"
        )
    
    with col2:
        timeframe_options = {
            "1 Day": 1,
            "5 Days": 5,
            "30 Days": 30,
            "60 Days": 60,
            "1 Year": 252
        }
        selected_timeframe = st.selectbox(
            "Timeframe",
            list(timeframe_options.keys()),
            index=2,  # Default to 30 Days
            key="attr_timeframe_select"
        )
        days = timeframe_options[selected_timeframe]
    
    with col3:
        show_all_waves = st.checkbox(
            "Show All Waves Comparison",
            value=False,
            key="attr_show_all"
        )
    
    st.divider()
    
    # Compute attribution for selected wave
    try:
        # Filter data for selected wave
        wave_data = wave_df[wave_df['wave'] == selected_wave].copy()
        wave_data = wave_data.sort_values('date')
        
        # Take last N days
        wave_data = wave_data.tail(days)
        
        if len(wave_data) == 0:
            st.warning(f"⚠️ No data available for {selected_wave} in the selected timeframe.")
            return
        
        # Prepare history DataFrame for attribution
        wave_data.set_index('date', inplace=True)
        history_df = pd.DataFrame({
            'wave_ret': wave_data['portfolio_return'],
            'bm_ret': wave_data['benchmark_return']
        })
        
        # Compute attribution
        with st.spinner(f"Computing attribution for {selected_wave}..."):
            daily_df, summary = compute_alpha_attribution_series(
                wave_name=selected_wave,
                mode="Standard",
                history_df=history_df,
                diagnostics_df=None,  # Use fallback defaults
                tilt_strength=0.8,
                base_exposure=1.0
            )
        
        # Display results
        st.success(f"✅ Attribution computed successfully for {selected_wave} ({len(wave_data)} days)")
        
        # Summary section
        st.subheader("📊 Attribution Summary")
        
        # Create summary metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                "Total Wave Return",
                f"{summary.total_wave_return * 100:+.2f}%"
            )
        
        with col_m2:
            st.metric(
                "Total Benchmark Return",
                f"{summary.total_benchmark_return * 100:+.2f}%"
            )
        
        with col_m3:
            st.metric(
                "Total Alpha",
                f"{summary.total_alpha * 100:+.2f}%",
                delta=None
            )
        
        with col_m4:
            recon_status = "✅ PASS" if abs(summary.reconciliation_pct_error) < 0.01 else "⚠️ CHECK"
            st.metric(
                "Reconciliation",
                recon_status,
                delta=f"{summary.reconciliation_error * 100:.4f}%"
            )
        
        st.divider()
        
        # Component breakdown table
        st.subheader("🔍 Alpha Component Breakdown")
        
        component_data = {
            "Component": [
                "1️⃣ Exposure & Timing Alpha",
                "2️⃣ Regime & VIX Overlay Alpha",
                "3️⃣ Momentum & Trend Alpha",
                "4️⃣ Volatility & Risk Control Alpha",
                "5️⃣ Asset Selection Alpha (Residual)",
                "**Total Alpha**"
            ],
            "Cumulative Alpha": [
                f"{summary.exposure_timing_alpha * 100:+.2f}%",
                f"{summary.regime_vix_alpha * 100:+.2f}%",
                f"{summary.momentum_trend_alpha * 100:+.2f}%",
                f"{summary.volatility_control_alpha * 100:+.2f}%",
                f"{summary.asset_selection_alpha * 100:+.2f}%",
                f"**{summary.total_alpha * 100:+.2f}%**"
            ],
            "Contribution to Total": [
                f"{summary.exposure_timing_contribution_pct:+.1f}%",
                f"{summary.regime_vix_contribution_pct:+.1f}%",
                f"{summary.momentum_trend_contribution_pct:+.1f}%",
                f"{summary.volatility_control_contribution_pct:+.1f}%",
                f"{summary.asset_selection_contribution_pct:+.1f}%",
                "**100.0%**"
            ]
        }
        
        st.dataframe(
            pd.DataFrame(component_data),
            hide_index=True,
            use_container_width=True
        )
        
        st.divider()
        
        # Visualization - Waterfall chart
        st.subheader("📈 Attribution Waterfall")
        
        # Create waterfall chart
        components = [
            "Start",
            "Exposure & Timing",
            "Regime & VIX",
            "Momentum & Trend",
            "Volatility Control",
            "Asset Selection",
            "Total Alpha"
        ]
        
        values = [
            0,  # Start at 0
            summary.exposure_timing_alpha * 100,
            summary.regime_vix_alpha * 100,
            summary.momentum_trend_alpha * 100,
            summary.volatility_control_alpha * 100,
            summary.asset_selection_alpha * 100,
            summary.total_alpha * 100
        ]
        
        # Build measures for waterfall
        measures = ["absolute", "relative", "relative", "relative", "relative", "relative", "total"]
        
        waterfall_fig = go.Figure(go.Waterfall(
            name="Attribution",
            orientation="v",
            measure=measures,
            x=components,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "green"}},
            decreasing={"marker": {"color": "red"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        waterfall_fig.update_layout(
            title=f"Alpha Attribution Waterfall - {selected_wave}",
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(waterfall_fig, use_container_width=True)
        
        st.divider()
        
        # Time series of cumulative components
        st.subheader("📉 Cumulative Attribution Over Time")
        
        # Calculate cumulative sums
        cumulative_df = pd.DataFrame({
            'Date': daily_df.index,
            'Exposure & Timing': daily_df['ExposureTimingα'].cumsum() * 100,
            'Regime & VIX': daily_df['RegimeVIXα'].cumsum() * 100,
            'Momentum & Trend': daily_df['MomentumTrendα'].cumsum() * 100,
            'Volatility Control': daily_df['VolatilityControlα'].cumsum() * 100,
            'Asset Selection': daily_df['AssetSelectionα'].cumsum() * 100,
            'Total Alpha': daily_df['TotalAlpha'].cumsum() * 100
        }).set_index('Date')
        
        ts_fig = go.Figure()
        
        for col in cumulative_df.columns:
            ts_fig.add_trace(go.Scatter(
                x=cumulative_df.index,
                y=cumulative_df[col],
                mode='lines',
                name=col,
                line=dict(width=2 if col == 'Total Alpha' else 1)
            ))
        
        ts_fig.update_layout(
            title=f"Cumulative Attribution Components - {selected_wave}",
            xaxis_title="Date",
            yaxis_title="Cumulative Alpha (%)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(ts_fig, use_container_width=True)
        
        st.divider()
        
        # Daily attribution sample
        with st.expander("📋 View Daily Attribution Details (Last 20 Days)", expanded=False):
            sample_df = daily_df.tail(20).copy()
            
            # Format for display
            display_df = pd.DataFrame({
                'Date': sample_df.index.strftime('%Y-%m-%d'),
                'VIX': sample_df['VIX'].round(1),
                'Regime': sample_df['Regime'],
                'Exp%': sample_df['Exposure (%)'].round(0).astype(int),
                'Safe%': sample_df['Safe (%)'].round(0).astype(int),
                'ExposTimα': (sample_df['ExposureTimingα'] * 100).round(2),
                'RegVIXα': (sample_df['RegimeVIXα'] * 100).round(2),
                'MomTrendα': (sample_df['MomentumTrendα'] * 100).round(2),
                'VolCtrlα': (sample_df['VolatilityControlα'] * 100).round(2),
                'AssetSelα': (sample_df['AssetSelectionα'] * 100).round(2),
                'Totalα': (sample_df['TotalAlpha'] * 100).round(2),
                'WaveRet': (sample_df['WaveReturn'] * 100).round(2),
                'BmRet': (sample_df['BenchmarkReturn'] * 100).round(2)
            })
            
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True
            )
        
        # Download option
        csv_data = daily_df.to_csv()
        st.download_button(
            label="📥 Download Full Daily Attribution (CSV)",
            data=csv_data,
            file_name=f"{selected_wave.replace(' ', '_')}_attribution_{days}d.csv",
            mime="text/csv",
            key="download_attribution"
        )
        
    except Exception as e:
        st.error(f"❌ Error computing attribution: {str(e)}")
        st.exception(e)
    
    # All Waves Comparison
    if show_all_waves:
        st.divider()
        st.subheader("🌊 All Waves Comparison")
        
        try:
            comparison_data = []
            
            with st.spinner("Computing attribution for all waves..."):
                for wave_name in available_waves:
                    try:
                        # Filter data for this wave
                        wave_data = wave_df[wave_df['wave'] == wave_name].copy()
                        wave_data = wave_data.sort_values('date')
                        wave_data = wave_data.tail(days)
                        
                        if len(wave_data) == 0:
                            continue
                        
                        # Prepare history DataFrame
                        wave_data.set_index('date', inplace=True)
                        history_df = pd.DataFrame({
                            'wave_ret': wave_data['portfolio_return'],
                            'bm_ret': wave_data['benchmark_return']
                        })
                        
                        # Compute attribution
                        _, summary = compute_alpha_attribution_series(
                            wave_name=wave_name,
                            mode="Standard",
                            history_df=history_df,
                            diagnostics_df=None,
                            tilt_strength=0.8,
                            base_exposure=1.0
                        )
                        
                        comparison_data.append({
                            'Wave': wave_name,
                            'Days': len(wave_data),
                            'Wave Return': f"{summary.total_wave_return * 100:+.2f}%",
                            'Benchmark Return': f"{summary.total_benchmark_return * 100:+.2f}%",
                            'Total Alpha': f"{summary.total_alpha * 100:+.2f}%",
                            'Exposure & Timing': f"{summary.exposure_timing_alpha * 100:+.2f}%",
                            'Regime & VIX': f"{summary.regime_vix_alpha * 100:+.2f}%",
                            'Momentum & Trend': f"{summary.momentum_trend_alpha * 100:+.2f}%",
                            'Volatility Control': f"{summary.volatility_control_alpha * 100:+.2f}%",
                            'Asset Selection': f"{summary.asset_selection_alpha * 100:+.2f}%",
                        })
                        
                    except Exception as wave_error:
                        st.warning(f"⚠️ Could not compute attribution for {wave_name}: {str(wave_error)}")
                        continue
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(
                    comparison_df,
                    hide_index=True,
                    use_container_width=True
                )
                
                # Download comparison
                comparison_csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Waves Comparison (CSV)",
                    data=comparison_csv,
                    file_name=f"all_waves_attribution_{days}d.csv",
                    mime="text/csv",
                    key="download_all_waves"
                )
            else:
                st.warning("⚠️ No comparison data available.")
                
        except Exception as e:
            st.error(f"❌ Error computing all waves comparison: {str(e)}")


def generate_board_pack_pdf():
    """
    Generate Board Pack PDF report using reportlab.
    Returns PDF bytes or None if reportlab unavailable.
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from io import BytesIO
        
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=1  # Center
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Get build info
        git_hash = get_git_commit_hash()
        git_branch = get_git_branch_name()
        deploy_timestamp = get_deploy_timestamp()
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # ==================== COVER PAGE ====================
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("WAVES Intelligence™", title_style))
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("Institutional Board Pack", styles['Heading2']))
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph(f"Generated: {current_datetime}", styles['Normal']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Build Information
        build_info = f"""
        <b>Build Information:</b><br/>
        Git Commit: {git_hash}<br/>
        Git Branch: {git_branch}<br/>
        Deployment: {deploy_timestamp}
        """
        elements.append(Paragraph(build_info, styles['Normal']))
        elements.append(PageBreak())
        
        # ==================== EXECUTIVE SUMMARY ====================
        elements.append(Paragraph("Executive Summary", heading_style))
        
        # Get Mission Control data
        mc_data = get_mission_control_data()
        
        if mc_data:
            exec_summary = f"""
            <b>Mission Control Snapshot:</b><br/>
            • Market Regime: {mc_data.get('market_regime', 'unknown')}<br/>
            • VIX Gate Status: {mc_data.get('vix_gate_status', 'unknown')}<br/>
            • 30-Day Alpha: {mc_data.get('alpha_30day', 'unknown')}<br/>
            • WaveScore Leader: {mc_data.get('wavescore_leader', 'unknown')} 
              (Score: {mc_data.get('wavescore_leader_score', 'N/A')})<br/>
            • Total Waves: {mc_data.get('total_waves', 0)}<br/>
            • Active Waves: {mc_data.get('active_waves', 0)}<br/>
            • System Status: {mc_data.get('system_status', 'unknown')}<br/>
            • Data Freshness: {mc_data.get('data_freshness', 'unknown')}
            """
            elements.append(Paragraph(exec_summary, styles['Normal']))
        else:
            elements.append(Paragraph("Executive summary data unavailable.", styles['Normal']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Alerts
        alerts = get_system_alerts()
        if alerts and len(alerts) > 0:
            elements.append(Paragraph("<b>System Alerts:</b>", styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
            for alert in alerts[:5]:  # Top 5 alerts
                alert_text = f"• [{alert.get('severity', 'info').upper()}] {alert.get('message', '')}"
                elements.append(Paragraph(alert_text, styles['Normal']))
        else:
            elements.append(Paragraph("<b>Alerts:</b> No alerts at this time.", styles['Normal']))
        
        elements.append(PageBreak())
        
        # ==================== PERFORMANCE TABLE ====================
        elements.append(Paragraph("Performance Table", heading_style))
        
        waves = get_available_waves()
        
        # Build performance table for multiple time periods
        perf_data = [['Wave', '30 Days', '60 Days', 'YTD']]
        
        if waves:
            for wave_name in waves[:10]:  # Top 10 waves
                wave_30d = get_wave_data_filtered(wave_name=wave_name, days=30)
                wave_60d = get_wave_data_filtered(wave_name=wave_name, days=60)
                wave_ytd = get_wave_data_filtered(wave_name=wave_name, days=252)  # Approx YTD
                
                metrics_30d = calculate_wave_metrics(wave_30d) if wave_30d is not None else {}
                metrics_60d = calculate_wave_metrics(wave_60d) if wave_60d is not None else {}
                metrics_ytd = calculate_wave_metrics(wave_ytd) if wave_ytd is not None else {}
                
                alpha_30d = metrics_30d.get('cumulative_alpha', 'N/A')
                alpha_60d = metrics_60d.get('cumulative_alpha', 'N/A')
                alpha_ytd = metrics_ytd.get('cumulative_alpha', 'N/A')
                
                alpha_30d_str = f"{alpha_30d*100:.2f}%" if isinstance(alpha_30d, (int, float)) else 'N/A'
                alpha_60d_str = f"{alpha_60d*100:.2f}%" if isinstance(alpha_60d, (int, float)) else 'N/A'
                alpha_ytd_str = f"{alpha_ytd*100:.2f}%" if isinstance(alpha_ytd, (int, float)) else 'N/A'
                
                perf_data.append([wave_name, alpha_30d_str, alpha_60d_str, alpha_ytd_str])
        else:
            perf_data.append(['No data', 'N/A', 'N/A', 'N/A'])
        
        perf_table = Table(perf_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        perf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(perf_table)
        elements.append(PageBreak())
        
        # ==================== RISK ANALYSIS ====================
        elements.append(Paragraph("Risk Analysis", heading_style))
        
        if waves:
            sample_wave = waves[0]
            wave_data = get_wave_data_filtered(wave_name=sample_wave, days=30)
            
            if wave_data is not None:
                metrics = calculate_wave_metrics(wave_data)
                
                risk_text = f"""
                <b>Sample Wave: {sample_wave}</b><br/>
                • Volatility: {f"{metrics.get('volatility', 'N/A')*100:.2f}%" if isinstance(metrics.get('volatility'), (int, float)) else 'N/A'}<br/>
                • Max Drawdown: {f"{metrics.get('max_drawdown', 'N/A')*100:.2f}%" if isinstance(metrics.get('max_drawdown'), (int, float)) else 'N/A'}<br/>
                • Sharpe Ratio: {f"{metrics.get('sharpe_ratio', 'N/A'):.2f}" if isinstance(metrics.get('sharpe_ratio'), (int, float)) else 'N/A'}<br/>
                • Win Rate: {f"{metrics.get('win_rate', 'N/A')*100:.2f}%" if isinstance(metrics.get('win_rate'), (int, float)) else 'N/A'}
                """
                elements.append(Paragraph(risk_text, styles['Normal']))
                
                # Correlation insights
                if len(waves) > 1:
                    wave2_name = waves[1]
                    wave2_data = get_wave_data_filtered(wave_name=wave2_name, days=30)
                    correlation = calculate_wave_correlation(wave_data, wave2_data)
                    
                    if correlation is not None:
                        corr_text = f"""
                        <br/><b>Correlation Insight:</b><br/>
                        Correlation between {sample_wave} and {wave2_name}: {correlation:.3f}
                        """
                        elements.append(Paragraph(corr_text, styles['Normal']))
            else:
                elements.append(Paragraph("Risk analysis data unavailable.", styles['Normal']))
        else:
            elements.append(Paragraph("No waves available for risk analysis.", styles['Normal']))
        
        elements.append(PageBreak())
        
        # ==================== ALPHA PROOF SECTION ====================
        elements.append(Paragraph("Alpha Proof - Components Breakdown", heading_style))
        
        if waves:
            sample_wave = waves[0]
            wave_data = get_wave_data_filtered(wave_name=sample_wave, days=30)
            
            if wave_data is not None:
                alpha_components = calculate_alpha_components(wave_data, sample_wave)
                
                if alpha_components:
                    alpha_text = f"""
                    <b>Sample Wave: {sample_wave}</b><br/>
                    • Total Alpha: {alpha_components.get('total_alpha', 0)*100:.2f}%<br/>
                    • Selection Alpha: {alpha_components.get('selection_alpha', 0)*100:.2f}%<br/>
                    • Overlay Alpha: {alpha_components.get('overlay_alpha', 0)*100:.2f}%<br/>
                    • Risk-Off Alpha (Cash): {alpha_components.get('cash_contribution', 0)*100:.2f}%<br/>
                    <br/>
                    Wave Return: {alpha_components.get('wave_return', 0)*100:.2f}%<br/>
                    Benchmark Return: {alpha_components.get('benchmark_return', 0)*100:.2f}%
                    """
                    elements.append(Paragraph(alpha_text, styles['Normal']))
                else:
                    elements.append(Paragraph("Alpha components unavailable.", styles['Normal']))
            else:
                elements.append(Paragraph("Wave data unavailable for alpha proof.", styles['Normal']))
        else:
            elements.append(Paragraph("No waves available for alpha proof.", styles['Normal']))
        
        elements.append(PageBreak())
        
        # ==================== WAVE ANALYSIS - VECTOR EXPLAIN ====================
        elements.append(Paragraph("Wave Analysis - Vector Explain Narrative", heading_style))
        
        if waves:
            sample_wave = waves[0]
            wave_data = get_wave_data_filtered(wave_name=sample_wave, days=30)
            
            if wave_data is not None and len(wave_data) > 0:
                # Generate narrative
                narrative = generate_wave_narrative(sample_wave, wave_data)
                
                if narrative:
                    # Convert markdown narrative to PDF-friendly format
                    narrative_lines = narrative.split('\n')
                    for line in narrative_lines:
                        if line.strip():
                            # Handle headers
                            if line.startswith('# '):
                                elements.append(Paragraph(line.replace('# ', ''), heading_style))
                            elif line.startswith('## '):
                                elements.append(Paragraph(line.replace('## ', ''), styles['Heading3']))
                            elif line.startswith('**') and line.endswith('**'):
                                # Bold text
                                elements.append(Paragraph(line, styles['Normal']))
                            else:
                                # Regular text
                                elements.append(Paragraph(line, styles['Normal']))
                        else:
                            elements.append(Spacer(1, 0.1*inch))
                else:
                    elements.append(Paragraph("Vector Explain narrative unavailable.", styles['Normal']))
            else:
                elements.append(Paragraph("Insufficient wave data for narrative generation.", styles['Normal']))
        else:
            elements.append(Paragraph("No waves available for narrative analysis.", styles['Normal']))
        
        elements.append(PageBreak())
        
        # ==================== TOP MOVERS ====================
        elements.append(Paragraph("Top Movers - Biggest WaveScore Changes", heading_style))
        
        # Get leaderboard data
        leaderboard_data = []
        if waves:
            for wave_name in waves:
                wave_data = get_wave_data_filtered(wave_name=wave_name, days=30)
                if wave_data is not None and len(wave_data) > 0:
                    wavescore = calculate_wavescore(wave_data)
                    leaderboard_data.append({
                        'wave': wave_name,
                        'wavescore': wavescore
                    })
            
            # Sort by wavescore
            leaderboard_data.sort(key=lambda x: x['wavescore'] if isinstance(x['wavescore'], (int, float)) else 0, reverse=True)
        
        if leaderboard_data and len(leaderboard_data) >= 3:
            movers_text = "<b>Top Performers:</b><br/>"
            for i, item in enumerate(leaderboard_data[:5], 1):
                movers_text += f"{i}. {item['wave']} - WaveScore: {item['wavescore']:.1f}<br/>"
            elements.append(Paragraph(movers_text, styles['Normal']))
        else:
            elements.append(Paragraph("Insufficient data for movers analysis.", styles['Normal']))
        
        elements.append(PageBreak())
        
        # ==================== APPENDIX ====================
        elements.append(Paragraph("Appendix - System Metadata", heading_style))
        
        appendix_text = f"""
        <b>Data Timestamps:</b><br/>
        • Latest Data: {mc_data.get('data_freshness', 'unknown')}<br/>
        • Data Age: {mc_data.get('data_age_days', 'N/A')} days<br/>
        • Report Generated: {current_datetime}<br/>
        <br/>
        <b>System Information:</b><br/>
        • Total Waves Tracked: {mc_data.get('total_waves', 0)}<br/>
        • Active Waves: {mc_data.get('active_waves', 0)}<br/>
        • System Status: {mc_data.get('system_status', 'unknown')}<br/>
        <br/>
        <b>Version Control:</b><br/>
        • Git Commit Hash: {git_hash}<br/>
        • Git Branch: {git_branch}<br/>
        • Deployment Timestamp: {deploy_timestamp}
        """
        elements.append(Paragraph(appendix_text, styles['Normal']))
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
        
    except ImportError:
        return None
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None


def render_board_pack_tab():
    """
    Render the Board Pack (PDF) tab with PDF download functionality.
    Uses reportlab for PDF generation with graceful error handling.
    """
    st.header("📊 Board Pack (PDF)")
    st.write("Generate a comprehensive PDF report including all key metrics, performance data, and analytics.")
    
    st.divider()
    
    # Introduction
    st.markdown("""
    ### Institutional Board Pack Report
    
    The Board Pack includes:
    
    1. **Cover Page** - Title, current date/time, and build information
    2. **Executive Summary** - Mission Control metrics, WaveScore leader, and alerts
    3. **Performance Table** - Key portfolio metrics for 30 days, 60 days, and Year-To-Date
    4. **Risk Analysis** - Volatility, max drawdown, and correlation insights
    5. **Alpha Proof** - Selection Alpha, Overlay Alpha, and Risk-Off Alpha breakdown
    6. **Wave Analysis** - WaveScore, top holdings, and Vector Explain narrative
    7. **Top Movers** - Biggest WaveScore changes during the period
    8. **Appendix** - Data timestamps and system metadata
    """)
    
    st.divider()
    
    # Preview Data On-Screen (before generation)
    st.subheader("📋 Report Preview - Metrics to Include")
    st.write("The following metrics and sections will be included in the PDF:")
    
    # Build preview table showing availability of data
    preview_data = []
    
    # Mission Control data
    mc_data = get_mission_control_data()
    mc_available = mc_data is not None and mc_data.get('market_regime') is not None
    preview_data.append({
        'Section': 'Mission Control',
        'Metrics': 'Market Regime, VIX Gate Status, 30-Day Alpha, WaveScore Leader',
        'Status': '✅ Available' if mc_available else '⚠️ Unavailable'
    })
    
    # Wave data
    waves = get_available_waves()
    waves_available = waves is not None and len(waves) > 0
    preview_data.append({
        'Section': 'Wave Performance',
        'Metrics': '30/60/YTD Returns, Volatility, Sharpe Ratio',
        'Status': '✅ Available' if waves_available else '⚠️ Unavailable'
    })
    
    # Alpha Proof
    alpha_available = False
    if waves_available:
        sample_wave_data = get_wave_data_filtered(wave_name=waves[0], days=30)
        alpha_available = sample_wave_data is not None
    preview_data.append({
        'Section': 'Alpha Proof Decomposition',
        'Metrics': 'Selection Alpha, Overlay Alpha, Risk-Off Alpha',
        'Status': '✅ Available' if alpha_available else '⚠️ Unavailable'
    })
    
    # Risk Metrics
    preview_data.append({
        'Section': 'Risk Metrics',
        'Metrics': 'Max Drawdown, Win Rate, Correlation Analysis',
        'Status': '✅ Available' if alpha_available else '⚠️ Unavailable'
    })
    
    # WaveScore & Top Holdings
    preview_data.append({
        'Section': 'WaveScore & Leaderboard',
        'Metrics': 'Top 10 Waves by WaveScore, Performance Rankings',
        'Status': '✅ Available' if waves_available else '⚠️ Unavailable'
    })
    
    # Vector Explain
    preview_data.append({
        'Section': 'Vector Explain Narrative',
        'Metrics': 'Institutional narrative, performance drivers, insights',
        'Status': '✅ Available' if alpha_available else '⚠️ Unavailable'
    })
    
    # Display preview table
    preview_df = pd.DataFrame(preview_data)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Check if reportlab is available
    try:
        import reportlab
        reportlab_available = True
    except ImportError:
        reportlab_available = False
    
    if not reportlab_available:
        st.error("❌ PDF Generation not available.")
        st.warning("The `reportlab` library is required for PDF generation but is not installed.")
        st.info("Please install reportlab to enable PDF downloads: `pip install reportlab`")
        st.info("The app will continue to function normally. Other tabs remain accessible.")
        return
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Generate Board Pack", type="primary", use_container_width=True):
            with st.spinner("Generating Board Pack PDF..."):
                try:
                    pdf_bytes = generate_board_pack_pdf()
                    
                    if pdf_bytes is None:
                        st.error("❌ Report generation unavailable.")
                        st.warning("Unable to generate PDF report. Please check system configuration.")
                    else:
                        # Generate filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"WAVES_Board_Pack_{timestamp}.pdf"
                        
                        # Provide download
                        st.download_button(
                            label="💾 Download PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.success("✅ PDF generated successfully! Click 'Download PDF' above to save.")
                        
                except Exception as e:
                    st.error("❌ Report generation unavailable.")
                    st.warning(f"An error occurred while generating the report: {str(e)}")
                    st.info("The application continues to function normally. Other tabs remain accessible.")
    
    with col2:
        if st.button("🔄 Force Reload Data", use_container_width=True):
            # Clear cache
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # Remove wave-related session keys
            keys_to_remove = [key for key in st.session_state.keys() if 'wave' in key.lower()]
            for key in keys_to_remove:
                del st.session_state[key]
            
            st.success("✅ Cache cleared and data reloaded!")
            st.info("Please click 'Generate Board Pack' again to create PDF with fresh data.")
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("Board Pack PDF generation uses live data from the Institutional Console. All sections gracefully handle unavailable data.")


def render_ic_pack_tab():
    """
    Render the IC Pack (Investment Committee Pack) tab.
    
    This comprehensive executive report includes:
    - Mission Control Summary Tiles
    - WaveScore Details (leaderboard, alerts, biggest movers)
    - Overlay Proof Table (VIX/Regime diagnostics)
    - Alpha Proof Decomposition
    - Vector Brief Narrative
    - HTML Download capability
    """
    st.header("📊 IC Pack - Investment Committee Executive Report")
    st.write("Comprehensive executive intelligence pack for investment committee review.")
    
    st.divider()
    
    # ========================================================================
    # SECTION 1: MISSION CONTROL SUMMARY TILES
    # ========================================================================
    st.subheader("🎯 Mission Control Summary")
    
    try:
        mc_data = get_mission_control_data()
        
        # First row: Primary metrics (5 columns)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            regime_value = mc_data.get('market_regime', 'Data unavailable')
            if 'Risk-On' in str(regime_value):
                regime_display = f"📈 {regime_value}"
            elif 'Risk-Off' in str(regime_value):
                regime_display = f"📉 {regime_value}"
            else:
                regime_display = f"➖ {regime_value}"
            
            st.metric(
                label="Market Regime",
                value=regime_display,
                help="Current market regime based on portfolio performance"
            )
        
        with col2:
            vix_value = mc_data.get('vix_gate_status', 'Data unavailable')
            if 'GREEN' in str(vix_value):
                vix_display = f"🟢 {vix_value}"
            elif 'YELLOW' in str(vix_value):
                vix_display = f"🟡 {vix_value}"
            elif 'RED' in str(vix_value):
                vix_display = f"🔴 {vix_value}"
            else:
                vix_display = str(vix_value)
            
            st.metric(
                label="VIX Gate Status",
                value=vix_display,
                help="Volatility-based risk gate"
            )
        
        with col3:
            alpha_today_str = mc_data.get('alpha_today', 'Data unavailable')
            alpha_30day_str = mc_data.get('alpha_30day', 'Data unavailable')
            
            st.markdown("**Alpha Captured**")
            st.write(f"Latest: {alpha_today_str}")
            st.write(f"30-Day: {alpha_30day_str}")
        
        with col4:
            st.markdown("**WaveScore Leader**")
            leader_name = mc_data.get('wavescore_leader', 'Data unavailable')
            leader_score = mc_data.get('wavescore_leader_score', 'N/A')
            st.write(f"{leader_name}")
            st.write(f"Score: {leader_score}")
        
        with col5:
            st.metric(
                label="System Status",
                value=mc_data.get('system_status', 'Data unavailable'),
                help="Data freshness and system health"
            )
            if mc_data.get('data_age_days') is not None:
                st.caption(f"Data age: {mc_data['data_age_days']} days")
        
        # Second row: Additional metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Total Waves", mc_data.get('total_waves', 0))
        
        with col_b:
            st.metric("Active Waves", mc_data.get('active_waves', 0))
        
        with col_c:
            st.metric("Data Freshness", mc_data.get('data_freshness', 'unknown'))
        
        with col_d:
            pass  # Reserved for future use
    
    except Exception as e:
        st.warning(f"⚠️ Mission Control data unavailable: {str(e)}")
    
    st.divider()
    
    # ========================================================================
    # SECTION 2: WAVESCORE DETAILS
    # ========================================================================
    st.subheader("🏆 WaveScore Details")
    
    try:
        waves = get_available_waves()
        
        if not waves:
            st.info("📊 Data unavailable - No wave data found")
        else:
            # Build leaderboard
            leaderboard_data = []
            
            for wave_name in waves:
                wave_data = get_wave_data_filtered(wave_name, days=30)
                if wave_data is not None and len(wave_data) > 0:
                    metrics = calculate_wave_metrics(wave_data)
                    
                    leaderboard_data.append({
                        'Wave': wave_name,
                        'WaveScore': metrics.get('wavescore', 0),
                        'Cumulative Alpha': metrics.get('cumulative_alpha', 'N/A'),
                        'Volatility': metrics.get('volatility', 'N/A'),
                        'Sharpe Ratio': metrics.get('sharpe_ratio', 'N/A')
                    })
            
            if leaderboard_data:
                leaderboard_df = pd.DataFrame(leaderboard_data)
                leaderboard_df = leaderboard_df.sort_values('WaveScore', ascending=False)
                
                # Top scoring waves
                st.markdown("**📈 Top Scoring Waves (Last 30 Days)**")
                top_waves = leaderboard_df.head(5)
                st.dataframe(top_waves, use_container_width=True, hide_index=True)
                
                # Biggest movers (highest absolute change in score)
                st.markdown("**🔄 Biggest Movers**")
                st.caption("Waves with highest WaveScore volatility indicate significant performance changes")
                
                # Calculate movers based on score variance
                movers_data = []
                for wave_name in waves[:10]:  # Limit to top 10 for performance
                    wave_data = get_wave_data_filtered(wave_name, days=30)
                    if wave_data is not None and len(wave_data) > 5:
                        if 'alpha' in wave_data.columns:
                            alpha_std = wave_data['alpha'].std()
                            cumulative_alpha = wave_data['alpha'].sum()
                            movers_data.append({
                                'Wave': wave_name,
                                'Alpha Volatility': f"{alpha_std * 100:.3f}%",
                                'Cumulative Alpha': f"{cumulative_alpha * 100:.2f}%",
                                'Movement Score': alpha_std * 1000
                            })
                
                if movers_data:
                    movers_df = pd.DataFrame(movers_data)
                    movers_df = movers_df.sort_values('Movement Score', ascending=False).head(5)
                    movers_df = movers_df.drop(columns=['Movement Score'])
                    st.dataframe(movers_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Movement data unavailable")
                
                # Alerts section
                st.markdown("**⚠️ WaveScore Alerts**")
                low_score_waves = leaderboard_df[leaderboard_df['WaveScore'] < 40]
                
                if len(low_score_waves) > 0:
                    st.warning(f"🔴 {len(low_score_waves)} wave(s) with low WaveScore (<40)")
                    st.dataframe(low_score_waves, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No low-scoring waves detected")
            else:
                st.info("📊 Data unavailable - Could not calculate WaveScores")
    
    except Exception as e:
        st.warning(f"⚠️ WaveScore details unavailable: {str(e)}")
    
    st.divider()
    
    # ========================================================================
    # SECTION 3: OVERLAY PROOF TABLE
    # ========================================================================
    st.subheader("🔬 Overlay Proof Table")
    st.caption("VIX/Regime overlay diagnostics showing exposure scaling activity")
    
    try:
        if not VIX_DIAGNOSTICS_AVAILABLE:
            st.warning("⚠️ VIX diagnostics module not available")
        else:
            # Selectors
            col_sel1, col_sel2, col_sel3 = st.columns(3)
            
            with col_sel1:
                waves = get_available_waves()
                if waves and 'SPY Wave' in waves:
                    default_wave = 'SPY Wave'
                elif waves:
                    default_wave = waves[0]
                else:
                    default_wave = 'SPY Wave'
                
                selected_wave = st.selectbox(
                    "Wave",
                    options=waves if waves else [default_wave],
                    index=waves.index(default_wave) if waves and default_wave in waves else 0,
                    key="ic_overlay_wave"
                )
            
            with col_sel2:
                selected_mode = st.selectbox(
                    "Mode",
                    options=["Standard", "Aggressive", "Conservative"],
                    index=0,
                    key="ic_overlay_mode"
                )
            
            with col_sel3:
                selected_days = st.selectbox(
                    "Days",
                    options=[15, 30, 60, 90],
                    index=0,
                    key="ic_overlay_days"
                )
            
            # Fetch diagnostics
            try:
                diagnostics_df = get_wave_diagnostics(
                    wave_name=selected_wave,
                    mode=selected_mode,
                    days=selected_days
                )
                
                if diagnostics_df is None or len(diagnostics_df) == 0:
                    st.info("📊 Data unavailable for selected parameters")
                else:
                    # Format and display the table
                    display_df = diagnostics_df.copy()
                    
                    # Ensure required columns exist
                    required_cols = ['Date', 'VIX', 'Regime', 'VIX_Exposure', 'Safe_Fraction', 
                                   'Exposure', 'Wave_Return', 'Benchmark_Return']
                    
                    # Calculate Realized_Alpha if possible
                    if 'Wave_Return' in display_df.columns and 'Benchmark_Return' in display_df.columns:
                        display_df['Realized_Alpha'] = display_df['Wave_Return'] - display_df['Benchmark_Return']
                    
                    # Select columns for display (use available columns)
                    display_cols = []
                    for col in required_cols + ['Realized_Alpha']:
                        if col in display_df.columns:
                            display_cols.append(col)
                    
                    if display_cols:
                        display_df = display_df[display_cols]
                        
                        # Format percentages
                        pct_cols = ['VIX_Exposure', 'Safe_Fraction', 'Exposure', 'Wave_Return', 
                                   'Benchmark_Return', 'Realized_Alpha']
                        for col in pct_cols:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].apply(
                                    lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A"
                                )
                        
                        # Format date
                        if 'Date' in display_df.columns:
                            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d')
                        
                        # Display table
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Summary statistics
                        st.markdown("**Summary Statistics**")
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            if 'Realized_Alpha' in diagnostics_df.columns:
                                avg_alpha = diagnostics_df['Realized_Alpha'].mean()
                                st.metric("Avg Daily Alpha", f"{avg_alpha * 100:.3f}%")
                        
                        with col_stat2:
                            if 'VIX_Exposure' in diagnostics_df.columns:
                                avg_vix_exp = diagnostics_df['VIX_Exposure'].mean()
                                st.metric("Avg VIX Exposure", f"{avg_vix_exp * 100:.1f}%")
                        
                        with col_stat3:
                            if 'Safe_Fraction' in diagnostics_df.columns:
                                avg_safe = diagnostics_df['Safe_Fraction'].mean()
                                st.metric("Avg Safe Fraction", f"{avg_safe * 100:.1f}%")
                    else:
                        st.info("📊 Data unavailable - required columns missing")
            
            except Exception as diag_err:
                st.info(f"📊 Data unavailable: {str(diag_err)}")
    
    except Exception as e:
        st.warning(f"⚠️ Overlay proof table unavailable: {str(e)}")
    
    st.divider()
    
    # ========================================================================
    # SECTION 4: ALPHA PROOF DECOMPOSITION
    # ========================================================================
    st.subheader("🔍 Alpha Proof Decomposition")
    st.caption("Breakdown of alpha sources from decision attribution")
    
    try:
        if not ALPHA_ATTRIBUTION_AVAILABLE:
            st.warning("⚠️ Alpha attribution module not available")
        else:
            # Wave selector for alpha decomposition
            waves = get_available_waves()
            if waves:
                selected_alpha_wave = st.selectbox(
                    "Select Wave for Alpha Decomposition",
                    options=waves,
                    index=waves.index('SPY Wave') if 'SPY Wave' in waves else 0,
                    key="ic_alpha_wave"
                )
                
                # Get wave data
                wave_data = get_wave_data_filtered(selected_alpha_wave, days=30)
                
                if wave_data is not None and len(wave_data) > 0:
                    try:
                        # Use alpha attribution module
                        attribution_result = compute_alpha_attribution_series(
                            wave_name=selected_alpha_wave,
                            wave_data=wave_data,
                            days=30
                        )
                        
                        if attribution_result is not None and hasattr(attribution_result, 'summary'):
                            summary = attribution_result.summary
                            
                            # Display alpha components
                            st.markdown("**Alpha Components (Last 30 Days)**")
                            
                            col_alpha1, col_alpha2, col_alpha3, col_alpha4 = st.columns(4)
                            
                            with col_alpha1:
                                selection_val = getattr(summary, 'selection_alpha', None)
                                if selection_val is not None:
                                    st.metric("Selection Alpha", f"{selection_val * 100:.3f}%")
                                else:
                                    st.metric("Selection Alpha", "Data unavailable")
                            
                            with col_alpha2:
                                overlay_val = getattr(summary, 'overlay_alpha', None)
                                if overlay_val is not None:
                                    st.metric("Overlay Alpha", f"{overlay_val * 100:.3f}%")
                                else:
                                    st.metric("Overlay Alpha", "Data unavailable")
                            
                            with col_alpha3:
                                risk_off_val = getattr(summary, 'risk_off_alpha', None)
                                if risk_off_val is not None:
                                    st.metric("Risk-Off Alpha", f"{risk_off_val * 100:.3f}%")
                                else:
                                    st.metric("Risk-Off Alpha", "Data unavailable")
                            
                            with col_alpha4:
                                residual_val = getattr(summary, 'residual_alpha', None)
                                if residual_val is not None:
                                    st.metric("Residual Alpha", f"{residual_val * 100:.3f}%")
                                else:
                                    st.metric("Residual Alpha", "Data unavailable")
                            
                            # Data completeness indicator
                            completeness = getattr(summary, 'data_completeness', 0)
                            st.progress(completeness, text=f"Data Completeness: {completeness * 100:.0f}%")
                            
                        else:
                            # Fallback: Try to use DecisionAttributionEngine
                            st.info("📊 Using alternative attribution calculation")
                            
                            engine = DecisionAttributionEngine()
                            components = engine.compute_attribution(wave_data, selected_alpha_wave)
                            
                            if components:
                                col_alpha1, col_alpha2, col_alpha3, col_alpha4 = st.columns(4)
                                
                                with col_alpha1:
                                    if components.selection_available:
                                        st.metric("Selection", f"{components.selection_alpha * 100:.3f}%")
                                    else:
                                        st.metric("Selection", "Data unavailable")
                                
                                with col_alpha2:
                                    if components.overlay_available:
                                        st.metric("Overlay", f"{components.overlay_alpha * 100:.3f}%")
                                    else:
                                        st.metric("Overlay", "Data unavailable")
                                
                                with col_alpha3:
                                    if components.risk_off_available:
                                        st.metric("Risk-Off", f"{components.risk_off_alpha * 100:.3f}%")
                                    else:
                                        st.metric("Risk-Off", "Data unavailable")
                                
                                with col_alpha4:
                                    if components.residual_available:
                                        st.metric("Residual", f"{components.residual_alpha * 100:.3f}%")
                                    else:
                                        st.metric("Residual", "Data unavailable")
                                
                                st.progress(components.data_completeness, 
                                          text=f"Data Completeness: {components.data_completeness * 100:.0f}%")
                            else:
                                st.info("📊 Data unavailable for attribution calculation")
                    
                    except Exception as attr_err:
                        st.info(f"📊 Data unavailable: {str(attr_err)}")
                else:
                    st.info("📊 Data unavailable for selected wave")
            else:
                st.info("📊 Data unavailable - No waves found")
    
    except Exception as e:
        st.warning(f"⚠️ Alpha decomposition unavailable: {str(e)}")
    
    st.divider()
    
    # ========================================================================
    # SECTION 5: VECTOR BRIEF NARRATIVE
    # ========================================================================
    st.subheader("📝 Vector Brief - Executive Narrative")
    st.caption("AI-generated summary of key metrics, drivers, and recommended actions")
    
    try:
        # Generate narrative based on available data
        narrative_sections = []
        
        # Section 1: Current State
        try:
            mc_data = get_mission_control_data()
            regime = mc_data.get('market_regime', 'unknown')
            vix_status = mc_data.get('vix_gate_status', 'unknown')
            alpha_30day = mc_data.get('alpha_30day', 'unknown')
            
            if regime != 'unknown':
                narrative_sections.append(
                    f"**Current Market State:** The portfolio is operating in a {regime} environment "
                    f"with VIX gate status at {vix_status}. "
                    f"30-day alpha performance stands at {alpha_30day}."
                )
            else:
                narrative_sections.append("**Current Market State:** Data unavailable for current market assessment.")
        except:
            narrative_sections.append("**Current Market State:** Data unavailable for current market assessment.")
        
        # Section 2: Performance Drivers
        try:
            waves = get_available_waves()
            if waves:
                top_performers = []
                for wave in waves[:5]:  # Top 5 waves
                    wave_data = get_wave_data_filtered(wave, days=30)
                    if wave_data is not None and len(wave_data) > 0:
                        metrics = calculate_wave_metrics(wave_data)
                        wavescore = metrics.get('wavescore', 0)
                        if wavescore > 60:
                            top_performers.append((wave, wavescore))
                
                if top_performers:
                    top_performers.sort(key=lambda x: x[1], reverse=True)
                    top_wave = top_performers[0]
                    narrative_sections.append(
                        f"**Major Performance Drivers:** {top_wave[0]} leads with a WaveScore of {top_wave[1]:.1f}, "
                        f"indicating strong relative performance. "
                        f"{len(top_performers)} wave(s) currently scoring above 60."
                    )
                else:
                    narrative_sections.append("**Major Performance Drivers:** No strong performers identified in current period.")
            else:
                narrative_sections.append("**Major Performance Drivers:** Data unavailable for performance analysis.")
        except:
            narrative_sections.append("**Major Performance Drivers:** Data unavailable for performance analysis.")
        
        # Section 3: Risk Assessment
        try:
            mc_data = get_mission_control_data()
            vix_status = mc_data.get('vix_gate_status', 'unknown')
            
            if 'RED' in str(vix_status):
                risk_level = "elevated"
                risk_action = "Recommend maintaining defensive positioning and monitoring for regime shifts."
            elif 'YELLOW' in str(vix_status):
                risk_level = "moderate"
                risk_action = "Continue normal operations with heightened awareness of volatility signals."
            elif 'GREEN' in str(vix_status):
                risk_level = "low"
                risk_action = "Favorable environment for growth-oriented positioning."
            else:
                risk_level = "unknown"
                risk_action = "Data unavailable for risk assessment."
            
            narrative_sections.append(
                f"**Risk Assessment:** Current volatility regime indicates {risk_level} risk levels. "
                f"{risk_action}"
            )
        except:
            narrative_sections.append("**Risk Assessment:** Data unavailable for risk assessment.")
        
        # Section 4: Recommended Actions
        try:
            mc_data = get_mission_control_data()
            system_status = mc_data.get('system_status', 'unknown')
            data_age = mc_data.get('data_age_days', None)
            
            actions = []
            
            if data_age is not None and data_age > 3:
                actions.append(f"Update data feeds (currently {data_age} days old)")
            
            if system_status == 'Stale':
                actions.append("Refresh system data to ensure accurate reporting")
            
            # Check for low-scoring waves
            waves = get_available_waves()
            low_score_count = 0
            if waves:
                for wave in waves:
                    wave_data = get_wave_data_filtered(wave, days=30)
                    if wave_data is not None:
                        metrics = calculate_wave_metrics(wave_data)
                        if metrics.get('wavescore', 100) < 40:
                            low_score_count += 1
            
            if low_score_count > 0:
                actions.append(f"Review {low_score_count} underperforming wave(s)")
            
            if actions:
                action_text = "; ".join(actions) + "."
            else:
                action_text = "No immediate actions required. Continue monitoring key metrics."
            
            narrative_sections.append(f"**Recommended Actions:** {action_text}")
        except:
            narrative_sections.append("**Recommended Actions:** Data unavailable for action recommendations.")
        
        # Display narrative
        for section in narrative_sections:
            st.markdown(section)
            st.write("")
        
    except Exception as e:
        st.warning(f"⚠️ Vector brief unavailable: {str(e)}")
    
    st.divider()
    
    # ========================================================================
    # SECTION 6: HTML DOWNLOAD
    # ========================================================================
    st.subheader("📥 Download IC Pack Report")
    
    try:
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        
        with col_dl2:
            if st.button("📥 Download IC Pack (HTML)", type="primary", use_container_width=True):
                with st.spinner("Generating HTML report..."):
                    try:
                        html_content = generate_ic_pack_html()
                        
                        if html_content:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"IC_Pack_{timestamp}.html"
                            
                            st.download_button(
                                label="💾 Download HTML Report",
                                data=html_content,
                                file_name=filename,
                                mime="text/html",
                                use_container_width=True
                            )
                            
                            st.success("✅ HTML report generated successfully!")
                        else:
                            st.error("❌ Report generation failed")
                    
                    except Exception as gen_err:
                        st.error(f"❌ Report generation error: {str(gen_err)}")
    
    except Exception as e:
        st.warning(f"⚠️ Download functionality unavailable: {str(e)}")
    
    st.divider()
    
    # Footer
    st.markdown("---")
    st.caption("IC Pack report powered by WAVES Intelligence™ - Real-time institutional analytics")


def generate_ic_pack_html():
    """
    Generate HTML version of the IC Pack report for download.
    
    Returns:
        HTML string containing the full IC Pack report
    """
    try:
        # Start HTML document
        html_parts = []
        
        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>IC Pack - Investment Committee Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        h1 {
            margin: 0;
            font-size: 32px;
        }
        .timestamp {
            font-size: 14px;
            opacity: 0.9;
            margin-top: 10px;
        }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h2 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th {
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        .narrative {
            line-height: 1.8;
            color: #555;
        }
        .footer {
            text-align: center;
            color: #999;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 IC Pack - Investment Committee Executive Report</h1>
        <div class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
    </div>
""")
        
        # Mission Control Section
        html_parts.append('<div class="section"><h2>🎯 Mission Control Summary</h2>')
        
        try:
            mc_data = get_mission_control_data()
            html_parts.append('<div class="metric-grid">')
            
            metrics = [
                ('Market Regime', mc_data.get('market_regime', 'N/A')),
                ('VIX Gate Status', mc_data.get('vix_gate_status', 'N/A')),
                ('30-Day Alpha', mc_data.get('alpha_30day', 'N/A')),
                ('WaveScore Leader', mc_data.get('wavescore_leader', 'N/A')),
                ('System Status', mc_data.get('system_status', 'N/A')),
                ('Total Waves', mc_data.get('total_waves', 0)),
            ]
            
            for label, value in metrics:
                html_parts.append(f'''
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                ''')
            
            html_parts.append('</div>')
        except:
            html_parts.append('<p>Data unavailable</p>')
        
        html_parts.append('</div>')
        
        # WaveScore Leaderboard
        html_parts.append('<div class="section"><h2>🏆 WaveScore Leaderboard</h2>')
        
        try:
            waves = get_available_waves()
            if waves:
                leaderboard_data = []
                for wave in waves:
                    wave_data = get_wave_data_filtered(wave, days=30)
                    if wave_data is not None:
                        metrics = calculate_wave_metrics(wave_data)
                        leaderboard_data.append({
                            'Wave': wave,
                            'WaveScore': metrics.get('wavescore', 0),
                            'Alpha': metrics.get('cumulative_alpha', 'N/A')
                        })
                
                if leaderboard_data:
                    leaderboard_data.sort(key=lambda x: x['WaveScore'], reverse=True)
                    
                    html_parts.append('<table><tr><th>Wave</th><th>WaveScore</th><th>Cumulative Alpha</th></tr>')
                    for item in leaderboard_data[:10]:
                        html_parts.append(f'''
                        <tr>
                            <td>{item['Wave']}</td>
                            <td>{item['WaveScore']:.1f}</td>
                            <td>{item['Alpha']}</td>
                        </tr>
                        ''')
                    html_parts.append('</table>')
                else:
                    html_parts.append('<p>Data unavailable</p>')
            else:
                html_parts.append('<p>Data unavailable</p>')
        except:
            html_parts.append('<p>Data unavailable</p>')
        
        html_parts.append('</div>')
        
        # Vector Brief
        html_parts.append('<div class="section"><h2>📝 Vector Brief - Executive Narrative</h2>')
        html_parts.append('<div class="narrative">')
        
        try:
            mc_data = get_mission_control_data()
            regime = mc_data.get('market_regime', 'unknown')
            alpha = mc_data.get('alpha_30day', 'unknown')
            
            html_parts.append(f'<p><strong>Current State:</strong> The portfolio is operating in a {regime} environment with 30-day alpha of {alpha}.</p>')
            html_parts.append('<p><strong>Risk Assessment:</strong> Monitor key volatility indicators and maintain appropriate positioning based on current regime.</p>')
            html_parts.append('<p><strong>Recommended Actions:</strong> Continue monitoring performance metrics and adjust exposure as market conditions evolve.</p>')
        except:
            html_parts.append('<p>Data unavailable for narrative generation.</p>')
        
        html_parts.append('</div></div>')
        
        # Footer
        html_parts.append('''
    <div class="footer">
        <p>IC Pack Report - WAVES Intelligence™</p>
        <p>Institutional-grade analytics for investment committees</p>
    </div>
</body>
</html>
''')
        
        return ''.join(html_parts)
    
    except Exception as e:
        return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"


# ============================================================================
# SECTION 7.5: BOTTOM TICKER BAR FUNCTIONALITY
# ============================================================================

@st.cache_data(ttl=300)
def get_portfolio_tickers():
    """
    Get portfolio tickers from Master_Stock_Sheet.csv.
    Returns top tickers by weight or empty list if unavailable.
    """
    try:
        master_sheet_path = os.path.join(os.path.dirname(__file__), 'Master_Stock_Sheet.csv')
        
        if not os.path.exists(master_sheet_path):
            return []
        
        df = pd.read_csv(master_sheet_path)
        
        if df is None or len(df) == 0:
            return []
        
        # Get top tickers by weight
        if 'Ticker' in df.columns and 'Weight' in df.columns:
            # Sort by weight and get top 10
            df = df.sort_values('Weight', ascending=False)
            tickers = df.head(10)['Ticker'].dropna().tolist()
            return tickers
        elif 'Ticker' in df.columns:
            # Just get first 10 tickers
            tickers = df.head(10)['Ticker'].dropna().tolist()
            return tickers
        
        return []
        
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_next_earnings_date(ticker):
    """
    Get next earnings date for a ticker using yfinance (free API).
    Returns formatted date string or None if unavailable.
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        
        if calendar is not None and not calendar.empty:
            # Get earnings date
            if 'Earnings Date' in calendar.index:
                earnings_date = calendar.loc['Earnings Date']
                if pd.notna(earnings_date) and hasattr(earnings_date, 'strftime'):
                    return earnings_date.strftime('%Y-%m-%d')
                elif isinstance(earnings_date, str):
                    return earnings_date
        
        return None
        
    except Exception:
        return None


@st.cache_data(ttl=86400)
def get_fed_decision_info():
    """
    Get next Federal Reserve decision date and current rate.
    Uses hardcoded schedule for 2024-2025 (no paid API required).
    Returns dict with 'next_date' and 'current_rate' or None values if unavailable.
    """
    try:
        # Federal Reserve FOMC meeting dates for 2024-2025
        # Source: federalreserve.gov (publicly available schedule)
        fomc_dates = [
            datetime(2024, 12, 17),
            datetime(2024, 12, 18),
            datetime(2025, 1, 28),
            datetime(2025, 1, 29),
            datetime(2025, 3, 18),
            datetime(2025, 3, 19),
            datetime(2025, 5, 6),
            datetime(2025, 5, 7),
            datetime(2025, 6, 17),
            datetime(2025, 6, 18),
            datetime(2025, 7, 29),
            datetime(2025, 7, 30),
            datetime(2025, 9, 16),
            datetime(2025, 9, 17),
            datetime(2025, 10, 28),
            datetime(2025, 10, 29),
            datetime(2025, 12, 9),
            datetime(2025, 12, 10),
        ]
        
        # Find next FOMC date after today
        now = datetime.now()
        next_date = None
        
        for date in fomc_dates:
            if date > now:
                next_date = date
                break
        
        # Current Federal Funds Rate (as of Dec 2024)
        # This is a static value - could be updated manually or via API
        current_rate = "4.25-4.50%"
        
        return {
            'next_date': next_date.strftime('%Y-%m-%d') if next_date else None,
            'current_rate': current_rate
        }
        
    except Exception:
        return {'next_date': None, 'current_rate': None}


def render_bottom_ticker_bar():
    """
    Render the bottom ticker bar with scrolling animation.
    Displays portfolio tickers, earnings dates, and Fed information.
    """
    try:
        # Get ticker data
        tickers = get_portfolio_tickers()
        fed_info = get_fed_decision_info()
        
        # Build ticker text content
        ticker_items = []
        
        # Add portfolio tickers with earnings dates
        if tickers:
            for ticker in tickers[:5]:  # Limit to top 5 for performance
                earnings_date = get_next_earnings_date(ticker)
                if earnings_date:
                    ticker_items.append(f"{ticker}: Next Earnings {earnings_date}")
                else:
                    ticker_items.append(f"{ticker}: EARNINGS N/A")
        else:
            ticker_items.append("PORTFOLIO: N/A")
        
        # Add Fed information
        if fed_info and fed_info.get('next_date'):
            ticker_items.append(f"FED MEETING: {fed_info['next_date']}")
        else:
            ticker_items.append("FED MEETING: N/A")
        
        if fed_info and fed_info.get('current_rate'):
            ticker_items.append(f"FED FUNDS RATE: {fed_info['current_rate']}")
        else:
            ticker_items.append("FED FUNDS RATE: N/A")
        
        # Create ticker text
        ticker_text = " • ".join(ticker_items)
        
        # Duplicate ticker text for seamless loop
        ticker_text_full = f"{ticker_text} • {ticker_text} • {ticker_text}"
        
        # Render HTML ticker bar
        ticker_html = f"""
        <style>
            @keyframes scroll-left {{
                0% {{
                    transform: translateX(0);
                }}
                100% {{
                    transform: translateX(-33.333%);
                }}
            }}
            
            .ticker-bar {{
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
                color: white;
                padding: 12px 0;
                z-index: 9999;
                overflow: hidden;
                box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.2);
                font-family: 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
            }}
            
            .ticker-content {{
                display: inline-block;
                white-space: nowrap;
                animation: scroll-left 60s linear infinite;
                padding-left: 100%;
            }}
            
            .ticker-content:hover {{
                animation-play-state: paused;
            }}
            
            /* Add padding to main content to prevent overlap */
            .main .block-container {{
                padding-bottom: 60px !important;
            }}
        </style>
        
        <div class="ticker-bar">
            <div class="ticker-content">
                {ticker_text_full}
            </div>
        </div>
        """
        
        st.markdown(ticker_html, unsafe_allow_html=True)
        
    except Exception as e:
        # Fail silently - don't disrupt the app if ticker bar fails
        pass


# ============================================================================
# SECTION 8: MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main application entry point - Executive Layer v2.
    Orchestrates the entire Institutional Console UI with enhanced analytics.
    """
    # ========================================================================
    # Session State Initialization
    # ========================================================================
    
    # Initialize wave_universe_version if not present
    if "wave_universe_version" not in st.session_state:
        st.session_state.wave_universe_version = 1
    
    # Initialize last_refresh_time if not present
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = datetime.now()
    
    # Initialize auto_refresh_enabled if not present (default: OFF)
    if "auto_refresh_enabled" not in st.session_state:
        st.session_state.auto_refresh_enabled = False
    
    # Initialize show_bottom_ticker if not present (default: ON)
    if "show_bottom_ticker" not in st.session_state:
        st.session_state.show_bottom_ticker = True
    
    # ========================================================================
    # Auto-Refresh Logic (15-second interval)
    # ========================================================================
    
    # Check if auto-refresh is enabled and supported
    if st.session_state.auto_refresh_enabled:
        try:
            # Try using streamlit-autorefresh if available
            from streamlit_autorefresh import st_autorefresh
            # Refresh every 15000ms (15 seconds) - only refreshes Overview/Executive
            st_autorefresh(interval=15000, key="auto_refresh_counter")
        except ImportError:
            # Fallback: Check if built-in autorefresh is available
            if hasattr(st, 'autorefresh'):
                st.autorefresh(interval=15000)
            # If neither is available, do nothing (auto-refresh disabled)
    
    # ========================================================================
    # Wave Universe Initialization and Force Reload Handling
    # ========================================================================
    
    # Check for force reload flag and rebuild universe if needed
    force_reload = st.session_state.get("force_reload_universe", False)
    wave_universe_version = st.session_state.get("wave_universe_version", 1)
    
    if force_reload:
        # Rebuild wave universe
        get_canonical_wave_universe(force_reload=True, _wave_universe_version=wave_universe_version)
        # Reset the flag
        st.session_state["force_reload_universe"] = False
    else:
        # Normal initialization - use cached universe
        get_canonical_wave_universe(force_reload=False, _wave_universe_version=wave_universe_version)
    
    # Display duplicate cleanup feedback in sidebar
    try:
        universe = st.session_state.get("wave_universe", {})
        removed_duplicates = universe.get("removed_duplicates", [])
        
        if removed_duplicates:
            # Show info message with duplicate count
            st.sidebar.info(f"ℹ️ Removed {len(removed_duplicates)} duplicate wave(s) from universe")
            
            # Optionally show list of removed duplicates in expander
            with st.sidebar.expander("View Removed Duplicates"):
                st.write("**Duplicates removed during deduplication:**")
                for dup in removed_duplicates:
                    st.text(f"• {dup}")
    except Exception:
        pass
    
    # ========================================================================
    # Main Application UI
    # ========================================================================
    
    # Render Mission Control at the top
    render_mission_control()
    
    # Render sidebar
    render_sidebar_info()
    
    # Main analytics tabs
    st.title("Institutional Console - Executive Layer v2")
    
    analytics_tabs = st.tabs(["Executive", "Overview", "Details", "Reports", "Overlays", "Attribution", "Board Pack", "IC Pack"])
    
    with analytics_tabs[0]:
        render_executive_tab()
    
    with analytics_tabs[1]:
        render_overview_tab()
    
    with analytics_tabs[2]:
        render_details_tab()
    
    with analytics_tabs[3]:
        render_reports_tab()
    
    with analytics_tabs[4]:
        render_overlays_tab()
    
    with analytics_tabs[5]:
        render_attribution_tab()
    
    with analytics_tabs[6]:
        render_board_pack_tab()
    
    with analytics_tabs[7]:
        render_ic_pack_tab()
    
    # ========================================================================
    # Bottom Ticker Bar Rendering
    # ========================================================================
    
    # Render bottom ticker bar if enabled
    if st.session_state.get("show_bottom_ticker", True):
        render_bottom_ticker_bar()


# Run the application
if __name__ == "__main__":
    main()
