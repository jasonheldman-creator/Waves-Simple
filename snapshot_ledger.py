"""
snapshot_ledger.py

WAVE SNAPSHOT LEDGER - Analytics Pipeline for 28/28 Waves Performance Metrics

This module provides a daily snapshot table with comprehensive Wave analytics
without depending on full ticker coverage or data-ready gating. It uses a
tiered fallback approach to ensure no Wave is excluded.

Key Features:
- Tiered data sourcing (Tier A-D) for resilient metric computation
- VIX-based exposure and cash percentage computation
- Multi-timeframe return and alpha calculations
- Partial data handling with appropriate flags
- Persistent snapshot caching for fast rendering

Directory Structure:
    data/
        live_snapshot.csv  (daily snapshot with all metrics)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Import from waves_engine
try:
    from waves_engine import (
        get_all_wave_ids,
        get_display_name_from_wave_id,
        compute_history_nav,
        get_vix_regime_diagnostics,
        WAVE_WEIGHTS,
        BENCHMARK_WEIGHTS_STATIC,
        get_auto_benchmark_holdings,
        get_wave_id_from_display_name,
        MODE_BASE_EXPOSURE,
    )
    WAVES_ENGINE_AVAILABLE = True
except ImportError:
    WAVES_ENGINE_AVAILABLE = False

# Import ticker diagnostics
try:
    from helpers.ticker_diagnostics import get_diagnostics_tracker
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False

# Import data cache
try:
    from data_cache import get_global_price_cache
    DATA_CACHE_AVAILABLE = True
except ImportError:
    DATA_CACHE_AVAILABLE = False

# Constants
SNAPSHOT_FILE = "data/live_snapshot.csv"
TRADING_DAYS_PER_YEAR = 252
MAX_SNAPSHOT_AGE_HOURS = 24  # Regenerate snapshot after 24 hours

# Timeframes for return calculation
TIMEFRAMES = {
    "1D": 1,
    "30D": 30,
    "60D": 60,
    "365D": 365,
}


def _safe_return(nav_series: pd.Series, days: int) -> float:
    """
    Safely compute return over N days from NAV series.
    
    Args:
        nav_series: NAV time series
        days: Number of days to look back
        
    Returns:
        Return as decimal, or NaN if insufficient data
    """
    if nav_series is None or len(nav_series) < 2:
        return float("nan")
    
    # Need at least days + 1 points
    if len(nav_series) < min(days + 1, 2):
        return float("nan")
    
    try:
        # Get the most recent value
        end_val = float(nav_series.iloc[-1])
        
        # Get value from 'days' ago, or earliest if not enough history
        start_idx = max(0, len(nav_series) - days - 1)
        start_val = float(nav_series.iloc[start_idx])
        
        if start_val <= 0:
            return float("nan")
        
        return (end_val / start_val) - 1.0
    except Exception:
        return float("nan")


def _compute_beta(wave_returns: pd.Series, bm_returns: pd.Series, min_points: int = 20) -> float:
    """
    Compute beta from aligned daily returns.
    
    Args:
        wave_returns: Wave daily returns
        bm_returns: Benchmark daily returns
        min_points: Minimum number of points required
        
    Returns:
        Beta value, or NaN if insufficient data
    """
    if wave_returns is None or bm_returns is None:
        return float("nan")
    
    # Align and drop NaN
    aligned = pd.DataFrame({"wave": wave_returns, "bm": bm_returns}).dropna()
    
    if len(aligned) < min_points:
        return float("nan")
    
    try:
        # Compute covariance and variance
        cov = aligned["wave"].cov(aligned["bm"])
        var = aligned["bm"].var()
        
        if var <= 0:
            return float("nan")
        
        return float(cov / var)
    except Exception:
        return float("nan")


def _compute_max_drawdown(nav_series: pd.Series) -> float:
    """
    Compute maximum drawdown from NAV series.
    
    Args:
        nav_series: NAV time series
        
    Returns:
        Maximum drawdown as negative decimal, or NaN if insufficient data
    """
    if nav_series is None or len(nav_series) < 2:
        return float("nan")
    
    try:
        running_max = nav_series.cummax()
        drawdown = (nav_series / running_max) - 1.0
        return float(drawdown.min())
    except Exception:
        return float("nan")


def _estimate_turnover(wave_id: str) -> float:
    """
    Estimate turnover from trades.csv if available.
    
    Args:
        wave_id: Wave identifier
        
    Returns:
        Estimated annual turnover, or NaN if not available
    """
    trades_path = f"data/waves/{wave_id}/trades.csv"
    
    if not os.path.exists(trades_path):
        return float("nan")
    
    try:
        trades_df = pd.read_csv(trades_path)
        
        if trades_df.empty or "date" not in trades_df.columns:
            return float("nan")
        
        # Simple estimate: sum of absolute trade values over period
        if "value" in trades_df.columns:
            total_traded = trades_df["value"].abs().sum()
            
            # Annualize based on date range
            trades_df["date"] = pd.to_datetime(trades_df["date"])
            date_range = (trades_df["date"].max() - trades_df["date"].min()).days
            
            if date_range > 0:
                return float(total_traded * 365 / date_range)
        
        return float("nan")
    except Exception:
        return float("nan")


def _get_vix_level_and_regime(price_df: Optional[pd.DataFrame] = None) -> Tuple[float, str]:
    """
    Get current VIX level and regime.
    
    Args:
        price_df: Optional price DataFrame with VIX data
        
    Returns:
        Tuple of (vix_level, vix_regime)
    """
    vix_level = float("nan")
    vix_regime = "unknown"
    
    try:
        if price_df is not None and "^VIX" in price_df.columns:
            vix_series = price_df["^VIX"].dropna()
            if not vix_series.empty:
                vix_level = float(vix_series.iloc[-1])
        
        # Determine regime from VIX level
        if not np.isnan(vix_level):
            if vix_level < 15:
                vix_regime = "low"
            elif vix_level < 20:
                vix_regime = "normal"
            elif vix_level < 30:
                vix_regime = "elevated"
            else:
                vix_regime = "high"
    except Exception:
        pass
    
    return vix_level, vix_regime


def _compute_exposure_and_cash(
    wave_name: str,
    mode: str,
    vix_level: float,
    vix_regime: str,
    price_df: Optional[pd.DataFrame] = None
) -> Tuple[float, float]:
    """
    Compute exposure and cash percentage using VIX ladder logic.
    
    This is independent of ticker availability and works even in degraded modes.
    
    Args:
        wave_name: Wave display name
        mode: Operating mode
        vix_level: Current VIX level
        vix_regime: Current VIX regime
        price_df: Optional price DataFrame
        
    Returns:
        Tuple of (exposure, cash_percent)
    """
    # Default values
    base_exposure = MODE_BASE_EXPOSURE.get(mode, 1.0)
    exposure = base_exposure
    cash_percent = 0.0
    
    try:
        # VIX-based adjustments (simplified ladder logic)
        if not np.isnan(vix_level):
            if vix_level < 15:
                # Low VIX - full exposure, minimal cash
                exposure_mult = 1.1
                cash_pct = 0.0
            elif vix_level < 20:
                # Normal VIX - standard exposure
                exposure_mult = 1.0
                cash_pct = 0.05
            elif vix_level < 25:
                # Elevated VIX - reduce exposure
                exposure_mult = 0.9
                cash_pct = 0.15
            elif vix_level < 30:
                # High VIX - further reduce exposure
                exposure_mult = 0.8
                cash_pct = 0.30
            else:
                # Very high VIX - defensive positioning
                exposure_mult = 0.7
                cash_pct = 0.50
            
            exposure = base_exposure * exposure_mult
            cash_percent = cash_pct
        
        # Mode-specific adjustments
        if mode == "Alpha-Minus-Beta":
            # More defensive
            cash_percent = min(cash_percent + 0.15, 0.75)
            exposure = min(exposure, 0.85)
        elif mode == "Private Logic":
            # More aggressive
            cash_percent = max(cash_percent - 0.05, 0.0)
            exposure = min(exposure * 1.1, 1.5)
        
    except Exception:
        # Fallback to base values
        exposure = base_exposure
        cash_percent = 0.0
    
    return exposure, cash_percent


def _build_snapshot_row_tier_a(
    wave_id: str,
    wave_name: str,
    mode: str,
    price_df: Optional[pd.DataFrame] = None
) -> Optional[Dict[str, Any]]:
    """
    Tier A: Use engine-provided Wave NAV and Benchmark NAV series.
    
    This is the preferred method when full history is available.
    
    Args:
        wave_id: Wave identifier
        wave_name: Wave display name
        mode: Operating mode
        price_df: Optional pre-fetched price DataFrame
        
    Returns:
        Snapshot row dictionary, or None if not available
    """
    try:
        # Attempt to compute full history
        hist_df = compute_history_nav(wave_name, mode=mode, days=365, price_df=price_df)
        
        if hist_df is None or hist_df.empty or len(hist_df) < 7:
            # Not enough data for Tier A
            return None
        
        # Extract NAV series
        wave_nav = hist_df["wave_nav"]
        bm_nav = hist_df["bm_nav"]
        wave_ret = hist_df.get("wave_ret", pd.Series())
        bm_ret = hist_df.get("bm_ret", pd.Series())
        
        # Get current values
        current_nav = float(wave_nav.iloc[-1])
        nav_1d_change = float(wave_nav.iloc[-1] - wave_nav.iloc[-2]) if len(wave_nav) >= 2 else float("nan")
        
        # Compute returns for all timeframes
        returns = {}
        bm_returns = {}
        alphas = {}
        
        for label, days in TIMEFRAMES.items():
            wave_return = _safe_return(wave_nav, days)
            bm_return = _safe_return(bm_nav, days)
            
            returns[f"Return_{label}"] = wave_return
            bm_returns[f"Benchmark_Return_{label}"] = bm_return
            
            # Alpha = Wave return - Benchmark return
            if not np.isnan(wave_return) and not np.isnan(bm_return):
                alphas[f"Alpha_{label}"] = wave_return - bm_return
            else:
                alphas[f"Alpha_{label}"] = float("nan")
        
        # Compute beta
        beta_real = _compute_beta(wave_ret, bm_ret, min_points=20)
        
        # Beta target (simplified - assume 1.0 for most waves)
        beta_target = 1.0
        
        # Beta drift
        if not np.isnan(beta_real):
            beta_drift = abs(beta_real - beta_target)
        else:
            beta_drift = float("nan")
        
        # Max drawdown
        max_dd = _compute_max_drawdown(wave_nav)
        
        # Turnover estimate
        turnover_est = _estimate_turnover(wave_id)
        
        # VIX and exposure
        vix_level, vix_regime = _get_vix_level_and_regime(price_df)
        exposure, cash_percent = _compute_exposure_and_cash(wave_name, mode, vix_level, vix_regime, price_df)
        
        # Determine data regime tag
        if len(hist_df) >= 365:
            data_regime_tag = "Full"
        elif len(hist_df) >= 60:
            data_regime_tag = "Partial"
        else:
            data_regime_tag = "Operational"
        
        # Coverage score (0-100)
        coverage_score = min(100, int(len(hist_df) / 365 * 100))
        
        # Flags
        flags = []
        if len(hist_df) < 365:
            flags.append("Limited History")
        if np.isnan(beta_real):
            flags.append("Beta N/A")
        if np.isnan(turnover_est):
            flags.append("Turnover N/A")
        
        flags_str = "; ".join(flags) if flags else "OK"
        
        # Build row
        row = {
            "Wave": wave_name,
            "Mode": mode,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "NAV": current_nav,
            "NAV_1D_Change": nav_1d_change,
            **returns,
            **bm_returns,
            **alphas,
            "Exposure": exposure,
            "CashPercent": cash_percent,
            "VIX_Level": vix_level,
            "VIX_Regime": vix_regime,
            "Beta_Real": beta_real,
            "Beta_Target": beta_target,
            "Beta_Drift": beta_drift,
            "Turnover_Est": turnover_est,
            "MaxDD": max_dd,
            "Flags": flags_str,
            "Data_Regime_Tag": data_regime_tag,
            "Coverage_Score": coverage_score,
        }
        
        return row
        
    except Exception as e:
        print(f"Tier A failed for {wave_name}: {e}")
        return None


def _build_snapshot_row_tier_b(
    wave_id: str,
    wave_name: str,
    mode: str,
    price_df: Optional[pd.DataFrame] = None
) -> Optional[Dict[str, Any]]:
    """
    Tier B: Compute returns from recent NAV points (7-30 days).
    
    Used when full history is unavailable but some NAV data exists.
    
    Args:
        wave_id: Wave identifier
        wave_name: Wave display name
        mode: Operating mode
        price_df: Optional pre-fetched price DataFrame
        
    Returns:
        Snapshot row dictionary, or None if not available
    """
    try:
        # Try to get limited history
        hist_df = compute_history_nav(wave_name, mode=mode, days=60, price_df=price_df)
        
        if hist_df is None or hist_df.empty or len(hist_df) < 1:
            return None
        
        # Extract NAV series
        wave_nav = hist_df["wave_nav"]
        bm_nav = hist_df["bm_nav"]
        wave_ret = hist_df.get("wave_ret", pd.Series())
        bm_ret = hist_df.get("bm_ret", pd.Series())
        
        # Get current values
        current_nav = float(wave_nav.iloc[-1])
        nav_1d_change = float(wave_nav.iloc[-1] - wave_nav.iloc[-2]) if len(wave_nav) >= 2 else float("nan")
        
        # Compute returns for available timeframes
        returns = {}
        bm_returns = {}
        alphas = {}
        
        for label, days in TIMEFRAMES.items():
            wave_return = _safe_return(wave_nav, days)
            bm_return = _safe_return(bm_nav, days)
            
            returns[f"Return_{label}"] = wave_return
            bm_returns[f"Benchmark_Return_{label}"] = bm_return
            
            if not np.isnan(wave_return) and not np.isnan(bm_return):
                alphas[f"Alpha_{label}"] = wave_return - bm_return
            else:
                alphas[f"Alpha_{label}"] = float("nan")
        
        # Beta (may not be available with limited data)
        beta_real = _compute_beta(wave_ret, bm_ret, min_points=10)  # Lower threshold
        beta_target = 1.0
        beta_drift = abs(beta_real - beta_target) if not np.isnan(beta_real) else float("nan")
        
        # Max drawdown
        max_dd = _compute_max_drawdown(wave_nav)
        
        # Turnover
        turnover_est = _estimate_turnover(wave_id)
        
        # VIX and exposure
        vix_level, vix_regime = _get_vix_level_and_regime(price_df)
        exposure, cash_percent = _compute_exposure_and_cash(wave_name, mode, vix_level, vix_regime, price_df)
        
        # Data regime
        data_regime_tag = "Operational"
        coverage_score = min(100, int(len(hist_df) / 365 * 100))
        
        # Flags
        flags = ["Limited History", "Tier B"]
        if np.isnan(beta_real):
            flags.append("Beta N/A")
        if np.isnan(turnover_est):
            flags.append("Turnover N/A")
        
        flags_str = "; ".join(flags)
        
        # Build row
        row = {
            "Wave": wave_name,
            "Mode": mode,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "NAV": current_nav,
            "NAV_1D_Change": nav_1d_change,
            **returns,
            **bm_returns,
            **alphas,
            "Exposure": exposure,
            "CashPercent": cash_percent,
            "VIX_Level": vix_level,
            "VIX_Regime": vix_regime,
            "Beta_Real": beta_real,
            "Beta_Target": beta_target,
            "Beta_Drift": beta_drift,
            "Turnover_Est": turnover_est,
            "MaxDD": max_dd,
            "Flags": flags_str,
            "Data_Regime_Tag": data_regime_tag,
            "Coverage_Score": coverage_score,
        }
        
        return row
        
    except Exception as e:
        print(f"Tier B failed for {wave_name}: {e}")
        return None


def _build_snapshot_row_tier_c(
    wave_id: str,
    wave_name: str,
    mode: str,
    price_df: Optional[pd.DataFrame] = None
) -> Optional[Dict[str, Any]]:
    """
    Tier C: Compute returns from portfolio weights and top N holdings that fetch.
    
    Used when NAV series is missing but we can reconstruct from holdings.
    
    Args:
        wave_id: Wave identifier
        wave_name: Wave display name
        mode: Operating mode
        price_df: Optional pre-fetched price DataFrame
        
    Returns:
        Snapshot row dictionary, or None if not available
    """
    try:
        # This is complex and requires holdings data
        # For now, return None and rely on Tier D
        # A full implementation would:
        # 1. Get holdings from WAVE_WEIGHTS
        # 2. Fetch prices for available tickers
        # 3. Compute weighted returns
        # 4. Renormalize weights for missing tickers
        
        return None
        
    except Exception as e:
        print(f"Tier C failed for {wave_name}: {e}")
        return None


def _build_snapshot_row_tier_d(
    wave_id: str,
    wave_name: str,
    mode: str,
    price_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Tier D: Benchmark fallback - use benchmark-only return, set wave return = benchmark.
    
    This ensures no Wave is excluded from the snapshot.
    
    Args:
        wave_id: Wave identifier
        wave_name: Wave display name
        mode: Operating mode
        price_df: Optional pre-fetched price DataFrame
        
    Returns:
        Snapshot row dictionary (always succeeds)
    """
    # VIX and exposure (can always compute)
    vix_level, vix_regime = _get_vix_level_and_regime(price_df)
    exposure, cash_percent = _compute_exposure_and_cash(wave_name, mode, vix_level, vix_regime, price_df)
    
    # All returns are NaN or 0
    returns = {f"Return_{label}": float("nan") for label in TIMEFRAMES.keys()}
    bm_returns = {f"Benchmark_Return_{label}": float("nan") for label in TIMEFRAMES.keys()}
    alphas = {f"Alpha_{label}": 0.0 for label in TIMEFRAMES.keys()}  # Alpha = 0 in fallback
    
    # Build row with fallback values
    row = {
        "Wave": wave_name,
        "Mode": mode,
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "NAV": float("nan"),
        "NAV_1D_Change": float("nan"),
        **returns,
        **bm_returns,
        **alphas,
        "Exposure": exposure,
        "CashPercent": cash_percent,
        "VIX_Level": vix_level,
        "VIX_Regime": vix_regime,
        "Beta_Real": float("nan"),
        "Beta_Target": 1.0,
        "Beta_Drift": float("nan"),
        "Turnover_Est": float("nan"),
        "MaxDD": float("nan"),
        "Flags": "Benchmark Fallback; Tier D; No Data",
        "Data_Regime_Tag": "Unavailable",
        "Coverage_Score": 0,
    }
    
    return row


def generate_snapshot(
    force_refresh: bool = False,
    max_runtime_seconds: int = 300,
    price_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Generate daily snapshot for all 28 Waves.
    
    Uses tiered fallback approach (A -> B -> C -> D) to ensure complete coverage.
    
    Args:
        force_refresh: If True, ignore cached snapshot
        max_runtime_seconds: Maximum time to spend generating snapshot
        price_df: Optional pre-fetched price DataFrame
        
    Returns:
        DataFrame with snapshot data for all waves
    """
    start_time = time.time()
    
    print("=" * 80)
    print("WAVE SNAPSHOT LEDGER - Generating Daily Snapshot")
    print("=" * 80)
    
    # Check if cached snapshot exists and is recent
    if not force_refresh and os.path.exists(SNAPSHOT_FILE):
        try:
            cached_df = pd.read_csv(SNAPSHOT_FILE)
            
            # Check if snapshot is recent enough
            if "Date" in cached_df.columns and not cached_df.empty:
                snapshot_date = pd.to_datetime(cached_df["Date"].iloc[0])
                age_hours = (datetime.now() - snapshot_date).total_seconds() / 3600
                
                if age_hours < MAX_SNAPSHOT_AGE_HOURS:
                    print(f"✓ Using cached snapshot (age: {age_hours:.1f} hours)")
                    return cached_df
                else:
                    print(f"⚠ Cached snapshot is stale (age: {age_hours:.1f} hours), regenerating...")
        except Exception as e:
            print(f"⚠ Failed to load cached snapshot: {e}")
    
    # Get all wave IDs
    if not WAVES_ENGINE_AVAILABLE:
        print("✗ Waves engine not available")
        # Return empty DataFrame with proper columns
        return pd.DataFrame(columns=[
            "Wave", "Mode", "Date", "NAV", "NAV_1D_Change",
            "Return_1D", "Return_30D", "Return_60D", "Return_365D",
            "Benchmark_Return_1D", "Benchmark_Return_30D", "Benchmark_Return_60D", "Benchmark_Return_365D",
            "Alpha_1D", "Alpha_30D", "Alpha_60D", "Alpha_365D",
            "Exposure", "CashPercent", "VIX_Level", "VIX_Regime",
            "Beta_Real", "Beta_Target", "Beta_Drift", "Turnover_Est", "MaxDD",
            "Flags", "Data_Regime_Tag", "Coverage_Score"
        ])
    
    try:
        all_wave_ids = get_all_wave_ids()
    except Exception as e:
        print(f"✗ Failed to get wave IDs: {e}")
        return pd.DataFrame()
    
    print(f"✓ Found {len(all_wave_ids)} waves in registry")
    
    # Get global price cache if available
    if price_df is None and DATA_CACHE_AVAILABLE and WAVES_ENGINE_AVAILABLE:
        try:
            cache_result = get_global_price_cache(wave_registry=WAVE_WEIGHTS, days=365, ttl_seconds=7200)
            price_df = cache_result.get("price_df")
            print(f"✓ Loaded global price cache with {len(price_df.columns) if price_df is not None else 0} tickers")
        except Exception as e:
            print(f"⚠ Failed to load global price cache: {e}")
    
    # Generate snapshot rows
    snapshot_rows = []
    tier_stats = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    for wave_id in all_wave_ids:
        # Check runtime limit
        elapsed = time.time() - start_time
        if elapsed > max_runtime_seconds:
            print(f"⚠ Max runtime ({max_runtime_seconds}s) exceeded, using fallback for remaining waves")
            # Use Tier D for remaining waves
            for remaining_wave_id in all_wave_ids[len(snapshot_rows):]:
                try:
                    remaining_wave_name = get_display_name_from_wave_id(remaining_wave_id)
                except Exception:
                    # Fallback: use wave_id as name if function fails
                    remaining_wave_name = remaining_wave_id
                row = _build_snapshot_row_tier_d(remaining_wave_id, remaining_wave_name, "Standard", price_df)
                snapshot_rows.append(row)
                tier_stats["D"] += 1
            break
        
        wave_name = get_display_name_from_wave_id(wave_id)
        mode = "Standard"  # Default mode
        
        print(f"\n[{len(snapshot_rows)+1}/{len(all_wave_ids)}] Processing {wave_name}...")
        
        # Try tiers in order: A -> B -> C -> D
        row = None
        
        # Tier A
        row = _build_snapshot_row_tier_a(wave_id, wave_name, mode, price_df)
        if row is not None:
            print(f"  ✓ Tier A successful")
            tier_stats["A"] += 1
        else:
            # Tier B
            row = _build_snapshot_row_tier_b(wave_id, wave_name, mode, price_df)
            if row is not None:
                print(f"  ✓ Tier B successful")
                tier_stats["B"] += 1
            else:
                # Tier C
                row = _build_snapshot_row_tier_c(wave_id, wave_name, mode, price_df)
                if row is not None:
                    print(f"  ✓ Tier C successful")
                    tier_stats["C"] += 1
                else:
                    # Tier D (always succeeds)
                    row = _build_snapshot_row_tier_d(wave_id, wave_name, mode, price_df)
                    print(f"  ⚠ Tier D fallback")
                    tier_stats["D"] += 1
        
        snapshot_rows.append(row)
    
    # Create DataFrame
    snapshot_df = pd.DataFrame(snapshot_rows)
    
    # Summary
    print("\n" + "=" * 80)
    print("SNAPSHOT GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total Waves: {len(snapshot_df)}")
    print(f"Tier A (Full History): {tier_stats['A']}")
    print(f"Tier B (Limited History): {tier_stats['B']}")
    print(f"Tier C (Holdings Reconstruction): {tier_stats['C']}")
    print(f"Tier D (Benchmark Fallback): {tier_stats['D']}")
    print(f"Generation Time: {time.time() - start_time:.1f}s")
    print("=" * 80)
    
    # Persist snapshot
    try:
        # Create directory if needed (handle case where dirname is empty)
        dirname = os.path.dirname(SNAPSHOT_FILE)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        snapshot_df.to_csv(SNAPSHOT_FILE, index=False)
        print(f"✓ Snapshot saved to {SNAPSHOT_FILE}")
    except Exception as e:
        print(f"✗ Failed to save snapshot: {e}")
    
    return snapshot_df


def load_snapshot(force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    Load snapshot from cache, or generate if not available.
    
    Args:
        force_refresh: If True, regenerate snapshot even if cached
        
    Returns:
        Snapshot DataFrame, or None if failed
    """
    try:
        if force_refresh:
            return generate_snapshot(force_refresh=True)
        
        # Try to load from cache
        if os.path.exists(SNAPSHOT_FILE):
            snapshot_df = pd.read_csv(SNAPSHOT_FILE)
            
            # Validate snapshot
            if "Date" in snapshot_df.columns and not snapshot_df.empty:
                snapshot_date = pd.to_datetime(snapshot_df["Date"].iloc[0])
                age_hours = (datetime.now() - snapshot_date).total_seconds() / 3600
                
                if age_hours < MAX_SNAPSHOT_AGE_HOURS:
                    return snapshot_df
        
        # Generate new snapshot
        return generate_snapshot()
        
    except Exception as e:
        print(f"Failed to load snapshot: {e}")
        return None


def get_snapshot_metadata() -> Dict[str, Any]:
    """
    Get metadata about the current snapshot.
    
    Returns:
        Dictionary with snapshot metadata
    """
    metadata = {
        "exists": False,
        "timestamp": None,
        "age_hours": None,
        "wave_count": 0,
        "is_stale": True,
    }
    
    try:
        if os.path.exists(SNAPSHOT_FILE):
            metadata["exists"] = True
            
            # Get file modification time
            mtime = os.path.getmtime(SNAPSHOT_FILE)
            metadata["timestamp"] = datetime.fromtimestamp(mtime)
            
            # Calculate age
            age_seconds = time.time() - mtime
            metadata["age_hours"] = age_seconds / 3600
            metadata["is_stale"] = metadata["age_hours"] > MAX_SNAPSHOT_AGE_HOURS
            
            # Get wave count
            snapshot_df = pd.read_csv(SNAPSHOT_FILE)
            metadata["wave_count"] = len(snapshot_df)
    
    except Exception:
        pass
    
    return metadata
