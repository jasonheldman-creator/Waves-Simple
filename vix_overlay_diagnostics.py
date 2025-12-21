# vix_overlay_diagnostics.py — VIX/Regime Overlay Diagnostic Validation
#
# Purpose: Prove VIX/regime overlay is materially active in equity Wave returns
# and provide auditable diagnostic logs for exposure scaling, safe fraction, and
# benchmark exposure assumptions.
#
# This module provides:
#   1. Per-day diagnostic logs (Date, VIX, Regime, Exposure, Safe Fraction, etc.)
#   2. Validation that exposure scaling is material (not cosmetic)
#   3. Reports showing overlay activity during high-VIX / risk-off periods

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from waves_engine import (
    get_all_waves,
    get_modes,
    simulate_history_nav,
    WAVE_WEIGHTS,
)


def get_wave_diagnostics(
    wave_name: str,
    mode: str = "Standard",
    days: int = 365,
) -> pd.DataFrame:
    """
    Get detailed per-day diagnostics for a Wave showing VIX/regime overlay activity.
    
    Returns DataFrame with columns:
        - Date: trading date
        - VIX: VIX level (or crypto vol proxy)
        - Regime: regime state (panic/downtrend/neutral/uptrend)
        - Exposure: final exposure used in return calculation (0..1+)
        - Safe_Fraction: portion allocated to safe assets (0..1)
        - Risk_Fraction: portion in risky assets = 1 - Safe_Fraction
        - Vol_Adjust: volatility targeting adjustment factor
        - VIX_Exposure: VIX-driven exposure adjustment factor
        - VIX_Gate: VIX-driven safe allocation boost
        - Regime_Gate: regime-driven safe allocation boost
        - Wave_Return: daily Wave return (decimal)
        - Benchmark_Return: daily benchmark return (decimal)
    
    Args:
        wave_name: name of the Wave
        mode: operating mode (Standard, Alpha-Minus-Beta, Private Logic)
        days: history window
        
    Returns:
        DataFrame with diagnostic columns indexed by Date
    """
    # Use simulate_history_nav which runs with shadow=True to capture diagnostics
    result = simulate_history_nav(wave_name=wave_name, mode=mode, days=days, overrides={})
    
    if result.empty:
        return pd.DataFrame(columns=[
            "Date", "VIX", "Regime", "Exposure", "Safe_Fraction", "Risk_Fraction",
            "Vol_Adjust", "VIX_Exposure", "VIX_Gate", "Regime_Gate",
            "Wave_Return", "Benchmark_Return"
        ])
    
    # Extract diagnostics from attrs
    diag_df = result.attrs.get("diagnostics")
    if diag_df is None or diag_df.empty:
        # Fallback: create basic diagnostic frame from return data
        return pd.DataFrame({
            "Date": result.index,
            "Wave_Return": result["wave_ret"].values,
            "Benchmark_Return": result["bm_ret"].values,
        }).set_index("Date")
    
    # Merge diagnostics with returns
    combined = diag_df.copy()
    combined["Wave_Return"] = result["wave_ret"].reindex(combined.index).fillna(0.0)
    combined["Benchmark_Return"] = result["bm_ret"].reindex(combined.index).fillna(0.0)
    
    # Add derived columns
    combined["Risk_Fraction"] = 1.0 - combined["safe_fraction"]
    
    # Rename columns for clarity
    combined = combined.rename(columns={
        "regime": "Regime",
        "vix": "VIX",
        "safe_fraction": "Safe_Fraction",
        "exposure": "Exposure",
        "vol_adjust": "Vol_Adjust",
        "vix_exposure": "VIX_Exposure",
        "vix_gate": "VIX_Gate",
        "regime_gate": "Regime_Gate",
    })
    
    # Reorder columns
    column_order = [
        "VIX", "Regime", "Exposure", "Safe_Fraction", "Risk_Fraction",
        "Vol_Adjust", "VIX_Exposure", "VIX_Gate", "Regime_Gate",
        "Wave_Return", "Benchmark_Return"
    ]
    
    # Keep only columns that exist
    available_cols = [c for c in column_order if c in combined.columns]
    combined = combined[available_cols]
    
    return combined


def validate_exposure_scaling(
    diagnostics: pd.DataFrame,
    min_material_change: float = 0.05,
) -> Dict[str, Any]:
    """
    Validate that exposure scaling is material (not cosmetic).
    
    Args:
        diagnostics: DataFrame from get_wave_diagnostics()
        min_material_change: minimum exposure change to consider material (default 5%)
        
    Returns:
        Dict with validation results:
            - is_material: bool, True if exposure changes are material
            - max_exposure: float, maximum exposure observed
            - min_exposure: float, minimum exposure observed
            - exposure_range: float, max - min
            - high_vix_avg_exposure: float, average exposure during VIX >= 25
            - low_vix_avg_exposure: float, average exposure during VIX < 20
            - risk_off_avg_exposure: float, average exposure during risk-off regimes
            - risk_on_avg_exposure: float, average exposure during risk-on regimes
            - material_changes_count: int, number of days with material exposure change
            - total_days: int, total days in sample
    """
    if diagnostics.empty or "Exposure" not in diagnostics.columns:
        return {
            "is_material": False,
            "error": "No exposure data available",
        }
    
    exp = diagnostics["Exposure"].dropna()
    if len(exp) == 0:
        return {
            "is_material": False,
            "error": "No valid exposure values",
        }
    
    max_exp = float(exp.max())
    min_exp = float(exp.min())
    exp_range = max_exp - min_exp
    
    # Check for material changes day-over-day
    exp_changes = exp.diff().abs()
    material_changes = (exp_changes >= min_material_change).sum()
    
    # High VIX vs low VIX exposure
    high_vix_avg = None
    low_vix_avg = None
    if "VIX" in diagnostics.columns:
        vix = diagnostics["VIX"].dropna()
        if len(vix) > 0:
            high_vix_mask = vix >= 25
            low_vix_mask = vix < 20
            
            if high_vix_mask.sum() > 0:
                high_vix_avg = float(exp[high_vix_mask].mean())
            if low_vix_mask.sum() > 0:
                low_vix_avg = float(exp[low_vix_mask].mean())
    
    # Risk-off vs risk-on exposure
    risk_off_avg = None
    risk_on_avg = None
    if "Regime" in diagnostics.columns:
        regime = diagnostics["Regime"]
        risk_off_mask = regime.isin(["panic", "downtrend"])
        risk_on_mask = regime.isin(["uptrend"])
        
        if risk_off_mask.sum() > 0:
            risk_off_avg = float(exp[risk_off_mask].mean())
        if risk_on_mask.sum() > 0:
            risk_on_avg = float(exp[risk_on_mask].mean())
    
    # Determine if scaling is material
    is_material = (
        exp_range >= min_material_change or
        material_changes > len(exp) * 0.05  # At least 5% of days have material changes
    )
    
    # If we have VIX data, also check high/low VIX difference
    if high_vix_avg is not None and low_vix_avg is not None:
        vix_exposure_diff = abs(high_vix_avg - low_vix_avg)
        is_material = is_material or vix_exposure_diff >= min_material_change
    
    return {
        "is_material": bool(is_material),
        "max_exposure": max_exp,
        "min_exposure": min_exp,
        "exposure_range": exp_range,
        "high_vix_avg_exposure": high_vix_avg,
        "low_vix_avg_exposure": low_vix_avg,
        "risk_off_avg_exposure": risk_off_avg,
        "risk_on_avg_exposure": risk_on_avg,
        "material_changes_count": int(material_changes),
        "total_days": len(exp),
    }


def get_stress_period_diagnostics(
    diagnostics: pd.DataFrame,
    vix_threshold: float = 25.0,
) -> pd.DataFrame:
    """
    Extract diagnostics for stress periods (high VIX or risk-off regime).
    
    Args:
        diagnostics: DataFrame from get_wave_diagnostics()
        vix_threshold: VIX level to consider as stress (default 25)
        
    Returns:
        DataFrame with only stress period rows
    """
    if diagnostics.empty:
        return diagnostics
    
    stress_mask = pd.Series(False, index=diagnostics.index)
    
    # High VIX periods
    if "VIX" in diagnostics.columns:
        stress_mask |= diagnostics["VIX"] >= vix_threshold
    
    # Risk-off regimes
    if "Regime" in diagnostics.columns:
        stress_mask |= diagnostics["Regime"].isin(["panic", "downtrend"])
    
    return diagnostics[stress_mask]


def generate_diagnostic_report(
    wave_name: str,
    mode: str = "Standard",
    days: int = 365,
    output_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive diagnostic report for a Wave showing VIX/regime overlay activity.
    
    Args:
        wave_name: name of the Wave
        mode: operating mode
        days: history window
        output_csv: optional path to save CSV report
        
    Returns:
        Dict with:
            - wave_name: str
            - mode: str
            - days: int
            - diagnostics: DataFrame with full diagnostic data
            - stress_diagnostics: DataFrame with only stress periods
            - validation: Dict from validate_exposure_scaling()
            - summary: Dict with key statistics
    """
    # Get full diagnostics
    diag = get_wave_diagnostics(wave_name, mode, days)
    
    if diag.empty:
        return {
            "wave_name": wave_name,
            "mode": mode,
            "days": days,
            "error": "Unable to generate diagnostics (insufficient data)",
        }
    
    # Get stress period diagnostics
    stress_diag = get_stress_period_diagnostics(diag)
    
    # Validate exposure scaling
    validation = validate_exposure_scaling(diag)
    
    # Compute summary statistics
    summary = {}
    
    if "VIX" in diag.columns:
        vix = diag["VIX"].dropna()
        if len(vix) > 0:
            summary["avg_vix"] = float(vix.mean())
            summary["max_vix"] = float(vix.max())
            summary["min_vix"] = float(vix.min())
            summary["high_vix_days"] = int((vix >= 25).sum())
            summary["low_vix_days"] = int((vix < 20).sum())
    
    if "Regime" in diag.columns:
        regime_counts = diag["Regime"].value_counts().to_dict()
        summary["regime_distribution"] = regime_counts
    
    if "Safe_Fraction" in diag.columns:
        safe_frac = diag["Safe_Fraction"].dropna()
        if len(safe_frac) > 0:
            summary["avg_safe_fraction"] = float(safe_frac.mean())
            summary["max_safe_fraction"] = float(safe_frac.max())
            summary["min_safe_fraction"] = float(safe_frac.min())
    
    if "Exposure" in diag.columns:
        exp = diag["Exposure"].dropna()
        if len(exp) > 0:
            summary["avg_exposure"] = float(exp.mean())
            summary["median_exposure"] = float(exp.median())
    
    # Add wave/benchmark performance
    if "Wave_Return" in diag.columns and "Benchmark_Return" in diag.columns:
        wave_ret = diag["Wave_Return"]
        bm_ret = diag["Benchmark_Return"]
        
        wave_cumret = (1 + wave_ret).prod() - 1
        bm_cumret = (1 + bm_ret).prod() - 1
        alpha = wave_cumret - bm_cumret
        
        summary["wave_total_return"] = float(wave_cumret)
        summary["benchmark_total_return"] = float(bm_cumret)
        summary["alpha"] = float(alpha)
    
    # Add stress period stats
    if not stress_diag.empty:
        if "Exposure" in stress_diag.columns:
            stress_exp = stress_diag["Exposure"].dropna()
            if len(stress_exp) > 0:
                summary["stress_avg_exposure"] = float(stress_exp.mean())
                summary["stress_min_exposure"] = float(stress_exp.min())
        
        if "Safe_Fraction" in stress_diag.columns:
            stress_safe = stress_diag["Safe_Fraction"].dropna()
            if len(stress_safe) > 0:
                summary["stress_avg_safe_fraction"] = float(stress_safe.mean())
                summary["stress_max_safe_fraction"] = float(stress_safe.max())
        
        summary["stress_period_days"] = len(stress_diag)
    
    # Save CSV if requested
    if output_csv:
        diag_save = diag.copy()
        diag_save.index.name = "Date"
        diag_save.to_csv(output_csv)
    
    return {
        "wave_name": wave_name,
        "mode": mode,
        "days": days,
        "diagnostics": diag,
        "stress_diagnostics": stress_diag,
        "validation": validation,
        "summary": summary,
    }


def print_diagnostic_report(report: Dict[str, Any]) -> None:
    """
    Print a human-readable diagnostic report to console.
    
    Args:
        report: output from generate_diagnostic_report()
    """
    if "error" in report:
        print(f"ERROR: {report['error']}")
        return
    
    print("=" * 80)
    print(f"VIX/REGIME OVERLAY DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"Wave: {report['wave_name']}")
    print(f"Mode: {report['mode']}")
    print(f"Period: {report['days']} days")
    print()
    
    # Summary stats
    print("-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)
    summary = report["summary"]
    
    if "avg_vix" in summary:
        print(f"VIX - Avg: {summary['avg_vix']:.1f}, "
              f"Max: {summary['max_vix']:.1f}, "
              f"Min: {summary['min_vix']:.1f}")
        print(f"VIX - High (>=25): {summary.get('high_vix_days', 0)} days, "
              f"Low (<20): {summary.get('low_vix_days', 0)} days")
    
    if "regime_distribution" in summary:
        print(f"Regime Distribution: {summary['regime_distribution']}")
    
    if "avg_exposure" in summary:
        print(f"Exposure - Avg: {summary['avg_exposure']:.2%}, "
              f"Median: {summary.get('median_exposure', 0):.2%}")
    
    if "avg_safe_fraction" in summary:
        print(f"Safe Fraction - Avg: {summary['avg_safe_fraction']:.2%}, "
              f"Max: {summary['max_safe_fraction']:.2%}, "
              f"Min: {summary['min_safe_fraction']:.2%}")
    
    if "wave_total_return" in summary:
        print(f"\nPerformance:")
        print(f"  Wave Return: {summary['wave_total_return']:.2%}")
        print(f"  Benchmark Return: {summary['benchmark_total_return']:.2%}")
        print(f"  Alpha: {summary['alpha']:.2%}")
    
    # Validation
    print()
    print("-" * 80)
    print("EXPOSURE SCALING VALIDATION")
    print("-" * 80)
    validation = report["validation"]
    
    print(f"Material Scaling: {'YES' if validation.get('is_material') else 'NO'}")
    print(f"Exposure Range: {validation.get('min_exposure', 0):.2%} - "
          f"{validation.get('max_exposure', 0):.2%} "
          f"(Δ = {validation.get('exposure_range', 0):.2%})")
    
    if validation.get("high_vix_avg_exposure") is not None:
        high_vix = validation["high_vix_avg_exposure"]
        low_vix = validation.get("low_vix_avg_exposure", 0)
        diff = abs(high_vix - low_vix) if low_vix else 0
        print(f"VIX Impact: High VIX Avg Exp = {high_vix:.2%}, "
              f"Low VIX Avg Exp = {low_vix:.2%}, "
              f"Difference = {diff:.2%}")
    
    if validation.get("risk_off_avg_exposure") is not None:
        risk_off = validation["risk_off_avg_exposure"]
        risk_on = validation.get("risk_on_avg_exposure", 0)
        diff = abs(risk_off - risk_on) if risk_on else 0
        print(f"Regime Impact: Risk-Off Avg Exp = {risk_off:.2%}, "
              f"Risk-On Avg Exp = {risk_on:.2%}, "
              f"Difference = {diff:.2%}")
    
    print(f"Material Changes: {validation.get('material_changes_count', 0)} days "
          f"out of {validation.get('total_days', 0)} "
          f"({validation.get('material_changes_count', 0) / max(1, validation.get('total_days', 1)) * 100:.1f}%)")
    
    # Stress periods
    if "stress_period_days" in summary and summary["stress_period_days"] > 0:
        print()
        print("-" * 80)
        print("STRESS PERIOD ANALYSIS (VIX >= 25 or Risk-Off Regime)")
        print("-" * 80)
        print(f"Stress Days: {summary['stress_period_days']}")
        if "stress_avg_exposure" in summary:
            print(f"Avg Exposure During Stress: {summary['stress_avg_exposure']:.2%} "
                  f"(Min: {summary.get('stress_min_exposure', 0):.2%})")
        if "stress_avg_safe_fraction" in summary:
            print(f"Avg Safe Fraction During Stress: {summary['stress_avg_safe_fraction']:.2%} "
                  f"(Max: {summary.get('stress_max_safe_fraction', 0):.2%})")
    
    # Sample recent data
    print()
    print("-" * 80)
    print("RECENT DIAGNOSTIC DATA (Last 10 Days)")
    print("-" * 80)
    
    diag = report["diagnostics"]
    if not diag.empty:
        recent = diag.tail(10).copy()
        
        # Format for display
        display_cols = []
        if "VIX" in recent.columns:
            recent["VIX"] = recent["VIX"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
            display_cols.append("VIX")
        if "Regime" in recent.columns:
            display_cols.append("Regime")
        if "Exposure" in recent.columns:
            recent["Exposure"] = recent["Exposure"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            display_cols.append("Exposure")
        if "Safe_Fraction" in recent.columns:
            recent["Safe_Fraction"] = recent["Safe_Fraction"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            display_cols.append("Safe_Fraction")
        if "Wave_Return" in recent.columns:
            recent["Wave_Return"] = recent["Wave_Return"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
            display_cols.append("Wave_Return")
        
        if display_cols:
            print(recent[display_cols].to_string())
    
    print()
    print("=" * 80)


def validate_equity_waves_overlay(
    modes: Optional[List[str]] = None,
    days: int = 365,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Validate VIX/regime overlay for all equity Waves.
    
    Args:
        modes: list of modes to test (default: ["Standard", "Alpha-Minus-Beta"])
        days: history window
        output_dir: optional directory to save individual CSV reports
        
    Returns:
        Dict mapping wave_name -> report for each equity Wave
    """
    if modes is None:
        modes = ["Standard", "Alpha-Minus-Beta"]
    
    # Identify equity Waves (exclude crypto, safe, and bond Waves)
    all_waves = get_all_waves()
    equity_waves = []
    
    for wave in all_waves:
        wave_lower = wave.lower()
        is_equity = True
        
        # Exclude non-equity waves
        if any(keyword in wave_lower for keyword in [
            "crypto", "bitcoin", "stable", "yield", "income",
            "smartsafe", "treasury", "muni", "gold"
        ]):
            is_equity = False
        
        if is_equity:
            equity_waves.append(wave)
    
    print(f"Validating {len(equity_waves)} equity Waves across {len(modes)} modes...")
    print(f"Equity Waves: {equity_waves}")
    print()
    
    reports = {}
    
    for wave in equity_waves:
        for mode in modes:
            key = f"{wave} ({mode})"
            print(f"\nProcessing: {key}")
            print("-" * 80)
            
            output_csv = None
            if output_dir:
                import os
                os.makedirs(output_dir, exist_ok=True)
                safe_name = wave.replace("/", "_").replace(" ", "_")
                output_csv = os.path.join(output_dir, f"{safe_name}_{mode}_diagnostics.csv")
            
            report = generate_diagnostic_report(
                wave_name=wave,
                mode=mode,
                days=days,
                output_csv=output_csv,
            )
            
            reports[key] = report
            
            # Print summary
            if "error" not in report:
                val = report["validation"]
                print(f"  Material Scaling: {'YES' if val.get('is_material') else 'NO'}")
                if val.get('exposure_range'):
                    print(f"  Exposure Range: {val['exposure_range']:.2%}")
                if val.get('high_vix_avg_exposure') and val.get('low_vix_avg_exposure'):
                    diff = abs(val['high_vix_avg_exposure'] - val['low_vix_avg_exposure'])
                    print(f"  High/Low VIX Exposure Δ: {diff:.2%}")
            else:
                print(f"  ERROR: {report.get('error', 'Unknown')}")
    
    return reports


# Convenience function for quick validation
def quick_validate(wave_name: str = "US MegaCap Core Wave", mode: str = "Standard", days: int = 365) -> None:
    """
    Quick validation for a single Wave. Prints full diagnostic report.
    
    Args:
        wave_name: Wave to validate
        mode: operating mode
        days: history window
    """
    report = generate_diagnostic_report(wave_name, mode, days)
    print_diagnostic_report(report)


if __name__ == "__main__":
    # Example usage: validate a representative equity Wave
    print("VIX/Regime Overlay Diagnostic Validation")
    print("=" * 80)
    print()
    
    # Quick test with one Wave
    quick_validate("US MegaCap Core Wave", "Standard", 365)
    
    print("\n\n")
    
    # Full validation of all equity Waves
    print("Running full equity Wave validation...")
    print()
    
    reports = validate_equity_waves_overlay(
        modes=["Standard", "Alpha-Minus-Beta"],
        days=365,
        output_dir="/tmp/vix_diagnostics",
    )
    
    # Summary
    print("\n\n")
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    material_count = 0
    total_count = 0
    
    for key, report in reports.items():
        if "error" not in report:
            total_count += 1
            if report["validation"].get("is_material"):
                material_count += 1
    
    print(f"Total Equity Waves Tested: {total_count}")
    print(f"Material Exposure Scaling: {material_count} / {total_count} "
          f"({material_count / max(1, total_count) * 100:.1f}%)")
    
    print("\nDiagnostic CSV files saved to: /tmp/vix_diagnostics/")
