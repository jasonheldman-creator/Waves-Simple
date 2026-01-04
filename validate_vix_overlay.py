#!/usr/bin/env python3
# validate_vix_overlay.py — Quick validation that VIX/regime overlay is active
#
# This script proves that the VIX/regime overlay is materially affecting
# equity Wave returns by showing exposure scaling in action.
#
# Usage:
#   python validate_vix_overlay.py [wave_name] [mode] [days]
#
# Example:
#   python validate_vix_overlay.py "US MegaCap Core Wave" Standard 365

import sys
from typing import Optional

try:
    from waves_engine import get_vix_regime_diagnostics, get_all_waves
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Please ensure pandas, numpy, and yfinance are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def validate_single_wave(wave_name: str, mode: str = "Standard", days: int = 365) -> None:
    """Validate VIX/regime overlay for a single Wave."""
    
    print("=" * 80)
    print(f"VIX/REGIME OVERLAY VALIDATION")
    print("=" * 80)
    print(f"Wave: {wave_name}")
    print(f"Mode: {mode}")
    print(f"Period: {days} days")
    print()
    
    try:
        # Get diagnostics
        diag = get_vix_regime_diagnostics(wave_name, mode, days)
        
        if diag.empty:
            print("ERROR: No diagnostic data available")
            print("This may occur if:")
            print("  - The Wave name is invalid")
            print("  - Market data cannot be fetched (network issue)")
            print("  - The time window is too short")
            return
        
        print(f"Retrieved {len(diag)} days of diagnostic data")
        print()
        
        # Validate that diagnostics contain required columns
        required_cols = ["vix", "regime", "exposure", "safe_fraction"]
        missing_cols = [c for c in required_cols if c not in diag.columns]
        if missing_cols:
            print(f"WARNING: Missing diagnostic columns: {missing_cols}")
            print("Available columns:", list(diag.columns))
            return
        
        # Extract key metrics
        exposure = diag["exposure"].dropna()
        safe_fraction = diag["safe_fraction"].dropna()
        vix = diag["vix"].dropna()
        regime = diag["regime"]
        
        if len(exposure) == 0:
            print("ERROR: No valid exposure data in diagnostics")
            return
        
        # Overall statistics
        print("-" * 80)
        print("OVERALL STATISTICS")
        print("-" * 80)
        print(f"Exposure:")
        print(f"  Average: {exposure.mean():.2%}")
        print(f"  Range: {exposure.min():.2%} - {exposure.max():.2%} (Δ = {exposure.max() - exposure.min():.2%})")
        print(f"  Std Dev: {exposure.std():.2%}")
        print()
        
        print(f"Safe Fraction:")
        print(f"  Average: {safe_fraction.mean():.2%}")
        print(f"  Range: {safe_fraction.min():.2%} - {safe_fraction.max():.2%}")
        print()
        
        if len(vix) > 0:
            print(f"VIX:")
            print(f"  Average: {vix.mean():.1f}")
            print(f"  Range: {vix.min():.1f} - {vix.max():.1f}")
            print()
        
        # Regime distribution
        if regime.notna().sum() > 0:
            regime_counts = regime.value_counts()
            print(f"Regime Distribution:")
            for r, count in regime_counts.items():
                pct = count / len(regime) * 100
                print(f"  {r}: {count} days ({pct:.1f}%)")
            print()
        
        # High VIX vs Low VIX comparison
        print("-" * 80)
        print("VIX IMPACT ANALYSIS")
        print("-" * 80)
        
        if len(vix) > 0:
            high_vix_mask = vix >= 25
            low_vix_mask = vix < 20
            
            high_vix_days = high_vix_mask.sum()
            low_vix_days = low_vix_mask.sum()
            
            if high_vix_days > 0:
                high_vix_exp = exposure[high_vix_mask].mean()
                high_vix_safe = safe_fraction[high_vix_mask].mean()
                print(f"High VIX (>= 25): {high_vix_days} days")
                print(f"  Average Exposure: {high_vix_exp:.2%}")
                print(f"  Average Safe Fraction: {high_vix_safe:.2%}")
            else:
                print(f"High VIX (>= 25): No days in period")
                high_vix_exp = None
            
            if low_vix_days > 0:
                low_vix_exp = exposure[low_vix_mask].mean()
                low_vix_safe = safe_fraction[low_vix_mask].mean()
                print(f"Low VIX (< 20): {low_vix_days} days")
                print(f"  Average Exposure: {low_vix_exp:.2%}")
                print(f"  Average Safe Fraction: {low_vix_safe:.2%}")
            else:
                print(f"Low VIX (< 20): No days in period")
                low_vix_exp = None
            
            # Calculate impact
            if high_vix_exp is not None and low_vix_exp is not None:
                exp_diff = abs(high_vix_exp - low_vix_exp)
                print()
                print(f"VIX Exposure Impact: {exp_diff:.2%} difference")
                if exp_diff >= 0.05:
                    print(f"  ✓ MATERIAL - Exposure scaling is >= 5%")
                elif exp_diff >= 0.03:
                    print(f"  ~ MODERATE - Exposure scaling is 3-5%")
                else:
                    print(f"  ✗ MINIMAL - Exposure scaling is < 3%")
        else:
            print("No VIX data available for analysis")
        
        print()
        
        # Risk-off vs Risk-on comparison
        print("-" * 80)
        print("REGIME IMPACT ANALYSIS")
        print("-" * 80)
        
        if regime.notna().sum() > 0:
            risk_off_mask = regime.isin(["panic", "downtrend"])
            risk_on_mask = regime.isin(["uptrend"])
            
            risk_off_days = risk_off_mask.sum()
            risk_on_days = risk_on_mask.sum()
            
            if risk_off_days > 0:
                risk_off_exp = exposure[risk_off_mask].mean()
                risk_off_safe = safe_fraction[risk_off_mask].mean()
                print(f"Risk-Off (panic/downtrend): {risk_off_days} days")
                print(f"  Average Exposure: {risk_off_exp:.2%}")
                print(f"  Average Safe Fraction: {risk_off_safe:.2%}")
            else:
                print(f"Risk-Off: No days in period")
                risk_off_exp = None
            
            if risk_on_days > 0:
                risk_on_exp = exposure[risk_on_mask].mean()
                risk_on_safe = safe_fraction[risk_on_mask].mean()
                print(f"Risk-On (uptrend): {risk_on_days} days")
                print(f"  Average Exposure: {risk_on_exp:.2%}")
                print(f"  Average Safe Fraction: {risk_on_safe:.2%}")
            else:
                print(f"Risk-On: No days in period")
                risk_on_exp = None
            
            # Calculate impact
            if risk_off_exp is not None and risk_on_exp is not None:
                exp_diff = abs(risk_off_exp - risk_on_exp)
                print()
                print(f"Regime Exposure Impact: {exp_diff:.2%} difference")
                if exp_diff >= 0.05:
                    print(f"  ✓ MATERIAL - Exposure scaling is >= 5%")
                elif exp_diff >= 0.03:
                    print(f"  ~ MODERATE - Exposure scaling is 3-5%")
                else:
                    print(f"  ✗ MINIMAL - Exposure scaling is < 3%")
        else:
            print("No regime data available for analysis")
        
        print()
        
        # Show recent examples
        print("-" * 80)
        print("RECENT DIAGNOSTIC EXAMPLES (Last 10 Days)")
        print("-" * 80)
        
        recent = diag.tail(10).copy()
        
        # Format for display
        display_df = pd.DataFrame(index=recent.index)
        display_df["VIX"] = recent["vix"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        display_df["Regime"] = recent["regime"].apply(lambda x: str(x) if pd.notna(x) else "N/A")
        display_df["Exposure"] = recent["exposure"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        display_df["Safe%"] = recent["safe_fraction"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        
        if "vix_gate" in recent.columns:
            display_df["VIX_Gate"] = recent["vix_gate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        if "regime_gate" in recent.columns:
            display_df["Reg_Gate"] = recent["regime_gate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        
        print(display_df.to_string())
        print()
        
        # Final assessment
        print("=" * 80)
        print("VALIDATION RESULT")
        print("=" * 80)
        
        # Check if overlay is materially active
        exp_range = exposure.max() - exposure.min()
        
        material_vix = False
        material_regime = False
        
        if len(vix) > 0:
            high_vix_mask = vix >= 25
            low_vix_mask = vix < 20
            if high_vix_mask.sum() > 0 and low_vix_mask.sum() > 0:
                vix_impact = abs(exposure[high_vix_mask].mean() - exposure[low_vix_mask].mean())
                material_vix = vix_impact >= 0.05
        
        if regime.notna().sum() > 0:
            risk_off_mask = regime.isin(["panic", "downtrend"])
            risk_on_mask = regime.isin(["uptrend"])
            if risk_off_mask.sum() > 0 and risk_on_mask.sum() > 0:
                regime_impact = abs(exposure[risk_off_mask].mean() - exposure[risk_on_mask].mean())
                material_regime = regime_impact >= 0.05
        
        is_material = exp_range >= 0.05 or material_vix or material_regime
        
        if is_material:
            print("✓ VIX/REGIME OVERLAY IS MATERIALLY ACTIVE")
            print()
            print("Evidence:")
            if exp_range >= 0.05:
                print(f"  • Exposure range is {exp_range:.2%} (>= 5%)")
            if material_vix:
                print(f"  • VIX-driven exposure difference is >= 5%")
            if material_regime:
                print(f"  • Regime-driven exposure difference is >= 5%")
            print()
            print("The overlay is dynamically adjusting exposure based on market conditions.")
            print("This affects daily returns through reduced exposure during high-VIX/risk-off periods.")
        else:
            print("⚠ VIX/REGIME OVERLAY ACTIVITY IS MINIMAL")
            print()
            print(f"Exposure range: {exp_range:.2%} (< 5% threshold)")
            print()
            print("Possible reasons:")
            print("  • Market conditions have been stable (low VIX volatility)")
            print("  • Time period doesn't include stress events")
            print("  • Mode settings may limit exposure adjustments")
            print()
            print("Note: This doesn't mean the overlay isn't working, just that")
            print("market conditions haven't triggered material scaling in this period.")
        
        print()
        print("=" * 80)
        
    except Exception as e:
        print(f"ERROR: Failed to validate Wave: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for validation script."""
    
    # Parse command line arguments
    wave_name = "US MegaCap Core Wave"  # default
    mode = "Standard"  # default
    days = 365  # default
    
    if len(sys.argv) > 1:
        wave_name = sys.argv[1]
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    if len(sys.argv) > 3:
        days = int(sys.argv[3])
    
    validate_single_wave(wave_name, mode, days)


if __name__ == "__main__":
    main()
