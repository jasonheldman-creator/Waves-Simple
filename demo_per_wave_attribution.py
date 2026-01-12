"""
Demo script for per-wave attribution feature (ENGINE v17.5).

This script demonstrates how to use the new attribution functionality to:
1. Decompose wave alpha into selection and overlay components
2. Understand where alpha comes from (stock selection vs. strategy overlay)
3. Track attribution across different time periods
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waves_engine import get_attribution, ENGINE_VERSION


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def demo_basic_attribution():
    """Demonstrate basic attribution computation."""
    print_separator()
    print("DEMO 1: Basic Attribution Computation")
    print_separator()
    
    wave_name = "S&P 500 Wave"
    periods = [30, 60, 365]
    
    print(f"\nComputing attribution for: {wave_name}")
    print(f"Periods: {periods} days")
    print(f"Engine Version: {ENGINE_VERSION}\n")
    
    result = get_attribution(wave_name, periods=periods)
    
    if not result["success"]:
        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
        return
    
    print("✓ Attribution computed successfully\n")
    
    # Display results
    for period_days, metrics in sorted(result["attribution"].items()):
        if "error" in metrics:
            print(f"Period {period_days}D: {metrics['error']}")
            continue
        
        print(f"Period: {period_days} days")
        print(f"  Benchmark Return:        {metrics['benchmark_return']*100:>8.2f}%")
        print(f"  Raw Wave Return:         {metrics['raw_wave_return']*100:>8.2f}%")
        print(f"  Strategy Wave Return:    {metrics['strategy_wave_return']*100:>8.2f}%")
        print(f"  {'─' * 45}")
        print(f"  Total Alpha:             {metrics['total_alpha']*100:>8.2f}%")
        print(f"    └─ Selection Alpha:    {metrics['selection_alpha']*100:>8.2f}%")
        print(f"    └─ Overlay Alpha:      {metrics['overlay_alpha']*100:>8.2f}%")
        print(f"  Reconciliation Error:    {metrics['reconciliation_error']*100:>8.4f}%")
        print()


def demo_multi_wave_comparison():
    """Compare attribution across multiple waves."""
    print_separator()
    print("DEMO 2: Multi-Wave Attribution Comparison")
    print_separator()
    
    waves = [
        "S&P 500 Wave",
        "US MegaCap Core Wave",
        "AI & Cloud MegaCap Wave",
        "Small Cap Growth Wave"
    ]
    
    period = 365  # 1 year
    
    print(f"\nComparing {len(waves)} waves over {period} days:\n")
    
    results = []
    for wave_name in waves:
        result = get_attribution(wave_name, periods=[period])
        if result["success"] and period in result["attribution"]:
            metrics = result["attribution"][period]
            if "error" not in metrics:
                results.append((wave_name, metrics))
    
    # Print comparison table
    print(f"{'Wave':<35} {'Total':<10} {'Selection':<12} {'Overlay':<10}")
    print(f"{'─'*35} {'─'*10} {'─'*12} {'─'*10}")
    
    for wave_name, metrics in results:
        total = metrics['total_alpha'] * 100
        selection = metrics['selection_alpha'] * 100
        overlay = metrics['overlay_alpha'] * 100
        
        print(f"{wave_name:<35} {total:>8.2f}%  {selection:>10.2f}%  {overlay:>8.2f}%")
    
    print()


def demo_alpha_sources():
    """Analyze where alpha comes from."""
    print_separator()
    print("DEMO 3: Alpha Source Analysis")
    print_separator()
    
    wave_name = "AI & Cloud MegaCap Wave"
    period = 365
    
    print(f"\nAnalyzing alpha sources for: {wave_name}")
    print(f"Period: {period} days\n")
    
    result = get_attribution(wave_name, periods=[period])
    
    if not result["success"] or period not in result["attribution"]:
        print(f"✗ Failed to compute attribution")
        return
    
    metrics = result["attribution"][period]
    if "error" in metrics:
        print(f"✗ Error: {metrics['error']}")
        return
    
    total_alpha = metrics['total_alpha']
    selection_alpha = metrics['selection_alpha']
    overlay_alpha = metrics['overlay_alpha']
    
    # Calculate contributions
    if abs(total_alpha) > 0.0001:  # Avoid division by near-zero
        selection_contrib = (selection_alpha / total_alpha) * 100
        overlay_contrib = (overlay_alpha / total_alpha) * 100
    else:
        selection_contrib = 0.0
        overlay_contrib = 0.0
    
    print(f"Total Alpha: {total_alpha*100:+.2f}%\n")
    print("Alpha Attribution:")
    print(f"  Stock Selection:   {selection_alpha*100:>8.2f}% ({selection_contrib:>5.1f}% of total)")
    print(f"  Strategy Overlay:  {overlay_alpha*100:>8.2f}% ({overlay_contrib:>5.1f}% of total)")
    print()
    
    # Interpretation
    print("Interpretation:")
    if abs(overlay_alpha) > abs(selection_alpha):
        print("  → Alpha is primarily driven by STRATEGY OVERLAY (regime, VIX, exposure timing)")
    elif abs(selection_alpha) > abs(overlay_alpha):
        print("  → Alpha is primarily driven by STOCK SELECTION (basket composition)")
    else:
        print("  → Alpha is balanced between stock selection and strategy overlay")
    print()


def demo_period_comparison():
    """Compare attribution across different time periods."""
    print_separator()
    print("DEMO 4: Period-by-Period Attribution")
    print_separator()
    
    wave_name = "US MegaCap Core Wave"
    periods = [7, 30, 60, 90, 180, 365]
    
    print(f"\nWave: {wave_name}")
    print(f"Analyzing attribution across {len(periods)} time periods\n")
    
    result = get_attribution(wave_name, periods=periods)
    
    if not result["success"]:
        print(f"✗ Failed: {result.get('error', 'Unknown')}")
        return
    
    # Print table
    print(f"{'Period':<10} {'Total Alpha':<15} {'Selection':<15} {'Overlay':<15}")
    print(f"{'─'*10} {'─'*15} {'─'*15} {'─'*15}")
    
    for period in periods:
        if period not in result["attribution"]:
            continue
        
        metrics = result["attribution"][period]
        if "error" in metrics:
            print(f"{period}D        Error: {metrics['error'][:40]}")
            continue
        
        total = metrics['total_alpha'] * 100
        selection = metrics['selection_alpha'] * 100
        overlay = metrics['overlay_alpha'] * 100
        
        print(f"{period}D        {total:>10.2f}%     {selection:>10.2f}%     {overlay:>10.2f}%")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print(" " * 20 + "PER-WAVE ATTRIBUTION DEMO")
    print(" " * 25 + f"ENGINE v{ENGINE_VERSION}")
    print("=" * 80 + "\n")
    
    try:
        demo_basic_attribution()
        demo_multi_wave_comparison()
        demo_alpha_sources()
        demo_period_comparison()
        
        print("=" * 80)
        print("All demos completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
