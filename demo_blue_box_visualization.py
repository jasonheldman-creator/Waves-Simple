#!/usr/bin/env python3
"""
Visual demonstration of the blue box display format.
This script simulates what users will see in the enhanced blue box.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def visualize_blue_box():
    """
    Demonstrate the blue box display format with sample data.
    """
    print("\n" + "=" * 80)
    print("PORTFOLIO SNAPSHOT BLUE BOX - VISUAL DEMONSTRATION")
    print("=" * 80)
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("\nLoading portfolio data...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Compute ledger
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[1, 30, 60, 365],
            benchmark_ticker='SPY',
            mode='Standard',
            vix_exposure_enabled=True
        )
        
        if not ledger['success']:
            print(f"❌ FAIL: Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        # Display metadata
        n_waves = len([w for w in ledger.get('period_results', {}).keys()])
        print(f"✓ Portfolio loaded (waves={n_waves}, VIX={'enabled' if ledger.get('overlay_available') else 'disabled'})")
        
        # Simulate blue box display
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 20 + "💼 Portfolio Snapshot (All Waves)" + " " * 24 + "│")
        print("│" + " " * 78 + "│")
        print("│  Equal-weight portfolio across all active waves                            │")
        print("│  Each period shows: Portfolio Return | Benchmark Return | Alpha           │")
        print("├" + "─" * 78 + "┤")
        
        # Display each period in columns
        periods = ['1D', '30D', '60D', '365D']
        
        # Header row
        header = "│  "
        for period_key in periods:
            header += f"{period_key:^18} "
        header += " │"
        print(header)
        
        print("├" + "─" * 78 + "┤")
        
        # Portfolio row
        portfolio_row = "│  "
        for period_key in periods:
            period_data = ledger['period_results'].get(period_key, {})
            if period_data.get('available'):
                cum_realized = period_data['cum_realized']
                value = f"{cum_realized:+.2%}"
            else:
                value = "N/A"
            portfolio_row += f"📈 Port: {value:>8} "
        portfolio_row += " │"
        print(portfolio_row)
        
        # Benchmark row
        benchmark_row = "│  "
        for period_key in periods:
            period_data = ledger['period_results'].get(period_key, {})
            if period_data.get('available'):
                cum_benchmark = period_data['cum_benchmark']
                value = f"{cum_benchmark:+.2%}"
            else:
                value = "N/A"
            benchmark_row += f"📊 Bmrk: {value:>8} "
        benchmark_row += " │"
        print(benchmark_row)
        
        # Alpha row
        alpha_row = "│  "
        for period_key in periods:
            period_data = ledger['period_results'].get(period_key, {})
            if period_data.get('available'):
                total_alpha = period_data['total_alpha']
                value = f"{total_alpha:+.2%}"
                # Color indicator (✓ for positive, ✗ for negative)
                indicator = "✓" if total_alpha >= 0 else "✗"
            else:
                value = "N/A"
                indicator = "⚠"
            alpha_row += f"🎯 {indicator} {value:>10} "
        alpha_row += " │"
        print(alpha_row)
        
        # Date range row
        date_row = "│  "
        for period_key in periods:
            period_data = ledger['period_results'].get(period_key, {})
            if period_data.get('available'):
                start = period_data['start_date']
                end = period_data['end_date']
                # Truncate dates to fit
                date_str = f"{start[-5:]}-{end[-5:]}"
            else:
                reason = period_data.get('reason', 'unknown')
                # Truncate reason to fit
                date_str = reason[:16]
            date_row += f"{date_str:^18} "
        date_row += " │"
        print(date_row)
        
        print("└" + "─" * 78 + "┘")
        
        # Show alpha attribution for 30D
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 22 + "🔬 Alpha Attribution (30D)" + " " * 29 + "│")
        print("├" + "─" * 78 + "┤")
        
        period_30d = ledger['period_results'].get('30D', {})
        if period_30d.get('available'):
            total = period_30d['total_alpha']
            selection = period_30d['selection_alpha']
            overlay = period_30d['overlay_alpha']
            residual = period_30d['residual']
            
            print(f"│  Total Alpha:      {total:+.2%}  (Realized - Benchmark)" + " " * 32 + "│")
            print(f"│  Selection Alpha:  {selection:+.2%}  (Wave selection)" + " " * 36 + "│")
            print(f"│  Overlay Alpha:    {overlay:+.2%}  (VIX exposure)" + " " * 37 + "│")
            
            # Residual with color coding
            residual_pct = abs(residual) * 100
            if residual_pct < 0.10:
                status = "🟢 Excellent"
            elif residual_pct < 0.5:
                status = "🟡 Acceptable"
            else:
                status = "🔴 Warning"
            
            print(f"│  Residual:         {residual:+.3%}  ({status})" + " " * (44 - len(status)) + "│")
        else:
            reason = period_30d.get('reason', 'unknown')
            print(f"│  ⚠️ Unavailable: {reason[:56]}" + " " * (56 - len(reason[:56])) + "│")
        
        print("└" + "─" * 78 + "┘")
        
        # Show warnings if any
        if ledger.get('warnings'):
            print("\n⚠️ Warnings:")
            for warning in ledger['warnings']:
                print(f"  • {warning}")
        
        print("\n" + "=" * 80)
        print("KEY FEATURES DEMONSTRATED:")
        print("=" * 80)
        print("✓ Each period shows Portfolio / Benchmark / Alpha in stacked format")
        print("✓ Positive alpha indicated with ✓, negative with ✗")
        print("✓ Unavailable periods show N/A with truncated reason")
        print("✓ Alpha attribution shows detailed breakdown for 30D period")
        print("✓ Residual is color-coded based on tolerance (🟢 < 0.10%, 🟡 < 0.5%, 🔴 >= 0.5%)")
        print("✓ All values come from single source of truth: compute_portfolio_alpha_ledger()")
        print("\n🎉 VISUALIZATION COMPLETE!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = visualize_blue_box()
    sys.exit(0 if success else 1)
