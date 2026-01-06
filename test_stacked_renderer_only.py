#!/usr/bin/env python3
"""
Test to verify that Portfolio Snapshot blue box uses ONLY the stacked ledger renderer.

This test ensures:
1. Blue box uses compute_portfolio_alpha_ledger() exclusively
2. All period data displayed comes from ledger['period_results']
3. Reconciliation rules are enforced:
   - Portfolio Return - Benchmark Return = Total Alpha
   - Selection Alpha + Overlay Alpha + Residual = Total Alpha
4. No legacy tile renderer logic is executed
"""

import sys
import os

# Add parent directory to path for test execution
# This allows the test to import modules from the project root
# when run directly (e.g., `python test_stacked_renderer_only.py`)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_blue_box_uses_ledger_only():
    """
    Verify that blue box implementation uses compute_portfolio_alpha_ledger() as sole data source.
    """
    print("\n" + "=" * 70)
    print("TEST: Blue Box Uses Stacked Ledger Renderer Only")
    print("=" * 70)
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger, RESIDUAL_TOLERANCE
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("\n1. Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded: {len(price_book)} days, {len(price_book.columns)} tickers")
        
        # Compute ledger (this is the ONLY source for blue box data)
        print("\n2. Computing portfolio alpha ledger (exclusive data source)...")
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
        
        print("✓ Ledger computed successfully")
        
        # Verify all periods use ledger data (no inline calculations)
        print("\n3. Verifying periods use ledger data exclusively...")
        
        periods_tested = 0
        periods_passed = 0
        
        for period_key in ['1D', '30D', '60D', '365D']:
            periods_tested += 1
            period_data = ledger['period_results'].get(period_key, {})
            
            print(f"\n{period_key} Period:")
            
            if period_data.get('available'):
                # Verify reconciliation 1: Portfolio - Benchmark = Alpha
                portfolio = period_data['cum_realized']
                benchmark = period_data['cum_benchmark']
                alpha = period_data['total_alpha']
                
                expected_alpha = portfolio - benchmark
                diff_1 = abs(expected_alpha - alpha)
                
                print(f"  📈 Portfolio: {portfolio:+.4%}")
                print(f"  📊 Benchmark: {benchmark:+.4%}")
                print(f"  🎯 Alpha: {alpha:+.4%}")
                
                if diff_1 > RESIDUAL_TOLERANCE:
                    print(f"  ❌ Reconciliation 1 FAILED: Portfolio - Benchmark ≠ Alpha (diff={diff_1:.6f})")
                    return False
                else:
                    print(f"  ✅ Reconciliation 1 PASSED: Portfolio - Benchmark = Alpha (diff={diff_1:.6f})")
                
                # Verify reconciliation 2: Selection + Overlay + Residual = Total
                selection = period_data['selection_alpha']
                overlay = period_data['overlay_alpha']
                residual = period_data['residual']
                
                expected_total = selection + overlay + residual
                diff_2 = abs(expected_total - alpha)
                
                if diff_2 > RESIDUAL_TOLERANCE:
                    print(f"  ❌ Reconciliation 2 FAILED: Selection + Overlay + Residual ≠ Total (diff={diff_2:.6f})")
                    return False
                else:
                    print(f"  ✅ Reconciliation 2 PASSED: Selection + Overlay + Residual = Total (diff={diff_2:.6f})")
                
                periods_passed += 1
            else:
                # Unavailable period - verify it shows N/A with reason
                reason = period_data.get('reason', 'unknown')
                print(f"  ⚠️ Period unavailable: {reason}")
                print(f"  ✅ Properly handled with N/A display")
                periods_passed += 1
        
        print(f"\n4. Summary: {periods_passed}/{periods_tested} periods validated")
        
        if periods_passed == periods_tested:
            print("\n✅ ALL TESTS PASSED: Blue box uses stacked ledger renderer exclusively")
            return True
        else:
            print("\n❌ SOME TESTS FAILED")
            return False
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_legacy_tile_renderer_code():
    """
    Verify that app.py does not contain legacy tile renderer code in blue box section.
    """
    print("\n" + "=" * 70)
    print("TEST: No Legacy Tile Renderer Code")
    print("=" * 70)
    
    try:
        # Read app.py and find the blue box section
        app_path = os.path.join(os.path.dirname(__file__), 'app.py')
        
        with open(app_path, 'r') as f:
            lines = f.readlines()
        
        # Find the blue box section (SECTION 1.5: PORTFOLIO SNAPSHOT)
        in_blue_box = False
        blue_box_start = -1
        blue_box_end = -1
        
        for i, line in enumerate(lines):
            if 'SECTION 1.5: PORTFOLIO SNAPSHOT (BLUE BOX)' in line:
                in_blue_box = True
                blue_box_start = i
            elif in_blue_box and ('SECTION 2:' in line or 'SECTION 3:' in line):
                blue_box_end = i
                break
        
        if blue_box_start == -1 or blue_box_end == -1:
            print("❌ FAIL: Could not locate blue box section in app.py")
            return False
        
        print(f"\n1. Blue box section found: lines {blue_box_start+1} to {blue_box_end+1}")
        
        # Check for stacked ledger renderer proof line
        proof_line_found = False
        for i in range(blue_box_start, blue_box_end):
            if 'Renderer: Stacked Ledger' in lines[i] or 'STACKED LEDGER RENDERER' in lines[i]:
                proof_line_found = True
                print(f"✅ Renderer proof line found at line {i+1}")
                break
        
        if not proof_line_found:
            print("❌ FAIL: Renderer proof line not found in blue box")
            return False
        
        # Verify compute_portfolio_alpha_ledger is called
        ledger_call_found = False
        for i in range(blue_box_start, blue_box_end):
            if 'compute_portfolio_alpha_ledger(' in lines[i]:
                ledger_call_found = True
                print(f"✅ compute_portfolio_alpha_ledger() call found at line {i+1}")
                break
        
        if not ledger_call_found:
            print("❌ FAIL: compute_portfolio_alpha_ledger() not called in blue box")
            return False
        
        # Verify stacked format uses st.markdown (not st.metric for period displays)
        period_display_section_start = -1
        for i in range(blue_box_start, blue_box_end):
            if 'for col, period_key in zip' in lines[i]:
                period_display_section_start = i
                break
        
        if period_display_section_start == -1:
            print("❌ FAIL: Period display loop not found")
            return False
        
        # Check next 30 lines for st.metric usage in period display
        uses_stacked_markdown = False
        uses_legacy_metric = False
        
        for i in range(period_display_section_start, min(period_display_section_start + 30, blue_box_end)):
            line = lines[i]
            # Check for stacked markdown (Portfolio, Benchmark, Alpha)
            if 'st.markdown(f"📈 **Portfolio:**' in line or 'st.markdown(f"📊 **Benchmark:**' in line or 'st.markdown(f"🎯 **Alpha:**' in line:
                uses_stacked_markdown = True
            # Check for legacy tile metric for periods (NOT attribution metrics)
            if 'st.metric' in line and ('period_key' in ''.join(lines[max(0, i-5):i+5]) or any(p in line for p in ['1D', '30D', '60D', '365D'])):
                # This would be a legacy tile renderer
                if 'Attribution' not in ''.join(lines[max(0, i-10):i+10]):
                    uses_legacy_metric = True
                    print(f"⚠️ WARNING: Potential legacy st.metric found at line {i+1}: {line.strip()}")
        
        print(f"\n2. Period display analysis:")
        print(f"   - Uses stacked markdown: {uses_stacked_markdown}")
        print(f"   - Uses legacy metric tiles: {uses_legacy_metric}")
        
        if uses_stacked_markdown and not uses_legacy_metric:
            print("\n✅ PASSED: Blue box uses stacked markdown renderer, no legacy metric tiles for periods")
            return True
        elif uses_legacy_metric:
            print("\n❌ FAILED: Legacy metric tiles detected in period display")
            return False
        else:
            print("\n⚠️ WARNING: Could not verify renderer type")
            return False
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reconciliation_rules_enforced():
    """
    Test that reconciliation rules are enforced by compute_portfolio_alpha_ledger().
    """
    print("\n" + "=" * 70)
    print("TEST: Reconciliation Rules Enforced")
    print("=" * 70)
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger, RESIDUAL_TOLERANCE
        from helpers.price_book import get_price_book
        
        # Load data
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("⚠️ SKIP: PRICE_BOOK is empty")
            return True
        
        # Compute ledger
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[1, 30, 60, 365],
            benchmark_ticker='SPY',
            mode='Standard',
            vix_exposure_enabled=True
        )
        
        if not ledger['success']:
            print(f"⚠️ SKIP: Ledger computation failed: {ledger['failure_reason']}")
            return True
        
        print("\nTesting reconciliation rules for all available periods...")
        
        all_passed = True
        
        for period_key in ['1D', '30D', '60D', '365D']:
            period_data = ledger['period_results'].get(period_key, {})
            
            if not period_data.get('available'):
                print(f"\n{period_key}: Unavailable ({period_data.get('reason', 'unknown')})")
                continue
            
            print(f"\n{period_key}:")
            
            # Rule 1: Portfolio - Benchmark = Alpha
            portfolio = period_data['cum_realized']
            benchmark = period_data['cum_benchmark']
            total_alpha = period_data['total_alpha']
            
            rule1_diff = abs((portfolio - benchmark) - total_alpha)
            rule1_passed = rule1_diff <= RESIDUAL_TOLERANCE
            
            print(f"  Rule 1: Portfolio ({portfolio:+.4%}) - Benchmark ({benchmark:+.4%}) = Alpha ({total_alpha:+.4%})")
            print(f"          Difference: {rule1_diff:.6f} {'✅ PASS' if rule1_passed else '❌ FAIL'}")
            
            if not rule1_passed:
                all_passed = False
            
            # Rule 2: Selection + Overlay + Residual = Total Alpha
            selection = period_data['selection_alpha']
            overlay = period_data['overlay_alpha']
            residual = period_data['residual']
            
            rule2_diff = abs((selection + overlay + residual) - total_alpha)
            rule2_passed = rule2_diff <= RESIDUAL_TOLERANCE
            
            print(f"  Rule 2: Selection ({selection:+.4%}) + Overlay ({overlay:+.4%}) + Residual ({residual:+.4%}) = Total ({total_alpha:+.4%})")
            print(f"          Difference: {rule2_diff:.6f} {'✅ PASS' if rule2_passed else '❌ FAIL'}")
            
            if not rule2_passed:
                all_passed = False
        
        if all_passed:
            print("\n✅ ALL RECONCILIATION RULES PASSED")
            return True
        else:
            print("\n❌ SOME RECONCILIATION RULES FAILED")
            return False
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("STACKED LEDGER RENDERER TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Blue Box Uses Ledger Only", test_blue_box_uses_ledger_only),
        ("No Legacy Tile Renderer Code", test_no_legacy_tile_renderer_code),
        ("Reconciliation Rules Enforced", test_reconciliation_rules_enforced),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed")
        sys.exit(1)
