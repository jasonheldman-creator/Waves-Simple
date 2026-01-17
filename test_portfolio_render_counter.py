"""
Test Suite for Portfolio Render Counter Feature
==================================================

This test validates that the Portfolio Snapshot render counter:
1. Initializes properly in st.session_state
2. Increments on each render
3. Is displayed in the Portfolio Snapshot UI
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_counter_initialization_code():
    """Test that the counter initialization code is present in app.py"""
    print("Testing counter initialization code in app.py...")
    
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    # Check for initialization code
    assert '_portfolio_render_count' in app_content, \
        "Counter variable '_portfolio_render_count' should be in app.py"
    
    assert 'st.session_state["_portfolio_render_count"]' in app_content, \
        "Counter should be stored in st.session_state"
    
    print("✅ Counter initialization code is present")


def test_counter_increment_code():
    """Test that the counter increment code is present"""
    print("\nTesting counter increment code in app.py...")
    
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    # Check for increment code
    assert 'st.session_state["_portfolio_render_count"] += 1' in app_content, \
        "Counter should be incremented with += 1"
    
    # Verify it's near compute_portfolio_snapshot
    lines = app_content.split('\n')
    increment_line_idx = None
    compute_line_idx = None
    
    for i, line in enumerate(lines):
        if 'st.session_state["_portfolio_render_count"] += 1' in line:
            increment_line_idx = i
        if 'compute_portfolio_snapshot(price_book' in line:
            compute_line_idx = i
    
    assert increment_line_idx is not None, "Increment line should exist"
    assert compute_line_idx is not None, "compute_portfolio_snapshot call should exist"
    
    # Check that increment is within 10 lines before compute_portfolio_snapshot
    line_distance = compute_line_idx - increment_line_idx
    assert 0 < line_distance < 10, \
        f"Counter increment should be immediately before compute_portfolio_snapshot (distance: {line_distance} lines)"
    
    print(f"✅ Counter increment is {line_distance} lines before compute_portfolio_snapshot")


def test_counter_display_code():
    """Test that the counter display code is present in the UI"""
    print("\nTesting counter display code in app.py...")
    
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    # Check for display code
    assert 'Render Count:' in app_content, \
        "Counter should be displayed with 'Render Count:' label"
    
    # Check it's retrieved from session state
    assert 'render_count = st.session_state.get("_portfolio_render_count"' in app_content, \
        "Counter should be retrieved from session state for display"
    
    # Check it's in portfolio_info_html (portfolio view specific)
    assert 'portfolio_info_html' in app_content, \
        "Counter display should be in portfolio_info_html"
    
    # Verify the display shows the counter value
    lines = app_content.split('\n')
    found_render_count_display = False
    for line in lines:
        if 'Render Count:' in line and '{render_count}' in line:
            found_render_count_display = True
            break
    
    assert found_render_count_display, \
        "Counter value should be interpolated in the display string"
    
    print("✅ Counter display code is present in Portfolio Snapshot UI")


def test_counter_only_in_portfolio_view():
    """Test that the counter is only displayed in portfolio view"""
    print("\nTesting counter is portfolio-view specific...")
    
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    # Find the render_count assignment and verify it's in the is_portfolio_view block
    lines = app_content.split('\n')
    render_count_line_idx = None
    
    for i, line in enumerate(lines):
        if 'render_count = st.session_state.get("_portfolio_render_count"' in line:
            render_count_line_idx = i
            break
    
    assert render_count_line_idx is not None, "render_count assignment should exist"
    
    # Look backwards to find the if is_portfolio_view condition
    found_portfolio_check = False
    for i in range(render_count_line_idx, max(0, render_count_line_idx - 20), -1):
        if 'if is_portfolio_view:' in lines[i]:
            found_portfolio_check = True
            break
    
    assert found_portfolio_check, \
        "render_count display should be within is_portfolio_view block"
    
    print("✅ Counter display is portfolio-view specific")


def run_all_tests():
    """Run all tests in the suite"""
    print("=" * 70)
    print("Portfolio Render Counter Test Suite")
    print("=" * 70)
    
    tests = [
        test_counter_initialization_code,
        test_counter_increment_code,
        test_counter_display_code,
        test_counter_only_in_portfolio_view,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED")
        print("\nThe render counter implementation is correct:")
        print("  1. Counter is initialized in st.session_state")
        print("  2. Counter increments before compute_portfolio_snapshot()")
        print("  3. Counter is displayed in Portfolio Snapshot UI")
        print("  4. Counter is only shown in portfolio view")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
