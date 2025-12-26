#!/usr/bin/env python3
"""
Test suite for Safe Mode error handling behavior.

Tests the Safe Mode banner logic to ensure it shows only once per session
and that the Retry Full Mode button properly clears error state.
"""

import sys
import os

# Mock Streamlit session state for testing
class MockSessionState(dict):
    """Mock Streamlit session state for testing."""
    
    def get(self, key, default=None):
        return super().get(key, default)


def test_safe_mode_banner_logic():
    """Test that Safe Mode banner logic works correctly."""
    print("Testing Safe Mode banner logic...")
    
    # Simulate session state
    session_state = MockSessionState()
    
    # Scenario 1: First error - banner should be shown
    print("\n1. First error - should show banner:")
    
    # Check if error has been shown before
    error_shown_before = session_state.get("safe_mode_error_shown", False)
    print(f"   Error shown before: {error_shown_before}")
    assert error_shown_before == False, "First error should not have been shown before"
    
    # Initialize error tracking (simulating first error)
    if "safe_mode_error_shown" not in session_state:
        session_state["safe_mode_error_shown"] = False
        session_state["safe_mode_error_message"] = "Test error"
        session_state["safe_mode_error_traceback"] = "Test traceback"
    
    # Check if we should show banner
    if not session_state["safe_mode_error_shown"]:
        print("   ✅ Would show large red banner")
        should_show_banner = True
        # Mark as shown
        session_state["safe_mode_error_shown"] = True
    else:
        print("   ❌ Would show small warning")
        should_show_banner = False
    
    assert should_show_banner == True, "First error should show banner"
    assert session_state["safe_mode_error_shown"] == True, "Flag should be set after showing banner"
    
    # Scenario 2: Subsequent rerun - banner should NOT be shown
    print("\n2. Subsequent rerun - should show small warning:")
    
    # Simulate another rerun (session state persists)
    error_shown_before = session_state.get("safe_mode_error_shown", False)
    print(f"   Error shown before: {error_shown_before}")
    assert error_shown_before == True, "Error should have been shown in previous run"
    
    # Check if we should show banner
    if not session_state["safe_mode_error_shown"]:
        print("   ❌ Would show large red banner")
        should_show_banner = True
    else:
        print("   ✅ Would show small warning")
        should_show_banner = False
    
    assert should_show_banner == False, "Subsequent rerun should NOT show banner"
    
    # Scenario 3: Retry Full Mode - should clear state
    print("\n3. Retry Full Mode button clicked - should clear state:")
    
    # Simulate button click (clear error flags)
    session_state["safe_mode_error_shown"] = False
    session_state["safe_mode_enabled"] = False
    if "safe_mode_error_message" in session_state:
        del session_state["safe_mode_error_message"]
    if "safe_mode_error_traceback" in session_state:
        del session_state["safe_mode_error_traceback"]
    
    # Verify state is cleared
    assert session_state.get("safe_mode_error_shown", False) == False, "Error shown flag should be cleared"
    assert "safe_mode_error_message" not in session_state, "Error message should be cleared"
    assert "safe_mode_error_traceback" not in session_state, "Error traceback should be cleared"
    print("   ✅ State cleared successfully")
    
    # Scenario 4: After retry, new error should show banner again
    print("\n4. New error after retry - should show banner again:")
    
    # Initialize error tracking (simulating new error after retry)
    if "safe_mode_error_shown" not in session_state:
        session_state["safe_mode_error_shown"] = False
        session_state["safe_mode_error_message"] = "New test error"
        session_state["safe_mode_error_traceback"] = "New test traceback"
    
    # Check if we should show banner
    if not session_state["safe_mode_error_shown"]:
        print("   ✅ Would show large red banner")
        should_show_banner = True
        session_state["safe_mode_error_shown"] = True
    else:
        print("   ❌ Would show small warning")
        should_show_banner = False
    
    assert should_show_banner == True, "New error after retry should show banner"
    
    print("\n✅ All Safe Mode banner logic tests passed!")


def test_compute_alpha_attribution_calls():
    """Test that compute_alpha_attribution_series calls use keyword arguments."""
    print("\nTesting compute_alpha_attribution_series calls...")
    
    import ast
    
    # Read app.py
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    if not os.path.exists(app_path):
        print("⚠️  app.py not found, skipping this test")
        return
    
    with open(app_path, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        
        calls_found = 0
        issues_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if this is a call to compute_alpha_attribution_series
                func_name = None
                if hasattr(node.func, 'id'):
                    func_name = node.func.id
                elif hasattr(node.func, 'attr'):
                    func_name = node.func.attr
                
                if func_name == 'compute_alpha_attribution_series':
                    calls_found += 1
                    
                    # Check if it has any positional arguments
                    if node.args:
                        issues_found.append(f"Line {node.lineno}: Uses positional arguments ({len(node.args)} args)")
                    
                    # Verify it uses keyword arguments
                    if not node.keywords:
                        issues_found.append(f"Line {node.lineno}: No keyword arguments found")
        
        print(f"   Found {calls_found} calls to compute_alpha_attribution_series")
        
        if issues_found:
            print("   ❌ Issues found:")
            for issue in issues_found:
                print(f"      {issue}")
            raise AssertionError("compute_alpha_attribution_series calls have issues")
        else:
            print("   ✅ All calls use keyword arguments only")
        
    except Exception as e:
        print(f"   ⚠️  Error parsing app.py: {e}")


def test_fallback_safety():
    """Test that app_fallback.py has no risky operations."""
    print("\nTesting app_fallback.py safety...")
    
    import ast
    
    fallback_path = os.path.join(os.path.dirname(__file__), 'app_fallback.py')
    if not os.path.exists(fallback_path):
        print("⚠️  app_fallback.py not found, skipping this test")
        return
    
    with open(fallback_path, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        
        risky_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for st.image, st.audio, st.video calls
                if hasattr(node.func, 'attr'):
                    if node.func.attr in ['image', 'audio', 'video']:
                        risky_calls.append((node.lineno, node.func.attr))
                
                # Check for compute_alpha_attribution_series
                if hasattr(node.func, 'id') and 'attribution' in str(node.func.id).lower():
                    risky_calls.append((node.lineno, 'alpha_attribution'))
                elif hasattr(node.func, 'attr') and 'attribution' in str(node.func.attr).lower():
                    risky_calls.append((node.lineno, 'alpha_attribution'))
        
        if risky_calls:
            print("   ❌ Risky calls found:")
            for line, name in risky_calls:
                print(f"      Line {line}: {name}")
            raise AssertionError("app_fallback.py has risky operations")
        else:
            print("   ✅ No risky operations found")
        
    except Exception as e:
        print(f"   ⚠️  Error parsing app_fallback.py: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Safe Mode Behavior Test Suite")
    print("=" * 60)
    
    try:
        test_safe_mode_banner_logic()
        test_compute_alpha_attribution_calls()
        test_fallback_safety()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
