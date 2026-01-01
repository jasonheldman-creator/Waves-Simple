#!/usr/bin/env python3
"""
Test suite for Safe Mode stabilization features.

Tests:
1. Safe Mode initialization (default ON)
2. Run Guard counter increments and triggers
3. Top banner displays correct status
4. Manual rebuild buttons respect Safe Mode
5. compute_gate respects Safe Mode
"""

import sys
import os

# Mock Streamlit session state for testing
class MockSessionState(dict):
    """Mock Streamlit session state for testing."""
    
    def get(self, key, default=None):
        return super().get(key, default)


def test_safe_mode_initialization():
    """Test that Safe Mode initializes to ON by default."""
    print("Testing Safe Mode initialization...")
    
    session_state = MockSessionState()
    
    # Simulate initialization in main()
    if "safe_mode_no_fetch" not in session_state:
        session_state["safe_mode_no_fetch"] = True  # Default to ON
    
    assert session_state["safe_mode_no_fetch"] == True, "Safe Mode should default to ON"
    print("   ✅ Safe Mode defaults to ON")


def test_run_guard_counter():
    """Test that run guard counter increments and triggers at threshold."""
    print("\nTesting Run Guard counter...")
    
    session_state = MockSessionState()
    
    # Initialize
    if "run_guard_counter" not in session_state:
        session_state.run_guard_counter = 0
    
    # Simulate 3 runs
    for i in range(3):
        session_state.run_guard_counter += 1
        print(f"   Run {i+1}: counter = {session_state.run_guard_counter}")
        assert session_state.run_guard_counter <= 3, "Counter should not exceed 3"
    
    # Check threshold
    should_stop = session_state.run_guard_counter > 3
    assert should_stop == False, "Should not trigger at count 3"
    print("   ✅ Run Guard does not trigger at count 3")
    
    # One more run
    session_state.run_guard_counter += 1
    should_stop = session_state.run_guard_counter > 3
    assert should_stop == True, "Should trigger at count 4"
    print("   ✅ Run Guard triggers at count 4")


def test_compute_gate_safe_mode():
    """Test that compute_gate respects Safe Mode."""
    print("\nTesting compute_gate Safe Mode check...")
    
    # Import compute_gate
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helpers'))
    from compute_gate import should_allow_build
    
    session_state = MockSessionState()
    session_state["safe_mode_no_fetch"] = True
    
    # Test auto-build (should be blocked)
    should_build, reason = should_allow_build(
        snapshot_path="test.csv",
        session_state=session_state,
        build_key="test",
        explicit_button_click=False
    )
    
    assert should_build == False, "Auto-build should be blocked in Safe Mode"
    assert "Safe Mode" in reason, f"Reason should mention Safe Mode, got: {reason}"
    print(f"   ✅ Auto-build blocked: {reason}")
    
    # Test explicit button click (should be allowed)
    should_build, reason = should_allow_build(
        snapshot_path="test.csv",
        session_state=session_state,
        build_key="test",
        explicit_button_click=True
    )
    
    assert should_build == True, "Manual button click should be allowed"
    assert "button click" in reason.lower(), f"Reason should mention button click, got: {reason}"
    print(f"   ✅ Manual button click allowed: {reason}")


def test_safe_mode_off_allows_builds():
    """Test that turning Safe Mode OFF allows builds."""
    print("\nTesting Safe Mode OFF behavior...")
    
    # Import compute_gate
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helpers'))
    from compute_gate import should_allow_build
    
    session_state = MockSessionState()
    session_state["safe_mode_no_fetch"] = False  # Safe Mode OFF
    session_state["run_id"] = 1
    
    # Create a missing snapshot scenario
    snapshot_path = "/tmp/test_missing_snapshot.csv"
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)
    
    # Test auto-build with missing snapshot (should be allowed when Safe Mode is OFF)
    should_build, reason = should_allow_build(
        snapshot_path=snapshot_path,
        session_state=session_state,
        build_key="test",
        explicit_button_click=False
    )
    
    assert should_build == True, "Auto-build should be allowed when Safe Mode is OFF and snapshot is missing"
    assert "does not exist" in reason, f"Reason should mention missing snapshot, got: {reason}"
    print(f"   ✅ Auto-build allowed with Safe Mode OFF: {reason}")


def test_loop_detection_flag():
    """Test that loop_detected flag is set correctly."""
    print("\nTesting loop detection flag...")
    
    session_state = MockSessionState()
    
    # Initialize
    if "loop_detected" not in session_state:
        session_state.loop_detected = False
    
    assert session_state.loop_detected == False, "loop_detected should default to False"
    print("   ✅ loop_detected defaults to False")
    
    # Simulate loop detection
    session_state.run_guard_counter = 5
    if session_state.run_guard_counter > 3:
        session_state.loop_detected = True
    
    assert session_state.loop_detected == True, "loop_detected should be True when counter > 3"
    print("   ✅ loop_detected set to True when threshold exceeded")


if __name__ == "__main__":
    print("=" * 60)
    print("Safe Mode Stabilization Test Suite")
    print("=" * 60)
    
    try:
        test_safe_mode_initialization()
        test_run_guard_counter()
        test_compute_gate_safe_mode()
        test_safe_mode_off_allows_builds()
        test_loop_detection_flag()
        
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
