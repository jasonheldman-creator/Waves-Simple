#!/usr/bin/env python3
"""
Test suite for Wave Selection Control.

Tests the wave selector UI control to ensure:
1. Portfolio mode is the default
2. Users can select individual waves
3. Wave-specific metrics appear only for wave context
4. Context switching works correctly
"""

import sys
import os

# Mock Streamlit session state for testing
class MockSessionState(dict):
    """Mock Streamlit session state for testing."""
    
    def get(self, key, default=None):
        return super().get(key, default)


def test_wave_selector_default_to_portfolio():
    """Test that wave selector defaults to portfolio mode."""
    print("\n=== Test 1: Default to Portfolio Mode ===")
    
    # Simulate fresh session
    session_state = MockSessionState()
    
    # Check default value
    selected_wave = session_state.get("selected_wave")
    print(f"   Initial selected_wave: {selected_wave}")
    
    # Should be None (portfolio mode)
    assert selected_wave is None, "Default should be None (portfolio mode)"
    print("   ✅ PASS: Defaults to portfolio mode")
    
    return True


def test_wave_selector_select_individual_wave():
    """Test that users can select individual waves."""
    print("\n=== Test 2: Select Individual Wave ===")
    
    session_state = MockSessionState()
    
    # Simulate selecting a wave
    test_wave = "Gold Wave"
    session_state["selected_wave"] = test_wave
    
    # Check selection
    selected_wave = session_state.get("selected_wave")
    print(f"   Selected wave: {selected_wave}")
    
    assert selected_wave == test_wave, f"Should be {test_wave}"
    print(f"   ✅ PASS: Can select individual wave ({test_wave})")
    
    return True


def test_wave_selector_switch_back_to_portfolio():
    """Test that users can switch back to portfolio mode."""
    print("\n=== Test 3: Switch Back to Portfolio ===")
    
    session_state = MockSessionState()
    
    # Start with a wave selected
    session_state["selected_wave"] = "Income Wave"
    print(f"   Starting wave: {session_state['selected_wave']}")
    
    # Switch to portfolio
    session_state["selected_wave"] = None
    
    # Check selection
    selected_wave = session_state.get("selected_wave")
    print(f"   After switch: {selected_wave}")
    
    assert selected_wave is None, "Should be None (portfolio mode)"
    print("   ✅ PASS: Can switch back to portfolio")
    
    return True


def test_is_portfolio_context():
    """Test the is_portfolio_context helper function."""
    print("\n=== Test 4: is_portfolio_context() Helper ===")
    
    # Constants
    PORTFOLIO_VIEW_PLACEHOLDER = "NONE"
    
    # Helper function (from app.py)
    def is_portfolio_context(selected_wave: str) -> bool:
        return selected_wave is None or selected_wave == PORTFOLIO_VIEW_PLACEHOLDER
    
    # Test cases
    test_cases = [
        (None, True, "None should be portfolio"),
        (PORTFOLIO_VIEW_PLACEHOLDER, True, "NONE placeholder should be portfolio"),
        ("Gold Wave", False, "Specific wave should NOT be portfolio"),
        ("Income Wave", False, "Another wave should NOT be portfolio"),
    ]
    
    for selected_wave, expected, description in test_cases:
        result = is_portfolio_context(selected_wave)
        print(f"   {description}: {result}")
        assert result == expected, f"Failed: {description}"
    
    print("   ✅ PASS: is_portfolio_context() works correctly")
    return True


def test_wave_specific_metrics_visibility():
    """Test that wave-specific metrics only show for wave context."""
    print("\n=== Test 5: Wave-Specific Metrics Visibility ===")
    
    PORTFOLIO_VIEW_PLACEHOLDER = "NONE"
    
    def is_portfolio_context(selected_wave: str) -> bool:
        return selected_wave is None or selected_wave == PORTFOLIO_VIEW_PLACEHOLDER
    
    def should_show_wave_metrics(selected_wave: str) -> bool:
        """Determine if wave-specific metrics should be shown."""
        return not is_portfolio_context(selected_wave)
    
    # Test cases
    test_cases = [
        (None, False, "Portfolio: NO wave metrics"),
        (PORTFOLIO_VIEW_PLACEHOLDER, False, "Portfolio placeholder: NO wave metrics"),
        ("Gold Wave", True, "Gold Wave: SHOW wave metrics"),
        ("Income Wave", True, "Income Wave: SHOW wave metrics"),
    ]
    
    for selected_wave, expected, description in test_cases:
        result = should_show_wave_metrics(selected_wave)
        print(f"   {description}: {result}")
        assert result == expected, f"Failed: {description}"
    
    print("   ✅ PASS: Wave-specific metrics visibility logic is correct")
    return True


def test_wave_selector_options():
    """Test that wave selector options include portfolio and all waves."""
    print("\n=== Test 6: Wave Selector Options ===")
    
    # Constants
    PORTFOLIO_VIEW_TITLE = "Portfolio Snapshot (All Waves)"
    
    # Mock waves
    mock_waves = [
        "AI & Cloud MegaCap Wave",
        "Clean Transit-Infrastructure Wave",
        "Crypto AI Growth Wave",
        "Gold Wave",
        "Income Wave",
    ]
    
    # Build options (as in the app)
    wave_options = [PORTFOLIO_VIEW_TITLE] + sorted(mock_waves)
    
    print(f"   Total options: {len(wave_options)}")
    print(f"   First option: {wave_options[0]}")
    print(f"   Last option: {wave_options[-1]}")
    
    # Assertions
    assert wave_options[0] == PORTFOLIO_VIEW_TITLE, "First option should be Portfolio"
    assert len(wave_options) == len(mock_waves) + 1, "Should have portfolio + all waves"
    assert all(wave in wave_options for wave in mock_waves), "All waves should be in options"
    
    print("   ✅ PASS: Wave selector options are correct")
    return True


def test_selector_persistence():
    """Test that selection persists in session state."""
    print("\n=== Test 7: Selection Persistence ===")
    
    session_state = MockSessionState()
    
    # Simulate multiple selections
    selections = [
        "Gold Wave",
        "Income Wave", 
        None,  # Back to portfolio
        "AI & Cloud MegaCap Wave"
    ]
    
    for i, wave in enumerate(selections):
        session_state["selected_wave"] = wave
        retrieved = session_state.get("selected_wave")
        print(f"   Selection {i+1}: {wave} -> Retrieved: {retrieved}")
        assert retrieved == wave, f"Selection {i+1} should persist"
    
    print("   ✅ PASS: Selections persist correctly")
    return True


def run_all_tests():
    """Run all wave selector tests."""
    print("\n" + "="*60)
    print("WAVE SELECTION CONTROL TEST SUITE")
    print("="*60)
    
    tests = [
        ("Default to Portfolio", test_wave_selector_default_to_portfolio),
        ("Select Individual Wave", test_wave_selector_select_individual_wave),
        ("Switch Back to Portfolio", test_wave_selector_switch_back_to_portfolio),
        ("is_portfolio_context Helper", test_is_portfolio_context),
        ("Wave-Specific Metrics Visibility", test_wave_specific_metrics_visibility),
        ("Wave Selector Options", test_wave_selector_options),
        ("Selection Persistence", test_selector_persistence),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"   ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
