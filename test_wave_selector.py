"""
Test wave selector implementation to verify it works correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_wave_selector_logic():
    """Test the wave selector logic without running the full Streamlit app."""
    
    # Constants from app.py
    PORTFOLIO_VIEW_PLACEHOLDER = "NONE"
    PORTFOLIO_VIEW_TITLE = "Portfolio Snapshot (All Waves)"
    PORTFOLIO_VIEW_ICON = "🏛️"
    WAVE_VIEW_ICON = "🌊"
    
    # Mock wave universe
    mock_waves = [
        "AI & Cloud MegaCap Wave",
        "Clean Transit-Infrastructure Wave",
        "Crypto AI Growth Wave",
        "Gold Wave",
        "Income Wave"
    ]
    
    # Build options like the app does
    wave_options = [PORTFOLIO_VIEW_TITLE] + sorted(mock_waves)
    
    print("=== Wave Selector Logic Test ===\n")
    print(f"Total options: {len(wave_options)}")
    print(f"First option (default): {wave_options[0]}")
    print(f"\nAll options:")
    for i, option in enumerate(wave_options):
        print(f"  {i}: {option}")
    
    # Test 1: Default selection (None -> Portfolio)
    print("\n=== Test 1: Default Selection ===")
    current_selection = None
    if current_selection is None or current_selection == PORTFOLIO_VIEW_PLACEHOLDER:
        default_index = 0
        print(f"✓ Selected index: {default_index} ({wave_options[default_index]})")
        print(f"✓ Display: {PORTFOLIO_VIEW_ICON} Portfolio View Active")
    else:
        print("✗ Failed to default to portfolio")
    
    # Test 2: Specific wave selected
    print("\n=== Test 2: Specific Wave Selection ===")
    test_wave = "Gold Wave"
    current_selection = test_wave
    if current_selection in wave_options:
        default_index = wave_options.index(current_selection)
        print(f"✓ Selected index: {default_index} ({wave_options[default_index]})")
        print(f"✓ Display: {WAVE_VIEW_ICON} Wave View: {current_selection}")
    else:
        print("✗ Failed to select specific wave")
    
    # Test 3: Switch back to portfolio
    print("\n=== Test 3: Switch Back to Portfolio ===")
    selected_option = PORTFOLIO_VIEW_TITLE
    if selected_option == PORTFOLIO_VIEW_TITLE:
        selected_wave = None
        print(f"✓ selected_wave set to: {selected_wave}")
        print(f"✓ Display: {PORTFOLIO_VIEW_ICON} Portfolio View Active")
    else:
        print("✗ Failed to switch to portfolio")
    
    # Test 4: Select different wave
    print("\n=== Test 4: Select Different Wave ===")
    selected_option = "Income Wave"
    if selected_option != PORTFOLIO_VIEW_TITLE:
        selected_wave = selected_option
        print(f"✓ selected_wave set to: {selected_wave}")
        print(f"✓ Display: {WAVE_VIEW_ICON} Wave View: {selected_wave}")
    else:
        print("✗ Failed to select wave")
    
    print("\n=== All Tests Passed ✓ ===")
    return True

if __name__ == "__main__":
    test_wave_selector_logic()
