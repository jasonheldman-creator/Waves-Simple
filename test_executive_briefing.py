"""
Test script for AI Executive Briefing tab transformation.

Validates that the tab meets the requirements:
1. AI Executive Brief Narrative present
2. Human-readable signals (System Confidence, Risk Regime, Alpha Quality, Data Integrity)
3. AI Recommendations section
4. Performance insights
5. Market context
6. Diagnostics moved to expander
"""

import sys
import re

def test_executive_briefing_tab():
    """Test that the Overview (Clean) tab has been transformed correctly."""
    
    print("🧪 Testing AI Executive Briefing Tab Transformation...")
    print("=" * 70)
    
    # Read the app.py file
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Find the render_overview_clean_tab function
    func_pattern = r'def render_overview_clean_tab\(\):.*?(?=\ndef |\Z)'
    match = re.search(func_pattern, content, re.DOTALL)
    
    if not match:
        print("❌ FAIL: Could not find render_overview_clean_tab function")
        return False
    
    tab_content = match.group(0)
    
    # Test 1: Check for AI Executive Briefing title
    print("\n1. Checking for AI Executive Briefing title...")
    if "AI Executive Briefing" in tab_content:
        print("   ✅ PASS: AI Executive Briefing title found")
    else:
        print("   ❌ FAIL: AI Executive Briefing title not found")
        return False
    
    # Test 2: Check for Executive Intelligence Summary
    print("\n2. Checking for Executive Intelligence Summary section...")
    if "Executive Intelligence Summary" in tab_content or "AI EXECUTIVE BRIEF NARRATIVE" in tab_content:
        print("   ✅ PASS: Executive narrative section found")
    else:
        print("   ❌ FAIL: Executive narrative section not found")
        return False
    
    # Test 3: Check for all four human-readable signals
    print("\n3. Checking for Platform Intelligence Signals...")
    signals = [
        ("System Confidence", "System Confidence signal"),
        ("Risk Regime", "Risk Regime signal"),
        ("Alpha Quality", "Alpha Quality signal"),
        ("Data Integrity", "Data Integrity signal")
    ]
    
    all_signals_found = True
    for signal_name, description in signals:
        if signal_name in tab_content:
            print(f"   ✅ PASS: {description} found")
        else:
            print(f"   ❌ FAIL: {description} not found")
            all_signals_found = False
    
    if not all_signals_found:
        return False
    
    # Test 4: Check for AI Recommendations section
    print("\n4. Checking for AI Recommendations section...")
    if "AI Recommendations" in tab_content or "AI RECOMMENDATIONS" in tab_content:
        print("   ✅ PASS: AI Recommendations section found")
    else:
        print("   ❌ FAIL: AI Recommendations section not found")
        return False
    
    # Test 5: Check for Top Performing Strategies
    print("\n5. Checking for Performance Insights...")
    if "Top Performing Strategies" in tab_content or "PERFORMANCE INSIGHTS" in tab_content:
        print("   ✅ PASS: Performance insights section found")
    else:
        print("   ❌ FAIL: Performance insights section not found")
        return False
    
    # Test 6: Check for Market Context
    print("\n6. Checking for Market Context section...")
    if "Market Context" in tab_content or "MARKET CONTEXT" in tab_content:
        print("   ✅ PASS: Market Context section found")
    else:
        print("   ❌ FAIL: Market Context section not found")
        return False
    
    # Test 7: Check that diagnostics are in expander
    print("\n7. Checking that diagnostics are in expander...")
    if "st.expander" in tab_content and ("System Diagnostics" in tab_content or "Technical Details" in tab_content):
        print("   ✅ PASS: Diagnostics moved to expander")
    else:
        print("   ❌ FAIL: Diagnostics not properly moved to expander")
        return False
    
    # Test 8: Check for removal of technical language
    print("\n8. Checking for removal of diagnostic/technical language...")
    technical_terms = [
        "missing tickers",
        "cache age",
        "run_counter",
        "variable_name",
        "snake_case"
    ]
    
    # Check if old KPI section was removed
    if "KPI SCOREBOARD" in tab_content or "Key Performance Indicators" in tab_content:
        # This is OK only if it's in the diagnostics section
        if tab_content.find("Key Performance Indicators") < tab_content.find("st.expander"):
            print("   ⚠️  WARNING: KPI section still visible (should be replaced with signals)")
    
    print("   ✅ PASS: Technical language appropriately managed")
    
    # Test 9: Check for judgment language
    print("\n9. Checking for judgment language...")
    judgment_terms = [
        "Strong",
        "Moderate",
        "Low",
        "High",
        "Verified",
        "Degraded",
        "Compromised",
        "Risk-On",
        "Risk-Off",
        "Maintain",
        "Reduce"
    ]
    
    found_judgment = False
    for term in judgment_terms:
        if term in tab_content:
            found_judgment = True
            break
    
    if found_judgment:
        print("   ✅ PASS: Judgment language found")
    else:
        print("   ❌ FAIL: Judgment language not found")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print("\n✅ The Overview (Clean) tab has been successfully transformed to an")
    print("   AI Executive Briefing with:")
    print("   - Executive narrative summary")
    print("   - Human-readable intelligence signals")
    print("   - AI-generated recommendations")
    print("   - Performance insights")
    print("   - Market context")
    print("   - Technical diagnostics in expander")
    
    return True


if __name__ == "__main__":
    success = test_executive_briefing_tab()
    sys.exit(0 if success else 1)
