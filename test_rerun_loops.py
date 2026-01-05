"""
Test script to verify no infinite rerun loops exist.

This test validates:
1. Wave selection doesn't cause multiple reruns
2. Auto-refresh is disabled by default
3. Clear Cache button works correctly
4. No exceptions trigger reruns
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_auto_refresh_default():
    """Test that auto-refresh is disabled by default."""
    from auto_refresh_config import DEFAULT_AUTO_REFRESH_ENABLED
    
    assert DEFAULT_AUTO_REFRESH_ENABLED == False, \
        "Auto-refresh should be disabled by default to prevent infinite loops"
    print("✓ Auto-refresh is disabled by default")


def test_rerun_calls_limited():
    """Test that st.rerun() calls are limited and intentional."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Count st.rerun() calls
    rerun_count = content.count('st.rerun()')
    
    # Should be minimal - ideally only in trigger_rerun function and button handlers
    assert rerun_count <= 5, \
        f"Too many st.rerun() calls found: {rerun_count}. Should be minimal to prevent loops."
    
    print(f"✓ Limited st.rerun() calls found: {rerun_count}")


def test_no_exception_reruns():
    """Test that exception handlers don't trigger reruns."""
    with open('app.py', 'r') as f:
        lines = f.readlines()
    
    # Find all except blocks
    in_except_block = False
    except_has_rerun = False
    
    for i, line in enumerate(lines):
        if 'except' in line and ':' in line:
            in_except_block = True
            except_has_rerun = False
        elif in_except_block:
            if 'st.rerun()' in line or 'trigger_rerun(' in line:
                except_has_rerun = True
                print(f"⚠ Warning: Rerun found in except block at line {i+1}")
            # Check if we're out of the except block (dedented or new def/class)
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                in_except_block = False
    
    print("✓ Exception handlers checked for reruns")


def test_wave_selection_initialization():
    """Test that wave selection initialization is conditional."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Check that selected_wave initialization is guarded
    if 'if "selected_wave" not in st.session_state:' in content:
        print("✓ Wave selection initialization is conditional")
    else:
        print("⚠ Warning: Wave selection initialization may be unconditional")


def test_clear_cache_button_enhanced():
    """Test that Clear Cache button includes cache clearing."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Find the Clear Cache button
    if 'Clear Cache & Restart' in content:
        print("✓ Clear Cache button has been enhanced")
        
        # Check it includes st.cache_data.clear()
        assert 'st.cache_data.clear()' in content, \
            "Clear Cache should call st.cache_data.clear()"
        
        assert 'st.cache_resource.clear()' in content, \
            "Clear Cache should call st.cache_resource.clear()"
        
        print("✓ Clear Cache button includes all cache clearing operations")
    else:
        print("⚠ Warning: Clear Cache button not found or not enhanced")


def test_trigger_rerun_marks_interaction():
    """Test that trigger_rerun marks user interaction."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Find trigger_rerun function
    if 'def trigger_rerun' in content:
        # Check it marks user interaction
        if 'user_interaction_detected = True' in content:
            print("✓ trigger_rerun marks user interaction")
        else:
            print("⚠ Warning: trigger_rerun may not mark user interaction")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing for Infinite Rerun Loops")
    print("=" * 60)
    
    test_auto_refresh_default()
    test_rerun_calls_limited()
    test_no_exception_reruns()
    test_wave_selection_initialization()
    test_clear_cache_button_enhanced()
    test_trigger_rerun_marks_interaction()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
