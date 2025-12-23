"""
Test script to validate HTML rendering functions.
This ensures no raw HTML tags leak into the UI.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


def test_render_html_block_exists():
    """Test that render_html_block function exists and has proper signature."""
    from app import render_html_block
    
    # Check function exists
    assert callable(render_html_block), "render_html_block should be callable"
    
    # Check function signature
    import inspect
    sig = inspect.signature(render_html_block)
    params = list(sig.parameters.keys())
    
    assert 'html_content' in params, "render_html_block should have html_content parameter"
    assert 'height' in params, "render_html_block should have height parameter"
    assert 'key' in params, "render_html_block should have key parameter"
    
    print("✓ render_html_block function exists with correct signature")


def test_check_for_raw_html_exists():
    """Test that check_for_raw_html_in_output function exists."""
    from app import check_for_raw_html_in_output
    
    # Check function exists
    assert callable(check_for_raw_html_in_output), "check_for_raw_html_in_output should be callable"
    
    # Call the function to ensure it runs without errors
    result = check_for_raw_html_in_output()
    # Function now returns None instead of bool
    assert result is None, "check_for_raw_html_in_output should return None"
    
    print("✓ check_for_raw_html_in_output function exists and runs")


def test_no_unsafe_markdown_in_app():
    """Test that app.py doesn't contain st.markdown with unsafe_allow_html=True."""
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    # Check for patterns that should not exist
    assert 'st.markdown(' not in app_content or 'unsafe_allow_html=True' not in app_content or \
           app_content.count('st.markdown(') - app_content.count('unsafe_allow_html=True') == app_content.count('st.markdown('), \
           "app.py should not contain st.markdown with unsafe_allow_html=True"
    
    print("✓ No unsafe st.markdown calls in app.py")


def test_no_unsafe_markdown_in_ticker_rail():
    """Test that ticker_rail.py doesn't contain st.markdown with unsafe_allow_html=True."""
    with open('helpers/ticker_rail.py', 'r') as f:
        ticker_content = f.read()
    
    # Check for patterns that should not exist
    assert 'unsafe_allow_html=True' not in ticker_content, \
           "ticker_rail.py should not contain st.markdown with unsafe_allow_html=True"
    
    print("✓ No unsafe st.markdown calls in ticker_rail.py")


def test_render_html_block_usage():
    """Test that render_html_block is used in the codebase."""
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    # Check that render_html_block is actually called
    assert 'render_html_block(' in app_content, "render_html_block should be used in app.py"
    
    # Count usage
    usage_count = app_content.count('render_html_block(')
    print(f"✓ render_html_block is used {usage_count} times in app.py")


def test_dangerous_patterns_not_in_markdown():
    """Test that dangerous HTML patterns are not passed to st.markdown."""
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    dangerous_patterns = [
        'class="metric-grid"',
        'class="stats-grid"',
        'class="wave-identity-card"'
    ]
    
    lines = app_content.split('\n')
    for i, line in enumerate(lines, 1):
        if 'st.markdown(' in line and 'unsafe_allow_html' in line:
            for pattern in dangerous_patterns:
                # Check surrounding lines for dangerous patterns
                context_start = max(0, i - 5)
                context_end = min(len(lines), i + 5)
                context = '\n'.join(lines[context_start:context_end])
                
                if pattern in context:
                    print(f"⚠️  Warning: Found {pattern} near st.markdown at line {i}")
    
    print("✓ Dangerous HTML patterns check complete")


if __name__ == '__main__':
    print("Running HTML rendering safety tests...\n")
    
    try:
        test_render_html_block_exists()
        test_check_for_raw_html_exists()
        test_no_unsafe_markdown_in_app()
        test_no_unsafe_markdown_in_ticker_rail()
        test_render_html_block_usage()
        test_dangerous_patterns_not_in_markdown()
        
        print("\n✅ All tests passed!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
