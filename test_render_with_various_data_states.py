"""
Test render_adaptive_intelligence_tab with various data states.

This test validates that the function renders correctly even when:
- Data is None
- Data is empty
- Data is missing certain fields
"""

import pandas as pd
import sys
from io import StringIO


def capture_output(func, *args, **kwargs):
    """Capture function output by redirecting stdout."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        func(*args, **kwargs)
        output = sys.stdout.getvalue()
        return output
    finally:
        sys.stdout = old_stdout


def test_render_with_none_data():
    """Test that the function handles None data gracefully."""
    print("Testing render_adaptive_intelligence_tab with None data...")
    
    try:
        import adaptive_intelligence
        
        # Mock Streamlit to avoid runtime errors
        class MockStreamlit:
            @staticmethod
            def header(text):
                print(f"HEADER: {text}")
            
            @staticmethod
            def caption(text):
                print(f"CAPTION: {text}")
            
            @staticmethod
            def markdown(text, **kwargs):
                text_str = str(text) if text is not None else ""
                preview = text_str[:50] if len(text_str) > 50 else text_str
                print(f"MARKDOWN: {preview}...")
            
            @staticmethod
            def warning(text):
                print(f"WARNING: {text}")
            
            @staticmethod
            def info(text):
                print(f"INFO: {text}")
            
            @staticmethod
            def error(text):
                print(f"ERROR: {text}")
            
            @staticmethod
            def success(text):
                print(f"SUCCESS: {text}")
            
            @staticmethod
            def divider():
                print("DIVIDER")
            
            @staticmethod
            def subheader(text):
                print(f"SUBHEADER: {text}")
            
            @staticmethod
            def expander(text, expanded=False):
                return MockExpander()
            
            @staticmethod
            def json(data):
                print(f"JSON: {len(str(data))} chars")
        
        class MockExpander:
            def __enter__(self):
                print("EXPANDER_ENTER")
                return self
            
            def __exit__(self, *args):
                print("EXPANDER_EXIT")
        
        # Monkey patch streamlit
        adaptive_intelligence.st = MockStreamlit()
        from helpers import diagnostics_review_signals
        diagnostics_review_signals.st = MockStreamlit()
        
        # Test with None data
        print("\n=== Test 1: Both None ===")
        adaptive_intelligence.render_adaptive_intelligence_tab(None, None)
        print("✓ Rendered successfully with None data\n")
        
        # Test with empty DataFrames
        print("=== Test 2: Empty DataFrames ===")
        empty_df = pd.DataFrame()
        adaptive_intelligence.render_adaptive_intelligence_tab(empty_df, empty_df)
        print("✓ Rendered successfully with empty DataFrames\n")
        
        # Test with one None
        print("=== Test 3: Mixed None/DataFrame ===")
        adaptive_intelligence.render_adaptive_intelligence_tab(None, empty_df)
        print("✓ Rendered successfully with mixed None/DataFrame\n")
        
        # Test with valid data
        print("=== Test 4: Valid data ===")
        snapshot_df = pd.DataFrame({
            'wave_name': ['Wave1', 'Wave2'],
            'return_30d': [0.05, 0.03]
        })
        attrib_df = pd.DataFrame({
            'wave': ['Wave1', 'Wave2'],
            'alpha': [0.02, 0.01]
        })
        adaptive_intelligence.render_adaptive_intelligence_tab(snapshot_df, attrib_df)
        print("✓ Rendered successfully with valid data\n")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_section_header_always_visible():
    """Test that 'Review & Adaptation Signals' header is always rendered."""
    print("Testing that section header is always visible...")
    
    try:
        from helpers import diagnostics_review_signals
        
        # Mock Streamlit
        class MockStreamlit:
            subheaders = []
            
            @staticmethod
            def subheader(text):
                MockStreamlit.subheaders.append(text)
                print(f"SUBHEADER: {text}")
            
            @staticmethod
            def caption(text):
                print(f"CAPTION: {text}")
            
            @staticmethod
            def info(text):
                print(f"INFO: {text}")
            
            @staticmethod
            def success(text):
                print(f"SUCCESS: {text}")
            
            @staticmethod
            def expander(text, expanded=False):
                class MockExp:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                return MockExp()
            
            @staticmethod
            def json(data):
                pass
        
        diagnostics_review_signals.st = MockStreamlit()
        
        # Test with None data
        print("\n=== Testing with None data ===")
        MockStreamlit.subheaders = []
        diagnostics_review_signals.render_review_and_adaptation_signals(None, None, {})
        
        # Check that header was rendered
        if "Review & Adaptation Signals" in MockStreamlit.subheaders:
            print("✓ Section header 'Review & Adaptation Signals' was rendered\n")
            return True
        else:
            print(f"✗ Section header not found. Headers: {MockStreamlit.subheaders}\n")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*60)
    print("Testing Robust Rendering with Various Data States")
    print("="*60)
    print()
    
    results = []
    results.append(("Render with various data states", test_render_with_none_data()))
    results.append(("Section header always visible", test_section_header_always_visible()))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        exit(1)
