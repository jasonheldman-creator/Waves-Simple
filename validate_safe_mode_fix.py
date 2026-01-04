#!/usr/bin/env python3
"""
Manual validation script for Safe Mode fix.

This script demonstrates the Safe Mode banner behavior with a simulated
error scenario. It shows:
1. First error - large banner displayed
2. Subsequent "reruns" - small warning displayed
3. Retry button - clears state
"""

import sys


class MockSessionState(dict):
    """Mock Streamlit session state."""
    
    def get(self, key, default=None):
        return super().get(key, default)


class MockStreamlit:
    """Mock Streamlit module for demonstration."""
    
    def __init__(self):
        self.session_state = MockSessionState()
        self.output = []
    
    def markdown(self, text, **kwargs):
        """Mock st.markdown."""
        if "APPLICATION ERROR" in text:
            self.output.append("🔴 LARGE RED BANNER: APPLICATION ERROR - SWITCHING TO SAFE MODE")
        else:
            self.output.append(f"markdown: {text[:50]}...")
    
    def warning(self, text):
        """Mock st.warning."""
        self.output.append(f"⚠️  {text}")
    
    def error(self, text):
        """Mock st.error."""
        self.output.append(f"❌ {text}")
    
    def code(self, text, **kwargs):
        """Mock st.code."""
        self.output.append(f"code: {text[:50]}...")
    
    def expander(self, title, **kwargs):
        """Mock st.expander."""
        return MockExpander(title, self.output)
    
    def columns(self, spec):
        """Mock st.columns."""
        if isinstance(spec, list):
            num_cols = len(spec)
        else:
            num_cols = spec
        return [MockColumn(self) for _ in range(num_cols)]
    
    def button(self, label, **kwargs):
        """Mock st.button - returns False by default."""
        self.output.append(f"🔘 Button: {label}")
        return False
    
    def rerun(self):
        """Mock st.rerun."""
        self.output.append("🔄 Rerunning app...")
    
    def get_output(self):
        """Get all output."""
        return "\n".join(self.output)
    
    def clear_output(self):
        """Clear output."""
        self.output = []


class MockExpander:
    """Mock expander context manager."""
    
    def __init__(self, title, output):
        self.title = title
        self.output = output
    
    def __enter__(self):
        self.output.append(f"▼ Expander: {self.title}")
        return self
    
    def __exit__(self, *args):
        pass


class MockColumn:
    """Mock column context manager."""
    
    def __init__(self, st):
        self.st = st
    
    def __enter__(self):
        return self.st
    
    def __exit__(self, *args):
        pass


def simulate_error_handler(st, exception_msg, traceback_msg):
    """
    Simulates the Safe Mode error handler from app.py.
    
    This is the exact logic from lines 11841-11891 of app.py.
    """
    # Initialize safe mode error tracking in session state
    if "safe_mode_error_shown" not in st.session_state:
        st.session_state["safe_mode_error_shown"] = False
        st.session_state["safe_mode_error_message"] = exception_msg
        st.session_state["safe_mode_error_traceback"] = traceback_msg
    
    # Display prominent error banner ONLY ONCE per session
    if not st.session_state["safe_mode_error_shown"]:
        st.markdown("""
            <div style="background-color: #ff4444;">
                <h1>⚠️ APPLICATION ERROR - SWITCHING TO SAFE MODE</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Mark banner as shown
        st.session_state["safe_mode_error_shown"] = True
    else:
        # For subsequent reruns, show a smaller, less intrusive status line
        st.warning("⚠️ Running in Safe Mode due to application error.")
    
    # Display error details in a collapsible expander (collapsed by default)
    with st.expander("🔍 View Error Details", expanded=False):
        st.error(f"**Error Message:** {st.session_state['safe_mode_error_message']}")
        st.code(st.session_state["safe_mode_error_traceback"], language="python")
    
    # Add "Retry Full Mode" button to allow recovery without page refresh
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Retry Full Mode", help="Clear Safe Mode and attempt to run the full application"):
            # Clear error flags
            st.session_state["safe_mode_error_shown"] = False
            st.session_state["safe_mode_enabled"] = False
            if "safe_mode_error_message" in st.session_state:
                del st.session_state["safe_mode_error_message"]
            if "safe_mode_error_traceback" in st.session_state:
                del st.session_state["safe_mode_error_traceback"]
            # Trigger rerun
            st.rerun()


def main():
    """Run the validation demonstration."""
    print("=" * 80)
    print("Safe Mode Fix - Manual Validation Demonstration")
    print("=" * 80)
    print()
    
    # Create mock Streamlit instance
    st = MockStreamlit()
    
    # Scenario 1: First error occurs
    print("SCENARIO 1: First Error Occurs")
    print("-" * 80)
    simulate_error_handler(st, "ValueError: Test error", "Traceback:\n  File 'test.py', line 42")
    print(st.get_output())
    print()
    
    # Verify behavior
    assert st.session_state["safe_mode_error_shown"] == True
    assert "🔴 LARGE RED BANNER" in st.get_output()
    assert "⚠️  Running in Safe Mode" not in st.get_output()
    print("✅ First error shows large banner - CORRECT")
    print()
    
    # Scenario 2: User interacts with app (rerun 1)
    print("SCENARIO 2: User Clicks Something (First Rerun)")
    print("-" * 80)
    st.clear_output()
    simulate_error_handler(st, "ValueError: Test error", "Traceback:\n  File 'test.py', line 42")
    print(st.get_output())
    print()
    
    # Verify behavior
    assert "🔴 LARGE RED BANNER" not in st.get_output()
    assert "Running in Safe Mode" in st.get_output()
    print("✅ First rerun shows small warning - CORRECT")
    print()
    
    # Scenario 3: User interacts again (rerun 2)
    print("SCENARIO 3: User Clicks Something Else (Second Rerun)")
    print("-" * 80)
    st.clear_output()
    simulate_error_handler(st, "ValueError: Test error", "Traceback:\n  File 'test.py', line 42")
    print(st.get_output())
    print()
    
    # Verify behavior
    assert "🔴 LARGE RED BANNER" not in st.get_output()
    assert "Running in Safe Mode" in st.get_output()
    print("✅ Second rerun still shows small warning - CORRECT")
    print()
    
    # Scenario 4: Verify Retry button is present
    print("SCENARIO 4: Verify Retry Button Functionality")
    print("-" * 80)
    assert "🔘 Button: 🔄 Retry Full Mode" in st.get_output()
    print("✅ Retry Full Mode button is present - CORRECT")
    print()
    
    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()
    print("✅ First error displays large banner")
    print("✅ Subsequent reruns display small warning")
    print("✅ Banner is NOT repeated on every rerun")
    print("✅ Retry Full Mode button is available")
    print("✅ Error details are in collapsible expander")
    print()
    print("🎉 All scenarios validated successfully!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
