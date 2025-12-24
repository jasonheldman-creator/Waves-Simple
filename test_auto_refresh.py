"""
Test script for Auto-Refresh feature configuration and components.
Validates that the Auto-Refresh feature is properly configured.
"""

import sys
import importlib.util

def test_autorefresh_dependency():
    """Test that streamlit-autorefresh is installed."""
    print("Testing streamlit-autorefresh dependency...")
    try:
        from streamlit_autorefresh import st_autorefresh
        print("✅ streamlit-autorefresh is installed and importable")
        return True
    except ImportError as e:
        print(f"❌ Failed to import streamlit-autorefresh: {e}")
        return False

def test_app_imports():
    """Test that app.py can be imported without errors."""
    print("\nTesting app.py imports...")
    try:
        # Import app module
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app = importlib.util.module_from_spec(spec)
        
        # Load the module to execute top-level code
        spec.loader.exec_module(app)
        
        # Check for AUTO_REFRESH_CONFIG
        if hasattr(app, 'AUTO_REFRESH_CONFIG'):
            config = app.AUTO_REFRESH_CONFIG
            print("✅ AUTO_REFRESH_CONFIG found in app.py")
            print(f"   Default enabled: {config.get('default_enabled')}")
            print(f"   Default interval: {config.get('default_interval_seconds')}s")
            print(f"   Allowed intervals: {config.get('allowed_intervals')}")
            print(f"   Pause on error: {config.get('pause_on_error')}")
            print(f"   Max errors: {config.get('max_consecutive_errors')}")
            return True
        else:
            print("❌ AUTO_REFRESH_CONFIG not found in app.py")
            return False
            
    except Exception as e:
        print(f"❌ Failed to import app.py: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_values():
    """Test that configuration values are correct."""
    print("\nTesting configuration values...")
    try:
        spec = importlib.util.spec_from_file_location("app", "app.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        
        config = app.AUTO_REFRESH_CONFIG
        
        # Test default enabled is True
        assert config['default_enabled'] == True, "default_enabled should be True"
        print("✅ Default enabled is True")
        
        # Test default interval is 60 seconds
        assert config['default_interval_seconds'] == 60, "default_interval_seconds should be 60"
        print("✅ Default interval is 60 seconds")
        
        # Test allowed intervals contains 30, 60, 120
        expected_intervals = [30, 60, 120]
        assert config['allowed_intervals'] == expected_intervals, f"allowed_intervals should be {expected_intervals}"
        print(f"✅ Allowed intervals are {expected_intervals}")
        
        # Test pause_on_error is True
        assert config['pause_on_error'] == True, "pause_on_error should be True"
        print("✅ Pause on error is enabled")
        
        # Test max_consecutive_errors is 3
        assert config['max_consecutive_errors'] == 3, "max_consecutive_errors should be 3"
        print("✅ Max consecutive errors is 3")
        
        return True
        
    except AssertionError as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

def test_requirements():
    """Test that requirements.txt includes streamlit-autorefresh."""
    print("\nTesting requirements.txt...")
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            
        if 'streamlit-autorefresh' in requirements:
            print("✅ streamlit-autorefresh found in requirements.txt")
            return True
        else:
            print("❌ streamlit-autorefresh not found in requirements.txt")
            return False
            
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def test_documentation():
    """Test that AUTO_REFRESH_DOCUMENTATION.md exists."""
    print("\nTesting documentation...")
    try:
        import os
        if os.path.exists('AUTO_REFRESH_DOCUMENTATION.md'):
            print("✅ AUTO_REFRESH_DOCUMENTATION.md exists")
            
            # Check file size
            size = os.path.getsize('AUTO_REFRESH_DOCUMENTATION.md')
            if size > 1000:  # Should be substantial documentation
                print(f"✅ Documentation file size: {size} bytes")
                return True
            else:
                print(f"⚠️ Documentation file seems small: {size} bytes")
                return False
        else:
            print("❌ AUTO_REFRESH_DOCUMENTATION.md not found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking documentation: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Auto-Refresh Feature Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Dependency Check", test_autorefresh_dependency()))
    results.append(("App Import", test_app_imports()))
    results.append(("Configuration Values", test_config_values()))
    results.append(("Requirements File", test_requirements()))
    results.append(("Documentation", test_documentation()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    # Exit with appropriate code
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
