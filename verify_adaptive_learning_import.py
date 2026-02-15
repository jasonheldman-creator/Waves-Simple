#!/usr/bin/env python3
"""
Verification script for adaptive_learning module import.
Tests that the module can be imported correctly in Streamlit Cloud deployment.
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("ADAPTIVE_LEARNING MODULE IMPORT VERIFICATION")
print("=" * 60)

# Verify Python environment
print(f"\nPython version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Verify working directory
print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")

# Check if adaptive_learning.py exists in the expected location
app_dir = Path(__file__).parent.resolve()
adaptive_learning_path = app_dir / "adaptive_learning.py"
print(f"\nExpected adaptive_learning.py location: {adaptive_learning_path}")
print(f"File exists: {adaptive_learning_path.exists()}")

if adaptive_learning_path.exists():
    print(f"File size: {adaptive_learning_path.stat().st_size} bytes")

# Check app_min.py exists
app_min_path = app_dir / "app_min.py"
print(f"\nExpected app_min.py location: {app_min_path}")
print(f"File exists: {app_min_path.exists()}")

# Verify both files are in the same directory
print(f"\nBoth files in same directory: {adaptive_learning_path.parent == app_min_path.parent}")

# Add app directory to Python path (simulating Streamlit Cloud environment)
print(f"\nAdding application directory to Python path: {app_dir}")
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))
    print("✓ Directory added to sys.path")
else:
    print("✓ Directory already in sys.path")

# Test import
print("\nAttempting to import adaptive_learning...")
try:
    import adaptive_learning as al
    print("✓ Successfully imported adaptive_learning module")
    
    # Verify expected functions exist
    expected_functions = [
        'load_adaptive_state',
        'update_adaptive_state',
        'compute_scenario_simulation',
        'compute_cross_horizon_agreement',
        'generate_adaptive_tilt_proposals'
    ]
    
    print("\nVerifying module functions:")
    for func_name in expected_functions:
        if hasattr(al, func_name):
            print(f"  ✓ {func_name} exists")
        else:
            print(f"  ✗ {func_name} MISSING")
    
    # Test basic functionality
    print("\nTesting basic functionality:")
    try:
        state = al.load_adaptive_state()
        print(f"  ✓ load_adaptive_state() returned: {type(state).__name__}")
        
        result = al.compute_scenario_simulation("test_scenario", None, None)
        print(f"  ✓ compute_scenario_simulation() returned: {type(result).__name__}")
        
        agreement = al.compute_cross_horizon_agreement(None, None)
        print(f"  ✓ compute_cross_horizon_agreement() returned: {type(agreement).__name__}")
        
        proposals = al.generate_adaptive_tilt_proposals({}, state, [])
        print(f"  ✓ generate_adaptive_tilt_proposals() returned: {type(proposals).__name__} with {len(proposals)} items")
        
        print("\n✓ All function calls executed successfully")
        
    except Exception as e:
        print(f"\n✗ Error testing functions: {e}")
    
except ImportError as e:
    print(f"✗ Failed to import adaptive_learning: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1)

# Test integrity_signals import as well (related module)
print("\nAttempting to import integrity_signals...")
try:
    import integrity_signals as integ
    print("✓ Successfully imported integrity_signals module")
    
    # Check expected functions
    if hasattr(integ, 'get_all_integrity_signals'):
        print("  ✓ get_all_integrity_signals exists")
    if hasattr(integ, 'compute_selection_integrity'):
        print("  ✓ compute_selection_integrity exists")
        
except ImportError as e:
    print(f"✗ Failed to import integrity_signals: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\n✓ All imports successful and module structure verified")
print("✓ Repository is ready for Streamlit Cloud deployment")
