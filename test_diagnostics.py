#!/usr/bin/env python3
"""
Test suite for diagnostics module

Tests that the diagnostics module imports cleanly and prevents syntax errors.
"""

import sys


def test_diagnostics_import():
    """Test that diagnostics.data_contact module imports without syntax errors."""
    print("Testing diagnostics.data_contact import...")
    
    try:
        import diagnostics.data_contact
        print("✅ diagnostics.data_contact imported successfully")
        return True
        
    except SyntaxError as e:
        print(f"❌ SyntaxError when importing diagnostics.data_contact: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"❌ Error when importing diagnostics.data_contact: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_diagnostics_classes():
    """Test that diagnostics classes can be instantiated."""
    print("\nTesting diagnostics classes...")
    
    try:
        from diagnostics.data_contact import SimArtifacts, SnapshotArtifacts, DefinitionArtifacts, diagnostics_support
        
        # Test SimArtifacts
        sim = SimArtifacts()
        assert sim is not None, "SimArtifacts should be instantiable"
        print("✅ SimArtifacts instantiated successfully")
        
        # Test SnapshotArtifacts
        snapshot = SnapshotArtifacts()
        assert snapshot is not None, "SnapshotArtifacts should be instantiable"
        print("✅ SnapshotArtifacts instantiated successfully")
        
        # Test DefinitionArtifacts
        definition = DefinitionArtifacts()
        assert definition is not None, "DefinitionArtifacts should be instantiable"
        print("✅ DefinitionArtifacts instantiated successfully")
        
        # Test diagnostics_support function
        assert callable(diagnostics_support), "diagnostics_support should be callable"
        diagnostics_support()  # Should not raise an error
        print("✅ diagnostics_support function is callable")
        
        print("✅ All diagnostics classes test passed")
        return True
        
    except Exception as e:
        print(f"❌ Diagnostics classes test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostics tests."""
    print("\n🧪 Diagnostics Module Tests\n")
    print("="*60)
    
    results = []
    
    # Test import
    results.append(test_diagnostics_import())
    
    # Test classes
    results.append(test_diagnostics_classes())
    
    print("\n" + "="*60)
    
    if all(results):
        print("✅ All diagnostics tests passed")
        return 0
    else:
        print("❌ Some diagnostics tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
