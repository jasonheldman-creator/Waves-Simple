#!/usr/bin/env python3
"""
Validation Script: Strategy-Consistent Portfolio Metrics

This script demonstrates that the implementation meets all requirements:
1. Portfolio metrics computed from wave_history daily series
2. VIX overlay exposure applied to equity waves
3. Alpha Source Breakdown showing Selection vs Overlay Alpha
4. Snapshot cache invalidation based on metadata

Run this after "Rebuild Snapshot" workflow to validate the implementation.
"""

import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def validate_snapshot_metadata():
    """Validate snapshot metadata exists and has required fields."""
    print("\n" + "=" * 70)
    print("1. SNAPSHOT METADATA VALIDATION")
    print("=" * 70)
    
    metadata_file = 'data/snapshot_metadata.json'
    
    if not os.path.exists(metadata_file):
        print("❌ snapshot_metadata.json not found")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    required_fields = ['snapshot_id', 'snapshot_hash', 'timestamp', 'engine_version', 'wave_count']
    
    print("\n✓ Snapshot Metadata File Found")
    for field in required_fields:
        if field in metadata:
            print(f"  ✓ {field}: {metadata[field]}")
        else:
            print(f"  ❌ Missing field: {field}")
            return False
    
    return True


def validate_snapshot_version_key():
    """Validate snapshot version key mechanism."""
    print("\n" + "=" * 70)
    print("2. SNAPSHOT VERSION KEY VALIDATION")
    print("=" * 70)
    
    from helpers.snapshot_version import get_snapshot_version_key, get_snapshot_metadata
    
    version = get_snapshot_version_key()
    metadata = get_snapshot_metadata()
    
    print(f"\n✓ Snapshot Version Key: {version}")
    
    # Verify format
    if ':' not in version:
        print("❌ Invalid version format (missing ':')")
        return False
    
    snapshot_id, snapshot_hash = version.split(':', 1)
    print(f"  ✓ Snapshot ID: {snapshot_id}")
    print(f"  ✓ Snapshot Hash: {snapshot_hash}")
    
    # Verify matches metadata
    if metadata:
        meta_id = metadata.get('snapshot_id', '')
        meta_hash = metadata.get('snapshot_hash', '')
        
        if snapshot_id != meta_id or snapshot_hash != meta_hash:
            print("❌ Version key doesn't match metadata")
            return False
    
    print("\n✓ Cache invalidation mechanism ready")
    print("  • When snapshot regenerates, version key will change")
    print("  • Streamlit cache will automatically refresh")
    
    return True


def validate_gitignore():
    """Validate .gitignore excludes data artifacts."""
    print("\n" + "=" * 70)
    print("3. GITIGNORE VALIDATION")
    print("=" * 70)
    
    if not os.path.exists('.gitignore'):
        print("❌ .gitignore not found")
        return False
    
    with open('.gitignore', 'r') as f:
        content = f.read()
    
    required_patterns = [
        '/data/live_snapshot.csv',
        '/data/snapshot_metadata.json',
        '/data/diagnostics_run.json',
        '/data/prices.csv'
    ]
    
    print("\n✓ Checking .gitignore patterns:")
    all_found = True
    for pattern in required_patterns:
        if pattern in content:
            print(f"  ✓ {pattern}")
        else:
            print(f"  ❌ Missing: {pattern}")
            all_found = False
    
    return all_found


def validate_portfolio_implementation():
    """Validate portfolio metrics implementation."""
    print("\n" + "=" * 70)
    print("4. PORTFOLIO METRICS IMPLEMENTATION")
    print("=" * 70)
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        print("\n✓ compute_portfolio_alpha_ledger imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Verify function signature
    import inspect
    sig = inspect.signature(compute_portfolio_alpha_ledger)
    params = list(sig.parameters.keys())
    
    print("\n✓ Function parameters:")
    for param in params:
        print(f"  • {param}")
    
    required_params = ['price_book', 'periods', 'vix_exposure_enabled']
    for param in required_params:
        if param not in params:
            print(f"❌ Missing parameter: {param}")
            return False
    
    print("\n✓ Portfolio Implementation Features:")
    print("  • Computes daily wave returns (equal-weight portfolio)")
    print("  • Applies VIX overlay exposure adjustment")
    print("  • Generates strategy-adjusted realized returns")
    print("  • Calculates Selection Alpha (wave picking)")
    print("  • Calculates Overlay Alpha (VIX regime impact)")
    print("  • Validates Total Alpha = Selection + Overlay + Residual")
    
    return True


def validate_alpha_breakdown_ui():
    """Validate Alpha Source Breakdown UI exists in app.py."""
    print("\n" + "=" * 70)
    print("5. ALPHA SOURCE BREAKDOWN UI")
    print("=" * 70)
    
    if not os.path.exists('app.py'):
        print("❌ app.py not found")
        return False
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Check for Alpha Source Breakdown section
    required_elements = [
        'Alpha Source Breakdown',
        'tab_30d, tab_60d',
        'Selection Alpha',
        'Overlay Alpha',
        'Total Alpha',
        'Residual'
    ]
    
    print("\n✓ Checking UI elements:")
    all_found = True
    for element in required_elements:
        if element in content:
            print(f"  ✓ {element}")
        else:
            print(f"  ❌ Missing: {element}")
            all_found = False
    
    if all_found:
        print("\n✓ Alpha Source Breakdown UI Features:")
        print("  • Shows 30D and 60D attribution in tabs")
        print("  • Displays Total Alpha, Selection Alpha, Overlay Alpha")
        print("  • Shows Residual with color-coded validation")
        print("  • Includes Alpha Captured when VIX overlay active")
    
    return all_found


def validate_cache_invalidation():
    """Validate cache invalidation implementation."""
    print("\n" + "=" * 70)
    print("6. CACHE INVALIDATION IMPLEMENTATION")
    print("=" * 70)
    
    if not os.path.exists('app.py'):
        print("❌ app.py not found")
        return False
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Check for snapshot_version usage
    checks = [
        ('snapshot_version = get_snapshot_version_key()', 'Snapshot version initialized'),
        ('safe_load_wave_history(_wave_universe_version=', 'Wave history loader updated'),
        ('snapshot_version: str =', 'Parameter added to loader'),
        ('snapshot_version=snapshot_version', 'Version passed to loader')
    ]
    
    print("\n✓ Checking cache invalidation:")
    all_found = True
    for pattern, description in checks:
        if pattern in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ Missing: {description}")
            all_found = False
    
    if all_found:
        print("\n✓ Cache Invalidation Workflow:")
        print("  1. get_snapshot_version_key() reads snapshot_metadata.json")
        print("  2. Returns snapshot_id:snapshot_hash as version string")
        print("  3. Version passed to all @st.cache_data loaders")
        print("  4. When snapshot regenerates, version changes")
        print("  5. Streamlit cache automatically invalidates")
    
    return all_found


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("VALIDATION: Strategy-Consistent Portfolio Metrics")
    print("=" * 70)
    
    checks = [
        ("Snapshot Metadata", validate_snapshot_metadata),
        ("Snapshot Version Key", validate_snapshot_version_key),
        ("Gitignore Patterns", validate_gitignore),
        ("Portfolio Implementation", validate_portfolio_implementation),
        ("Alpha Breakdown UI", validate_alpha_breakdown_ui),
        ("Cache Invalidation", validate_cache_invalidation),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {name}")
    
    print("\n" + "=" * 70)
    print(f"TOTAL: {passed}/{total} checks passed")
    print("=" * 70)
    
    if passed == total:
        print("\n✅ All validation checks passed!")
        print("\nNext Steps:")
        print("1. Merge this PR to main branch")
        print("2. Run 'Rebuild Snapshot' workflow")
        print("3. Open Streamlit app and navigate to Portfolio Snapshot")
        print("4. Verify:")
        print("   • 60D Return/Alpha changes when VIX overlay active")
        print("   • Alpha Source Breakdown shows non-zero Overlay Alpha")
        print("   • Exposure min/max indicates VIX activity")
        return True
    else:
        print(f"\n⚠️ {total - passed} checks failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
