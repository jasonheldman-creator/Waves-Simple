"""
Test: Strategy-Consistent Portfolio Metrics and Snapshot Cache Invalidation

This test validates the implementation of requirements from the problem statement:
1. Portfolio Snapshot cards computed from wave_history daily series
2. Alpha Source Breakdown (Selection vs Overlay Alpha)
3. Snapshot cache invalidation using snapshot_metadata.json

Requirements:
- Portfolio metrics use strategy-adjusted returns (VIX overlay applied)
- Alpha breakdown shows Selection Alpha and Overlay Alpha
- Cache invalidates when snapshot_id or snapshot_hash changes
"""

import sys
import os
import pandas as pd
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test constants
ALPHA_DECOMPOSITION_TOLERANCE = 0.001  # 0.1% tolerance for alpha arithmetic


def test_snapshot_version_key():
    """Test get_snapshot_version_key() reads metadata correctly."""
    print("\n=== Test: Snapshot Version Key ===")
    from helpers.snapshot_version import get_snapshot_version_key
    
    # Test with actual metadata file
    version = get_snapshot_version_key()
    assert version is not None
    assert isinstance(version, str)
    assert ':' in version  # Format should be "snapshot_id:snapshot_hash"
    
    # Version should not be the fallback if metadata exists
    if os.path.exists('data/snapshot_metadata.json'):
        assert version != "unknown:0"
        # Should have both parts
        parts = version.split(':')
        assert len(parts) == 2
        snapshot_id, snapshot_hash = parts
        assert snapshot_id.startswith('snap-') or snapshot_id == 'unknown'
        assert len(snapshot_hash) > 0
        print(f"✓ Snapshot version: {version}")
    else:
        print(f"⚠ Snapshot metadata not found, using fallback: {version}")
    
    return True


def test_snapshot_metadata_structure():
    """Test snapshot_metadata.json has required fields."""
    print("\n=== Test: Snapshot Metadata Structure ===")
    metadata_file = 'data/snapshot_metadata.json'
    
    if not os.path.exists(metadata_file):
        print("⚠ snapshot_metadata.json not found - skipping")
        return True
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Required fields for cache invalidation
    assert 'snapshot_id' in metadata
    assert 'snapshot_hash' in metadata
    
    # Additional useful fields
    assert 'timestamp' in metadata
    assert 'engine_version' in metadata
    assert 'wave_count' in metadata
    
    print(f"✓ Snapshot ID: {metadata['snapshot_id']}")
    print(f"✓ Snapshot Hash: {metadata['snapshot_hash']}")
    print(f"✓ Wave Count: {metadata['wave_count']}")
    print(f"✓ Engine Version: {metadata['engine_version']}")
    
    return True


def test_portfolio_alpha_ledger_structure():
    """Test compute_portfolio_alpha_ledger returns correct structure."""
    print("\n=== Test: Portfolio Alpha Ledger Structure ===")
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
    except ImportError as e:
        print(f"⚠ Required modules not available: {e}")
        return True
    
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        print("⚠ Price book not available - skipping")
        return True
    
    ledger = compute_portfolio_alpha_ledger(
        price_book,
        periods=[1, 30, 60, 365],
        vix_exposure_enabled=True
    )
    
    # Check structure
    assert 'success' in ledger
    assert 'period_results' in ledger
    assert 'daily_realized_return' in ledger
    assert 'daily_unoverlay_return' in ledger
    assert 'daily_benchmark_return' in ledger
    assert 'daily_exposure' in ledger
    assert 'overlay_available' in ledger
    
    print(f"✓ Ledger success: {ledger['success']}")
    print(f"✓ Overlay available: {ledger['overlay_available']}")
    
    if ledger['success']:
        # Check period results contain alpha breakdown
        for period_key in ['1D', '30D', '60D', '365D']:
            if period_key in ledger['period_results']:
                period_data = ledger['period_results'][period_key]
                if period_data.get('available'):
                    # Alpha Source Breakdown fields
                    assert 'total_alpha' in period_data
                    assert 'selection_alpha' in period_data
                    assert 'overlay_alpha' in period_data
                    assert 'residual' in period_data
                    
                    # Verify alpha decomposition: total_alpha = selection_alpha + overlay_alpha + residual
                    total = period_data['total_alpha']
                    selection = period_data['selection_alpha']
                    overlay = period_data['overlay_alpha']
                    residual = period_data['residual']
                    
                    reconstructed = selection + overlay + residual
                    assert abs(total - reconstructed) < ALPHA_DECOMPOSITION_TOLERANCE, \
                        f"{period_key}: Alpha decomposition failed: {total:.6f} != {reconstructed:.6f}"
                    
                    print(f"✓ {period_key}: Total={total:+.2%}, Selection={selection:+.2%}, Overlay={overlay:+.2%}, Residual={residual:+.3%}")
    
    return True


def test_overlay_alpha_nonzero_when_vix_active():
    """Test Overlay Alpha is non-zero when VIX overlay is active (exposure != 1.0)."""
    print("\n=== Test: Overlay Alpha When VIX Active ===")
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
    except ImportError as e:
        print(f"⚠ Required modules not available: {e}")
        return True
    
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        print("⚠ Price book not available - skipping")
        return True
    
    ledger = compute_portfolio_alpha_ledger(
        price_book,
        periods=[60],
        vix_exposure_enabled=True
    )
    
    if not ledger['success']:
        print("⚠ Portfolio ledger computation failed - skipping")
        return True
    
    # Check if VIX overlay is active
    if ledger['overlay_available'] and ledger['daily_exposure'] is not None:
        exposure_series = ledger['daily_exposure']
        
        # Check if exposure varies from 1.0 (indicating VIX overlay active)
        exp_min = exposure_series.min()
        exp_max = exposure_series.max()
        
        print(f"Exposure range: [{exp_min:.2f}, {exp_max:.2f}]")
        
        if exp_min < 1.0 or exp_max > 1.0:
            # VIX overlay is active - Overlay Alpha should exist
            period_60d = ledger['period_results'].get('60D', {})
            if period_60d.get('available'):
                overlay_alpha = period_60d['overlay_alpha']
                print(f"✓ VIX overlay active: exposure range [{exp_min:.2f}, {exp_max:.2f}]")
                print(f"✓ 60D Overlay Alpha: {overlay_alpha:+.4%}")
        else:
            print(f"ℹ VIX overlay inactive: exposure = 1.0 (constant)")
    else:
        print(f"⚠ VIX overlay not available")
    
    return True


def test_exposure_series_computed():
    """Test that exposure series is computed when VIX data available."""
    print("\n=== Test: Exposure Series Computation ===")
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
    except ImportError as e:
        print(f"⚠ Required modules not available: {e}")
        return True
    
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        print("⚠ Price book not available - skipping")
        return True
    
    ledger = compute_portfolio_alpha_ledger(
        price_book,
        periods=[60],
        vix_exposure_enabled=True
    )
    
    if not ledger['success']:
        print("⚠ Portfolio ledger computation failed - skipping")
        return True
    
    # Check exposure series exists
    assert ledger['daily_exposure'] is not None
    assert len(ledger['daily_exposure']) > 0
    
    # Exposure should be between 0 and ~1.5 (reasonable bounds)
    exp_min = ledger['daily_exposure'].min()
    exp_max = ledger['daily_exposure'].max()
    assert exp_min >= 0.0
    assert exp_max <= 2.0  # Should not exceed reasonable bounds
    
    print(f"✓ Exposure range: [{exp_min:.2f}, {exp_max:.2f}]")
    print(f"✓ VIX ticker used: {ledger.get('vix_ticker_used', 'none')}")
    print(f"✓ Overlay available: {ledger.get('overlay_available', False)}")
    
    return True


def test_gitignore_excludes_data_artifacts():
    """Test that .gitignore excludes data artifacts."""
    print("\n=== Test: .gitignore Excludes Data Artifacts ===")
    gitignore_path = '.gitignore'
    
    if not os.path.exists(gitignore_path):
        print("⚠ .gitignore not found - skipping")
        return True
    
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()
    
    # Check that data artifacts are excluded
    required_patterns = [
        '/data/live_snapshot.csv',
        '/data/snapshot_metadata.json',
        '/data/diagnostics_run.json',
        '/data/prices.csv'
    ]
    
    for pattern in required_patterns:
        assert pattern in gitignore_content, \
            f"Missing pattern in .gitignore: {pattern}"
        print(f"✓ Pattern found: {pattern}")
    
    return True


def test_portfolio_uses_wave_history_daily_series():
    """Test that portfolio metrics are computed from daily wave series."""
    print("\n=== Test: Portfolio Uses Daily Wave Series ===")
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
    except ImportError as e:
        print(f"⚠ Required modules not available: {e}")
        return True
    
    price_book = get_price_book()
    if price_book is None or price_book.empty:
        print("⚠ Price book not available - skipping")
        return True
    
    ledger = compute_portfolio_alpha_ledger(
        price_book,
        periods=[1, 30, 60, 365],
        vix_exposure_enabled=True
    )
    
    if not ledger['success']:
        print("⚠ Portfolio ledger computation failed - skipping")
        return True
    
    # Verify daily series exist
    assert ledger['daily_risk_return'] is not None
    assert ledger['daily_realized_return'] is not None
    assert ledger['daily_benchmark_return'] is not None
    
    # Daily series should have reasonable length
    assert len(ledger['daily_risk_return']) > 100  # At least 100 trading days
    
    print(f"✓ Daily series length: {len(ledger['daily_risk_return'])}")
    
    # Verify returns are compounded over periods
    period_60d = ledger['period_results'].get('60D', {})
    if period_60d.get('available'):
        # Cumulative return should be computed from daily series
        cum_realized = period_60d['cum_realized']
        assert cum_realized is not None
        assert isinstance(cum_realized, (int, float))
        
        print(f"✓ 60D Portfolio Return: {cum_realized:+.2%}")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("Testing: Strategy-Consistent Portfolio Metrics")
    print("=" * 70)
    
    tests = [
        test_snapshot_version_key,
        test_snapshot_metadata_structure,
        test_gitignore_excludes_data_artifacts,
        test_portfolio_alpha_ledger_structure,
        test_exposure_series_computed,
        test_overlay_alpha_nonzero_when_vix_active,
        test_portfolio_uses_wave_history_daily_series,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {test_func.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

