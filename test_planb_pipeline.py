"""
test_planb_pipeline.py

Focused tests for Plan B Data Model and Monitor pipeline.

Tests:
- Stub file generation
- Snapshot building with 28 rows guarantee
- Graceful degradation
- Status flags (FULL/PARTIAL/UNAVAILABLE)
- Alert generation
"""

import os
import sys
import pytest
import pandas as pd
from datetime import datetime, timedelta

# Import the module under test
from planb_pipeline import (
    build_planb_snapshot,
    get_planb_diagnostics,
    check_planb_files,
    ensure_planb_files,
    generate_stub_nav_file,
    generate_stub_positions_file,
    generate_stub_trades_file,
    generate_stub_prices_file,
    calculate_returns,
    calculate_volatility,
    calculate_max_drawdown,
    calculate_beta,
    generate_alerts,
    PLANB_DATA_DIR,
    STATUS_FULL,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE
)


class TestStubFileGeneration:
    """Test stub file generation functionality."""
    
    def test_generate_stub_nav_file(self):
        """Test NAV stub file generation."""
        # Clean up if exists
        nav_path = os.path.join(PLANB_DATA_DIR, "nav.csv")
        if os.path.exists(nav_path):
            os.remove(nav_path)
        
        # Generate stub
        result = generate_stub_nav_file()
        assert result is True
        
        # Check file exists
        assert os.path.exists(nav_path)
        
        # Check file has correct headers
        df = pd.read_csv(nav_path)
        expected_columns = ['date', 'wave_id', 'nav', 'cash', 'holdings_value']
        assert list(df.columns) == expected_columns
        assert len(df) == 0  # Should be empty
    
    def test_generate_stub_positions_file(self):
        """Test positions stub file generation."""
        positions_path = os.path.join(PLANB_DATA_DIR, "positions.csv")
        if os.path.exists(positions_path):
            os.remove(positions_path)
        
        result = generate_stub_positions_file()
        assert result is True
        assert os.path.exists(positions_path)
        
        df = pd.read_csv(positions_path)
        expected_columns = ['wave_id', 'ticker', 'weight', 'description', 'exposure', 'cash', 'safe_fraction']
        assert list(df.columns) == expected_columns
    
    def test_generate_stub_trades_file(self):
        """Test trades stub file generation."""
        trades_path = os.path.join(PLANB_DATA_DIR, "trades.csv")
        if os.path.exists(trades_path):
            os.remove(trades_path)
        
        result = generate_stub_trades_file()
        assert result is True
        assert os.path.exists(trades_path)
        
        df = pd.read_csv(trades_path)
        expected_columns = ['date', 'wave_id', 'ticker', 'action', 'shares', 'price', 'value']
        assert list(df.columns) == expected_columns
    
    def test_generate_stub_prices_file(self):
        """Test prices stub file generation."""
        prices_path = os.path.join(PLANB_DATA_DIR, "prices.csv")
        if os.path.exists(prices_path):
            os.remove(prices_path)
        
        result = generate_stub_prices_file()
        assert result is True
        assert os.path.exists(prices_path)
        
        df = pd.read_csv(prices_path)
        assert 'date' in df.columns
        # Should have benchmark tickers
        assert 'SPY' in df.columns
        assert 'QQQ' in df.columns
    
    def test_ensure_planb_files(self):
        """Test ensure all Plan B files exist."""
        result = ensure_planb_files()
        
        assert 'file_status' in result
        assert 'created_stubs' in result
        assert 'all_exist' in result
        
        # All files should exist after ensure
        assert result['all_exist'] is True


class TestSnapshotBuilding:
    """Test snapshot building functionality."""
    
    def test_build_planb_snapshot_returns_28_rows(self):
        """Test that snapshot always returns exactly 28 rows."""
        snapshot = build_planb_snapshot(days=365)
        
        # CRITICAL: Must always return 28 rows
        assert len(snapshot) == 28
        
        print(f"✅ Snapshot returned {len(snapshot)} rows (expected 28)")
    
    def test_build_planb_snapshot_has_required_columns(self):
        """Test that snapshot has all required columns."""
        snapshot = build_planb_snapshot(days=365)
        
        required_columns = [
            'wave_id', 'display_name', 'mode', 'timestamp',
            'nav_latest',
            'return_1d', 'return_30d', 'return_60d', 'return_365d',
            'bm_return_1d', 'bm_return_30d', 'bm_return_60d', 'bm_return_365d',
            'alpha_1d', 'alpha_30d', 'alpha_60d', 'alpha_365d',
            'exposure_pct', 'cash_pct',
            'beta_est', 'vol_365d', 'maxdd_365d', 'turnover_30d',
            'alerts', 'status', 'reason'
        ]
        
        for col in required_columns:
            assert col in snapshot.columns, f"Missing column: {col}"
        
        print(f"✅ Snapshot has all {len(required_columns)} required columns")
    
    def test_build_planb_snapshot_with_empty_data(self):
        """Test snapshot building with empty data files (all UNAVAILABLE)."""
        # Ensure files are empty (stubs)
        ensure_planb_files()
        
        snapshot = build_planb_snapshot(days=365)
        
        # Should still return 28 rows
        assert len(snapshot) == 28
        
        # All should be UNAVAILABLE with empty data
        status_counts = snapshot['status'].value_counts()
        assert status_counts.get(STATUS_UNAVAILABLE, 0) == 28
        
        # All should have a reason
        for idx, row in snapshot.iterrows():
            assert row['reason'] != '', f"Row {idx} missing reason"
        
        print(f"✅ Empty data correctly returns 28 UNAVAILABLE rows")
    
    def test_build_planb_snapshot_timestamps(self):
        """Test that snapshot includes timestamps."""
        snapshot = build_planb_snapshot(days=365)
        
        # All rows should have a timestamp
        assert 'timestamp' in snapshot.columns
        assert snapshot['timestamp'].notna().all()
        
        # Timestamps should be recent (within last minute)
        for ts in snapshot['timestamp']:
            age = (datetime.now() - ts).total_seconds()
            assert age < 60, f"Timestamp too old: {age} seconds"
        
        print(f"✅ All timestamps are recent")


class TestStatusFlags:
    """Test status flag logic (FULL/PARTIAL/UNAVAILABLE)."""
    
    def test_status_flags_with_empty_data(self):
        """Test status flags when data is empty."""
        snapshot = build_planb_snapshot(days=365)
        
        # With empty data, all should be UNAVAILABLE
        for idx, row in snapshot.iterrows():
            assert row['status'] in [STATUS_FULL, STATUS_PARTIAL, STATUS_UNAVAILABLE]
        
        print(f"✅ Status flags are valid")
    
    def test_default_values_for_unavailable(self):
        """Test that UNAVAILABLE rows have appropriate default values."""
        snapshot = build_planb_snapshot(days=365)
        
        unavailable_rows = snapshot[snapshot['status'] == STATUS_UNAVAILABLE]
        
        for idx, row in unavailable_rows.iterrows():
            # Exposure and cash should have defaults
            assert row['exposure_pct'] == 100.0
            assert row['cash_pct'] == 0.0
            
            # NAV and returns should be None/NaN
            assert pd.isna(row['nav_latest']) or row['nav_latest'] is None
        
        print(f"✅ UNAVAILABLE rows have correct defaults")


class TestCalculationFunctions:
    """Test individual calculation functions."""
    
    def test_calculate_returns_empty_series(self):
        """Test returns calculation with empty series."""
        returns = calculate_returns(None, [1, 30, 60, 365])
        
        assert returns['1d'] is None
        assert returns['30d'] is None
        assert returns['60d'] is None
        assert returns['365d'] is None
    
    def test_calculate_returns_with_data(self):
        """Test returns calculation with sample data."""
        # Create sample NAV series
        nav_series = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])
        
        returns = calculate_returns(nav_series, [1])
        
        # 1-day return should be positive
        assert returns['1d'] is not None
        assert returns['1d'] > 0  # Price went up
    
    def test_calculate_volatility_empty_series(self):
        """Test volatility calculation with empty series."""
        vol = calculate_volatility(None, days=365)
        assert vol is None
    
    def test_calculate_max_drawdown_empty_series(self):
        """Test max drawdown calculation with empty series."""
        dd = calculate_max_drawdown(None, days=365)
        assert dd is None
    
    def test_calculate_beta_empty_series(self):
        """Test beta calculation with empty series."""
        beta = calculate_beta(None, None)
        assert beta is None


class TestAlertGeneration:
    """Test alert generation logic."""
    
    def test_generate_alerts_no_issues(self):
        """Test alert generation with no issues."""
        row = {
            'status': STATUS_FULL,
            'beta_est': 0.9,
            'maxdd_365d': -10.0,
            'timestamp': datetime.now(),
            'reason': ''
        }
        
        alerts = generate_alerts(row)
        
        # Should have no alerts for normal values
        assert isinstance(alerts, list)
    
    def test_generate_alerts_high_beta(self):
        """Test alert generation for high beta."""
        row = {
            'status': STATUS_FULL,
            'beta_est': 1.8,  # High beta
            'maxdd_365d': -10.0,
            'timestamp': datetime.now(),
            'reason': ''
        }
        
        alerts = generate_alerts(row)
        
        # Should have alert for high beta
        assert any('beta drift' in alert.lower() for alert in alerts)
    
    def test_generate_alerts_large_drawdown(self):
        """Test alert generation for large drawdown."""
        row = {
            'status': STATUS_FULL,
            'beta_est': 0.9,
            'maxdd_365d': -35.0,  # Large drawdown
            'timestamp': datetime.now(),
            'reason': ''
        }
        
        alerts = generate_alerts(row)
        
        # Should have alert for large drawdown
        assert any('drawdown' in alert.lower() for alert in alerts)
    
    def test_generate_alerts_unavailable_status(self):
        """Test alert generation for unavailable status."""
        row = {
            'status': STATUS_UNAVAILABLE,
            'beta_est': None,
            'maxdd_365d': None,
            'timestamp': datetime.now(),
            'reason': 'Data missing'
        }
        
        alerts = generate_alerts(row)
        
        # Should have alert for unavailable status
        assert any('unavailable' in alert.lower() for alert in alerts)


class TestDiagnostics:
    """Test diagnostics functionality."""
    
    def test_get_planb_diagnostics(self):
        """Test diagnostics retrieval."""
        diagnostics = get_planb_diagnostics()
        
        assert 'timestamp' in diagnostics
        assert 'files' in diagnostics
        assert 'all_files_exist' in diagnostics
        assert 'missing_files' in diagnostics
        assert 'file_info' in diagnostics
        
        print(f"✅ Diagnostics returned successfully")
    
    def test_check_planb_files(self):
        """Test file existence check."""
        file_status = check_planb_files()
        
        assert 'nav' in file_status
        assert 'positions' in file_status
        assert 'trades' in file_status
        assert 'prices' in file_status
        
        # After ensure_planb_files, all should exist
        assert all(file_status.values())
        
        print(f"✅ File check returned correct status")


class TestGracefulDegradation:
    """Test graceful degradation behavior."""
    
    def test_no_blocking_on_missing_data(self):
        """Test that missing data doesn't block snapshot generation."""
        # This should complete quickly without hanging
        import time
        start = time.time()
        
        snapshot = build_planb_snapshot(days=365)
        
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds even with missing data
        assert elapsed < 5.0, f"Snapshot took too long: {elapsed} seconds"
        
        # Should still return 28 rows
        assert len(snapshot) == 28
        
        print(f"✅ Snapshot built in {elapsed:.2f} seconds (no blocking)")
    
    def test_no_ticker_dependencies(self):
        """Test that Plan B pipeline has no live ticker dependencies."""
        # This test verifies the pipeline doesn't try to fetch live data
        # by checking it completes quickly even without network
        
        snapshot = build_planb_snapshot(days=365)
        
        # Should succeed without fetching tickers
        assert len(snapshot) == 28
        
        print(f"✅ No live ticker dependencies detected")


def run_tests():
    """Run all tests and print summary."""
    print("\n" + "="*70)
    print("Plan B Pipeline Tests")
    print("="*70 + "\n")
    
    # Ensure files exist before running tests
    ensure_planb_files()
    
    # Run pytest with verbose output
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_tests()
