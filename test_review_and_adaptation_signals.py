"""
test_review_and_adaptation_signals.py

Unit tests for the Review & Adaptation Signals functionality.

Tests:
1. diagnostics_review_signals.py helper module
2. adaptive_intelligence.py rendering function
3. Graceful degradation with missing data
4. Signal computation logic
"""

import pandas as pd
import numpy as np
import pytest
from helpers.diagnostics_review_signals import (
    get_review_signals,
    _compute_data_availability_signal,
    _compute_wave_count_signal,
    _compute_attribution_coverage_signal,
    _compute_alpha_consistency_signal,
    _compute_adaptive_state_signal
)


def create_sample_snapshot():
    """Create a sample snapshot DataFrame for testing."""
    return pd.DataFrame({
        'wave_name': ['sp500_wave', 'income_wave', 'crypto_growth_wave', 'growth_wave', 'value_wave'],
        'display_name': ['S&P 500 Wave', 'Income Wave', 'Crypto Growth', 'Growth Wave', 'Value Wave'],
        'alpha_30d': [0.02, -0.01, 0.03, 0.015, 0.01],
        'alpha_60d': [0.025, -0.005, 0.028, 0.02, 0.012],
        'alpha_365d': [0.03, 0.005, 0.035, 0.025, 0.015],
        'weight': [0.30, 0.20, 0.25, 0.15, 0.10]
    })


def create_sample_attribution():
    """Create a sample attribution DataFrame for testing."""
    waves = ['sp500_wave', 'income_wave', 'crypto_growth_wave', 'growth_wave']
    horizons = [30, 60, 365]
    data = []
    for wave in waves:
        for horizon in horizons:
            data.append({
                'wave': wave,
                'horizon': horizon,
                'alpha': np.random.uniform(-0.02, 0.03)
            })
    return pd.DataFrame(data)


def create_sample_adaptive_state():
    """Create a sample adaptive state dictionary for testing."""
    return {
        'initialized': True,
        'last_update': '2026-02-03T10:00:00',
        'learning_history': [
            {'timestamp': '2026-02-01', 'signal': 'test'},
            {'timestamp': '2026-02-02', 'signal': 'test2'}
        ],
        'scenario_results': {
            'scenario1': {'result': 'test'}
        }
    }


class TestDataAvailabilitySignal:
    """Test data availability signal computation."""
    
    def test_both_data_sources_available(self):
        """Test when both snapshot and attribution data are available."""
        snapshot = create_sample_snapshot()
        attrib = create_sample_attribution()
        
        signal = _compute_data_availability_signal(snapshot, attrib)
        
        assert signal['title'] == 'Data Availability'
        assert signal['status'] == 'Stable'
        assert signal['scope'] == 'Portfolio-level'
        assert 'All primary data sources available' in signal['observation']
        assert signal['actionability'] == 'Observational only — human evaluation recommended'
    
    def test_partial_data_availability(self):
        """Test when only one data source is available."""
        snapshot = create_sample_snapshot()
        
        signal = _compute_data_availability_signal(snapshot, None)
        
        assert signal['status'] == 'Review Recommended'
        assert 'Partial data availability' in signal['observation']
    
    def test_no_data_available(self):
        """Test when no data is available."""
        signal = _compute_data_availability_signal(None, None)
        
        assert signal['status'] == 'Monitoring'
        assert 'Awaiting data initialization' in signal['observation']


class TestWaveCountSignal:
    """Test wave count stability signal computation."""
    
    def test_stable_wave_count(self):
        """Test when wave count is stable (>= 5 waves)."""
        snapshot = create_sample_snapshot()  # 5 waves
        
        signal = _compute_wave_count_signal(snapshot)
        
        assert signal['title'] == 'Wave Portfolio Composition'
        assert signal['status'] == 'Stable'
        assert '5 active waves' in signal['observation']
    
    def test_monitoring_wave_count(self):
        """Test when wave count is in monitoring range (3-4 waves)."""
        snapshot = create_sample_snapshot().iloc[:4]  # 4 waves
        
        signal = _compute_wave_count_signal(snapshot)
        
        assert signal['status'] == 'Monitoring'
        assert 'below typical range' in signal['observation']
    
    def test_review_recommended_wave_count(self):
        """Test when wave count requires review (< 3 waves)."""
        snapshot = create_sample_snapshot().iloc[:2]  # 2 waves
        
        signal = _compute_wave_count_signal(snapshot)
        
        assert signal['status'] == 'Review Recommended'
        assert 'Limited wave diversification' in signal['observation']
    
    def test_graceful_degradation(self):
        """Test graceful handling of errors."""
        snapshot = pd.DataFrame({'bad_column': [1, 2, 3]})
        
        signal = _compute_wave_count_signal(snapshot)
        
        assert signal is None


class TestAttributionCoverageSignal:
    """Test attribution coverage signal computation."""
    
    def test_high_coverage(self):
        """Test when attribution coverage is high (>= 90%)."""
        snapshot = create_sample_snapshot()
        attrib = create_sample_attribution()  # 4/5 waves = 80%, need all 5
        
        # Add 5th wave to attribution
        new_row = pd.DataFrame([{
            'wave': 'value_wave',
            'horizon': 30,
            'alpha': 0.01
        }])
        attrib = pd.concat([attrib, new_row], ignore_index=True)
        
        signal = _compute_attribution_coverage_signal(attrib, snapshot)
        
        assert signal['status'] == 'Stable'
        assert '100%' in signal['observation']
    
    def test_monitoring_coverage(self):
        """Test when coverage is in monitoring range (70-89%)."""
        snapshot = create_sample_snapshot()
        attrib = create_sample_attribution()  # 4/5 = 80%
        
        signal = _compute_attribution_coverage_signal(attrib, snapshot)
        
        assert signal['status'] == 'Monitoring'
        assert '80%' in signal['observation']
    
    def test_low_coverage(self):
        """Test when coverage is low (< 70%)."""
        snapshot = create_sample_snapshot()
        attrib = create_sample_attribution().iloc[:6]  # Only 2/5 waves = 40%
        
        signal = _compute_attribution_coverage_signal(attrib, snapshot)
        
        assert signal['status'] == 'Review Recommended'
        assert 'significant gaps' in signal['observation']


class TestAlphaConsistencySignal:
    """Test alpha consistency signal computation."""
    
    def test_high_consistency(self):
        """Test when alpha is consistent across horizons (>= 60% agreement)."""
        snapshot = create_sample_snapshot()
        
        signal = _compute_alpha_consistency_signal(snapshot)
        
        # All 5 waves have positive alpha across all horizons
        assert signal['status'] == 'Stable'
        assert 'positive alpha across all horizons' in signal['observation']
    
    def test_monitoring_consistency(self):
        """Test when consistency is moderate (40-59% agreement)."""
        snapshot = pd.DataFrame({
            'wave_name': ['w1', 'w2', 'w3', 'w4', 'w5'],
            'alpha_30d': [0.01, -0.01, 0.02, -0.01, 0.01],
            'alpha_60d': [0.01, 0.01, -0.01, 0.01, -0.01],
            'alpha_365d': [0.01, -0.01, 0.01, 0.01, -0.01]
        })
        
        signal = _compute_alpha_consistency_signal(snapshot)
        
        assert signal['status'] in ['Monitoring', 'Review Recommended']
    
    def test_missing_columns(self):
        """Test graceful handling when required columns are missing."""
        snapshot = pd.DataFrame({
            'wave_name': ['w1', 'w2'],
            'alpha_30d': [0.01, 0.02]
        })
        
        signal = _compute_alpha_consistency_signal(snapshot)
        
        assert signal is None


class TestAdaptiveStateSignal:
    """Test adaptive state health signal computation."""
    
    def test_healthy_adaptive_state(self):
        """Test when adaptive state is healthy and active."""
        state = create_sample_adaptive_state()
        
        signal = _compute_adaptive_state_signal(state)
        
        assert signal['status'] == 'Stable'
        assert 'active with historical context' in signal['observation']
    
    def test_initializing_state(self):
        """Test when state is initialized but accumulating data."""
        state = {
            'initialized': True,
            'learning_history': [],
            'scenario_results': {}
        }
        
        signal = _compute_adaptive_state_signal(state)
        
        assert signal['status'] == 'Monitoring'
        assert 'accumulating learning data' in signal['observation']
    
    def test_uninitialized_state(self):
        """Test when state is not initialized."""
        state = {
            'initialized': False
        }
        
        signal = _compute_adaptive_state_signal(state)
        
        assert signal['status'] == 'Review Recommended'
        assert 'not yet initialized' in signal['observation']


class TestGetReviewSignals:
    """Test the main get_review_signals function."""
    
    def test_all_signals_with_full_data(self):
        """Test that all signals are generated when all data is available."""
        snapshot = create_sample_snapshot()
        attrib = create_sample_attribution()
        state = create_sample_adaptive_state()
        
        signals = get_review_signals(snapshot, attrib, state)
        
        # Should have 5 signals when all data is available
        assert len(signals) >= 5
        
        # Check that all signals have required fields
        for signal in signals:
            assert 'title' in signal
            assert 'status' in signal
            assert 'observation' in signal
            assert 'scope' in signal
            assert 'actionability' in signal
            assert signal['status'] in ['Monitoring', 'Review Recommended', 'Stable']
            assert signal['scope'] in ['Portfolio-level', 'Wave-level']
            assert signal['actionability'] == 'Observational only — human evaluation recommended'
    
    def test_graceful_degradation_no_data(self):
        """Test that function handles no data gracefully."""
        signals = get_review_signals(None, None, None)
        
        # Should still return at least the data availability signal
        assert len(signals) >= 1
        assert signals[0]['title'] == 'Data Availability'
    
    def test_partial_data(self):
        """Test with partial data availability."""
        snapshot = create_sample_snapshot()
        
        signals = get_review_signals(snapshot_df=snapshot)
        
        # Should get some signals even with partial data
        assert len(signals) >= 2
        
        # Signals should handle missing data gracefully
        for signal in signals:
            assert signal is not None
            assert isinstance(signal, dict)
    
    def test_signal_filtering(self):
        """Test that None signals are filtered out."""
        # Create invalid snapshot that will produce None signals
        bad_snapshot = pd.DataFrame({'invalid': [1, 2, 3]})
        
        signals = get_review_signals(snapshot_df=bad_snapshot)
        
        # All signals should be valid (None filtered out)
        assert all(signal is not None for signal in signals)


class TestSignalStructure:
    """Test signal structure and format."""
    
    def test_signal_format_consistency(self):
        """Test that all signals follow the same format."""
        snapshot = create_sample_snapshot()
        attrib = create_sample_attribution()
        state = create_sample_adaptive_state()
        
        signals = get_review_signals(snapshot, attrib, state)
        
        for signal in signals:
            # Check required keys
            assert 'title' in signal
            assert 'status' in signal
            assert 'observation' in signal
            assert 'scope' in signal
            assert 'actionability' in signal
            
            # Check data types
            assert isinstance(signal['title'], str)
            assert isinstance(signal['status'], str)
            assert isinstance(signal['observation'], str)
            assert isinstance(signal['scope'], str)
            assert isinstance(signal['actionability'], str)
            
            # Check values
            assert len(signal['title']) > 0
            assert len(signal['observation']) > 0
    
    def test_observational_only_actionability(self):
        """Test that all signals are marked as observational only."""
        snapshot = create_sample_snapshot()
        
        signals = get_review_signals(snapshot_df=snapshot)
        
        for signal in signals:
            assert signal['actionability'] == 'Observational only — human evaluation recommended'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
