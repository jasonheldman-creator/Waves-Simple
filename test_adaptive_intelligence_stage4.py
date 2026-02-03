"""
test_adaptive_intelligence_stage4.py

Test suite for Stage 4 Decision Support Layer (Human-in-the-Loop) features.

This test suite validates:
1. Deterministic action recommendations
2. Risk of inaction calculations
3. Wave-level attention flags
4. Time & trend context enhancements
5. Decision support summary generation
6. Read-only compliance
"""

import unittest
import pandas as pd
import numpy as np
from adaptive_intelligence import (
    generate_recommended_action,
    calculate_risk_of_inaction,
    compute_attention_flag,
    enhance_narrative_with_time_context,
    get_decision_support_summary,
    cluster_signals,
    detect_learning_signals,
    get_priority_insights
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

def get_sample_truth_df():
    """Create sample TruthFrame for testing."""
    return pd.DataFrame({
        'wave_id': ['sp500_wave', 'tech_wave', 'value_wave'],
        'display_name': ['S&P 500 Wave', 'Tech Growth', 'Value Wave'],
        'alpha_1d': [0.001, -0.002, 0.003],
        'alpha_30d': [-0.015, -0.025, 0.010],
        'alpha_60d': [-0.020, -0.030, 0.005],
        'beta_target': [1.0, 1.2, 0.8],
        'beta_real': [1.0, 1.4, 0.8],
        'beta_drift': [0.00, 0.20, 0.00],
        'exposure_pct': [0.95, 0.99, 0.45],
        'drawdown_60d': [-0.10, -0.25, -0.05],
        'data_regime_tag': ['LIVE', 'SANDBOX', 'LIVE']
    })


def get_sample_beta_drift_cluster():
    """Create sample beta drift cluster."""
    return {
        'cluster_type': 'beta_drift',
        'cluster_name': 'Beta Drift Cluster',
        'cluster_severity': 65,
        'affected_waves': ['tech_wave'],
        'wave_count': 1,
        'persistence': 0.6,
        'narrative': 'Beta Drift Detected: 1 wave showing tracking error vs target beta.'
    }


def get_sample_regime_mismatch_cluster():
    """Create sample regime mismatch cluster."""
    return {
        'cluster_type': 'regime_mismatch',
        'cluster_name': 'Regime Mismatch Cluster',
        'cluster_severity': 85,
        'affected_waves': ['tech_wave'],
        'wave_count': 1,
        'persistence': 0.9,
        'narrative': 'Regime Mismatch Alert: 1 wave operating in non-LIVE data regimes.'
    }


def get_sample_alpha_decay_cluster():
    """Create sample alpha decay cluster."""
    return {
        'cluster_type': 'alpha_decay',
        'cluster_name': 'Alpha Decay Cluster',
        'cluster_severity': 55,
        'affected_waves': ['sp500_wave', 'tech_wave'],
        'wave_count': 2,
        'persistence': 0.8,
        'narrative': 'Sustained Alpha Decay: 2 waves underperforming benchmark over 30+ days.'
    }


# ============================================================================
# TEST SUITE
# ============================================================================

class TestStage4DecisionSupport(unittest.TestCase):
    """Test suite for Stage 4 Decision Support Layer."""
    
    # ========================================================================
    # TEST: RECOMMENDED ACTIONS (Deterministic)
    # ========================================================================
    
    def test_recommended_action_beta_drift_critical(self):
        """Test recommended action for critical beta drift."""
        cluster = {
            'cluster_type': 'beta_drift',
            'cluster_severity': 85,
            'wave_count': 3
        }
        action = generate_recommended_action(cluster)
        self.assertEqual(action, "Urgent: Review beta targets and rebalancing thresholds for all affected waves")
        self.assertIn("Urgent", action)
        self.assertIn("beta targets", action)
    
    def test_recommended_action_beta_drift_high(self):
        """Test recommended action for high severity beta drift."""
        cluster = {
            'cluster_type': 'beta_drift',
            'cluster_severity': 60,
            'wave_count': 2
        }
        action = generate_recommended_action(cluster)
        self.assertIn("Review beta targets", action)
        self.assertIn("tactical rebalancing", action)
    
    def test_recommended_action_regime_mismatch_critical(self):
        """Test recommended action for critical regime mismatch."""
        cluster = {
            'cluster_type': 'regime_mismatch',
            'cluster_severity': 90,
            'wave_count': 2
        }
        action = generate_recommended_action(cluster)
        self.assertIn("Critical", action)
        self.assertIn("data pipeline", action)
        self.assertIn("excluding", action)
    
    def test_recommended_action_alpha_decay_high(self):
        """Test recommended action for high severity alpha decay."""
        cluster = {
            'cluster_type': 'alpha_decay',
            'cluster_severity': 65,
            'wave_count': 3
        }
        action = generate_recommended_action(cluster)
        self.assertIn("strategy assumptions", action)
        self.assertIn("market conditions", action)
    
    def test_recommended_action_deterministic(self):
        """Test that recommended actions are deterministic."""
        cluster = {
            'cluster_type': 'beta_drift',
            'cluster_severity': 65,
            'wave_count': 1
        }
        action1 = generate_recommended_action(cluster)
        action2 = generate_recommended_action(cluster)
        self.assertEqual(action1, action2)
    
    # ========================================================================
    # TEST: RISK OF INACTION (Deterministic)
    # ========================================================================
    
    def test_risk_of_inaction_high(self):
        """Test risk of inaction for high-risk cluster."""
        cluster = {
            'cluster_severity': 90,
            'persistence': 0.9,
            'wave_count': 8
        }
        risk = calculate_risk_of_inaction(cluster)
        self.assertEqual(risk, "High")
    
    def test_risk_of_inaction_medium(self):
        """Test risk of inaction for medium-risk cluster."""
        cluster = {
            'cluster_severity': 50,
            'persistence': 0.5,
            'wave_count': 3
        }
        risk = calculate_risk_of_inaction(cluster)
        self.assertEqual(risk, "Medium")
    
    def test_risk_of_inaction_low(self):
        """Test risk of inaction for low-risk cluster."""
        cluster = {
            'cluster_severity': 20,
            'persistence': 0.3,
            'wave_count': 1
        }
        risk = calculate_risk_of_inaction(cluster)
        self.assertEqual(risk, "Low")
    
    def test_risk_of_inaction_deterministic(self):
        """Test that risk calculations are deterministic."""
        cluster = {
            'cluster_severity': 65,
            'persistence': 0.7,
            'wave_count': 4
        }
        risk1 = calculate_risk_of_inaction(cluster)
        risk2 = calculate_risk_of_inaction(cluster)
        self.assertEqual(risk1, risk2)
    
    # ========================================================================
    # TEST: ATTENTION FLAGS (Deterministic)
    # ========================================================================
    
    def test_attention_flag_escalating_risk(self):
        """Test escalating risk flag for high severity regime mismatch."""
        sample_truth_df = get_sample_truth_df()
        sample_regime_mismatch_cluster = get_sample_regime_mismatch_cluster()
        clusters = [sample_regime_mismatch_cluster]
        flag = compute_attention_flag('tech_wave', clusters, sample_truth_df)
        self.assertEqual(flag, "⚠️ Escalating Risk")
    
    def test_attention_flag_needs_review(self):
        """Test needs review flag for medium-high severity cluster."""
        sample_truth_df = get_sample_truth_df()
        cluster = get_sample_beta_drift_cluster()
        cluster['cluster_severity'] = 55
        clusters = [cluster]
        flag = compute_attention_flag('tech_wave', clusters, sample_truth_df)
        self.assertEqual(flag, "🔎 Needs Review")
    
    def test_attention_flag_monitor(self):
        """Test monitor flag for medium severity cluster."""
        sample_truth_df = get_sample_truth_df()
        cluster = {
            'cluster_type': 'concentration_risk',
            'cluster_severity': 30,
            'affected_waves': ['sp500_wave'],
            'wave_count': 1,
            'persistence': 0.4
        }
        clusters = [cluster]
        flag = compute_attention_flag('sp500_wave', clusters, sample_truth_df)
        self.assertEqual(flag, "⏳ Monitor")
    
    def test_attention_flag_none(self):
        """Test no flag for wave not in any cluster."""
        sample_truth_df = get_sample_truth_df()
        clusters = []
        flag = compute_attention_flag('sp500_wave', clusters, sample_truth_df)
        self.assertEqual(flag, "")
    
    # ========================================================================
    # TEST: TIME CONTEXT ENHANCEMENTS
    # ========================================================================
    
    def test_enhance_narrative_new_cluster(self):
        """Test narrative enhancement for new cluster."""
        sample_beta_drift_cluster = get_sample_beta_drift_cluster()
        enhanced = enhance_narrative_with_time_context(
            sample_beta_drift_cluster,
            prior_clusters=[]
        )
        self.assertIn("Beta Drift Detected", enhanced)
        self.assertIn("🆕 Newly detected", enhanced)
    
    def test_enhance_narrative_escalating(self):
        """Test narrative enhancement for escalating cluster."""
        sample_beta_drift_cluster = get_sample_beta_drift_cluster()
        prior_cluster = sample_beta_drift_cluster.copy()
        prior_cluster['cluster_severity'] = 45
        prior_cluster['wave_count'] = 1
        
        enhanced = enhance_narrative_with_time_context(
            sample_beta_drift_cluster,
            prior_clusters=[prior_cluster]
        )
        self.assertIn("⬆️ Escalating", enhanced)
    
    def test_enhance_narrative_improving(self):
        """Test narrative enhancement for improving cluster."""
        sample_beta_drift_cluster = get_sample_beta_drift_cluster()
        prior_cluster = sample_beta_drift_cluster.copy()
        prior_cluster['cluster_severity'] = 85
        prior_cluster['wave_count'] = 3
        
        current = sample_beta_drift_cluster.copy()
        current['cluster_severity'] = 50
        current['wave_count'] = 1
        
        enhanced = enhance_narrative_with_time_context(
            current,
            prior_clusters=[prior_cluster]
        )
        self.assertIn("⬇️ Improving", enhanced)
        self.assertIn("Contracted by 2 waves", enhanced)
    
    # ========================================================================
    # TEST: DECISION SUPPORT SUMMARY
    # ========================================================================
    
    def test_decision_support_summary_generation(self):
        """Test generation of decision support summary for top insights."""
        sample_truth_df = get_sample_truth_df()
        # Generate signals and clusters
        signals = detect_learning_signals(sample_truth_df)
        clusters = cluster_signals(signals, sample_truth_df)
        priority_insights = get_priority_insights(clusters)
        
        # Generate decision support summary
        decision_support = get_decision_support_summary(priority_insights, prior_clusters=[])
        
        # Verify structure
        self.assertIsInstance(decision_support, list)
        self.assertLessEqual(len(decision_support), 3)  # Top 3 max
        
        # Verify each item has required fields
        for item in decision_support:
            self.assertIn('recommended_action', item)
            self.assertIn('risk_of_inaction', item)
            self.assertIn('enhanced_narrative', item)
            self.assertIn('rank', item)
            self.assertIn('cluster_type', item)
            
            # Verify action is non-empty string
            self.assertIsInstance(item['recommended_action'], str)
            self.assertGreater(len(item['recommended_action']), 0)
            
            # Verify risk is valid level
            self.assertIn(item['risk_of_inaction'], ['Low', 'Medium', 'High'])
    
    def test_decision_support_summary_empty_insights(self):
        """Test decision support summary with no insights."""
        decision_support = get_decision_support_summary([], prior_clusters=[])
        self.assertEqual(decision_support, [])
    
    def test_decision_support_summary_deterministic(self):
        """Test that decision support summary is deterministic."""
        insight = {
            'rank': 1,
            'cluster_type': 'regime_mismatch',
            'cluster_name': 'Regime Mismatch Cluster',
            'cluster_severity': 85,
            'wave_count': 3,
            'priority_score': 88.0,
            'narrative': 'Test narrative',
            'justification': 'Test justification',
            'affected_waves': ['wave1', 'wave2', 'wave3'],
            'persistence': 0.9
        }
        
        result1 = get_decision_support_summary([insight], prior_clusters=[])
        result2 = get_decision_support_summary([insight], prior_clusters=[])
        
        self.assertEqual(result1[0]['recommended_action'], result2[0]['recommended_action'])
        self.assertEqual(result1[0]['risk_of_inaction'], result2[0]['risk_of_inaction'])
    
    # ========================================================================
    # TEST: READ-ONLY COMPLIANCE
    # ========================================================================
    
    def test_stage4_functions_read_only(self):
        """Test that Stage 4 functions do not modify input data."""
        sample_truth_df = get_sample_truth_df()
        sample_beta_drift_cluster = get_sample_beta_drift_cluster()
        
        # Create copies to verify no mutation
        cluster_copy = sample_beta_drift_cluster.copy()
        df_copy = sample_truth_df.copy()
        
        # Call all Stage 4 functions
        generate_recommended_action(cluster_copy)
        calculate_risk_of_inaction(cluster_copy)
        compute_attention_flag('tech_wave', [cluster_copy], df_copy)
        enhance_narrative_with_time_context(cluster_copy, [])
        
        # Verify originals unchanged
        self.assertEqual(sample_beta_drift_cluster, cluster_copy)
        self.assertTrue(sample_truth_df.equals(df_copy))
    
    def test_decision_support_no_side_effects(self):
        """Test that decision support generation has no side effects."""
        sample_truth_df = get_sample_truth_df()
        original_df = sample_truth_df.copy()
        
        # Run full pipeline
        signals = detect_learning_signals(sample_truth_df)
        clusters = cluster_signals(signals, sample_truth_df)
        insights = get_priority_insights(clusters)
        get_decision_support_summary(insights, [])
        
        # Verify TruthFrame unchanged
        self.assertTrue(sample_truth_df.equals(original_df))


# ============================================================================
# SUMMARY
# ============================================================================

if __name__ == "__main__":
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestStage4DecisionSupport)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("STAGE 4 TEST SUMMARY")
    print("="*70)
    if result.wasSuccessful():
        print("✅ All Stage 4 Decision Support Layer tests passed!")
        print(f"   Total tests: {result.testsRun}")
        print("   - Deterministic action recommendations")
        print("   - Risk of inaction calculations")
        print("   - Wave-level attention flags")
        print("   - Time & trend context enhancements")
        print("   - Decision support summary generation")
        print("   - Read-only compliance")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} test(s) had errors")
    print("="*70)
