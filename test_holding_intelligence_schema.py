"""
Unit tests for holding_intelligence and strategy_security_optimizer helper modules.

These tests validate the function signatures and return value schemas to ensure
they match what the app.py expects, preventing silent rendering failures.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.holding_intelligence import (
    evaluate_holdings,
    evaluate_secondary_candidates,
    generate_holding_observations,
    get_holdings_summary,
    get_governance_eligible_holdings,
)
from helpers.strategy_security_optimizer import (
    evaluate_strategy_fit,
    evaluate_replacement_candidates,
    generate_strategy_observations,
    get_strategy_fit_summary,
    check_governance_triggers,
    execute_governance_proposals,
)


# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------

def _make_holding(ticker="AAPL", wave="Growth Wave", drift=0.5):
    return {
        "ticker": ticker,
        "wave": wave,
        "weight": 5.0,
        "target_weight": 5.0 + drift,
        "drift": abs(drift),
        "drift_direction": "Under" if drift > 0 else "Flat",
        "review_cycles": 0,
        "status": "Stable",
        "observation": "No significant drift detected.",
        "vol_30d": None,
        "momentum_30d": None,
        "drawdown_90d": None,
        "trend_stability": None,
    }


# ---------------------------------------------------------------------------
# evaluate_holdings
# ---------------------------------------------------------------------------

class TestEvaluateHoldings:
    def test_returns_list(self, tmp_path):
        result = evaluate_holdings(str(tmp_path))
        assert isinstance(result, list)

    def test_empty_when_no_csv(self, tmp_path):
        result = evaluate_holdings(str(tmp_path))
        assert result == []

    def test_required_fields_present(self, tmp_path):
        csv_path = tmp_path / "wave_weights.csv"
        csv_path.write_text("wave_id,ticker,weight\nGrowth Wave,AAPL,5.0\n")
        result = evaluate_holdings(str(tmp_path))
        assert len(result) == 1
        holding = result[0]
        required_fields = [
            "ticker", "wave", "weight", "target_weight", "drift",
            "drift_direction", "review_cycles", "status", "observation",
            "vol_30d", "momentum_30d", "drawdown_90d", "trend_stability",
        ]
        for field in required_fields:
            assert field in holding, f"Missing field: {field}"

    def test_new_metric_fields_are_none_by_default(self, tmp_path):
        csv_path = tmp_path / "wave_weights.csv"
        csv_path.write_text("wave_id,ticker,weight\nIncome Wave,MSFT,3.0\n")
        result = evaluate_holdings(str(tmp_path))
        h = result[0]
        assert h["vol_30d"] is None
        assert h["momentum_30d"] is None
        assert h["drawdown_90d"] is None
        assert h["trend_stability"] is None


# ---------------------------------------------------------------------------
# evaluate_secondary_candidates
# ---------------------------------------------------------------------------

class TestEvaluateSecondaryCandidates:
    def test_no_args_returns_empty_list(self):
        """App calls evaluate_secondary_candidates() with no arguments."""
        result = evaluate_secondary_candidates()
        assert isinstance(result, list)
        assert result == []

    def test_with_holdings_arg(self):
        holdings = [_make_holding(drift=3.0), _make_holding(ticker="GOOG", drift=0.5)]
        result = evaluate_secondary_candidates(holdings)
        assert isinstance(result, list)
        # Only holdings with drift > 2.0 are secondary candidates
        assert len(result) == 1

    def test_wave_filter(self):
        holdings = [
            _make_holding(ticker="AAPL", wave="Growth", drift=3.0),
            _make_holding(ticker="GOOG", wave="Income", drift=3.0),
        ]
        result = evaluate_secondary_candidates(holdings, wave_name="Growth")
        assert all(h["wave"] == "Growth" for h in result)


# ---------------------------------------------------------------------------
# generate_holding_observations
# ---------------------------------------------------------------------------

class TestGenerateHoldingObservations:
    def test_returns_list_of_dicts(self):
        holdings = [_make_holding()]
        obs = generate_holding_observations(holdings)
        assert isinstance(obs, list)
        assert len(obs) > 0
        assert isinstance(obs[0], dict)

    def test_dicts_have_security_and_observation_keys(self):
        holdings = [_make_holding()]
        obs = generate_holding_observations(holdings)
        for o in obs:
            assert "security" in o, "Missing 'security' key in observation"
            assert "observation" in o, "Missing 'observation' key in observation"

    def test_accepts_two_args(self):
        holdings = [_make_holding()]
        secondary = []
        obs = generate_holding_observations(holdings, secondary)
        assert isinstance(obs, list)

    def test_empty_holdings_returns_fallback(self):
        obs = generate_holding_observations([])
        assert isinstance(obs, list)
        assert len(obs) > 0
        assert isinstance(obs[0], dict)


# ---------------------------------------------------------------------------
# get_holdings_summary
# ---------------------------------------------------------------------------

class TestGetHoldingsSummary:
    REQUIRED_KEYS = {"total", "with_data", "coverage_pct", "stable", "monitoring",
                     "review_candidate", "data_pending"}

    def test_empty_returns_correct_keys(self):
        s = get_holdings_summary([])
        assert self.REQUIRED_KEYS.issubset(s.keys())

    def test_non_empty_returns_correct_keys(self):
        holdings = [_make_holding(), _make_holding(ticker="GOOG")]
        s = get_holdings_summary(holdings)
        assert self.REQUIRED_KEYS.issubset(s.keys())

    def test_stable_count(self):
        holdings = [_make_holding(ticker="A"), _make_holding(ticker="B")]
        s = get_holdings_summary(holdings)
        assert s["stable"] == 2
        assert s["monitoring"] == 0
        assert s["review_candidate"] == 0

    def test_coverage_pct_range(self):
        holdings = [_make_holding()]
        s = get_holdings_summary(holdings)
        assert 0.0 <= s["coverage_pct"] <= 100.0

    def test_total_matches_len(self):
        holdings = [_make_holding(ticker=f"T{i}") for i in range(5)]
        s = get_holdings_summary(holdings)
        assert s["total"] == 5


# ---------------------------------------------------------------------------
# evaluate_strategy_fit
# ---------------------------------------------------------------------------

class TestEvaluateStrategyFit:
    VALID_CLASSIFICATIONS = {"Optimal Fit", "Acceptable Fit", "Weak Fit", "Review Candidate", "Data Pending"}

    def test_returns_list(self):
        result = evaluate_strategy_fit([])
        assert isinstance(result, list)

    def test_uses_classification_key(self):
        holdings = [_make_holding()]
        result = evaluate_strategy_fit(holdings)
        assert len(result) > 0
        assert "classification" in result[0], "Expected 'classification' key, not 'fit_label'"

    def test_classification_values_valid(self):
        holdings = [_make_holding(drift=d) for d in [0.5, 5.0, 10.0, 15.0]]
        result = evaluate_strategy_fit(holdings)
        for r in result:
            assert r["classification"] in self.VALID_CLASSIFICATIONS

    def test_fit_score_present(self):
        holdings = [_make_holding()]
        result = evaluate_strategy_fit(holdings)
        for r in result:
            assert "fit_score" in r
            assert isinstance(r["fit_score"], float)


# ---------------------------------------------------------------------------
# evaluate_replacement_candidates
# ---------------------------------------------------------------------------

class TestEvaluateReplacementCandidates:
    REQUIRED_KEYS = {"current_security", "candidate_security", "wave",
                     "current_score", "candidate_score", "relative_fit_improvement"}

    def test_correct_keys(self):
        holdings = [_make_holding(drift=15.0)]  # high drift → Review Candidate
        fit = evaluate_strategy_fit(holdings)
        upgrades = evaluate_replacement_candidates(fit)
        for uc in upgrades:
            assert self.REQUIRED_KEYS.issubset(uc.keys()), \
                f"Missing keys: {self.REQUIRED_KEYS - set(uc.keys())}"

    def test_no_upgrades_for_optimal_fit(self):
        holdings = [_make_holding(drift=0.1)]
        fit = evaluate_strategy_fit(holdings)
        upgrades = evaluate_replacement_candidates(fit)
        assert upgrades == []


# ---------------------------------------------------------------------------
# generate_strategy_observations
# ---------------------------------------------------------------------------

class TestGenerateStrategyObservations:
    def test_returns_list_of_dicts(self):
        holdings = [_make_holding()]
        fit = evaluate_strategy_fit(holdings)
        upgrades = evaluate_replacement_candidates(fit)
        obs = generate_strategy_observations(fit, upgrades)
        assert isinstance(obs, list)
        assert len(obs) > 0
        for o in obs:
            assert isinstance(o, dict), f"Expected dict, got {type(o)}"

    def test_dicts_have_required_keys(self):
        holdings = [_make_holding()]
        fit = evaluate_strategy_fit(holdings)
        upgrades = evaluate_replacement_candidates(fit)
        obs = generate_strategy_observations(fit, upgrades)
        for o in obs:
            assert "security" in o, "Missing 'security' key"
            assert "observation" in o, "Missing 'observation' key"

    def test_obs_map_construction(self):
        """Simulate the app's dict-comprehension to ensure no TypeError."""
        holdings = [_make_holding()]
        fit = evaluate_strategy_fit(holdings)
        upgrades = evaluate_replacement_candidates(fit)
        obs = generate_strategy_observations(fit, upgrades)
        # This is what helpers/app.py does - must not raise TypeError
        obs_map = {o["security"]: o["observation"] for o in obs if "\u2192" not in o.get("security", "")}
        assert isinstance(obs_map, dict)


# ---------------------------------------------------------------------------
# get_strategy_fit_summary
# ---------------------------------------------------------------------------

class TestGetStrategyFitSummary:
    REQUIRED_KEYS = {"optimal", "acceptable", "weak_fit", "review_candidate",
                     "data_pending", "total"}

    def test_empty_returns_correct_keys(self):
        s = get_strategy_fit_summary([])
        assert self.REQUIRED_KEYS.issubset(s.keys())

    def test_non_empty_returns_correct_keys(self):
        holdings = [_make_holding()]
        fit = evaluate_strategy_fit(holdings)
        s = get_strategy_fit_summary(fit)
        assert self.REQUIRED_KEYS.issubset(s.keys())

    def test_counts_match_classifications(self):
        holdings = [_make_holding(drift=d) for d in [0.5, 5.0, 10.0, 15.0]]
        fit = evaluate_strategy_fit(holdings)
        s = get_strategy_fit_summary(fit)
        total_classified = s["optimal"] + s["acceptable"] + s["weak_fit"] + s["review_candidate"] + s["data_pending"]
        assert total_classified == s["total"]
