"""
test_adaptive_learning_pipeline.py

Tests for the adaptive_learning.py module focusing on the canonical pipeline:
- load_governance_decisions() normalization
- compute_learning_curve() with NOTE 036/037 weights and bootstrap path
- compute_efficiency_curve() with aliased keys and engaged-signal counting
- compute_cross_horizon_stability() per-horizon sign-agreement stability
- update_adaptive_state() dynamic learning_rate
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import adaptive_learning as al


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_GOVERNANCE_DECISIONS = [
    {
        "id": "startup_wave_a_001",
        "wave": "Alpha Wave",
        "decision_type": "system_generated",
        "trigger_source": "startup_pipeline",
        "context_snapshot": {"action": "Monitor alpha decay", "item_type": "monitor"},
        "source": "initialize_live_system",
        "status": "Awaiting Approval",
        "created": "2026-01-15T10:00:00.000Z",
    },
    {
        "id": "startup_wave_b_002",
        "wave": "Beta Wave",
        "decision_type": "system_generated",
        "trigger_source": "startup_pipeline",
        "context_snapshot": {"action": "Short-term alpha weak", "item_type": "monitor"},
        "source": "initialize_live_system",
        "status": "Awaiting Approval",
        "created": "2026-01-16T10:00:00.000Z",
    },
    {
        "id": "startup_wave_c_003",
        "wave": "Gamma Wave",
        "decision_type": "system_generated",
        "trigger_source": "startup_pipeline",
        "context_snapshot": {"action": "Long-term alpha positive", "item_type": "monitor"},
        "source": "initialize_live_system",
        "status": "Awaiting Approval",
        "created": "2026-01-17T10:00:00.000Z",
    },
]

SAMPLE_OUTCOME_DECISIONS = [
    {
        "id": "dec_001",
        "date": "2026-01-10",
        "wave": "Alpha Wave",
        "decision_type": "governance",
        "status": "Recorded",
        "approval_status": "Approved",
        "outcome_30d": "Positive",
        "regime_at_decision": "Normal",
    },
    {
        "id": "dec_002",
        "date": "2026-01-12",
        "wave": "Beta Wave",
        "decision_type": "governance",
        "status": "Recorded",
        "approval_status": "Approved",
        "outcome_30d": "Neutral",
        "regime_at_decision": "Normal",
    },
    {
        "id": "dec_003",
        "date": "2026-01-14",
        "wave": "Gamma Wave",
        "decision_type": "governance",
        "status": "Recorded",
        "approval_status": "Approved",
        "outcome_30d": "Positive",
        "regime_at_decision": "Normal",
    },
]


def _make_attrib_df():
    """Minimal attribution DataFrame with 30D and 365D horizons."""
    rows = []
    for wave, s30, s365 in [
        ("Alpha Wave", 0.01, 0.03),
        ("Beta Wave", -0.005, 0.02),
        ("Gamma Wave", 0.008, 0.015),
    ]:
        for horizon, sel in [(30, s30), (365, s365)]:
            rows.append({
                "wave": wave,
                "horizon": horizon,
                "total_alpha": sel,
                "selection_alpha": sel,
                "momentum_alpha": sel * 0.5,
                "volatility_alpha": sel * 0.1,
                "regime_alpha": sel * 0.1,
                "exposure_alpha": sel * 0.1,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# load_governance_decisions
# ---------------------------------------------------------------------------

def test_load_governance_decisions_from_temp_file():
    """load_governance_decisions() normalises fields correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_GOVERNANCE_DECISIONS, f)
        tmp_path = f.name

    try:
        decisions = al.load_governance_decisions(path=tmp_path)
        assert len(decisions) == 3

        d = decisions[0]
        assert d["id"] == "startup_wave_a_001"
        assert d["date"] == "2026-01-15"
        assert d["wave"] == "Alpha Wave"
        assert d["status"] == "Awaiting Approval"
        assert d["outcome_30d"] is None
        # context should be the action string extracted from context_snapshot
        assert d["context"] == "Monitor alpha decay"
    finally:
        os.unlink(tmp_path)


def test_load_governance_decisions_missing_file():
    """load_governance_decisions() returns empty list when file not found."""
    result = al.load_governance_decisions(path="/nonexistent/path.json")
    assert result == []


# ---------------------------------------------------------------------------
# compute_learning_curve
# ---------------------------------------------------------------------------

def test_compute_learning_curve_with_outcome_decisions():
    """NOTE 036/037: 60/25/15 weights produce correct learning index."""
    state = al.load_adaptive_state()
    result = al.compute_learning_curve(None, None, state, SAMPLE_OUTCOME_DECISIONS)

    assert result["has_data"] is True
    assert "learning_index" in result
    assert "grade" in result

    # 3 recorded; 2 positive (67%), 1 neutral (33%); 0 negative
    doa = 2 / 3 * 100          # ~66.67%
    oc = 3 / 3 * 100            # 100%
    si = min(100, doa * 0.8 + oc * 0.2)
    expected_li = doa * 0.60 + oc * 0.25 + si * 0.15
    assert abs(result["learning_index"] - expected_li) < 0.1

    # Aliased keys must be present
    assert "decision_alignment_pct" in result
    assert "outcome_consistency_pct" in result
    assert "structural_improvement_pct" in result


def test_compute_learning_curve_bootstrap_path():
    """When no outcome decisions exist, bootstrap path must populate the curve."""
    state = {"confidence": 0.6, "regime_state": "normal", "pattern_memory": {}}
    result = al.compute_learning_curve(None, None, state, SAMPLE_GOVERNANCE_DECISIONS)

    assert result["has_data"] is True
    assert result["learning_index"] > 0
    assert len(result["monthly_points"]) >= 1
    # All aliased keys present
    assert result["decision_alignment_pct"] > 0
    assert result["outcome_consistency_pct"] > 0


def test_compute_learning_curve_empty_decisions():
    """Empty decision list returns has_data=False."""
    state = al.load_adaptive_state()
    result = al.compute_learning_curve(None, None, state, [])
    assert result["has_data"] is False


# ---------------------------------------------------------------------------
# compute_efficiency_curve
# ---------------------------------------------------------------------------

def test_compute_efficiency_curve_awaiting_approval_counts_as_engaged():
    """'Awaiting Approval' decisions must count as engaged signals (signal_engagement_pct > 0)."""
    state = al.load_adaptive_state()
    result = al.compute_efficiency_curve(SAMPLE_GOVERNANCE_DECISIONS, state)

    assert result["has_data"] is True
    # All 3 decisions have 'Awaiting Approval' → 100% engagement
    assert result["signal_engagement_pct"] == 100
    assert result["implementation_rate_pct"] == 0  # none approved yet

    # Aliased keys must be present
    assert "signal_engagement_pct" in result
    assert "implementation_rate_pct" in result
    assert "avg_decision_latency_hours" in result


def test_compute_efficiency_curve_outcome_decisions():
    """Approved decisions register in implementation_rate_pct."""
    state = al.load_adaptive_state()
    result = al.compute_efficiency_curve(SAMPLE_OUTCOME_DECISIONS, state)

    assert result["has_data"] is True
    # All 3 have status 'Recorded' → engaged
    assert result["signal_engagement_pct"] == 100
    # All 3 have approval_status 'Approved' → implemented
    assert result["implementation_rate_pct"] == 100


def test_compute_efficiency_curve_empty():
    """Fewer than 2 decisions returns has_data=False."""
    state = al.load_adaptive_state()
    assert al.compute_efficiency_curve([], state)["has_data"] is False
    assert al.compute_efficiency_curve([SAMPLE_GOVERNANCE_DECISIONS[0]], state)["has_data"] is False


# ---------------------------------------------------------------------------
# compute_cross_horizon_stability
# ---------------------------------------------------------------------------

def test_compute_cross_horizon_stability_sign_agreement():
    """Stability is 'Stable' when 30D and 365D share the same sign."""
    attrib_df = _make_attrib_df()
    result = al.compute_cross_horizon_stability(None, attrib_df)

    assert len(result["drivers"]) == 5
    # All drivers in the fixture have positive values at both 30D and 365D
    for driver in result["drivers"]:
        assert driver["Stability"] in ("Stable", "Moderate", "Volatile")

    stable_count = sum(1 for d in result["drivers"] if d["Stability"] == "Stable")
    assert stable_count >= 1, "At least one driver should be stable with matching signs"
    assert "stable" in result["summary"].lower()


def test_compute_cross_horizon_stability_opposite_signs():
    """Driver with opposite 30D / 365D sign should be 'Volatile'."""
    rows = []
    for horizon, val in [(30, -0.01), (365, 0.01)]:
        rows.append({
            "wave": "Test Wave",
            "horizon": horizon,
            "total_alpha": val,
            "selection_alpha": val,
            "momentum_alpha": val * 0.5,
            "volatility_alpha": val * 0.1,
            "regime_alpha": val * 0.1,
            "exposure_alpha": val * 0.1,
        })
    attrib_df = pd.DataFrame(rows)
    result = al.compute_cross_horizon_stability(None, attrib_df)

    selection_driver = next((d for d in result["drivers"] if d["Driver"] == "Selection"), None)
    assert selection_driver is not None
    assert selection_driver["Stability"] == "Volatile"


def test_compute_cross_horizon_stability_no_data():
    """Returns empty drivers list with informative summary when no data."""
    result = al.compute_cross_horizon_stability(None, None)
    assert result["drivers"] == []
    assert "Insufficient" in result["summary"]


# ---------------------------------------------------------------------------
# update_adaptive_state – dynamic learning_rate
# ---------------------------------------------------------------------------

def test_update_adaptive_state_sets_learning_rate():
    """update_adaptive_state must set a non-zero learning_rate."""
    state = {"confidence": 0.5, "regime_state": "normal", "pattern_memory": {}}
    updated_state, _ = al.update_adaptive_state(None, None, state)

    assert "learning_rate" in updated_state
    assert updated_state["learning_rate"] > 0, "learning_rate must be > 0 when system is active"


def test_update_adaptive_state_learning_rate_scales_with_confidence():
    """Higher confidence should produce a higher learning_rate."""
    state_low = {"confidence": 0.2, "regime_state": "normal", "pattern_memory": {}}
    state_high = {"confidence": 0.9, "regime_state": "normal", "pattern_memory": {}}

    updated_low, _ = al.update_adaptive_state(None, None, state_low)
    updated_high, _ = al.update_adaptive_state(None, None, state_high)

    assert updated_high["learning_rate"] > updated_low["learning_rate"]


# ---------------------------------------------------------------------------
# Governance + decision_log merge integration
# ---------------------------------------------------------------------------

def test_governance_merge_no_duplicates():
    """When decision_log and governance decisions overlap on id, no duplicates appear."""
    decision_log = [
        {"id": "startup_wave_a_001", "date": "2026-01-15", "wave": "Alpha Wave",
         "decision_type": "governance", "status": "Recorded", "outcome_30d": "Positive"},
    ]

    existing_ids = {d.get("id") for d in decision_log if d.get("id")}
    merged = decision_log + [d for d in SAMPLE_GOVERNANCE_DECISIONS
                             if d.get("id") not in existing_ids]

    ids = [d["id"] for d in merged]
    assert len(ids) == len(set(ids)), "Merged decision list must not have duplicate IDs"
    assert len(merged) == 3  # 1 from log + 2 new from governance


if __name__ == "__main__":
    import sys
    # Run tests manually
    tests = [
        test_load_governance_decisions_from_temp_file,
        test_load_governance_decisions_missing_file,
        test_compute_learning_curve_with_outcome_decisions,
        test_compute_learning_curve_bootstrap_path,
        test_compute_learning_curve_empty_decisions,
        test_compute_efficiency_curve_awaiting_approval_counts_as_engaged,
        test_compute_efficiency_curve_outcome_decisions,
        test_compute_efficiency_curve_empty,
        test_compute_cross_horizon_stability_sign_agreement,
        test_compute_cross_horizon_stability_opposite_signs,
        test_compute_cross_horizon_stability_no_data,
        test_update_adaptive_state_sets_learning_rate,
        test_update_adaptive_state_learning_rate_scales_with_confidence,
        test_governance_merge_no_duplicates,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"✗ {t.__name__}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
