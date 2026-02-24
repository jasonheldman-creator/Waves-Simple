"""
test_bootstrap_data.py

Tests for the deterministic synthetic bootstrap data generators in
helpers/bootstrap_data.py and the runtime validation guards in
helpers/runtime_validation.py.
"""

import pytest
import pandas as pd
import numpy as np

from helpers.bootstrap_data import (
    bootstrap_alpha_quality_ranking,
    bootstrap_capital_pressure_regime,
    bootstrap_rotation_velocity,
    bootstrap_alpha_ignition_surface,
    bootstrap_cross_horizon_stability,
    bootstrap_learning_diagnostics,
    bootstrap_adaptive_regime_diagnostics,
)
from helpers.runtime_validation import assert_not_empty, assert_has_columns


# ---------------------------------------------------------------------------
# runtime_validation tests
# ---------------------------------------------------------------------------

def test_assert_not_empty_raises_on_none():
    with pytest.raises(RuntimeError, match="Panel 'TestPanel'"):
        assert_not_empty(None, "TestPanel")


def test_assert_not_empty_raises_on_empty_df():
    with pytest.raises(RuntimeError, match="Panel 'TestPanel'"):
        assert_not_empty(pd.DataFrame(), "TestPanel")


def test_assert_not_empty_passes_on_nonempty_df():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert_not_empty(df, "TestPanel")  # should not raise


def test_assert_has_columns_raises_on_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(RuntimeError, match="missing required columns"):
        assert_has_columns(df, "TestPanel", ["a", "b", "c"])


def test_assert_has_columns_passes_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert_has_columns(df, "TestPanel", ["a", "b", "c"])  # should not raise


# ---------------------------------------------------------------------------
# bootstrap_alpha_quality_ranking
# ---------------------------------------------------------------------------

def test_alpha_quality_ranking_nonempty():
    df = bootstrap_alpha_quality_ranking()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 0


def test_alpha_quality_ranking_columns():
    df = bootstrap_alpha_quality_ranking()
    for col in ["wave", "alpha_quality_score", "consistency",
                "total_alpha_30D", "total_alpha_60D", "total_alpha_90D", "total_alpha_365D"]:
        assert col in df.columns, f"Missing column: {col}"


def test_alpha_quality_ranking_consistency_range():
    df = bootstrap_alpha_quality_ranking()
    assert df["consistency"].between(0.0, 1.0).all()


def test_alpha_quality_ranking_sorted_descending():
    df = bootstrap_alpha_quality_ranking()
    scores = df["alpha_quality_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_alpha_quality_ranking_deterministic():
    df1 = bootstrap_alpha_quality_ranking()
    df2 = bootstrap_alpha_quality_ranking()
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# bootstrap_capital_pressure_regime
# ---------------------------------------------------------------------------

def test_capital_pressure_regime_nonempty():
    df = bootstrap_capital_pressure_regime()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_capital_pressure_regime_columns():
    df = bootstrap_capital_pressure_regime()
    for col in ["wave", "capital_pressure_score", "regime_bin",
                "residual_alpha_30D", "regime_alpha_30D"]:
        assert col in df.columns, f"Missing column: {col}"


def test_capital_pressure_regime_bins():
    df = bootstrap_capital_pressure_regime()
    assert df["regime_bin"].isin(["High", "Neutral", "Low"]).all()


def test_capital_pressure_regime_deterministic():
    df1 = bootstrap_capital_pressure_regime()
    df2 = bootstrap_capital_pressure_regime()
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# bootstrap_rotation_velocity
# ---------------------------------------------------------------------------

def test_rotation_velocity_nonempty():
    df = bootstrap_rotation_velocity()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_rotation_velocity_columns():
    df = bootstrap_rotation_velocity()
    for col in ["wave", "total_alpha_30D", "total_alpha_365D",
                "rotation_velocity", "direction"]:
        assert col in df.columns, f"Missing column: {col}"


def test_rotation_velocity_direction_values():
    df = bootstrap_rotation_velocity()
    assert df["direction"].isin(["Accelerating", "Decelerating"]).all()


def test_rotation_velocity_formula():
    df = bootstrap_rotation_velocity()
    computed = (df["total_alpha_30D"] - df["total_alpha_365D"]).round(6)
    assert np.allclose(computed.values, df["rotation_velocity"].values, atol=1e-6)


def test_rotation_velocity_deterministic():
    df1 = bootstrap_rotation_velocity()
    df2 = bootstrap_rotation_velocity()
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# bootstrap_alpha_ignition_surface
# ---------------------------------------------------------------------------

def test_alpha_ignition_surface_nonempty():
    df = bootstrap_alpha_ignition_surface()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_alpha_ignition_surface_columns():
    df = bootstrap_alpha_ignition_surface()
    for col in ["wave", "horizon", "ignition_score",
                "selection_alpha", "momentum_alpha", "volatility_alpha"]:
        assert col in df.columns, f"Missing column: {col}"


def test_alpha_ignition_surface_horizon_labels():
    df = bootstrap_alpha_ignition_surface()
    assert df["horizon"].isin(["30D", "60D", "90D", "365D"]).all()


def test_alpha_ignition_surface_sorted_descending():
    df = bootstrap_alpha_ignition_surface()
    scores = df["ignition_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_alpha_ignition_surface_deterministic():
    df1 = bootstrap_alpha_ignition_surface()
    df2 = bootstrap_alpha_ignition_surface()
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# bootstrap_cross_horizon_stability
# ---------------------------------------------------------------------------

def test_cross_horizon_stability_nonempty():
    result = bootstrap_cross_horizon_stability()
    assert isinstance(result, dict)
    assert "drivers" in result
    assert "summary" in result
    assert len(result["drivers"]) > 0


def test_cross_horizon_stability_driver_keys():
    result = bootstrap_cross_horizon_stability()
    for driver in result["drivers"]:
        for key in ["Driver", "30D State", "90D State", "365D State", "Stability"]:
            assert key in driver, f"Driver missing key: {key}"


def test_cross_horizon_stability_summary_nonempty():
    result = bootstrap_cross_horizon_stability()
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_cross_horizon_stability_deterministic():
    r1 = bootstrap_cross_horizon_stability()
    r2 = bootstrap_cross_horizon_stability()
    assert r1["drivers"] == r2["drivers"]
    assert r1["summary"] == r2["summary"]


# ---------------------------------------------------------------------------
# bootstrap_learning_diagnostics
# ---------------------------------------------------------------------------

def test_learning_diagnostics_nonempty():
    lc, ec = bootstrap_learning_diagnostics()
    assert isinstance(lc, dict)
    assert isinstance(ec, dict)
    assert lc.get("has_data") is True
    assert ec.get("has_data") is True


def test_learning_diagnostics_lc_keys():
    lc, _ = bootstrap_learning_diagnostics()
    for key in ["has_data", "learning_index", "grade", "zone", "monthly_points"]:
        assert key in lc, f"learning_curve_data missing key: {key}"


def test_learning_diagnostics_ec_keys():
    _, ec = bootstrap_learning_diagnostics()
    for key in ["has_data", "efficiency_index", "grade", "monthly_points"]:
        assert key in ec, f"efficiency_curve_data missing key: {key}"


def test_learning_diagnostics_monthly_points_length():
    lc, ec = bootstrap_learning_diagnostics()
    assert len(lc["monthly_points"]) >= 2
    assert len(ec["monthly_points"]) >= 2


def test_learning_diagnostics_grade_valid():
    lc, ec = bootstrap_learning_diagnostics()
    assert lc["grade"] in ("A", "B", "C", "D", "F")
    assert ec["grade"] in ("A", "B", "C", "D", "F")


def test_learning_diagnostics_deterministic():
    lc1, ec1 = bootstrap_learning_diagnostics()
    lc2, ec2 = bootstrap_learning_diagnostics()
    assert lc1 == lc2
    assert ec1 == ec2


# ---------------------------------------------------------------------------
# bootstrap_adaptive_regime_diagnostics
# ---------------------------------------------------------------------------

def test_adaptive_regime_diagnostics_nonempty():
    params = bootstrap_adaptive_regime_diagnostics()
    assert isinstance(params, list)
    assert len(params) > 0


def test_adaptive_regime_diagnostics_keys():
    params = bootstrap_adaptive_regime_diagnostics()
    for p in params:
        assert "name" in p, "param entry missing 'name'"
        assert "status" in p, "param entry missing 'status'"


def test_adaptive_regime_diagnostics_status_values():
    params = bootstrap_adaptive_regime_diagnostics()
    valid_statuses = {"Stable", "Monitoring", "Review"}
    for p in params:
        assert p["status"] in valid_statuses, f"Unexpected status: {p['status']}"


def test_adaptive_regime_diagnostics_deterministic():
    p1 = bootstrap_adaptive_regime_diagnostics()
    p2 = bootstrap_adaptive_regime_diagnostics()
    assert p1 == p2


if __name__ == "__main__":
    import sys

    tests = [
        test_assert_not_empty_raises_on_none,
        test_assert_not_empty_raises_on_empty_df,
        test_assert_not_empty_passes_on_nonempty_df,
        test_assert_has_columns_raises_on_missing,
        test_assert_has_columns_passes_when_all_present,
        test_alpha_quality_ranking_nonempty,
        test_alpha_quality_ranking_columns,
        test_alpha_quality_ranking_consistency_range,
        test_alpha_quality_ranking_sorted_descending,
        test_alpha_quality_ranking_deterministic,
        test_capital_pressure_regime_nonempty,
        test_capital_pressure_regime_columns,
        test_capital_pressure_regime_bins,
        test_capital_pressure_regime_deterministic,
        test_rotation_velocity_nonempty,
        test_rotation_velocity_columns,
        test_rotation_velocity_direction_values,
        test_rotation_velocity_formula,
        test_rotation_velocity_deterministic,
        test_alpha_ignition_surface_nonempty,
        test_alpha_ignition_surface_columns,
        test_alpha_ignition_surface_horizon_labels,
        test_alpha_ignition_surface_sorted_descending,
        test_alpha_ignition_surface_deterministic,
        test_cross_horizon_stability_nonempty,
        test_cross_horizon_stability_driver_keys,
        test_cross_horizon_stability_summary_nonempty,
        test_cross_horizon_stability_deterministic,
        test_learning_diagnostics_nonempty,
        test_learning_diagnostics_lc_keys,
        test_learning_diagnostics_ec_keys,
        test_learning_diagnostics_monthly_points_length,
        test_learning_diagnostics_grade_valid,
        test_learning_diagnostics_deterministic,
        test_adaptive_regime_diagnostics_nonempty,
        test_adaptive_regime_diagnostics_keys,
        test_adaptive_regime_diagnostics_status_values,
        test_adaptive_regime_diagnostics_deterministic,
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
