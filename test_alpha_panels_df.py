"""
test_alpha_panels_df.py

Tests for the four DataFrame-producing wrappers added to adaptive_learning.py:
- compute_alpha_quality_df
- compute_capital_pressure_df
- compute_rotation_velocity_df
- compute_alpha_ignition_df
- _validate_attrib_df (via the public wrappers)
"""

import pandas as pd
import numpy as np
import pytest

import adaptive_learning as al


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_attrib_df(include_residual=True):
    """Minimal valid attribution DataFrame covering all four horizons."""
    rows = []
    waves = [
        ("Wave A", 0.02, 0.01, 0.005, 0.002, 0.001),
        ("Wave B", -0.01, -0.005, -0.003, -0.001, -0.0005),
        ("Wave C", 0.005, 0.002, 0.001, 0.0005, 0.00025),
    ]
    for wave, total, sel, mom, vol, exp in waves:
        for horizon in [30, 60, 90, 365]:
            row = {
                "wave": wave,
                "horizon": horizon,
                "total_alpha": total * (1 + horizon / 1000),
                "selection_alpha": sel,
                "momentum_alpha": mom,
                "volatility_alpha": vol,
                "regime_alpha": vol * 0.5,
                "exposure_alpha": exp,
                "residual_alpha": total * 0.05,
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _validate_attrib_df (via wrappers) – error-path tests
# ---------------------------------------------------------------------------

def test_validate_raises_on_none():
    with pytest.raises(ValueError, match="empty"):
        al.compute_alpha_quality_df(None)


def test_validate_raises_on_empty_df():
    with pytest.raises(ValueError, match="empty"):
        al.compute_alpha_quality_df(pd.DataFrame())


def test_validate_raises_on_missing_columns():
    df = pd.DataFrame({"wave": ["X"], "horizon": [30], "total_alpha": [0.01]})
    with pytest.raises(ValueError, match="missing required columns"):
        al.compute_alpha_quality_df(df)


def test_validate_raises_on_all_unknown_wave():
    df = _make_attrib_df()
    df["wave"] = "UNKNOWN_WAVE"
    with pytest.raises(ValueError, match="UNKNOWN_WAVE"):
        al.compute_alpha_quality_df(df)


def test_validate_filters_unknown_wave_rows():
    """Rows with UNKNOWN_WAVE are dropped; valid rows still produce output."""
    df = _make_attrib_df()
    # Mix one UNKNOWN_WAVE row with valid rows
    modified_row = df.iloc[0].copy()
    modified_row["wave"] = "UNKNOWN_WAVE"
    df = pd.concat([df, pd.DataFrame([modified_row])], ignore_index=True)
    result = al.compute_alpha_quality_df(df)
    assert "UNKNOWN_WAVE" not in result["wave"].values


# ---------------------------------------------------------------------------
# compute_alpha_quality_df
# ---------------------------------------------------------------------------

def test_alpha_quality_df_columns():
    df = _make_attrib_df()
    result = al.compute_alpha_quality_df(df)
    assert isinstance(result, pd.DataFrame)
    assert "wave" in result.columns
    assert "alpha_quality_score" in result.columns
    assert "consistency" in result.columns
    for h in ["30D", "60D", "90D", "365D"]:
        assert f"total_alpha_{h}" in result.columns


def test_alpha_quality_df_sorted_descending():
    df = _make_attrib_df()
    result = al.compute_alpha_quality_df(df)
    scores = result["alpha_quality_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_alpha_quality_df_one_row_per_wave():
    df = _make_attrib_df()
    result = al.compute_alpha_quality_df(df)
    assert len(result) == df["wave"].nunique()


def test_alpha_quality_df_consistency_range():
    df = _make_attrib_df()
    result = al.compute_alpha_quality_df(df)
    assert result["consistency"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# compute_capital_pressure_df
# ---------------------------------------------------------------------------

def test_capital_pressure_df_columns():
    df = _make_attrib_df()
    result = al.compute_capital_pressure_df(df)
    assert isinstance(result, pd.DataFrame)
    for col in ["wave", "capital_pressure_score", "regime_bin",
                "residual_alpha_30D", "regime_alpha_30D"]:
        assert col in result.columns


def test_capital_pressure_df_regime_bins():
    df = _make_attrib_df()
    result = al.compute_capital_pressure_df(df)
    assert result["regime_bin"].isin(["High", "Neutral", "Low"]).all()


def test_capital_pressure_df_one_row_per_wave():
    df = _make_attrib_df()
    result = al.compute_capital_pressure_df(df)
    assert len(result) == df["wave"].nunique()


# ---------------------------------------------------------------------------
# compute_rotation_velocity_df
# ---------------------------------------------------------------------------

def test_rotation_velocity_df_columns():
    df = _make_attrib_df()
    result = al.compute_rotation_velocity_df(df)
    assert isinstance(result, pd.DataFrame)
    for col in ["wave", "total_alpha_30D", "total_alpha_365D",
                "rotation_velocity", "direction"]:
        assert col in result.columns


def test_rotation_velocity_df_direction_values():
    df = _make_attrib_df()
    result = al.compute_rotation_velocity_df(df)
    assert result["direction"].isin(["Accelerating", "Decelerating"]).all()


def test_rotation_velocity_df_velocity_formula():
    """Rotation velocity = total_alpha_30D - total_alpha_365D."""
    df = _make_attrib_df()
    result = al.compute_rotation_velocity_df(df)
    computed = (result["total_alpha_30D"] - result["total_alpha_365D"]).round(6)
    assert (computed == result["rotation_velocity"]).all()


def test_rotation_velocity_df_sorted_by_abs():
    df = _make_attrib_df()
    result = al.compute_rotation_velocity_df(df)
    abs_vals = result["rotation_velocity"].abs().tolist()
    assert abs_vals == sorted(abs_vals, reverse=True)


# ---------------------------------------------------------------------------
# compute_alpha_ignition_df
# ---------------------------------------------------------------------------

def test_alpha_ignition_df_columns():
    df = _make_attrib_df()
    result = al.compute_alpha_ignition_df(df)
    assert isinstance(result, pd.DataFrame)
    for col in ["wave", "horizon", "ignition_score",
                "selection_alpha", "momentum_alpha", "volatility_alpha"]:
        assert col in result.columns


def test_alpha_ignition_df_score_formula():
    """Ignition score = 0.5*sel + 0.3*mom - 0.2*abs(vol)."""
    df = _make_attrib_df()
    result = al.compute_alpha_ignition_df(df)
    computed = (
        0.5 * result["selection_alpha"]
        + 0.3 * result["momentum_alpha"]
        - 0.2 * result["volatility_alpha"].abs()
    ).round(6)
    assert np.allclose(computed.values, result["ignition_score"].values, atol=1e-6)


def test_alpha_ignition_df_sorted_descending():
    df = _make_attrib_df()
    result = al.compute_alpha_ignition_df(df)
    scores = result["ignition_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_alpha_ignition_df_horizon_labels():
    """Horizon values must be normalised to label strings (30D/60D/90D/365D)."""
    df = _make_attrib_df()
    result = al.compute_alpha_ignition_df(df)
    assert result["horizon"].isin(["30D", "60D", "90D", "365D"]).all()


# ---------------------------------------------------------------------------
# Canonical CSV integration
# ---------------------------------------------------------------------------

def test_all_four_functions_with_real_csv():
    """All four wrappers produce non-empty DataFrames from the canonical CSV."""
    csv_path = "data/alpha_attribution_summary.csv"
    try:
        attrib_df = pd.read_csv(csv_path)
        attrib_df.columns = [c.strip().lower() for c in attrib_df.columns]
    except FileNotFoundError:
        pytest.skip(f"Canonical CSV not found at {csv_path}")

    quality = al.compute_alpha_quality_df(attrib_df)
    pressure = al.compute_capital_pressure_df(attrib_df)
    velocity = al.compute_rotation_velocity_df(attrib_df)
    ignition = al.compute_alpha_ignition_df(attrib_df)

    assert not quality.empty, "compute_alpha_quality_df returned empty DataFrame from canonical CSV"
    assert not pressure.empty, "compute_capital_pressure_df returned empty DataFrame from canonical CSV"
    assert not velocity.empty, "compute_rotation_velocity_df returned empty DataFrame from canonical CSV"
    assert not ignition.empty, "compute_alpha_ignition_df returned empty DataFrame from canonical CSV"


if __name__ == "__main__":
    import sys
    tests = [
        test_validate_raises_on_none,
        test_validate_raises_on_empty_df,
        test_validate_raises_on_missing_columns,
        test_validate_raises_on_all_unknown_wave,
        test_validate_filters_unknown_wave_rows,
        test_alpha_quality_df_columns,
        test_alpha_quality_df_sorted_descending,
        test_alpha_quality_df_one_row_per_wave,
        test_alpha_quality_df_consistency_range,
        test_capital_pressure_df_columns,
        test_capital_pressure_df_regime_bins,
        test_capital_pressure_df_one_row_per_wave,
        test_rotation_velocity_df_columns,
        test_rotation_velocity_df_direction_values,
        test_rotation_velocity_df_velocity_formula,
        test_rotation_velocity_df_sorted_by_abs,
        test_alpha_ignition_df_columns,
        test_alpha_ignition_df_score_formula,
        test_alpha_ignition_df_sorted_descending,
        test_alpha_ignition_df_horizon_labels,
        test_all_four_functions_with_real_csv,
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
