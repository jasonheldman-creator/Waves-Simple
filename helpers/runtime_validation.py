"""
runtime_validation.py
Runtime validation guards for panel DataFrames.

Each function raises ``RuntimeError`` when validation fails, ensuring
that empty panels surface as explicit failures rather than silent
placeholder messages.
"""

from __future__ import annotations

import pandas as pd


def assert_not_empty(df: pd.DataFrame | None, name: str) -> None:
    """Raise RuntimeError if *df* is None or has no rows.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    name:
        Human-readable panel name used in the error message.

    Raises
    ------
    RuntimeError
        When *df* is ``None``, not a DataFrame, or has zero rows.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(
            f"Panel '{name}': required dataset is empty. "
            "Ensure bootstrap data is loaded or live data is available."
        )


def assert_has_columns(
    df: pd.DataFrame | None, name: str, required_cols: list[str]
) -> None:
    """Raise RuntimeError if *df* is missing any of *required_cols*.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    name:
        Human-readable panel name used in the error message.
    required_cols:
        Column names that must be present in *df*.

    Raises
    ------
    RuntimeError
        When *df* is invalid or one or more *required_cols* are absent.
    """
    assert_not_empty(df, name)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Panel '{name}': DataFrame missing required columns: {sorted(missing)}."
        )
