"""
helpers/runtime_validation.py
Runtime validation utilities for intelligence pipeline outputs.
"""


def assert_not_empty(df, name):
    """Raise RuntimeError if df is None or has zero rows.

    Used as a hard assertion after pipeline execution to guarantee that
    panels always receive populated data.  Empty output is treated as a
    pipeline failure rather than a silent no-data state.
    """
    if df is None or len(df) == 0:
        raise RuntimeError(
            f"{name} produced empty dataset — pipeline failed."
        )
