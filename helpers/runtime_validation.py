"""
helpers/runtime_validation.py
Runtime validation utilities for intelligence pipeline outputs.
"""

import logging


def assert_not_empty(df, name):
    """Raise RuntimeError if df is None or has zero rows.

    Used as a hard assertion after pipeline execution to guarantee that
    panels always receive populated data.  Empty output is treated as a
    pipeline failure rather than a silent no-data state.

    The bootstrap layer (helpers/intelligence_bootstrap.py) must ensure
    that all DataFrames and dicts passed to rendering are pre-populated
    with synthetic fallback data so that this assertion is never triggered
    under normal operation.
    """
    if df is None:
        logging.error("[assert_not_empty] %s: received None — pipeline produced no data.", name)
        raise RuntimeError(
            f"{name} produced empty dataset — pipeline failed."
        )
    has_len = hasattr(df, "__len__")
    length = len(df) if has_len else None
    if has_len and length == 0:
        logging.error(
            "[assert_not_empty] %s: received zero-length %s — pipeline produced no data.",
            name, type(df).__name__,
        )
        raise RuntimeError(
            f"{name} produced empty dataset — pipeline failed."
        )
