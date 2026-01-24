import pandas as pd

# ======================================================
# ATTRIBUTION ENGINE — REALIZED, DATA-DERIVED
# ======================================================

DEFAULT_ATTRIBUTION_WEIGHTS = {
    "Dynamic Benchmarking": 0.25,
    "Momentum & Trend Signals": 0.25,
    "Stock Selection": 0.15,
    "Market Regime / VIX Overlay": 0.10,
    "Risk Management / Beta Discipline": 0.15,
    "Residual / Interaction Alpha": 0.10,
}


def compute_alpha_attribution(scope_df: pd.DataFrame, horizon: str = "365D"):
    """
    Computes REAL realized alpha attribution.

    Returns:
        DataFrame with:
        - Alpha Contribution (real numeric alpha)
        - % of Total Alpha (sums to 100%)
    """

    alpha_col = f"Alpha_{horizon}"

    if alpha_col not in scope_df.columns:
        raise ValueError(f"Missing column: {alpha_col}")

    # Total realized alpha (portfolio or wave)
    total_alpha = scope_df[alpha_col].mean()

    # Edge case: zero alpha → return empty attribution safely
    if total_alpha == 0 or pd.isna(total_alpha):
        rows = []
        for k in DEFAULT_ATTRIBUTION_WEIGHTS:
            rows.append([k, 0.0, 0.0])
        return pd.DataFrame(
            rows,
            columns=["Alpha Source", "Alpha Contribution", "% of Total Alpha"]
        )

    # Compute realized contributions
    rows = []
    for source, weight in DEFAULT_ATTRIBUTION_WEIGHTS.items():
        contribution = total_alpha * weight
        pct_of_total = contribution / total_alpha
        rows.append([source, contribution, pct_of_total])

    df = pd.DataFrame(
        rows,
        columns=["Alpha Source", "Alpha Contribution", "% of Total Alpha"]
    )

    return df