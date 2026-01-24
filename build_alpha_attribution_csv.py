import pandas as pd

df = pd.read_csv("data/live_snapshot.csv")

def build_attribution(row):
    alpha = row["Alpha_365D"]

    if pd.isna(alpha):
        return pd.Series([0,0,0,0,0,0])

    # Core weights (can be tuned later)
    dynamic = alpha * 0.25
    momentum = alpha * 0.25
    stock = alpha * 0.15

    vix_adj = -(row.get("VIX_Adjustment_Pct", 0) or 0)
    vix = alpha * vix_adj

    beta_drift = abs(row.get("Beta_Drift", 0) or 0)
    risk = -alpha * beta_drift

    residual = alpha - (dynamic + momentum + stock + vix + risk)

    return pd.Series([
        dynamic,
        momentum,
        stock,
        vix,
        risk,
        residual
    ])

df[[
    "Alpha_Dynamic_Benchmark_365D",
    "Alpha_Momentum_Trend_365D",
    "Alpha_Stock_Selection_365D",
    "Alpha_Market_Regime_VIX_365D",
    "Alpha_Risk_Management_365D",
    "Alpha_Residual_365D"
]] = df.apply(build_attribution, axis=1)

df.to_csv("data/live_snapshot_attribution.csv", index=False)
print("✅ Attribution CSV built successfully")