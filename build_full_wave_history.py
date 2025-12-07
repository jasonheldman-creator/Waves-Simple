import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)  # stable demo data

# ---------------------------------------------------------
# 1) Load wave names from your existing wave_weights.csv
# ---------------------------------------------------------
wave_weights = pd.read_csv("wave_weights.csv")

if "wave" not in wave_weights.columns:
    raise ValueError("wave_weights.csv must have a 'wave' column before building wave_history.csv")

waves = sorted(wave_weights["wave"].dropna().unique())
print("Waves detected:", waves)

# ---------------------------------------------------------
# 2) Build a 252-day date range (approx 1 trading year)
# ---------------------------------------------------------
NUM_DAYS = 252
end_date = datetime.today().date()
dates = [end_date - timedelta(days=i) for i in range(NUM_DAYS)]
dates = sorted(dates)

# ---------------------------------------------------------
# 3) Generate realistic portfolio & benchmark returns
# ---------------------------------------------------------
history_rows = []

for wave in waves:
    name = wave.lower()

    if "crypto" in name:
        port_mu = 0.0012   # higher expected return
        port_sigma = 0.03  # high volatility
        bench_mu = 0.0009
        bench_sigma = 0.025
    elif "income" in name:
        port_mu = 0.0003
        port_sigma = 0.005
        bench_mu = 0.00025
        bench_sigma = 0.004
    elif "s&p" in name:
        port_mu = 0.00045
        port_sigma = 0.011
        bench_mu = 0.00040
        bench_sigma = 0.010
    else:
        # default equity profile (growth / sector / thematic)
        port_mu = 0.0006
        port_sigma = 0.012
        bench_mu = 0.00045
        bench_sigma = 0.010

    for d in dates:
        port_ret = float(np.random.normal(port_mu, port_sigma))
        bench_ret = float(np.random.normal(bench_mu, bench_sigma))

        history_rows.append(
            {
                "date": d.isoformat(),
                "wave": wave,
                "portfolio_return": port_ret,
                "benchmark_return": bench_ret,
            }
        )

# ---------------------------------------------------------
# 4) Write wave_history.csv
# ---------------------------------------------------------
wave_history_df = pd.DataFrame(history_rows)
wave_history_df = wave_history_df.sort_values(["wave", "date"])

wave_history_df.to_csv("wave_history.csv", index=False)

print("✔ wave_history.csv written.")
print("   Rows:", len(wave_history_df))
print("   Waves:", len(waves))
print("   Dates:", wave_history_df['date'].nunique())
print("Columns:", list(wave_history_df.columns))
