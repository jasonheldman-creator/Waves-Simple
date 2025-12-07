import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

wave_weights = pd.read_csv("wave_weights.csv")
if "wave" not in wave_weights.columns:
    raise ValueError("wave_weights.csv must have a 'wave' column before building wave_history.csv")

waves = sorted(wave_weights["wave"].dropna().unique())
print("Waves detected:", waves)

NUM_DAYS = 756  # ~3 years
end_date = datetime.today().date()
dates = [end_date - timedelta(days=i) for i in range(NUM_DAYS)]
dates = sorted(dates)

rows = []

for wave in waves:
    name = wave.lower()
    if "crypto" in name:
        port_mu, port_sigma = 0.0012, 0.03
        bench_mu, bench_sigma = 0.0009, 0.025
    elif "income" in name:
        port_mu, port_sigma = 0.0003, 0.005
        bench_mu, bench_sigma = 0.00025, 0.004
    elif "s&p" in name:
        port_mu, port_sigma = 0.00045, 0.011
        bench_mu, bench_sigma = 0.00040, 0.010
    else:
        port_mu, port_sigma = 0.0006, 0.012
        bench_mu, bench_sigma = 0.00045, 0.010

    for d in dates:
        pr = float(np.random.normal(port_mu, port_sigma))
        br = float(np.random.normal(bench_mu, bench_sigma))
        rows.append(
            {
                "date": d.isoformat(),
                "wave": wave,
                "portfolio_return": pr,
                "benchmark_return": br,
            }
        )

df = pd.DataFrame(rows)
df = df.sort_values(["wave", "date"])
df.to_csv("wave_history.csv", index=False)

print("✔ wave_history.csv written.")
print("   Rows:", len(df))
print("   Waves:", len(waves))
print("   Dates:", df['date'].nunique())
print("Columns:", list(df.columns))
