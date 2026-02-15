import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load your existing wave weights so we get the REAL wave names
wave_weights = pd.read_csv("wave_weights.csv")
waves = sorted(wave_weights["wave"].unique())

# How many days of history to create (you can change this)
NUM_DAYS = 180  # ~9 months of trading days

end_date = datetime.today().date()
dates = [end_date - timedelta(days=i) for i in range(NUM_DAYS)]
dates = sorted(dates)

rows = []

for wave in waves:
    # You can tweak these if you want different profiles per wave
    port_mu = 0.0005   # ~+12–13% annualized
    port_sigma = 0.01  # daily volatility

    bench_mu = 0.0004  # ~+10% annualized
    bench_sigma = 0.008

    for d in dates:
        port_ret = float(np.random.normal(port_mu, port_sigma))
        bench_ret = float(np.random.normal(bench_mu, bench_sigma))
        rows.append(
            {
                "date": d.isoformat(),
                "wave": wave,
                "portfolio_return": port_ret,
                "benchmark_return": bench_ret,
            }
        )

df = pd.DataFrame(rows)
df.to_csv("wave_history.csv", index=False)
print("wave_history.csv generated with", len(df), "rows and", len(waves), "waves.")
