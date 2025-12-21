import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)  # stable demo data

# -------------------------------------------------------------------
# 1) DEFINE YOUR 9 INSTITUTIONAL WAVES + HOLDINGS
# -------------------------------------------------------------------
wave_tickers = {
    "S&P Wave": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "BRK.B", "JPM", "JNJ", "V", "PG",
    ],
    "Growth Wave": [
        "TSLA", "NVDA", "AMD", "AVGO", "ADBE",
        "CRM", "SHOP", "NOW", "NFLX", "INTU",
    ],
    "Future Power & Energy Wave": [
        "XOM", "CVX", "NEE", "DUK", "ENPH",
        "FSLR", "LNG", "SLB", "PSX", "PLUG",
    ],
    "Small Cap Growth Wave": [
        "TTD", "PLTR", "TWLO", "OKTA", "ZS",
        "DDOG", "APPF", "MDB", "ESTC", "SMAR",
    ],
    "Small-Mid Cap Growth Wave": [
        "CRWD", "ZS", "DDOG", "OKTA", "TEAM",
        "SNOW", "NET", "BILL", "DOCU", "PATH",
    ],
    "Clean Transit-Infrastructure Wave": [
        "TSLA", "NIO", "GM", "F", "BLDP",
        "PLUG", "CHPT", "ABB", "DE", "URI",
    ],
    "Quantum Computing Wave": [
        "NVDA", "AMD", "IBM", "MSFT", "GOOGL",
        "AMZN", "IONQ", "QUBT", "BRKS", "INTC",
    ],
    "Income Wave": [
        "TLT", "LQD", "HYG", "JNJ", "PG",
        "KO", "PEP", "XLU", "O", "VZ",
    ],
}

waves = list(wave_tickers.keys())

# -------------------------------------------------------------------
# 2) BUILD wave_weights.csv
# -------------------------------------------------------------------
weight_rows = []
for wave, tickers in wave_tickers.items():
    n = len(tickers)
    w = 1.0 / n
    for t in tickers:
        weight_rows.append(
            {
                "wave": wave,
                "ticker": t,
                "weight": w,
            }
        )

wave_weights_df = pd.DataFrame(weight_rows)
wave_weights_df.to_csv("wave_weights.csv", index=False)
print("✔ wave_weights.csv written with", len(wave_weights_df), "rows.")

# -------------------------------------------------------------------
# 3) BUILD market_history.csv  (SPY + VIX demo series)
# -------------------------------------------------------------------
NUM_DAYS = 180  # ~9 months of trading days

end_date = datetime.today().date()
dates = [end_date - timedelta(days=i) for i in range(NUM_DAYS)]
dates = sorted(dates)

market_rows = []

# SPY price series (simple random walk)
spy_price = 450.0
for d in dates:
    daily_ret = np.random.normal(0.0005, 0.01)  # ~12% annual, 16% vol
    spy_price *= (1 + daily_ret)
    market_rows.append({"date": d.isoformat(), "symbol": "SPY", "close": spy_price})

# VIX level series
vix_level = 18.0
for d in dates:
    daily_change = np.random.normal(0.0, 0.4)
    vix_level = max(10.0, vix_level + daily_change)  # keep it >=10
    market_rows.append({"date": d.isoformat(), "symbol": "VIX", "close": vix_level})

market_df = pd.DataFrame(market_rows)
market_df.to_csv("market_history.csv", index=False)
print("✔ market_history.csv written with", len(market_df), "rows.")

# -------------------------------------------------------------------
# 4) BUILD wave_history.csv  (returns for each wave vs benchmark)
# -------------------------------------------------------------------
history_rows = []

for wave in waves:
    # Slightly different profiles per wave (you can tweak these later)
    if "Income" in wave:
        port_mu = 0.0003
        port_sigma = 0.005
        bench_mu = 0.00025
        bench_sigma = 0.004
    else:
        port_mu = 0.0006
        port_sigma = 0.012
        bench_mu = 0.00045
        bench_sigma = 0.01

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

wave_history_df = pd.DataFrame(history_rows)
wave_history_df.to_csv("wave_history.csv", index=False)
print("✔ wave_history.csv written with", len(wave_history_df), "rows and", len(waves), "waves.")

print("\nAll demo CSVs generated successfully:")
print(" - wave_weights.csv")
print(" - market_history.csv")
print(" - wave_history.csv")
print("You can now commit these and refresh the Streamlit app.")
