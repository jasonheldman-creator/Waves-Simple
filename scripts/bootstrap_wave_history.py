import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Constants
WAVE_REGISTRY_FILE = "data/wave_registry.csv"
WAVE_HISTORY_FILE = "data/wave_history.csv"
BUSINESS_DAYS = 180
OUTPUT_COLUMNS = [
    "date", "wave_id", "nav", "wave_return", "bm_return", "alpha", 
    "exposure", "safe_fraction", "vix", "regime", "beta", "is_synthetic_data"
]


def get_wave_ids():
    if not os.path.exists(WAVE_REGISTRY_FILE):
        raise FileNotFoundError(f"Wave registry file not found: {WAVE_REGISTRY_FILE}")
    wave_registry = pd.read_csv(WAVE_REGISTRY_FILE)
    if "wave_id" not in wave_registry.columns:
        raise ValueError("Missing 'wave_id' column in wave registry file.")
    return wave_registry["wave_id"].unique()


def generate_synthetic_data(wave_ids):
    start_date = datetime.today() - timedelta(days=BUSINESS_DAYS * 1.2)  # Approximation
    business_dates = pd.date_range(start=start_date, periods=BUSINESS_DAYS, freq='B')
    synthetic_data = []
    
    for wave_id in wave_ids:
        np.random.seed(hash(wave_id) % (2**32))
        nav = 100  # Starting NAV
        for date in business_dates:
            wave_return = np.random.normal(0, 0.01)
            bm_return = np.random.normal(0, 0.01)
            alpha = wave_return - bm_return
            exposure = np.random.uniform(0, 1)
            safe_fraction = np.random.uniform(0, 1)
            vix = np.random.uniform(10, 50)
            regime = np.random.choice(["stable", "volatile"], p=[0.7, 0.3])
            beta = np.random.uniform(0.5, 1.5)

            synthetic_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "wave_id": wave_id,
                "nav": nav,
                "wave_return": wave_return,
                "bm_return": bm_return,
                "alpha": alpha,
                "exposure": exposure,
                "safe_fraction": safe_fraction,
                "vix": vix,
                "regime": regime,
                "beta": beta,
                "is_synthetic_data": True
            })

            nav *= (1 + wave_return)  # Update NAV
    
    return pd.DataFrame(synthetic_data, columns=OUTPUT_COLUMNS)


def save_wave_history():
    wave_ids = get_wave_ids()
    synthetic_data = generate_synthetic_data(wave_ids)
    synthetic_data.to_csv(WAVE_HISTORY_FILE, index=False)


if __name__ == "__main__":
    save_wave_history()