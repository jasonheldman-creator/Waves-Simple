import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass

# --------------------------
# Wave Renaming / Normalization
# --------------------------

WAVE_NAME_MAP = {
    "Crypto Wave": "Crypto Equity Wave (mid/large cap)",
    "Crypto Income Wave": "Crypto Equity Wave (mid/large cap)",
    "Future Power & Energy Wave": None,
    "AI Wave": "AI Wave",
    "SmartSafe Wave": "SmartSafe Wave"
}

def normalize_wave_name(name: str):
    if pd.isna(name):
        return None
    name = name.strip()
    if name in WAVE_NAME_MAP:
        return WAVE_NAME_MAP[name]
    return name


# --------------------------
# Load Weights
# --------------------------

def load_wave_weights(path="wave_weights.csv"):
    df = pd.read_csv(path)

    # Drop empty rows
    df = df.dropna(how="all")

    # Strip formatting
    df['wave'] = df['wave'].astype(str).str.strip()
    df['ticker'] = df['ticker'].astype(str).str.strip()

    # Remove blank waves and blank tickers
    df = df[(df['wave'] != "") & (df['ticker'] != "")]

    # Normalize names (remove Future Wave, rename others)
    df['wave'] = df['wave'].apply(normalize_wave_name)

    # Drop removed waves
    df = df[df['wave'].notna()]

    # Convert weight to float
    df['weight'] = df['weight'].astype(float)

    # Normalize each wave to ensure weights sum to 1.0
    df['weight'] = df.groupby('wave')['weight'].transform(lambda x: x / x.sum())

    return df


# --------------------------
# Compute returns
# --------------------------

def price_history(tickers, start="2022-01-01"):
    data = yf.download(list(tickers), start=start)['Adj Close']
    return data


def compute_wave_returns(df):
    waves = df['wave'].unique()
    wave_returns = {}

    for wave in waves:
        subset = df[df['wave'] == wave]
        tickers = subset['ticker'].tolist()
        weights = subset['weight'].values

        prices = price_history(tickers)
        prices = prices.dropna(how="all", axis=1)

        if prices.empty:
            raise RuntimeError(f"No price data found for tickers: {tickers} in {wave}")

        rets = prices.pct_change().fillna(0)
        wave_ret = (rets * weights).sum(axis=1)

        wave_returns[wave] = wave_ret

    return wave_returns


# --------------------------
# Main Engine Build
# --------------------------

def build_engine():
    weights_df = load_wave_weights()
    rets = compute_wave_returns(weights_df)
    return rets