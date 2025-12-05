import csv
import pandas as pd
from pathlib import Path

def load_universe(path: Path) -> pd.DataFrame:
    # 1. Detect delimiter automatically
    with open(path, "r", errors="ignore") as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter
        except:
            delimiter = ","  # fallback

    # 2. Read CSV but tolerate broken lines
    df = pd.read_csv(
        path,
        sep=delimiter,
        engine="python",
        on_bad_lines="skip",  # <-- THIS FIXES YOUR ERROR
        dtype=str,
        header=0
    )

    # 3. Normalize headers
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    # 4. Map headers
    col_map = {
        "ticker": "Ticker",
        "symbol": "Ticker",
        "company": "Company",
        "name": "Company",
        "weight": "Weight",
        "index_weight": "Weight",
        "sector": "Sector",
        "market_value": "MarketValue",
        "price": "Price"
    }
    df = df.rename(columns=col_map)

    # 5. Ensure missing columns exist
    for col in ["Ticker", "Company", "Sector", "Weight", "MarketValue", "Price"]:
        if col not in df.columns:
            df[col] = None

    # 6. Clean types
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Company"] = df["Company"].astype(str).str.strip()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # 7. Drop blank rows
    df = df[df["Ticker"].notna()]
    df = df[df["Ticker"] != ""]

    # 8. Final reorder
    df = df[["Ticker", "Company", "Sector", "Weight", "MarketValue", "Price"]]

    return df