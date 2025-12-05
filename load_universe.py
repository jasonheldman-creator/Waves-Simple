from pathlib import Path
import pandas as pd

UNIVERSE_PATH = Path("Master_Stock_Sheet.csv")

def load_universe(path: Path = UNIVERSE_PATH) -> pd.DataFrame:
    # 1) Read raw CSV
    df = pd.read_csv(path)

    # 2) Normalize column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    # 3) Map columns to required names
    col_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            col_map[col] = "Ticker"
        elif col in ["company", "company_name", "name"]:
            col_map[col] = "Company"
        elif col in ["weight", "index_weight", "wgt"]:
            col_map[col] = "Weight"
        elif col in ["sector", "gics_sector"]:
            col_map[col] = "Sector"
        elif col in ["marketvalue", "market_value", "mv"]:
            col_map[col] = "MarketValue"
        elif col in ["price", "last_price"]:
            col_map[col] = "Price"

    df = df.rename(columns=col_map)

    # 4) Required columns
    required = ["Ticker", "Company", "Weight"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # 5) Optional columns
    for optional in ["Sector", "MarketValue", "Price"]:
        if optional not in df.columns:
            df[optional] = None

    # 6) Clean and type-cast
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df["Company"] = df["Company"].astype(str).str.strip()

    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["Ticker", "Weight"])
    df = df[df["Ticker"] != ""]

    # Final clean order
    df = df[["Ticker", "Company", "Sector", "Weight", "MarketValue", "Price"]]

    return df
