import pandas as pd

# === 1. Google Sheets CSV URLs (fill these in) ======================

SP500_CSV_URL       = "<<<paste SP500 CSV URL here>>>"
R2000_CSV_URL       = "<<<paste R2000 CSV URL here>>>"
R3000_CSV_URL       = "<<<paste R3000 CSV URL here>>>"
TOTALMARKET_CSV_URL = "<<<paste TotalMarket CSV URL here>>>"
MASTER_CSV_URL      = "<<<paste Master CSV URL here>>>"

# === 2. Loader helper ===============================================

def load_csv(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        if col.startswith("ticker"):
            rename_map[col] = "ticker"
        if col.startswith("name"):
            rename_map[col] = "name"

    df = df.rename(columns=rename_map)
    df = df[["ticker","name"]].copy()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["name"]   = df["name"].astype(str).str.strip()
    df = df[df["ticker"] != ""]
    df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df


# === 3. Universe getters =============================================

def get_master_universe():    return load_csv(MASTER_CSV_URL)
def get_sp500_universe():      return load_csv(SP500_CSV_URL)
def get_r2000_universe():      return load_csv(R2000_CSV_URL)
def get_r3000_universe():      return load_csv(R3000_CSV_URL)
def get_totalmarket_universe():return load_csv(TOTALMARKET_CSV_URL)

# === 4. Wave definitions =============================================

def wave_sp500():                 return get_sp500_universe()
def wave_russell_2000():          return get_r2000_universe()
def wave_russell_3000():          return get_r3000_universe()
def wave_total_us_equity():       return get_totalmarket_universe()
def wave_smid_growth():           return get_r2000_universe()
def wave_large_cap_growth():      return get_sp500_universe()
def wave_equity_income():         return get_totalmarket_universe()
def wave_dividend_focus():        return get_totalmarket_universe()
def wave_future_power_energy():   return get_master_universe()
def wave_quality_core():          return get_sp500_universe()

# === 5. Wave registry ================================================

WAVES_REGISTRY = {
    "SP500_WAVE":              wave_sp500,
    "RUSSELL_2000_WAVE":       wave_russell_2000,
    "RUSSELL_3000_WAVE":       wave_russell_3000,
    "TOTAL_US_EQUITY_WAVE":    wave_total_us_equity,
    "SMID_GROWTH_WAVE":        wave_smid_growth,
    "LARGE_CAP_GROWTH_WAVE":   wave_large_cap_growth,
    "EQUITY_INCOME_WAVE":      wave_equity_income,
    "DIVIDEND_FOCUS_WAVE":     wave_dividend_focus,
    "FUTURE_POWER_ENERGY_WAVE":wave_future_power_energy,
    "QUALITY_CORE_WAVE":       wave_quality_core,
}

if __name__ == "__main__":
    for name, fn in WAVES_REGISTRY.items():
        df = fn()
        print(f"{name}: {len(df)} holdings")