import os
import logging
import pandas as pd
from datetime import datetime

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CONSTANT PATHS
# -----------------------------------------------------------------------------
WAVE_POSITIONS_PATH = "data/wave_positions.csv"
UNIVERSAL_UNIVERSE_PATH = "universal_universe.csv"

# -----------------------------------------------------------------------------
# TICKER LOADING LOGIC (DETERMINISTIC, CI-SAFE)
# -----------------------------------------------------------------------------
def load_tickers() -> list[str]:
    """
    Load tickers for price cache generation.

    Priority:
    1. data/wave_positions.csv (if present)
    2. universal_universe.csv (fallback)

    Guarantees:
    - Returns a non-empty list of tickers
    - Logs source selection
    - Raises a clear error if no tickers are found
    """

    tickers: list[str] = []

    # -------------------------------------------------------------------------
    # PRIMARY SOURCE: wave_positions.csv
    # -------------------------------------------------------------------------
    if os.path.exists(WAVE_POSITIONS_PATH):
        logger.info("Using wave_positions.csv as primary ticker source")

        df = pd.read_csv(WAVE_POSITIONS_PATH)

        if "ticker" not in df.columns:
            raise ValueError("wave_positions.csv is missing required 'ticker' column")

        tickers = (
            df["ticker"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

    # -------------------------------------------------------------------------
    # FALLBACK SOURCE: universal_universe.csv
    # -------------------------------------------------------------------------
    else:
        logger.warning(
            "wave_positions.csv not found — falling back to universal_universe.csv"
        )

        if not os.path.exists(UNIVERSAL_UNIVERSE_PATH):
            raise FileNotFoundError(
                "Neither wave_positions.csv nor universal_universe.csv could be found"
            )

        df = pd.read_csv(UNIVERSAL_UNIVERSE_PATH)

        if "ticker" not in df.columns:
            raise ValueError("universal_universe.csv is missing required 'ticker' column")

        tickers = (
            df["ticker"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

    # -------------------------------------------------------------------------
    # FINAL VALIDATION (HARD FAIL IF EMPTY)
    # -------------------------------------------------------------------------
    tickers_total = len(tickers)

    logger.info("Resolved ticker count: %d", tickers_total)

    if tickers_total <= 0:
        raise ValueError(
            "No tickers found after resolving ticker sources — cannot build price cache"
        )

    return tickers


# -----------------------------------------------------------------------------
# ENTRY POINT (SAFE FOR CI)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=== Building Price Cache ===")
    logger.info("Started at UTC: %s", datetime.utcnow().isoformat())

    tickers = load_tickers()

    logger.info("Ticker load successful")
    logger.info("Sample tickers: %s", tickers[:10])