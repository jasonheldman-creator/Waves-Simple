import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

# ---------- Paths & Config ----------

LIST_PATH = "list.csv"                 # Universe file (Tickers, Names, Sectors)
WEIGHTS_PATH = "wave_weights.csv"      # Wave weights definition

LOGS_POSITIONS_DIR = os.path.join("logs", "positions")
LOGS_PERFORMANCE_DIR = os.path.join("logs", "performance")

os.makedirs(LOGS_POSITIONS_DIR, exist_ok=True)
os.makedirs(LOGS_PERFORMANCE_DIR, exist_ok=True)

# Starting notional per Wave (just a scaling constant)
DEFAULT_CAPITAL = 1_000_000.0

# Benchmark mapping (can be customized)
WAVE_BENCHMARK_MAP: Dict[str, str] = {
    "S&P 500 Wave": "SPY",
    "Growth Wave": "QQQ",
    "AI Megacap Wave": "QQQ",
    "Quantum Computing Wave": "QQQ",
    "Small Cap Growth Wave": "IWM",
    "Small–Mid Cap Growth Wave": "IJH",
    "Income Wave": "AGG",
    "Future Power & Energy Wave": "XLE",
    "Clean Transit & Infrastructure Wave": "IDEV",
    "Crypto Income Wave": "BTC-USD",  # spot; alpha logic still works
    "Emerging Markets Wave": "EEM",
    "Global Opportunities Wave": "VT",
    "Infinite Alpha Wave": "VT",
    "Technology Innovators Wave": "XLK",
    "SmartSafe Wave": "BIL",
}


# ---------- Core Loaders ----------

def load_universe() -> Optional[pd.DataFrame]:
    if not os.path.exists(LIST_PATH):
        return None
    df = pd.read_csv(LIST_PATH)
    # Normalize
    cols = {c.lower(): c for c in df.columns}
    if "ticker" not in cols:
        return None
    return df


def load_wave_weights() -> Optional[pd.DataFrame]:
    if not os.path.exists(WEIGHTS_PATH):
        return None
    df = pd.read_csv(WEIGHTS_PATH)
    # Expect at least Wave, Ticker, Weight
    cols = {c.lower(): c for c in df.columns}
    required = ["wave", "ticker", "weight"]
    for r in required:
        if r not in cols:
            raise ValueError(
                f"wave_weights.csv is missing required column '{r}'. "
                "It must have: Wave, Ticker, Weight (case-insensitive)."
            )
    return df


def discover_waves_from_weights(weights_df: pd.DataFrame) -> List[str]:
    cols = {c.lower(): c for c in weights_df.columns}
    wave_col = cols["wave"]
    return sorted(weights_df[wave_col].dropna().unique().tolist())


# ---------- Price & Return Helpers ----------

def _fetch_prices(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Fetch last 2 days of daily prices for a list of tickers using yfinance.
    Returns dict {ticker: DataFrame}.
    """
    result: Dict[str, pd.DataFrame] = {}
    uniq = sorted(set(tickers))
    if not uniq:
        return result

    data = yf.download(uniq, period="2d", interval="1d", auto_adjust=False, progress=False)

    # yfinance returns different shapes for 1 vs many tickers
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex: (field, ticker)
        closes = data["Adj Close"]
        for t in uniq:
            if t in closes.columns:
                df = closes[[t]].rename(columns={t: "price"})
                result[t] = df
    else:
        # Single ticker
        df = data.rename(columns={"Adj Close": "price"})
        result[uniq[0]] = df[["price"]]

    return result


def _latest_return_from_price_df(df: pd.DataFrame) -> Optional[float]:
    """
    Given a DataFrame with a 'price' column and DateTimeIndex, return the
    last daily return (most recent vs previous). If not available, None.
    """
    if df is None or df.empty or "price" not in df.columns:
        return None
    df = df.dropna(subset=["price"]).copy()
    if len(df) < 2:
        return None
    df["return"] = df["price"].pct_change()
    return float(df["return"].iloc[-1])


# ---------- Performance File Helper ----------

def _performance_path_for_wave(wave: str) -> str:
    safe_wave = wave.replace(" ", "_")
    return os.path.join(LOGS_PERFORMANCE_DIR, f"{safe_wave}_performance_daily.csv")


def _positions_path_for_wave(wave: str, date: datetime) -> str:
    safe_wave = wave.replace(" ", "_")
    stamp = date.strftime("%Y%m%d")
    return os.path.join(LOGS_POSITIONS_DIR, f"{safe_wave}_positions_{stamp}.csv")


# ---------- Core Engine Logic ----------

def run_wave(
    wave_name: str,
    mode: str = "Standard",
    capital: float = DEFAULT_CAPITAL,
    debug: bool = False,
) -> None:
    """
    Run the engine for a single Wave:
    - Loads weights
    - Fetches live prices
    - Computes today's return + alpha vs benchmark
    - Appends performance entry
    - Writes positions snapshot
    """

    weights_df = load_wave_weights()
    if weights_df is None:
        raise RuntimeError("wave_weights.csv not found")

    universe_df = load_universe()  # optional, for name/sector enrichment

    cols = {c.lower(): c for c in weights_df.columns}
    wave_col = cols["wave"]
    ticker_col = cols["ticker"]
    weight_col = cols["weight"]
    basket_col = cols.get("basket")   # optional (Primary / Secondary / etc.)

    wv = weights_df[weights_df[wave_col] == wave_name].copy()
    if wv.empty:
        raise RuntimeError(f"No weights found for Wave '{wave_name}'")

    # Deduplicate & normalize weights
    wv[ticker_col] = wv[ticker_col].astype(str).str.strip().str.upper()
    wv = (
        wv.groupby(ticker_col, as_index=False)[weight_col]
        .sum()
        .rename(columns={weight_col: "Weight"})
    )
    total_w = wv["Weight"].sum()
    if total_w <= 0:
        raise RuntimeError(f"Total weight for Wave '{wave_name}' is non-positive.")
    wv["Weight"] = wv["Weight"] / total_w

    tickers = wv[ticker_col].tolist()

    # Fetch prices for tickers + benchmark
    bench_ticker = WAVE_BENCHMARK_MAP.get(wave_name, "SPY")
    all_tickers = tickers + [bench_ticker]
    px = _fetch_prices(all_tickers)

    # Compute today's return per position
    returns: Dict[str, float] = {}
    latest_prices: Dict[str, float] = {}
    for t in tickers:
        df = px.get(t)
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["price"])
        latest_prices[t] = float(df["price"].iloc[-1])
        r = _latest_return_from_price_df(df)
        if r is not None:
            returns[t] = r

    bench_ret: Optional[float] = None
    if bench_ticker in px:
        bench_ret = _latest_return_from_price_df(px[bench_ticker])

    # Wave daily return (weighted sum of individual returns)
    wave_ret_today = 0.0
    total_weight_for_ret = 0.0
    for _, row in wv.iterrows():
        t = row[ticker_col]
        w = row["Weight"]
        r = returns.get(t)
        if r is None:
            continue
        wave_ret_today += w * r
        total_weight_for_ret += w

    if total_weight_for_ret > 0:
        wave_ret_today = wave_ret_today / total_weight_for_ret
    else:
        wave_ret_today = 0.0

    if bench_ret is None:
        alpha_today = None
    else:
        alpha_today = wave_ret_today - bench_ret

    # Load existing performance history & append
    perf_path = _performance_path_for_wave(wave_name)
    if os.path.exists(perf_path):
        perf_df = pd.read_csv(perf_path)
    else:
        perf_df = pd.DataFrame()

    today = datetime.utcnow().date()

    new_row = {
        "date": today.isoformat(),
        "mode": mode,
        "return_1d": wave_ret_today,
        "bench_return_1d": bench_ret,
        "alpha_1d": alpha_today,
    }

    perf_df = pd.concat([perf_df, pd.DataFrame([new_row])], ignore_index=True)
    perf_df["date"] = pd.to_datetime(perf_df["date"])
    perf_df = perf_df.sort_values("date")

    # Compute cumulative returns
    perf_df["cum_return"] = (1 + perf_df["return_1d"].fillna(0.0)).cumprod() - 1
    if "bench_return_1d" in perf_df.columns:
        perf_df["cum_bench_return"] = (1 + perf_df["bench_return_1d"].fillna(0.0)).cumprod() - 1
        perf_df["cum_alpha"] = perf_df["cum_return"] - perf_df["cum_bench_return"]
    else:
        perf_df["cum_bench_return"] = None
        perf_df["cum_alpha"] = None

    # 30d/60d rolling (using rows as proxy for days)
    perf_df["return_30d"] = (
        (1 + perf_df["return_1d"].fillna(0.0)).rolling(window=30).apply(lambda x: (x.prod() - 1.0), raw=False)
    )
    perf_df["return_60d"] = (
        (1 + perf_df["return_1d"].fillna(0.0)).rolling(window=60).apply(lambda x: (x.prod() - 1.0), raw=False)
    )

    if "bench_return_1d" in perf_df.columns:
        perf_df["bench_return_30d"] = (
            (1 + perf_df["bench_return_1d"].fillna(0.0)).rolling(window=30).apply(lambda x: (x.prod() - 1.0), raw=False)
        )
        perf_df["bench_return_60d"] = (
            (1 + perf_df["bench_return_1d"].fillna(0.0)).rolling(window=60).apply(lambda x: (x.prod() - 1.0), raw=False)
        )
        perf_df["alpha_30d"] = perf_df["return_30d"] - perf_df["bench_return_30d"]
        perf_df["alpha_60d"] = perf_df["return_60d"] - perf_df["bench_return_60d"]
    else:
        perf_df["bench_return_30d"] = None
        perf_df["bench_return_60d"] = None
        perf_df["alpha_30d"] = None
        perf_df["alpha_60d"] = None

    perf_df.to_csv(perf_path, index=False)

    # Build positions snapshot
    if universe_df is not None:
        ucols = {c.lower(): c for c in universe_df.columns}
        u_ticker_col = ucols.get("ticker")
        name_col = ucols.get("name") or ucols.get("company") or ucols.get("security")
        sector_col = ucols.get("sector")
        if u_ticker_col:
            universe_df[u_ticker_col] = universe_df[u_ticker_col].astype(str).str.strip().str.upper()
            wv = wv.merge(
                universe_df,
                left_on=ticker_col,
                right_on=u_ticker_col,
                how="left"
            )
            if name_col:
                wv.rename(columns={name_col: "Name"}, inplace=True)
            if sector_col:
                wv.rename(columns={sector_col: "Sector"}, inplace=True)

    wv["Price"] = wv[ticker_col].map(latest_prices).fillna(0.0)
    wv["Weight"] = wv["Weight"].astype(float)
    wv["Capital"] = capital * wv["Weight"]
    wv["Shares"] = (wv["Capital"] / wv["Price"]).replace([float("inf"), -float("inf")], 0.0).fillna(0.0)

    # Save positions file
    dt_now = datetime.utcnow()
    positions_path = _positions_path_for_wave(wave_name, dt_now)
    wv["AsOf"] = dt_now.isoformat()
    wv.to_csv(positions_path, index=False)

    if debug:
        print(f"[{wave_name}] return_1d={wave_ret_today:.5f}, alpha_1d={alpha_today}")


def run_all_waves(
    mode: str = "Standard",
    capital: float = DEFAULT_CAPITAL,
    debug: bool = False
) -> None:
    """
    Convenience function: run engine for all waves discovered in wave_weights.csv.
    """
    weights_df = load_wave_weights()
    if weights_df is None:
        raise RuntimeError("wave_weights.csv not found")

    waves = discover_waves_from_weights(weights_df)
    for wave in waves:
        try:
            run_wave(wave, mode=mode, capital=capital, debug=debug)
        except Exception as e:
            if debug:
                print(f"Error running {wave}: {e}")


if __name__ == "__main__":
    # Quick CLI entry: python waves_engine.py
    run_all_waves(mode="Standard", capital=DEFAULT_CAPITAL, debug=True)