"""
waves_engine.py — WAVES Intelligence™ Vector Engine (Stage 4)

Features
--------
- Internal WAVE_WEIGHTS (no CSV dependency).
- Modes:
    • Standard
    • Alpha-Minus-Beta (de-risked via lower return scaling)
    • Private Logic (enhanced via higher return scaling)
- Optional Full_Wave_History.csv (Date/Wave/NAV), else live/simulated prices.
- Robust price fetching with yfinance + simulator fallback.
- Daily performance logging:
    • logs/performance/<Wave>_performance_daily.csv
      with Date, Wave, Mode, NAV, Return, CumReturn.
- Per-Wave Benchmarks (NEW, Stage 4):
    • BENCHMARK_MAP selects a benchmark Wave per Wave.
    • All alpha metrics are now vs each Wave’s own benchmark.
"""

import os
import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # App will fall back to simulator.


# ============================================================
# INTERNAL MASTER WEIGHTS (no CSV required)
# ============================================================

WAVE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "AI Wave": {
        "NVDA": 0.15,
        "MSFT": 0.10,
        "GOOGL": 0.10,
        "AMD": 0.10,
        "PLTR": 0.10,
        "META": 0.10,
        "CRWD": 0.10,
        "SNOW": 0.10,
        "TSLA": 0.05,
        "AVGO": 0.10,
    },
    "Cloud & Software Wave": {
        "MSFT": 0.20,
        "CRM": 0.10,
        "ADBE": 0.10,
        "NOW": 0.10,
        "GOOGL": 0.10,
        "AMZN": 0.20,
        "PANW": 0.10,
        "DDOG": 0.10,
    },
    "Crypto Income Wave": {
        "COIN": 0.33,
        "MSTR": 0.33,
        "RIOT": 0.34,
    },
    "Future Power & Energy Wave": {
        "NEE": 0.10,
        "DUK": 0.10,
        "ENPH": 0.10,
        "PSX": 0.10,
        "SLB": 0.10,
        "CVX": 0.10,
        "HAL": 0.10,
        "LIN": 0.10,
        "MPC": 0.10,
        "XOM": 0.10,
    },
    "Small Cap Growth Wave": {
        "PLTR": 0.10,
        "ROKU": 0.10,
        "UPST": 0.10,
        "DOCN": 0.10,
        "FSLY": 0.10,
        "AI": 0.10,
        "SMCI": 0.10,
        "TTD": 0.10,
        "AFRM": 0.10,
        "PATH": 0.10,
    },
    "Quantum Computing Wave": {
        "IBM": 0.25,
        "IONQ": 0.25,
        "NVDA": 0.25,
        "AMD": 0.25,
    },
    "Clean Transit-Infrastructure Wave": {
        "TSLA": 0.25,
        "NIO": 0.15,
        "BLNK": 0.15,
        "CHPT": 0.15,
        "RIVN": 0.15,
        "F": 0.15,
    },
    "S&P 500 Wave": {
        "AAPL": 0.06,
        "MSFT": 0.06,
        "AMZN": 0.06,
        "NVDA": 0.06,
        "GOOGL": 0.06,
        "META": 0.06,
        "TSLA": 0.06,
        "BRK.B": 0.06,
        "LLY": 0.06,
        "JPM": 0.06,
    },
    "Income Wave": {
        "HDV": 0.20,
        "SCHD": 0.20,
        "JEPI": 0.20,
        "JEPQ": 0.20,
        "PFF": 0.20,
    },
    "SmartSafe Wave": {
        "BIL": 0.50,
        "SHV": 0.50,
    },
}

FULL_HISTORY_FILE = "Full_Wave_History.csv"  # optional, Date/Wave/NAV

DEFAULT_LOOKBACK_DAYS = 365
SHORT_LOOKBACK_DAYS = 30

# Mode behaviour: these scale daily returns (not NAV directly)
MODE_MULTIPLIERS = {
    "Standard": 1.0,
    "Alpha-Minus-Beta": 0.80,   # ~20% de-risked vs Standard
    "Private Logic": 1.15,      # modestly enhanced
}

# ============================================================
# Per-Wave Benchmark Map (can be refined later)
# ============================================================

BENCHMARK_MAP: Dict[str, str] = {
    # Core equity Waves benchmarked to S&P 500 Wave (can change to sector/QQQ later)
    "AI Wave": "S&P 500 Wave",
    "Cloud & Software Wave": "S&P 500 Wave",
    "Crypto Income Wave": "S&P 500 Wave",
    "Future Power & Energy Wave": "S&P 500 Wave",
    "Small Cap Growth Wave": "S&P 500 Wave",
    "Quantum Computing Wave": "S&P 500 Wave",
    "Clean Transit-Infrastructure Wave": "S&P 500 Wave",
    "Income Wave": "S&P 500 Wave",
    # Self-benchmark for S&P itself and SmartSafe
    "S&P 500 Wave": "S&P 500 Wave",
    "SmartSafe Wave": "SmartSafe Wave",
}


def get_benchmark_wave_for(wave_name: str) -> str:
    """Return the benchmark Wave name for a given Wave."""
    return BENCHMARK_MAP.get(wave_name, "S&P 500 Wave")
    # ============================================================
# Helpers
# ============================================================

def _load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[waves_engine] Error loading CSV '{path}': {e}")
        return None


def _ensure_log_dirs():
    """Create logs/performance directory if it doesn't exist."""
    try:
        os.makedirs(os.path.join("logs", "performance"), exist_ok=True)
    except Exception as e:
        print(f"[waves_engine] Warning: could not create log directories: {e}")


# ============================================================
# Wave discovery / positions
# ============================================================

def get_all_waves() -> List[str]:
    """Return sorted list of all Waves defined in WAVE_WEIGHTS."""
    return sorted(list(WAVE_WEIGHTS.keys()))


def get_wave_positions(wave_name: str) -> pd.DataFrame:
    """
    Return positions for a specific Wave as a DataFrame with
    columns: Wave, Ticker, Weight.
    """
    if wave_name not in WAVE_WEIGHTS:
        raise ValueError(f"Wave '{wave_name}' not found in internal weights.")

    tickers = []
    weights = []
    for t, w in WAVE_WEIGHTS[wave_name].items():
        tickers.append(t.strip().upper())
        weights.append(float(w))

    df = pd.DataFrame(
        {
            "Wave": wave_name,
            "Ticker": tickers,
            "Weight": weights,
        }
    )

    total = df["Weight"].sum()
    if total > 0:
        df["Weight"] = df["Weight"] / total

    return df


# ============================================================
# Full_Wave_History (optional)
# ============================================================

def _load_full_history() -> Optional[pd.DataFrame]:
    """
    Load Full_Wave_History.csv if it exists AND has proper
    Date/Wave/NAV columns. Otherwise returns None.
    """
    hist = _load_csv_safe(FULL_HISTORY_FILE)
    if hist is None:
        return None

    lower_map = {c.lower(): c for c in hist.columns}
    date_col = lower_map.get("date")
    wave_col = lower_map.get("wave")
    nav_col = lower_map.get("nav")

    if not all([date_col, wave_col, nav_col]):
        print(
            "[waves_engine] Full_Wave_History.csv present but missing date/wave/nav; "
            "ignoring and using price-based NAV instead."
        )
        return None

    hist = hist.rename(columns={date_col: "date", wave_col: "Wave", nav_col: "NAV"})
    hist["Wave"] = hist["Wave"].astype(str).str.strip()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values(["Wave", "date"])
    return hist


# ============================================================
# Price / NAV from yfinance + simulator fallback
# ============================================================

def _compute_nav_from_prices(price_df: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Compute NAV and daily returns given price_df (date x tickers) and weights."""
    price_df = price_df.copy().sort_index()
    price_df = price_df[weights.index]

    returns = price_df.pct_change().fillna(0.0)
    port_ret = (returns * weights).sum(axis=1)
    nav = (1.0 + port_ret).cumprod()

    return pd.DataFrame({"NAV": nav, "Return": port_ret})


def _simulate_price_history(tickers: List[str], lookback_days: int) -> pd.DataFrame:
    """
    Synthetic price history for demo mode when all external price APIs fail.
    Uses a geometric random walk with modest drift and volatility.
    """
    end_date = dt.datetime.utcnow().date()
    start_date = end_date - dt.timedelta(days=lookback_days)
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    if not tickers:
        return pd.DataFrame()

    np.random.seed(42)  # deterministic demos
    mu = 0.08 / 252.0
    sigma = 0.18 / (252.0 ** 0.5)

    prices = {}
    for t in sorted(set([t.strip().upper() for t in tickers if t.strip()])):
        start_price = 100 + (hash(t) % 200)
        rets = np.random.normal(loc=mu, scale=sigma, size=len(dates))
        prices[t] = start_price * np.cumprod(1.0 + rets)

    df = pd.DataFrame(prices, index=dates)
    return df


def _fetch_price_history(tickers: List[str], lookback_days: int) -> pd.DataFrame:
    """
    Fetch price history from yfinance; if that fails, fall back to simulator.
    """
    tickers = sorted(set([t.strip().upper() for t in tickers if t.strip()]))
    if not tickers:
        return pd.DataFrame()

    prices = pd.DataFrame()

    # 1) yfinance batched
    if yf is not None:
        end_date = dt.datetime.utcnow().date()
        start_date = end_date - dt.timedelta(days=lookback_days + 5)

        def from_multiindex(data: pd.DataFrame, tickers_list: List[str]) -> pd.DataFrame:
            frames = []
            for t in tickers_list:
                if (t, "Adj Close") in data.columns:
                    frames.append(data[(t, "Adj Close")].rename(t))
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, axis=1)

        try:
            data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date + dt.timedelta(days=1),
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )
            if isinstance(data.columns, pd.MultiIndex):
                prices = from_multiindex(data, tickers)
            else:
                if "Adj Close" in data.columns:
                    prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        except Exception as e:
            print(f"[waves_engine] Batched yfinance download failed: {e}")

        # 2) per-ticker fallback
        if prices.empty:
            frames = []
            for t in tickers:
                try:
                    d = yf.download(
                        tickers=t,
                        start=start_date,
                        end=end_date + dt.timedelta(days=1),
                        progress=False,
                        auto_adjust=True,
                    )
                    if not d.empty and "Adj Close" in d.columns:
                        frames.append(d["Adj Close"].rename(t))
                except Exception as e:
                    print(f"[waves_engine] Ticker {t} failed: {e}")
            if frames:
                prices = pd.concat(frames, axis=1)

    # 3) simulator fallback
    if prices.empty:
        print("[waves_engine] No real price data; using simulated prices for demo mode.")
        prices = _simulate_price_history(tickers, lookback_days)

    prices.index = pd.to_datetime(prices.index)
    return prices
    # ============================================================
# Performance logging
# ============================================================

def _log_wave_nav_history(wave_name: str, mode: str, nav_df: pd.DataFrame) -> None:
    """
    Append/update daily NAV history for a Wave+Mode into
    logs/performance/<Wave>_performance_daily.csv.

    Columns written:
        Date, Wave, Mode, NAV, Return, CumReturn
    """
    try:
        _ensure_log_dirs()
        if nav_df is None or nav_df.empty:
            return

        df = nav_df.copy()
        # Reset index to Date column
        if df.index.name is None:
            df = df.reset_index().rename(columns={"index": "Date"})
        else:
            df = df.reset_index().rename(columns={df.index.name or "index": "Date"})

        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df["Wave"] = wave_name
        df["Mode"] = mode

        path = os.path.join(
            "logs", "performance", f"{wave_name.replace(' ', '_')}_performance_daily.csv"
        )

        if os.path.exists(path):
            old = pd.read_csv(path)
            old["Date"] = pd.to_datetime(old["Date"]).dt.date
            combined = pd.concat([old, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Date", "Mode"])
            combined = combined.sort_values("Date")
        else:
            combined = df

        combined.to_csv(path, index=False)

    except Exception as e:
        print(f"[waves_engine] Warning: failed to log performance for {wave_name}: {e}")


# ============================================================
# Core: compute_history_nav (with modes + logging)
# ============================================================

def compute_history_nav(
    wave_name: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    mode: str = "Standard",
) -> pd.DataFrame:
    """
    Compute NAV history for a Wave over the given lookback window.

    Priority:
      1) Use Full_Wave_History.csv if present & valid for this Wave.
      2) Otherwise, build from price history & internal weights.
    Modes scale daily returns before re-building NAV.
    Always logs the resulting NAV/Return/CumReturn to logs/performance.
    """
    mode = mode or "Standard"
    multiplier = MODE_MULTIPLIERS.get(mode, 1.0)

    # 1) Full_Wave_History option
    full_hist = _load_full_history()
    if full_hist is not None:
        mask_wave = full_hist["Wave"].str.lower() == wave_name.lower()
        hist_wave = full_hist.loc[mask_wave].copy()
        if not hist_wave.empty:
            cutoff = dt.datetime.utcnow().date() - dt.timedelta(days=lookback_days)
            hist_wave = hist_wave[hist_wave["date"] >= pd.to_datetime(cutoff)]
            hist_wave = hist_wave.sort_values("date")
            if not hist_wave.empty:
                hist_wave = hist_wave.set_index("date")
                base_nav = hist_wave["NAV"] / float(hist_wave["NAV"].iloc[0])
                ret = base_nav.pct_change().fillna(0.0) * multiplier
                nav_scaled = (1.0 + ret).cumprod()
                cum_ret = nav_scaled / nav_scaled.iloc[0] - 1.0
                nav_df = pd.DataFrame(
                    {"NAV": nav_scaled, "Return": ret, "CumReturn": cum_ret}
                )
                _log_wave_nav_history(wave_name, mode, nav_df)
                return nav_df

    # 2) Price-based NAV using internal weights
    positions = get_wave_positions(wave_name)
    tickers = positions["Ticker"].tolist()
    weights = positions.set_index("Ticker")["Weight"]

    price_df = _fetch_price_history(tickers, lookback_days)
    if price_df.empty:
        raise RuntimeError(
            f"Unable to compute NAV history for '{wave_name}': no price data available."
        )

    valid_cols = [c for c in price_df.columns if c in weights.index]
    if not valid_cols:
        price_df = price_df.copy()
        valid_cols = list(price_df.columns)
        weights = pd.Series([1.0 / len(valid_cols)] * len(valid_cols), index=valid_cols)
    else:
        price_df = price_df[valid_cols]
        weights = weights.loc[valid_cols]
        weights = weights / weights.sum()

    nav_df = _compute_nav_from_prices(price_df, weights)
    nav_df["Return"] = nav_df["Return"] * multiplier
    nav_df["NAV"] = (1.0 + nav_df["Return"]).cumprod()
    nav_df["CumReturn"] = nav_df["NAV"] / nav_df["NAV"].iloc[0] - 1.0

    _log_wave_nav_history(wave_name, mode, nav_df)
    return nav_df
    # ============================================================
# Summary / Overview + Alpha vs Benchmark
# ============================================================

def get_wave_summary_metrics(
    wave_name: str,
    mode: str = "Standard",
    long_lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    short_lookback_days: int = SHORT_LOOKBACK_DAYS,
) -> Dict[str, float]:
    """
    Basic metrics for a single Wave (no alpha).
    """
    hist_long = compute_history_nav(wave_name, long_lookback_days, mode)
    nav_last = float(hist_long["NAV"].iloc[-1])
    ret_long = float(hist_long["CumReturn"].iloc[-1])

    if len(hist_long) > short_lookback_days:
        short_slice = hist_long.iloc[-short_lookback_days:]
        ret_short = float(
            short_slice["NAV"].iloc[-1] / short_slice["NAV"].iloc[0] - 1.0
        )
    else:
        ret_short = float(ret_long)

    return {
        "nav_last": nav_last,
        "ret_long": ret_long,
        "ret_short": ret_short,
    }


def get_portfolio_overview(
    mode: str = "Standard",
    long_lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    short_lookback_days: int = SHORT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Build a portfolio-level overview table for all Waves, including
    alpha vs each Wave's benchmark (from BENCHMARK_MAP).

    Columns:
        Wave
        Benchmark
        NAV_last
        Return_365D
        Return_30D
        Alpha_365D
        Alpha_30D
    """
    waves = get_all_waves()

    # Precompute benchmark histories (Standard mode for benchmarks)
    benchmark_waves = {get_benchmark_wave_for(w) for w in waves}
    benchmark_histories: Dict[str, Optional[pd.DataFrame]] = {}
    for bench_name in benchmark_waves:
        try:
            benchmark_histories[bench_name] = compute_history_nav(
                bench_name, long_lookback_days, mode="Standard"
            )
        except Exception as e:
            print(f"[waves_engine] Failed to compute benchmark NAV for '{bench_name}': {e}")
            benchmark_histories[bench_name] = None

    rows = []

    for w in waves:
        bench_name = get_benchmark_wave_for(w)
        bench_hist = benchmark_histories.get(bench_name)

        try:
            hist_long = compute_history_nav(w, long_lookback_days, mode)
            nav_last = float(hist_long["NAV"].iloc[-1])
            ret_long = float(hist_long["CumReturn"].iloc[-1])

            if len(hist_long) > short_lookback_days:
                short_slice = hist_long.iloc[-short_lookback_days:]
                ret_short = float(
                    short_slice["NAV"].iloc[-1] / short_slice["NAV"].iloc[0] - 1.0
                )
            else:
                ret_short = float(ret_long)

            alpha_long = float("nan")
            alpha_short = float("nan")

            if bench_hist is not None:
                combo = pd.concat(
                    [
                        hist_long[["NAV"]].rename(columns={"NAV": "NAV_wave"}),
                        bench_hist[["NAV"]].rename(columns={"NAV": "NAV_bench"}),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()

                if not combo.empty:
                    nav_w = combo["NAV_wave"]
                    nav_b = combo["NAV_bench"]

                    wave_ret = nav_w.iloc[-1] / nav_w.iloc[0] - 1.0
                    bench_ret = nav_b.iloc[-1] / nav_b.iloc[0] - 1.0
                    alpha_long = float(wave_ret - bench_ret)

                    if len(combo) > short_lookback_days:
                        slice_j = combo.iloc[-short_lookback_days:]
                    else:
                        slice_j = combo

                    nav_w_s = slice_j["NAV_wave"]
                    nav_b_s = slice_j["NAV_bench"]
                    wave_ret_s = nav_w_s.iloc[-1] / nav_w_s.iloc[0] - 1.0
                    bench_ret_s = nav_b_s.iloc[-1] / nav_b_s.iloc[0] - 1.0
                    alpha_short = float(wave_ret_s - bench_ret_s)

            rows.append(
                {
                    "Wave": w,
                    "Benchmark": bench_name,
                    "NAV_last": nav_last,
                    "Return_365D": ret_long,
                    "Return_30D": ret_short,
                    "Alpha_365D": alpha_long,
                    "Alpha_30D": alpha_short,
                }
            )

        except Exception as e:
            print(f"[waves_engine] Error computing overview for Wave '{w}': {e}")
            rows.append(
                {
                    "Wave": w,
                    "Benchmark": bench_name,
                    "NAV_last": float("nan"),
                    "Return_365D": float("nan"),
                    "Return_30D": float("nan"),
                    "Alpha_365D": float("nan"),
                    "Alpha_30D": float("nan"),
                }
            )

    df = pd.DataFrame(rows).sort_values("Wave").reset_index(drop=True)
    return df