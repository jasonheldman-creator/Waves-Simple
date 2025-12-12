# waves_engine.py — WAVES Intelligence™ Engine (Guaranteed Performance Logger)
#
# Purpose:
#   Generate required logs so the Streamlit console populates:
#     logs/performance/<Wave>_performance_daily.csv
#     logs/positions/<Wave>_positions_YYYYMMDD.csv
#
# Inputs:
#   wave_weights.csv  (required)
#     Expected columns (flexible):
#       - Wave / wave / Portfolio / portfolio / Name / name
#       - Ticker / ticker / Symbol / symbol
#       - Weight / weight / Alloc / alloc / allocation
#     Optional:
#       - Benchmark / benchmark  (per-row or per-wave)
#
# Notes:
#   - Uses yfinance to fetch prices.
#   - Computes portfolio returns from weighted daily returns (static weights).
#   - Intraday return uses today’s regularMarketPrice vs previousClose.
#   - Rolling return windows: 30D, 60D, 1Y (~252 trading days)
#   - Alpha = portfolio return - benchmark return (same window)

import os
import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    yf = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOGS_POS_DIR = os.path.join(LOGS_DIR, "positions")
LOGS_PERF_DIR = os.path.join(LOGS_DIR, "performance")

WAVE_WEIGHTS_PATH = os.path.join(BASE_DIR, "wave_weights.csv")

DEFAULT_NOTIONAL = 100000.0  # used only for positions sizing display

# ----------------------------
# Helpers
# ----------------------------

def ts_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dirs():
    os.makedirs(LOGS_POS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERF_DIR, exist_ok=True)

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        if isinstance(x, str):
            s = x.strip()
            if s.endswith("%"):
                return float(s.replace("%", "").strip()) / 100.0
        return float(x)
    except Exception:
        return None

def normalize_wave_name(name: str) -> str:
    return " ".join(str(name).strip().split())

def normalize_ticker(t: str) -> str:
    return str(t).strip().upper()

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def infer_default_benchmark(wave: str) -> str:
    """Heuristic benchmark chooser if none provided."""
    w = wave.lower()
    if "crypto" in w or "bitcoin" in w:
        return "IBIT"  # bitcoin ETF proxy (change if needed)
    if "ai" in w or "tech" in w or "software" in w or "cloud" in w or "quantum" in w:
        return "QQQ"
    if "small" in w:
        return "IWM"
    if "energy" in w or "power" in w:
        return "XLE"
    if "muni" in w:
        return "MUB"
    # default broad
    return "SPY"

def yf_required():
    if yf is None:
        raise RuntimeError("yfinance is not installed/available in this environment.")

def fetch_daily_adjclose(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """
    Returns Adj Close prices (business days) for tickers.
    Robust to single ticker output shape.
    """
    yf_required()
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame()

    # yfinance can be flaky; retry a couple times
    last_err = None
    for _ in range(3):
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
            if df is None or df.empty:
                return pd.DataFrame()

            # If multiple tickers, columns are multiindex (PriceType, Ticker)
            if isinstance(df.columns, pd.MultiIndex):
                if ("Adj Close" in df.columns.get_level_values(0)):
                    out = df["Adj Close"].copy()
                elif ("Close" in df.columns.get_level_values(0)):
                    out = df["Close"].copy()
                else:
                    # take any first level
                    out = df.xs(df.columns.levels[0][0], axis=1, level=0)
            else:
                # Single ticker: columns are like ["Open","High","Low","Close","Adj Close","Volume"]
                col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
                out = df[[col]].copy()
                out.columns = [tickers[0]]

            out = out.dropna(how="all")
            return out
        except Exception as e:
            last_err = e
            time.sleep(1.0)

    raise RuntimeError(f"yfinance download failed: {repr(last_err)}")

def fetch_intraday_fields(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Uses yf.Ticker().fast_info when possible.
    Returns {ticker: {'price': ..., 'prev_close': ...}}
    """
    yf_required()
    out: Dict[str, Dict[str, float]] = {}
    for t in tickers:
        t = normalize_ticker(t)
        try:
            tk = yf.Ticker(t)
            fi = getattr(tk, "fast_info", {}) or {}
            price = fi.get("last_price", None)
            prev = fi.get("previous_close", None)

            # fallbacks
            if price is None or prev is None:
                info = tk.info or {}
                if price is None:
                    price = info.get("regularMarketPrice", None)
                if prev is None:
                    prev = info.get("regularMarketPreviousClose", None)

            out[t] = {
                "price": float(price) if price is not None else float("nan"),
                "prev_close": float(prev) if prev is not None else float("nan"),
            }
        except Exception:
            out[t] = {"price": float("nan"), "prev_close": float("nan")}
    return out

# ----------------------------
# Core calculations
# ----------------------------

def build_wave_table() -> Tuple[pd.DataFrame, str, str, str, Optional[str]]:
    """
    Load wave_weights.csv and return:
      df with standardized columns: Wave, Ticker, Weight, Benchmark(optional)
    """
    if not os.path.exists(WAVE_WEIGHTS_PATH):
        raise FileNotFoundError("wave_weights.csv not found in repo root.")

    df = pd.read_csv(WAVE_WEIGHTS_PATH)
    if df is None or df.empty:
        raise ValueError("wave_weights.csv is empty or unreadable.")

    wave_col = pick_col(df, ["Wave", "wave", "Portfolio", "portfolio", "Name", "name"])
    tick_col = pick_col(df, ["Ticker", "ticker", "Symbol", "symbol"])
    wt_col   = pick_col(df, ["Weight", "weight", "Alloc", "alloc", "allocation", "Allocation"])
    bm_col   = pick_col(df, ["Benchmark", "benchmark"])

    if wave_col is None or tick_col is None or wt_col is None:
        raise ValueError(
            "wave_weights.csv must have columns for Wave / Ticker / Weight "
            "(case-insensitive)."
        )

    out = pd.DataFrame({
        "Wave": df[wave_col].astype(str).map(normalize_wave_name),
        "Ticker": df[tick_col].astype(str).map(normalize_ticker),
        "Weight": pd.to_numeric(df[wt_col], errors="coerce"),
    })

    if bm_col is not None:
        out["Benchmark"] = df[bm_col].astype(str).map(normalize_ticker)
    else:
        out["Benchmark"] = None

    out = out.dropna(subset=["Wave", "Ticker", "Weight"])
    return out, wave_col, tick_col, wt_col, bm_col

def normalize_weights(df_wave: pd.DataFrame) -> pd.DataFrame:
    df_wave = df_wave.copy()
    df_wave["Weight"] = df_wave["Weight"].clip(lower=0.0)
    # combine duplicates
    df_wave = df_wave.groupby(["Wave", "Ticker"], as_index=False).agg({"Weight": "sum", "Benchmark": "first"})
    total = df_wave["Weight"].sum()
    if total <= 0:
        df_wave["Weight"] = 0.0
    else:
        df_wave["Weight"] = df_wave["Weight"] / total
    return df_wave

def compute_weighted_returns(price_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    price_df: columns tickers, index dates, values prices
    weights: {ticker: weight}
    Returns: portfolio daily return series aligned to price_df index.
    """
    # daily returns per ticker
    rets = price_df.pct_change().dropna(how="all")
    # align weights to available tickers
    w = {t: float(weights.get(t, 0.0)) for t in rets.columns}
    wsum = sum(w.values())
    if wsum <= 0:
        return pd.Series(index=rets.index, dtype=float)
    # normalized
    w = {t: v / wsum for t, v in w.items()}
    port = sum(rets[t] * w[t] for t in rets.columns if t in w)
    port.name = "portfolio_return"
    return port.dropna()

def rolling_total_return(daily_ret: pd.Series, window_trading_days: int) -> Optional[float]:
    """
    total return over last N trading days: (1+r).prod - 1
    """
    if daily_ret is None or daily_ret.empty:
        return None
    tail = daily_ret.dropna().iloc[-window_trading_days:]
    if len(tail) < max(5, int(window_trading_days * 0.6)):
        return None
    return float((1.0 + tail).prod() - 1.0)

def intraday_return_from_fastinfo(fast: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Optional[float]:
    """
    Weighted intraday return using price vs previous close.
    """
    vals = []
    for t, w in weights.items():
        f = fast.get(t, {})
        price = f.get("price", float("nan"))
        prev = f.get("prev_close", float("nan"))
        if price is None or prev is None:
            continue
        try:
            price = float(price)
            prev = float(prev)
            if math.isnan(price) or math.isnan(prev) or prev == 0:
                continue
            r = (price / prev) - 1.0
            vals.append(w * r)
        except Exception:
            continue
    if not vals:
        return None
    return float(sum(vals))

def write_positions_log(wave: str, df_wave: pd.DataFrame, fast: Dict[str, Dict[str, float]]):
    """
    Writes positions for today with weights + indicative $ value sizing.
    """
    today = datetime.now().strftime("%Y%m%d")
    path = os.path.join(LOGS_POS_DIR, f"{wave}_positions_{today}.csv")

    rows = []
    for _, r in df_wave.iterrows():
        t = r["Ticker"]
        w = float(r["Weight"])
        f = fast.get(t, {})
        price = f.get("price", float("nan"))
        if price is None or (isinstance(price, float) and math.isnan(price)) or price <= 0:
            price = float("nan")

        dollars = DEFAULT_NOTIONAL * w
        shares = (dollars / price) if (price == price and price > 0) else float("nan")
        rows.append({
            "Wave": wave,
            "Ticker": t,
            "Weight": w,
            "Price": price,
            "NotionalUSD": dollars,
            "Shares": shares,
            "timestamp": ts_now(),
        })

    pd.DataFrame(rows).to_csv(path, index=False)

def append_performance_log(wave: str, row: Dict):
    """
    Append to logs/performance/<Wave>_performance_daily.csv
    """
    path = os.path.join(LOGS_PERF_DIR, f"{wave}_performance_daily.csv")
    df_new = pd.DataFrame([row])

    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new

    # keep last 2000 rows
    if len(df) > 2000:
        df = df.iloc[-2000:].copy()

    df.to_csv(path, index=False)

# ----------------------------
# Main
# ----------------------------

def run_engine():
    ensure_dirs()

    if yf is None:
        raise RuntimeError("yfinance not available. Add yfinance to requirements.txt.")

    table, *_ = build_wave_table()
    if table.empty:
        raise ValueError("No wave rows found in wave_weights.csv.")

    waves = sorted(table["Wave"].unique().tolist())
    print(f"[{ts_now()}] WAVES Engine starting. Waves discovered: {len(waves)}")

    # Determine per-wave benchmark (use provided Benchmark if any row has it; else infer)
    wave_benchmark: Dict[str, str] = {}
    for w in waves:
        sub = table[table["Wave"] == w]
        bm_vals = [x for x in sub["Benchmark"].dropna().unique().tolist() if str(x).strip().upper() not in ["NONE", "NAN", ""]]
        if bm_vals:
            wave_benchmark[w] = normalize_ticker(bm_vals[0])
        else:
            wave_benchmark[w] = infer_default_benchmark(w)

    # Build ticker universe for price fetch
    all_port_tickers = sorted({t for t in table["Ticker"].dropna().astype(str).map(normalize_ticker).tolist()})
    all_bm_tickers = sorted(set(wave_benchmark.values()))
    all_tickers = sorted(set(all_port_tickers) | set(all_bm_tickers))

    print(f"[{ts_now()}] Fetching prices for {len(all_tickers)} tickers (portfolio + benchmarks)")

    # Pull daily prices for rolling windows
    prices = fetch_daily_adjclose(all_tickers, period="2y")  # enough for 252d
    # Pull intraday fields
    fast = fetch_intraday_fields(all_tickers)

    # Compute benchmark daily returns series cache
    bm_daily_ret: Dict[str, pd.Series] = {}
    for bm in all_bm_tickers:
        if bm in prices.columns:
            bm_daily_ret[bm] = prices[bm].pct_change().dropna()
        else:
            bm_daily_ret[bm] = pd.Series(dtype=float)

    for w in waves:
        dfw = normalize_weights(table[table["Wave"] == w][["Wave", "Ticker", "Weight", "Benchmark"]])
        tickers = dfw["Ticker"].tolist()
        weights = {r["Ticker"]: float(r["Weight"]) for _, r in dfw.iterrows()}

        bm = wave_benchmark[w]

        # positions log
        try:
            write_positions_log(w, dfw, fast)
        except Exception as e:
            print(f"[WARN] positions log failed for {w}: {repr(e)}")

        # portfolio daily returns (from daily prices)
        available = [t for t in tickers if t in prices.columns]
        port_daily = None
        if available:
            port_price = prices[available].copy().dropna(how="all")
            port_daily = compute_weighted_returns(port_price, weights)
        else:
            port_daily = pd.Series(dtype=float)

        # benchmark daily
        bm_series = bm_daily_ret.get(bm, pd.Series(dtype=float))

        # rolling returns
        r30 = rolling_total_return(port_daily, 30)
        r60 = rolling_total_return(port_daily, 60)
        r1y = rolling_total_return(port_daily, 252)

        b30 = rolling_total_return(bm_series, 30)
        b60 = rolling_total_return(bm_series, 60)
        b1y = rolling_total_return(bm_series, 252)

        a30 = (r30 - b30) if (r30 is not None and b30 is not None) else None
        a60 = (r60 - b60) if (r60 is not None and b60 is not None) else None
        a1y = (r1y - b1y) if (r1y is not None and b1y is not None) else None

        # intraday
        ir = intraday_return_from_fastinfo(fast, weights)
        # benchmark intraday
        bm_fast = fast.get(bm, {})
        bm_ir = None
        try:
            p = float(bm_fast.get("price", float("nan")))
            prev = float(bm_fast.get("prev_close", float("nan")))
            if not math.isnan(p) and not math.isnan(prev) and prev != 0:
                bm_ir = (p / prev) - 1.0
        except Exception:
            bm_ir = None

        ia = (ir - bm_ir) if (ir is not None and bm_ir is not None) else None

        row = {
            "timestamp": ts_now(),
            "Wave": w,
            "benchmark": bm,

            # Intraday
            "intraday_return": ir,
            "intraday_alpha": ia,

            # Rolling windows
            "return_30d": r30,
            "alpha_30d": a30,

            "return_60d": r60,
            "alpha_60d": a60,

            "return_1y": r1y,
            "alpha_1y": a1y,

            # Benchmark window returns (helpful for debugging)
            "benchmark_return_30d": b30,
            "benchmark_return_60d": b60,
            "benchmark_return_1y": b1y,
        }

        append_performance_log(w, row)
        print(f"[{ts_now()}] Wrote performance for: {w} | bm={bm}")

    print(f"[{ts_now()}] Engine complete. Perf logs in: {LOGS_PERF_DIR}")

def main():
    run_engine()

if __name__ == "__main__":
    main()