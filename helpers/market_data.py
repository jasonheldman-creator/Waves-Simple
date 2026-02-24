"""Market data utilities - stub implementation (no live API calls)."""
import os
from datetime import datetime, timedelta

BENCHMARK_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "MDY"]
REQUIRED_BENCHMARKS = ["SPY", "QQQ", "IWM", "VIX"]
SECTOR_TICKERS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC"]
INDEX_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "VTI", "EFA", "EEM"]
YIELD_TICKERS = ["TLT", "IEF", "SHY", "HYG", "LQD"]

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "prices_cache.parquet")
_PRICES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "prices.csv")


def _load_prices_csv_as_dict(lookback_days=400):
    """Load canonical data/prices.csv into {ticker: [float, ...]} dict."""
    try:
        import pandas as pd
        df = pd.read_csv(_PRICES_CSV_PATH)
        if "ticker" not in df.columns or "close" not in df.columns:
            return {}
        result = {}
        for ticker, group in df.groupby("ticker"):
            series = group.sort_values("date")["close"].dropna().tolist()
            result[str(ticker)] = series[-lookback_days:] if len(series) > lookback_days else series
        return result
    except Exception as exc:
        print(f"[MI] _load_prices_csv_as_dict: failed ({exc})")
        return {}


def fetch_all_prices(tickers=None, lookback_days=400):
    """Return dict of {ticker: list of floats} from cache if available, else empty dict.

    When *tickers* is None (default), all columns in the cache are returned so
    that the Market Intelligence pipeline has data without needing an explicit
    ticker list at the call site.  Missing tickers are back-filled from the
    canonical data/prices.csv source.
    """
    load_all = tickers is None
    tickers = tickers or []
    result = {}
    try:
        import pandas as pd
        df = pd.read_parquet(_CACHE_PATH)
        print(f"[MI] fetch_all_prices: cache loaded, {len(df.columns)} tickers, "
              f"load_all={load_all}, requested={'all' if load_all else len(tickers)}")
        if load_all:
            tickers = list(df.columns)
        for t in tickers:
            if t in df.columns:
                series = df[t].dropna().tolist()
                result[t] = series[-lookback_days:] if len(series) > lookback_days else series
            else:
                result[t] = None
    except Exception as exc:
        print(f"[MI] fetch_all_prices: cache read failed ({exc}); will rely on prices.csv")

    # Merge canonical data/prices.csv for any tickers missing from the parquet cache
    try:
        csv_dict = _load_prices_csv_as_dict(lookback_days=lookback_days)
        if csv_dict:
            if load_all:
                # Add all CSV tickers not already in result
                for sym, series in csv_dict.items():
                    if not result.get(sym) and series:
                        result[sym] = series
            else:
                # Fill missing requested tickers from CSV
                for t in tickers:
                    if not result.get(t) and t in csv_dict and csv_dict[t]:
                        result[t] = csv_dict[t]
                        print(f"[MI] fetch_all_prices: filled {t} from prices.csv ({len(csv_dict[t])} pts)")
    except Exception as csv_exc:
        print(f"[MI] fetch_all_prices: prices.csv merge failed ({csv_exc})")

    # yfinance fallback for REQUIRED_BENCHMARKS still missing (None or empty list)
    if load_all:
        for sym in REQUIRED_BENCHMARKS:
            if not result.get(sym):  # None or [] both indicate missing/invalid data
                try:
                    import yfinance as yf
                    end_dt = datetime.utcnow()
                    start_dt = end_dt - timedelta(days=lookback_days + 30)
                    hist = yf.download(
                        sym,
                        start=start_dt.strftime("%Y-%m-%d"),
                        end=end_dt.strftime("%Y-%m-%d"),
                        auto_adjust=True,
                        progress=False,
                    )
                    if hist is not None and not hist.empty and "Close" in hist.columns:
                        series = hist["Close"].dropna().tolist()
                        if series:
                            result[sym] = series[-lookback_days:] if len(series) > lookback_days else series
                            print(f"[MI] fetch_all_prices: fallback fetched {sym} ({len(result[sym])} pts)")
                except Exception as sym_exc:
                    print(f"[MI] fetch_all_prices: fallback fetch failed for {sym} ({sym_exc})")

    populated = sum(1 for v in result.values() if v)
    print(f"[MI] fetch_all_prices: returning {populated} populated / {len(result)} total")
    return result if result else {t: None for t in tickers}


def compute_returns(prices, window=None):
    """Return dict of period returns for a price series (list of floats)."""
    if not prices or len(prices) < 2:
        return {"1d": None, "5d": None, "30d": None, "90d": None, "365d": None}

    def _ret(n):
        if len(prices) >= n + 1:
            try:
                return round(prices[-1] / prices[-n - 1] - 1, 6)
            except (ZeroDivisionError, TypeError):
                return None
        return None

    return {
        "1d": _ret(1),
        "5d": _ret(5),
        "30d": _ret(21),
        "90d": _ret(63),
        "365d": _ret(252),
    }


def compute_slope(prices, window=None):
    """Return normalised linear regression slope of the last 30 prices."""
    if not prices or len(prices) < 2:
        return 0.0
    try:
        import numpy as np
        series = prices[-30:]
        x = np.arange(len(series), dtype=float)
        y = np.array(series, dtype=float)
        if y.std() == 0:
            return 0.0
        slope = float(np.polyfit(x, y, 1)[0])
        return round(slope / (y.mean() or 1), 6)
    except Exception:
        return 0.0


def compute_realized_vol(prices, window=21):
    """Return annualised realised volatility."""
    if not prices or len(prices) < 2:
        return 0.0
    try:
        import numpy as np
        series = np.array(prices[-window - 1:], dtype=float)
        log_rets = np.diff(np.log(series))
        return float(np.std(log_rets) * (252 ** 0.5))
    except Exception:
        return 0.0


def compute_vol_of_vol(prices, window=21):
    """Return vol-of-vol (std of rolling 5-day vol)."""
    if not prices or len(prices) < window + 5:
        return 0.0
    try:
        import numpy as np
        series = np.array(prices, dtype=float)
        log_rets = np.diff(np.log(series))
        rolling_vols = [
            np.std(log_rets[max(0, i - 5): i]) * (252 ** 0.5)
            for i in range(5, len(log_rets))
        ]
        return float(np.std(rolling_vols[-window:])) if rolling_vols else 0.0
    except Exception:
        return 0.0


def compute_drawdown(prices, window=None):
    """Return maximum drawdown as a negative float."""
    if not prices or len(prices) < 2:
        return 0.0
    try:
        import numpy as np
        series = np.array(prices, dtype=float)
        peak = np.maximum.accumulate(series)
        drawdowns = (series - peak) / peak
        return float(np.min(drawdowns))
    except Exception:
        return 0.0


def compute_pct_up_days(prices, window=30):
    """Return fraction of up days in the last `window` sessions."""
    if not prices or len(prices) < 2:
        return 0.0
    try:
        import numpy as np
        series = np.array(prices[-window - 1:], dtype=float)
        rets = np.diff(series)
        return float(np.sum(rets > 0) / len(rets))
    except Exception:
        return 0.0


def compute_above_ma(prices, window=50):
    """Return True if the last price is above its `window`-period moving average."""
    if not prices or len(prices) < window:
        return False
    try:
        import numpy as np
        ma = float(np.mean(prices[-window:]))
        return prices[-1] > ma
    except Exception:
        return False


def compute_relative_strength(prices, benchmark_prices, window=None):
    """Return relative strength of prices vs benchmark (ratio of recent returns)."""
    if not prices or not benchmark_prices or len(prices) < 2 or len(benchmark_prices) < 2:
        return 0.0
    try:
        asset_ret = prices[-1] / prices[-min(22, len(prices))] - 1
        bench_ret = benchmark_prices[-1] / benchmark_prices[-min(22, len(benchmark_prices))] - 1
        return round(asset_ret - bench_ret, 6)
    except (ZeroDivisionError, TypeError):
        return 0.0
