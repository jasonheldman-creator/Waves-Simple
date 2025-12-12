# waves_engine.py — WAVES Intelligence™ Engine
# Purpose:
# - Read wave_weights.csv
# - Normalize wave names into a stable slug for filenames
# - Fetch prices via yfinance
# - Compute basic returns & alpha vs benchmark
# - Write logs/performance/<slug>_performance_daily.csv
# - Write logs/positions/<slug>_positions_YYYYMMDD.csv
# - Write logs/performance/_wave_manifest.csv (display name <-> slug mapping)

import os
import re
import sys
import time
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise RuntimeError("yfinance is required. Check requirements.txt") from e


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOGS_PERF_DIR = os.path.join(LOGS_DIR, "performance")
LOGS_POS_DIR = os.path.join(LOGS_DIR, "positions")

WAVE_WEIGHTS_PATH = os.path.join(BASE_DIR, "wave_weights.csv")

# ---- Benchmark mapping (safe defaults; adjust anytime) ----
BENCHMARK_MAP = {
    "S&P 500 Wave": "SPY",
    "Growth Wave": "QQQ",
    "AI Wave": "QQQ",
    "Cloud & Software Wave": "QQQ",
    "Quantum Computing Wave": "QQQ",
    "Small Cap Growth Wave": "IWM",
    "Future Power & Energy Wave": "XLE",
    "Clean Transit-Infrastructure Wave": "IDRV",  # alt: CARZ
    "SmartSafe Wave": "BIL",
    "Income Wave": "SCHD",
    "Crypto Income Wave": "BTC-USD",
}

DEFAULT_BENCHMARK = "SPY"


# -------------------------
# Canonical normalization
# -------------------------
def normalize_wave_name(name: str) -> str:
    """
    Canonical slug used EVERYWHERE for filenames.
    Rules:
      - lowercase
      - & -> and
      - replace non-alphanumeric with underscores
      - collapse underscores
      - strip underscores
    """
    s = (name or "").strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def ensure_dirs() -> None:
    os.makedirs(LOGS_PERF_DIR, exist_ok=True)
    os.makedirs(LOGS_POS_DIR, exist_ok=True)


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_wave_weights(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing weights file: {path}")

    df = pd.read_csv(path)

    # Normalize headers
    cols = {c.lower().strip(): c for c in df.columns}
    if "wave" not in cols or "ticker" not in cols or "weight" not in cols:
        raise ValueError("wave_weights.csv must have columns: wave,ticker,weight")

    df = df.rename(columns={cols["wave"]: "wave", cols["ticker"]: "ticker", cols["weight"]: "weight"})
    df["wave"] = df["wave"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    df = df.dropna(subset=["wave", "ticker", "weight"])
    df = df[df["weight"] > 0]

    # Quick placeholder safety
    if df["ticker"].str.upper().str.contains("REPLACE_").any():
        bad = df[df["ticker"].str.upper().str.contains("REPLACE_")]["ticker"].unique().tolist()
        raise ValueError(f"Placeholder tickers found (must replace with real tickers): {bad[:10]}")

    return df


def group_waves(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    waves = {}
    for wave_name, g in df.groupby("wave"):
        g2 = g.copy()

        # Deduplicate tickers within wave by summing weights
        g2["ticker"] = g2["ticker"].astype(str).str.strip()
        g2 = g2.groupby(["wave", "ticker"], as_index=False)["weight"].sum()

        # Normalize weights to sum to 1.0
        total = g2["weight"].sum()
        if total <= 0:
            continue
        g2["weight"] = g2["weight"] / total

        waves[wave_name] = g2.sort_values("weight", ascending=False).reset_index(drop=True)

    return waves


def safe_yf_download(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    """
    Wrapper that tolerates yfinance quirks. Returns dataframe with columns by ticker.
    """
    if not tickers:
        return pd.DataFrame()

    # yfinance prefers space-separated string or list
    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        return data
    except Exception:
        # Last-resort: try one-by-one
        frames = []
        for t in tickers:
            try:
                d = yf.download(tickers=t, period=period, interval=interval, progress=False)
                d["__ticker__"] = t
                frames.append(d)
            except Exception:
                pass
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0)


def get_last_close_and_prev_close(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (last_price, prev_close) using daily history.
    """
    try:
        h = yf.Ticker(ticker).history(period="6d", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            return None, None
        closes = h["Close"].dropna()
        if len(closes) < 2:
            return float(closes.iloc[-1]), None
        return float(closes.iloc[-1]), float(closes.iloc[-2])
    except Exception:
        return None, None


def get_close_n_days_ago(ticker: str, n: int) -> Optional[float]:
    """
    Get close price about n trading days ago using daily history.
    """
    try:
        h = yf.Ticker(ticker).history(period=f"{max(n+10, 40)}d", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            return None
        closes = h["Close"].dropna()
        if len(closes) <= n:
            # not enough history
            return None
        return float(closes.iloc[-(n+1)])
    except Exception:
        return None


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    try:
        if old == 0:
            return None
        return (float(new) / float(old)) - 1.0
    except Exception:
        return None


def portfolio_return(weights: pd.DataFrame, price_now: Dict[str, float], price_then: Dict[str, float]) -> Optional[float]:
    """
    Weighted return: sum(w * (now/then - 1)).
    Requires price_then for each ticker; missing tickers are skipped but renormalized.
    """
    rows = []
    for _, r in weights.iterrows():
        t = r["ticker"]
        w = float(r["weight"])
        pn = price_now.get(t)
        pt = price_then.get(t)
        if pn is None or pt is None or pt == 0:
            continue
        rows.append((w, (pn / pt) - 1.0))

    if not rows:
        return None

    wsum = sum(w for w, _ in rows)
    if wsum <= 0:
        return None

    return sum((w / wsum) * ret for w, ret in rows)


def write_manifest(waves: Dict[str, pd.DataFrame]) -> None:
    rows = []
    for wave_name in sorted(waves.keys()):
        rows.append({"wave": wave_name, "slug": normalize_wave_name(wave_name)})
    out = pd.DataFrame(rows)
    out_path = os.path.join(LOGS_PERF_DIR, "_wave_manifest.csv")
    out.to_csv(out_path, index=False)


def write_positions_log(wave_name: str, weights: pd.DataFrame, prices: Dict[str, float]) -> None:
    slug = normalize_wave_name(wave_name)
    fname = f"{slug}_positions_{date.today().strftime('%Y%m%d')}.csv"
    path = os.path.join(LOGS_POS_DIR, fname)

    out = weights.copy()
    out["price"] = out["ticker"].map(prices)
    out["value_weighted"] = out["weight"] * out["price"]
    out.to_csv(path, index=False)


def append_perf_row(wave_name: str, row: Dict[str, object]) -> None:
    slug = normalize_wave_name(wave_name)
    path = os.path.join(LOGS_PERF_DIR, f"{slug}_performance_daily.csv")

    df_row = pd.DataFrame([row])

    if os.path.exists(path):
        try:
            df_old = pd.read_csv(path)
            df_new = pd.concat([df_old, df_row], ignore_index=True)
            df_new.to_csv(path, index=False)
            return
        except Exception:
            # If old file corrupt, overwrite safely
            pass

    df_row.to_csv(path, index=False)


def compute_wave_metrics(wave_name: str, weights: pd.DataFrame) -> Dict[str, object]:
    # Resolve benchmark
    bench = BENCHMARK_MAP.get(wave_name, DEFAULT_BENCHMARK)

    tickers = weights["ticker"].tolist()
    all_needed = sorted(set(tickers + [bench]))

    # Prices now + prev close for intraday
    price_now = {}
    prev_close = {}

    for t in all_needed:
        last_p, prev_p = get_last_close_and_prev_close(t)
        if last_p is not None:
            price_now[t] = last_p
        if prev_p is not None:
            prev_close[t] = prev_p

    # Intraday return uses (last vs prev close) as proxy
    wave_intraday = portfolio_return(
        weights,
        price_now={t: price_now.get(t) for t in tickers},
        price_then={t: prev_close.get(t) for t in tickers},
    )
    bench_intraday = pct_change(price_now.get(bench), prev_close.get(bench))
    intraday_alpha = None
    if wave_intraday is not None and bench_intraday is not None:
        intraday_alpha = wave_intraday - bench_intraday

    # 30d/60d/1y closes
    # Use ~trading days: 30d≈21, 60d≈42, 1y≈252
    horizons = {"30d": 21, "60d": 42, "1y": 252}

    price_then = {k: {} for k in horizons.keys()}
    bench_then = {}

    for label, n in horizons.items():
        # holdings
        for t in tickers:
            p = get_close_n_days_ago(t, n)
            if p is not None:
                price_then[label][t] = p
        # benchmark
        bp = get_close_n_days_ago(bench, n)
        if bp is not None:
            bench_then[label] = bp

    ret_30 = portfolio_return(weights, {t: price_now.get(t) for t in tickers}, price_then["30d"])
    ret_60 = portfolio_return(weights, {t: price_now.get(t) for t in tickers}, price_then["60d"])
    ret_1y = portfolio_return(weights, {t: price_now.get(t) for t in tickers}, price_then["1y"])

    bench_30 = pct_change(price_now.get(bench), bench_then.get("30d"))
    bench_60 = pct_change(price_now.get(bench), bench_then.get("60d"))
    bench_1y = pct_change(price_now.get(bench), bench_then.get("1y"))

    alpha_30 = None if ret_30 is None or bench_30 is None else (ret_30 - bench_30)
    alpha_60 = None if ret_60 is None or bench_60 is None else (ret_60 - bench_60)
    alpha_1y = None if ret_1y is None or bench_1y is None else (ret_1y - bench_1y)

    return {
        "timestamp": now_ts(),
        "wave": wave_name,
        "wave_slug": normalize_wave_name(wave_name),
        "benchmark": bench,

        "intraday_return": wave_intraday,
        "intraday_alpha": intraday_alpha,

        "return_30d": ret_30,
        "alpha_30d": alpha_30,

        "return_60d": ret_60,
        "alpha_60d": alpha_60,

        "return_1y": ret_1y,
        "alpha_1y": alpha_1y,

        "notes": "",
    }


def main() -> int:
    ensure_dirs()

    print("\n=== WAVES Intelligence™ Engine ===")
    print("Time:", now_ts())
    print("Weights:", WAVE_WEIGHTS_PATH)

    df = load_wave_weights(WAVE_WEIGHTS_PATH)
    waves = group_waves(df)

    if not waves:
        print("No waves found in wave_weights.csv")
        return 1

    # Write manifest every run
    write_manifest(waves)

    print(f"Discovered waves: {len(waves)}")
    for w in sorted(waves.keys()):
        print(" -", w, "->", normalize_wave_name(w))

    # Compute each wave and write logs
    ok = 0
    for wave_name, weights in waves.items():
        try:
            metrics = compute_wave_metrics(wave_name, weights)

            # Prices for positions log (latest)
            tickers = weights["ticker"].tolist()
            prices = {}
            for t in tickers:
                lp, _ = get_last_close_and_prev_close(t)
                if lp is not None:
                    prices[t] = lp

            write_positions_log(wave_name, weights, prices)
            append_perf_row(wave_name, metrics)

            ok += 1
            print(f"[OK] {wave_name} | intraday={metrics.get('intraday_return')} 30d={metrics.get('return_30d')} 60d={metrics.get('return_60d')} 1y={metrics.get('return_1y')}")
        except Exception as e:
            print(f"[ERR] {wave_name}: {repr(e)}")

    print(f"Completed: {ok}/{len(waves)} waves")
    print("Perf logs:", LOGS_PERF_DIR)
    print("Pos logs :", LOGS_POS_DIR)
    return 0 if ok > 0 else 2


if __name__ == "__main__":
    sys.exit(main())