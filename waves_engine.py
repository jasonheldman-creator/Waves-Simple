# waves_engine.py — WAVES Engine (Benchmark-LOCKED, Stable Alpha)
import os, math, time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOGS_POS_DIR = os.path.join(LOGS_DIR, "positions")
LOGS_PERF_DIR = os.path.join(LOGS_DIR, "performance")
WAVE_WEIGHTS_PATH = os.path.join(BASE_DIR, "wave_weights.csv")
DEFAULT_NOTIONAL = 100000.0

def ts_now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def ensure_dirs():
    os.makedirs(LOGS_POS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERF_DIR, exist_ok=True)

def pick_col(df, cands):
    m = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand.lower() in m: return m[cand.lower()]
    return None

def norm_wave(x): return " ".join(str(x).strip().split())
def norm_tkr(x):  return str(x).strip().upper()

def fetch_prices(tickers: List[str], period="5y") -> pd.DataFrame:
    tickers = [t for t in tickers if t]
    if not tickers: return pd.DataFrame()
    df = yf.download(
        tickers=tickers, period=period, interval="1d",
        auto_adjust=True, progress=False, group_by="column", threads=True
    )
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"] if "Close" in df.columns.get_level_values(0) else df.xs(df.columns.levels[0][0], axis=1, level=0)
        return close.dropna(how="all")
    else:
        col = "Close" if "Close" in df.columns else df.columns[0]
        out = df[[col]].copy()
        out.columns = [tickers[0]]
        return out.dropna(how="all")

def roll_total_return(daily_ret: pd.Series, n: int) -> Optional[float]:
    if daily_ret is None or daily_ret.empty: return None
    tail = daily_ret.dropna().iloc[-n:]
    # require decent sample
    if len(tail) < max(10, int(n * 0.7)): return None
    return float((1.0 + tail).prod() - 1.0)

def intraday_return(tickers: List[str], weights: Dict[str,float]) -> Optional[float]:
    vals = []
    for t in tickers:
        w = weights.get(t, 0.0)
        if w <= 0: continue
        try:
            tk = yf.Ticker(t)
            fi = getattr(tk, "fast_info", {}) or {}
            price = fi.get("last_price", None)
            prev  = fi.get("previous_close", None)
            if price is None or prev is None:
                info = tk.info or {}
                price = price if price is not None else info.get("regularMarketPrice", None)
                prev  = prev  if prev  is not None else info.get("regularMarketPreviousClose", None)
            if price is None or prev is None or prev == 0: 
                continue
            r = (float(price)/float(prev)) - 1.0
            vals.append(w * r)
        except Exception:
            continue
    return float(sum(vals)) if vals else None

def write_positions(wave: str, dfw: pd.DataFrame):
    today = datetime.now().strftime("%Y%m%d")
    path = os.path.join(LOGS_POS_DIR, f"{wave}_positions_{today}.csv")
    rows = []
    for _, r in dfw.iterrows():
        rows.append({
            "Wave": wave,
            "Ticker": r["Ticker"],
            "Weight": float(r["Weight"]),
            "NotionalUSD": DEFAULT_NOTIONAL * float(r["Weight"]),
            "timestamp": ts_now(),
        })
    pd.DataFrame(rows).to_csv(path, index=False)

def append_perf(wave: str, row: Dict):
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
    if len(df) > 2000:
        df = df.iloc[-2000:].copy()
    df.to_csv(path, index=False)

def run_engine():
    ensure_dirs()
    if not os.path.exists(WAVE_WEIGHTS_PATH):
        raise FileNotFoundError("wave_weights.csv not found")

    raw = pd.read_csv(WAVE_WEIGHTS_PATH)

    wave_col = pick_col(raw, ["Wave","Portfolio","Name"])
    tick_col = pick_col(raw, ["Ticker","Symbol"])
    wt_col   = pick_col(raw, ["Weight","Alloc","Allocation"])
    bm_col   = pick_col(raw, ["Benchmark"])

    if wave_col is None or tick_col is None or wt_col is None:
        raise ValueError("wave_weights.csv must contain Wave/Ticker/Weight columns.")
    if bm_col is None:
        raise ValueError("Add a 'Benchmark' column to wave_weights.csv (required).")

    df = pd.DataFrame({
        "Wave": raw[wave_col].astype(str).map(norm_wave),
        "Ticker": raw[tick_col].astype(str).map(norm_tkr),
        "Weight": pd.to_numeric(raw[wt_col], errors="coerce"),
        "Benchmark": raw[bm_col].astype(str).map(norm_tkr),
    }).dropna(subset=["Wave","Ticker","Weight","Benchmark"])

    waves = sorted(df["Wave"].unique().tolist())
    print(f"[{ts_now()}] Running engine for {len(waves)} waves")

    # pull all tickers + benchmarks
    all_tickers = sorted(set(df["Ticker"].tolist()) | set(df["Benchmark"].tolist()))
    prices = fetch_prices(all_tickers, period="5y")

    for w in waves:
        sub = df[df["Wave"] == w].copy()
        # normalize weights + combine dups
        sub["Weight"] = sub["Weight"].clip(lower=0.0)
        sub = sub.groupby(["Wave","Ticker","Benchmark"], as_index=False)["Weight"].sum()
        total = sub["Weight"].sum()
        sub["Weight"] = sub["Weight"] / total if total > 0 else 0.0

        bm = sub["Benchmark"].iloc[0]
        tickers = sub["Ticker"].tolist()
        weights = {r["Ticker"]: float(r["Weight"]) for _, r in sub.iterrows()}

        write_positions(w, sub[["Wave","Ticker","Weight"]])

        # daily return series
        port_cols = [t for t in tickers if t in prices.columns]
        if port_cols:
            rets = prices[port_cols].pct_change().dropna(how="all")
            # weighted sum
            port = None
            for t in port_cols:
                series = rets[t].dropna()
                if series.empty: 
                    continue
                port = series * weights.get(t, 0.0) if port is None else port.add(series * weights.get(t, 0.0), fill_value=0.0)
            port = port.dropna() if port is not None else pd.Series(dtype=float)
        else:
            port = pd.Series(dtype=float)

        bm_ret = prices[bm].pct_change().dropna() if bm in prices.columns else pd.Series(dtype=float)

        r30 = roll_total_return(port, 30);  b30 = roll_total_return(bm_ret, 30)
        r60 = roll_total_return(port, 60);  b60 = roll_total_return(bm_ret, 60)
        r1y = roll_total_return(port, 252); b1y = roll_total_return(bm_ret, 252)

        ir  = intraday_return(tickers, weights)
        # benchmark intraday (simple)
        bm_ir = intraday_return([bm], {bm: 1.0})

        row = {
            "timestamp": ts_now(),
            "Wave": w,
            "benchmark": bm,
            "intraday_return": ir,
            "intraday_alpha": (ir - bm_ir) if (ir is not None and bm_ir is not None) else None,
            "return_30d": r30,
            "alpha_30d": (r30 - b30) if (r30 is not None and b30 is not None) else None,
            "return_60d": r60,
            "alpha_60d": (r60 - b60) if (r60 is not None and b60 is not None) else None,
            "return_1y": r1y,
            "alpha_1y": (r1y - b1y) if (r1y is not None and b1y is not None) else None,
        }

        append_perf(w, row)
        print(f"[{ts_now()}] wrote {w} perf (bm={bm})")

    print(f"[{ts_now()}] done. perf logs: {LOGS_PERF_DIR}")

def main():
    run_engine()

if __name__ == "__main__":
    main()