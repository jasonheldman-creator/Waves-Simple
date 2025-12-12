# app.py — WAVES Intelligence™ Institutional Console (Auto-Discover Waves)
# Restores: full console, all waves matrix (returns + alpha), per-wave pages,
# top-10 holdings with Google quote links, and auto-includes any "sandbox" wave
# as long as it exists in wave_weights.csv and/or logs/.

import os
import glob
import math
import json
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

APP_TITLE = "WAVES Intelligence™ — Institutional Console"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Expected folders (engine writes here)
LOGS_POS_DIR = os.path.join(BASE_DIR, "logs", "positions")
LOGS_PERF_DIR = os.path.join(BASE_DIR, "logs", "performance")

# Inputs (engine uses these)
WAVE_WEIGHTS_PATH = os.path.join(BASE_DIR, "wave_weights.csv")
UNIVERSE_LIST_PATH = os.path.join(BASE_DIR, "list.csv")

# Optional: if you maintain a sandbox-only results file, it will be picked up if present
SANDBOX_OVERRIDE_PATH = os.path.join(BASE_DIR, "logs", "sandbox", "sandbox_summary.csv")

# ---------------------------
# Helpers
# ---------------------------

def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def _pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x*100:.{digits}f}%"

def _ts_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def google_quote_link(ticker: str) -> str:
    q = urllib.parse.quote(ticker.strip().upper())
    return f"https://www.google.com/finance/quote/{q}:NASDAQ"

def google_quote_link_fallback(ticker: str) -> str:
    # If exchange is unknown, Google still resolves many tickers via search
    q = urllib.parse.quote(ticker.strip().upper())
    return f"https://www.google.com/search?q=Google+Finance+{q}"

def normalize_wave_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    n = name.strip()
    # Light normalization only (keeps your branding)
    n = n.replace("WAVE", "Wave").replace("wave", "Wave")
    n = " ".join(n.split())
    return n

def find_latest_file(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def load_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def list_waves_from_weights() -> List[str]:
    df = load_csv_if_exists(WAVE_WEIGHTS_PATH)
    if df is None or df.empty:
        return []
    # Try common column names
    for col in ["Wave", "wave", "WAVE", "portfolio", "Portfolio", "name", "Name"]:
        if col in df.columns:
            waves = sorted({normalize_wave_name(x) for x in df[col].dropna().astype(str).tolist()})
            return waves
    # If not found, try first column
    waves = sorted({normalize_wave_name(x) for x in df.iloc[:, 0].dropna().astype(str).tolist()})
    return waves

def list_waves_from_logs() -> List[str]:
    waves = set()
    if os.path.isdir(LOGS_POS_DIR):
        for p in glob.glob(os.path.join(LOGS_POS_DIR, "*_positions_*.csv")):
            base = os.path.basename(p)
            # "<Wave>_positions_YYYYMMDD.csv"
            wave = base.split("_positions_")[0]
            if wave:
                waves.add(normalize_wave_name(wave))
    if os.path.isdir(LOGS_PERF_DIR):
        for p in glob.glob(os.path.join(LOGS_PERF_DIR, "*_performance_*.csv")):
            base = os.path.basename(p)
            wave = base.split("_performance_")[0]
            if wave:
                waves.add(normalize_wave_name(wave))
    return sorted(waves)

def discover_all_waves() -> List[str]:
    waves = set(list_waves_from_weights()) | set(list_waves_from_logs())
    return sorted(waves)

def load_latest_positions(wave: str) -> Optional[pd.DataFrame]:
    # support wave names with spaces in filesystem: engine typically writes exact wave string;
    # we try both raw and sanitized patterns.
    patterns = [
        os.path.join(LOGS_POS_DIR, f"{wave}_positions_*.csv"),
        os.path.join(LOGS_POS_DIR, f"{wave.replace(' ', '')}_positions_*.csv"),
        os.path.join(LOGS_POS_DIR, f"{wave.replace(' ', '_')}_positions_*.csv"),
    ]
    for pat in patterns:
        latest = find_latest_file(pat)
        if latest:
            try:
                return pd.read_csv(latest)
            except Exception:
                return None
    return None

def load_latest_performance(wave: str) -> Optional[pd.DataFrame]:
    patterns = [
        os.path.join(LOGS_PERF_DIR, f"{wave}_performance_*.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave.replace(' ', '')}_performance_*.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave.replace(' ', '_')}_performance_*.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave}_performance_daily.csv"),
    ]
    for pat in patterns:
        latest = find_latest_file(pat)
        if latest and os.path.exists(latest):
            try:
                return pd.read_csv(latest)
            except Exception:
                return None
    return None

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

@dataclass
class WaveSnapshot:
    wave: str
    tag: str  # LIVE/SANDBOX/HYBRID
    updated: str
    intraday_return: Optional[float] = None
    intraday_alpha: Optional[float] = None
    r30: Optional[float] = None
    a30: Optional[float] = None
    r60: Optional[float] = None
    a60: Optional[float] = None
    r1y: Optional[float] = None
    a1y: Optional[float] = None

def build_snapshot_from_perf(wave: str, perf: Optional[pd.DataFrame]) -> WaveSnapshot:
    snap = WaveSnapshot(wave=wave, tag="HYBRID", updated="—")

    if perf is None or perf.empty:
        return snap

    # Updated timestamp (best-effort)
    dt_col = _pick_col(perf, ["timestamp", "datetime", "date", "asof", "time"])
    if dt_col:
        try:
            snap.updated = str(perf[dt_col].iloc[-1])
        except Exception:
            snap.updated = "—"
    else:
        snap.updated = "latest row"

    # Tag
    tag_col = _pick_col(perf, ["regime", "data_regime", "tag", "mode_tag", "live_sandbox"])
    if tag_col:
        t = str(perf[tag_col].iloc[-1]).upper()
        if "LIVE" in t:
            snap.tag = "LIVE"
        elif "SANDBOX" in t or "BACKTEST" in t:
            snap.tag = "SANDBOX"
        else:
            snap.tag = "HYBRID"

    row = perf.iloc[-1]

    # Flexible column names for returns/alphas
    snap.intraday_return = _safe_float(row.get(_pick_col(perf, ["intraday_return", "return_intraday", "day_return", "return_1d", "ret_1d"]) or "", None))
    snap.intraday_alpha  = _safe_float(row.get(_pick_col(perf, ["intraday_alpha", "alpha_intraday", "alpha_1d", "alpha_day", "alpha_capture_1d"]) or "", None))

    snap.r30 = _safe_float(row.get(_pick_col(perf, ["return_30d", "ret_30d", "r30", "return30"]) or "", None))
    snap.a30 = _safe_float(row.get(_pick_col(perf, ["alpha_30d", "a30", "alpha30", "alpha_capture_30d"]) or "", None))

    snap.r60 = _safe_float(row.get(_pick_col(perf, ["return_60d", "ret_60d", "r60", "return60"]) or "", None))
    snap.a60 = _safe_float(row.get(_pick_col(perf, ["alpha_60d", "a60", "alpha60", "alpha_capture_60d"]) or "", None))

    snap.r1y = _safe_float(row.get(_pick_col(perf, ["return_1y", "return_1yr", "ret_1y", "ret_1yr", "return_252d"]) or "", None))
    snap.a1y = _safe_float(row.get(_pick_col(perf, ["alpha_1y", "alpha_1yr", "a1y", "alpha_capture_1y", "alpha_capture_252d"]) or "", None))

    return snap

def apply_sandbox_override(snapshots: Dict[str, WaveSnapshot]) -> Dict[str, WaveSnapshot]:
    # If you have a sandbox summary file, we merge it by wave name.
    if not os.path.exists(SANDBOX_OVERRIDE_PATH):
        return snapshots

    try:
        sdf = pd.read_csv(SANDBOX_OVERRIDE_PATH)
    except Exception:
        return snapshots

    # Must have a wave column
    wave_col = None
    for c in ["Wave", "wave", "name", "Name", "portfolio", "Portfolio"]:
        if c in sdf.columns:
            wave_col = c
            break
    if wave_col is None:
        return snapshots

    for _, r in sdf.iterrows():
        w = normalize_wave_name(str(r[wave_col]))
        if w not in snapshots:
            snapshots[w] = WaveSnapshot(wave=w, tag="SANDBOX", updated="sandbox override")

        snap = snapshots[w]
        snap.tag = "SANDBOX"
        snap.updated = "sandbox override"

        # best-effort merges
        for k, cand in [
            ("intraday_return", ["intraday_return", "return_1d", "ret_1d"]),
            ("intraday_alpha",  ["intraday_alpha", "alpha_1d", "alpha_capture_1d"]),
            ("r30", ["return_30d", "ret_30d"]),
            ("a30", ["alpha_30d", "alpha_capture_30d"]),
            ("r60", ["return_60d", "ret_60d"]),
            ("a60", ["alpha_60d", "alpha_capture_60d"]),
            ("r1y", ["return_1y", "ret_1y", "return_1yr"]),
            ("a1y", ["alpha_1y", "alpha_capture_1y", "alpha_1yr"]),
        ]:
            val = None
            for c in cand:
                if c in sdf.columns:
                    val = _safe_float(r.get(c))
                    break
            if val is not None:
                setattr(snap, k, val)

    return snapshots

def top10_from_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker", "Weight", "Value", "Link"])
    # Guess columns
    tcol = None
    for c in ["ticker", "Ticker", "symbol", "Symbol"]:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        tcol = df.columns[0]

    wcol = None
    for c in ["weight", "Weight", "w", "alloc", "allocation"]:
        if c in df.columns:
            wcol = c
            break

    vcol = None
    for c in ["value", "Value", "market_value", "MarketValue", "dollar_value", "Dollars"]:
        if c in df.columns:
            vcol = c
            break

    out = df.copy()
    out[tcol] = out[tcol].astype(str).str.upper().str.strip()

    if wcol:
        out[wcol] = pd.to_numeric(out[wcol], errors="coerce")
        out = out.sort_values(wcol, ascending=False)
    elif vcol:
        out[vcol] = pd.to_numeric(out[vcol], errors="coerce")
        out = out.sort_values(vcol, ascending=False)

    out = out.head(10).copy()
    out["Link"] = out[tcol].apply(lambda x: google_quote_link_fallback(x))

    # Standardize columns
    out_df = pd.DataFrame({
        "Ticker": out[tcol].values,
        "Weight": out[wcol].values if wcol else [None]*len(out),
        "Value": out[vcol].values if vcol else [None]*len(out),
        "Link": out["Link"].values
    })
    return out_df

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Controls")
    st.caption(f"Last loaded: {_ts_now()}")
    show_1y = st.toggle("Show 1-Year", value=True)
    show_60 = st.toggle("Show 60-Day", value=True)
    show_30 = st.toggle("Show 30-Day", value=True)
    show_intraday = st.toggle("Show Intraday", value=True)

    st.divider()
    st.subheader("Data status")
    st.write("Weights:", "✅" if os.path.exists(WAVE_WEIGHTS_PATH) else "❌", "wave_weights.csv")
    st.write("Universe:", "✅" if os.path.exists(UNIVERSE_LIST_PATH) else "❌", "list.csv")
    st.write("Logs/positions:", "✅" if os.path.isdir(LOGS_POS_DIR) else "❌", LOGS_POS_DIR)
    st.write("Logs/performance:", "✅" if os.path.isdir(LOGS_PERF_DIR) else "❌", LOGS_PERF_DIR)

st.divider()

waves = discover_all_waves()
if not waves:
    st.error(
        "No Waves found yet.\n\n"
        "Fix: make sure wave_weights.csv exists and/or your engine has written logs to logs/positions and logs/performance."
    )
    st.stop()

# Build snapshots
snapshots: Dict[str, WaveSnapshot] = {}
for w in waves:
    perf = load_latest_performance(w)
    snapshots[w] = build_snapshot_from_perf(w, perf)

snapshots = apply_sandbox_override(snapshots)

# Build table
rows = []
for w in waves:
    s = snapshots[w]
    row = {
        "Wave": s.wave,
        "Tag": s.tag,
        "Updated": s.updated,
    }
    if show_intraday:
        row["Intraday Return"] = s.intraday_return
        row["Intraday Alpha"] = s.intraday_alpha
    if show_30:
        row["30D Return"] = s.r30
        row["30D Alpha"] = s.a30
    if show_60:
        row["60D Return"] = s.r60
        row["60D Alpha"] = s.a60
    if show_1y:
        row["1Y Return"] = s.r1y
        row["1Y Alpha"] = s.a1y
    rows.append(row)

df = pd.DataFrame(rows)

# Pretty format (keep numeric for sorting; show formatted view separately)
sort_col = None
if show_30 and "30D Alpha" in df.columns:
    sort_col = "30D Alpha"
elif show_intraday and "Intraday Alpha" in df.columns:
    sort_col = "Intraday Alpha"
elif show_1y and "1Y Alpha" in df.columns:
    sort_col = "1Y Alpha"

if sort_col:
    df = df.sort_values(sort_col, ascending=False, na_position="last")

st.subheader("All Waves — Returns & Alpha Capture")
st.caption("Auto-discovered from wave_weights.csv and/or logs. Add a new Wave to wave_weights.csv and it will appear here automatically.")

# Display formatted
display_df = df.copy()
for c in display_df.columns:
    if "Return" in c or "Alpha" in c:
        display_df[c] = display_df[c].apply(lambda x: _pct(_safe_float(x)))

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Wave Detail")

# Choose wave
selected_wave = st.selectbox("Select a Wave", waves, index=0)

colA, colB, colC = st.columns([1.2, 1.2, 1.6])
s = snapshots[selected_wave]

with colA:
    st.markdown(f"**Wave:** {s.wave}")
    st.markdown(f"**Tag:** {s.tag}")
    st.markdown(f"**Updated:** {s.updated}")

with colB:
    if show_intraday:
        st.metric("Intraday Return", _pct(s.intraday_return), delta=_pct(s.intraday_alpha))
        st.caption("Delta shown = Intraday Alpha (best-effort)")
    if show_30:
        st.metric("30D Return", _pct(s.r30), delta=_pct(s.a30))
    if show_60:
        st.metric("60D Return", _pct(s.r60), delta=_pct(s.a60))
    if show_1y:
        st.metric("1Y Return", _pct(s.r1y), delta=_pct(s.a1y))

with colC:
    st.markdown("**Top 10 Holdings (latest positions log)**")
    pos = load_latest_positions(selected_wave)
    if pos is None or pos.empty:
        st.info("No positions file found yet for this Wave. Run the engine to generate logs/positions.")
    else:
        top10 = top10_from_positions(pos)
        # render links as markdown
        link_col = []
        for _, r in top10.iterrows():
            t = str(r["Ticker"])
            link = str(r["Link"])
            link_col.append(f"[{t}]({link})")
        top10_view = top10.copy()
        top10_view["Ticker"] = link_col
        # format weights/value
        if "Weight" in top10_view.columns:
            top10_view["Weight"] = top10_view["Weight"].apply(lambda x: _pct(_safe_float(x)) if x is not None else "—")
        if "Value" in top10_view.columns:
            top10_view["Value"] = top10_view["Value"].apply(lambda x: f"${_safe_float(x):,.0f}" if _safe_float(x) is not None else "—")
        top10_view = top10_view.drop(columns=["Link"])
        st.markdown(top10_view.to_markdown(index=False), unsafe_allow_html=True)

st.divider()
with st.expander("Diagnostics (click to open)"):
    st.write("Discovered Waves:", waves)
    st.write("Performance files present:", len(glob.glob(os.path.join(LOGS_PERF_DIR, "*"))))
    st.write("Positions files present:", len(glob.glob(os.path.join(LOGS_POS_DIR, "*"))))
    if os.path.exists(SANDBOX_OVERRIDE_PATH):
        st.write("Sandbox override:", SANDBOX_OVERRIDE_PATH)