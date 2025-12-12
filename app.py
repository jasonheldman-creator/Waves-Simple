# app.py — WAVES Intelligence™ Institutional Console (Full Console Restore Shell)
# - Auto-discovers ALL waves from wave_weights.csv and logs/
# - Overview matrix: Intraday / 30D / 60D / 1Y returns + alpha
# - Wave detail: metrics + Top-10 holdings with Google links
# - WaveScore Leaderboard (reads logs/wavescore/*.csv if present; otherwise placeholder)
# - Correlation matrix (computed from per-wave daily returns logs)
# - Multi-Wave Portfolio Constructor (blends waves + shows blended return/alpha)
#
# NOTE: This is designed to "snap back" to a full console experience
# without requiring you to have the exact prior app.py. If you DO have
# a prior commit (Fallback 1), restoring that is still fastest.

import os
import glob
import math
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

APP_TITLE = "WAVES Intelligence™ — Institutional Console"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOGS_POS_DIR = os.path.join(LOGS_DIR, "positions")
LOGS_PERF_DIR = os.path.join(LOGS_DIR, "performance")
LOGS_WAVESCORE_DIR = os.path.join(LOGS_DIR, "wavescore")
LOGS_SANDBOX_DIR = os.path.join(LOGS_DIR, "sandbox")

WAVE_WEIGHTS_PATH = os.path.join(BASE_DIR, "wave_weights.csv")
UNIVERSE_LIST_PATH = os.path.join(BASE_DIR, "list.csv")

SANDBOX_OVERRIDE_PATH = os.path.join(LOGS_SANDBOX_DIR, "sandbox_summary.csv")  # optional

# -----------------------------
# Utility
# -----------------------------
def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
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

def normalize_wave_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    n = " ".join(name.strip().split())
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

def google_quote_link(ticker: str) -> str:
    q = urllib.parse.quote(ticker.strip().upper())
    # Exchange-agnostic fallback: search Google Finance
    return f"https://www.google.com/search?q=Google+Finance+{q}"

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

# -----------------------------
# Wave discovery
# -----------------------------
def list_waves_from_weights() -> List[str]:
    df = load_csv_if_exists(WAVE_WEIGHTS_PATH)
    if df is None or df.empty:
        return []
    for col in ["Wave", "wave", "WAVE", "portfolio", "Portfolio", "name", "Name"]:
        if col in df.columns:
            return sorted({normalize_wave_name(x) for x in df[col].dropna().astype(str).tolist()})
    return sorted({normalize_wave_name(x) for x in df.iloc[:, 0].dropna().astype(str).tolist()})

def list_waves_from_logs() -> List[str]:
    waves = set()
    if os.path.isdir(LOGS_POS_DIR):
        for p in glob.glob(os.path.join(LOGS_POS_DIR, "*_positions_*.csv")):
            base = os.path.basename(p)
            wave = base.split("_positions_")[0]
            if wave:
                waves.add(normalize_wave_name(wave))
    if os.path.isdir(LOGS_PERF_DIR):
        for p in glob.glob(os.path.join(LOGS_PERF_DIR, "*_performance_*.csv")):
            base = os.path.basename(p)
            wave = base.split("_performance_")[0]
            if wave:
                waves.add(normalize_wave_name(wave))
        for p in glob.glob(os.path.join(LOGS_PERF_DIR, "*_performance_daily.csv")):
            base = os.path.basename(p)
            wave = base.split("_performance_daily.csv")[0]
            if wave:
                waves.add(normalize_wave_name(wave))
    return sorted(waves)

def discover_all_waves() -> List[str]:
    return sorted(set(list_waves_from_weights()) | set(list_waves_from_logs()))

# -----------------------------
# Load logs
# -----------------------------
def load_latest_positions(wave: str) -> Optional[pd.DataFrame]:
    patterns = [
        os.path.join(LOGS_POS_DIR, f"{wave}_positions_*.csv"),
        os.path.join(LOGS_POS_DIR, f"{wave.replace(' ', '_')}_positions_*.csv"),
        os.path.join(LOGS_POS_DIR, f"{wave.replace(' ', '')}_positions_*.csv"),
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
        os.path.join(LOGS_PERF_DIR, f"{wave}_performance_daily.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave}_performance_*.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave.replace(' ', '_')}_performance_*.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave.replace(' ', '')}_performance_*.csv"),
    ]
    for pat in patterns:
        latest = find_latest_file(pat)
        if latest and os.path.exists(latest):
            try:
                return pd.read_csv(latest)
            except Exception:
                return None
    return None

# -----------------------------
# Snapshot model
# -----------------------------
@dataclass
class WaveSnapshot:
    wave: str
    tag: str = "HYBRID"
    updated: str = "—"
    intraday_return: Optional[float] = None
    intraday_alpha: Optional[float] = None
    r30: Optional[float] = None
    a30: Optional[float] = None
    r60: Optional[float] = None
    a60: Optional[float] = None
    r1y: Optional[float] = None
    a1y: Optional[float] = None

def build_snapshot_from_perf(wave: str, perf: Optional[pd.DataFrame]) -> WaveSnapshot:
    s = WaveSnapshot(wave=wave)
    if perf is None or perf.empty:
        return s

    dt_col = _pick_col(perf, ["timestamp", "datetime", "date", "asof", "time"])
    if dt_col:
        try:
            s.updated = str(perf[dt_col].iloc[-1])
        except Exception:
            s.updated = "latest row"
    else:
        s.updated = "latest row"

    tag_col = _pick_col(perf, ["regime", "data_regime", "tag", "mode_tag", "live_sandbox"])
    if tag_col:
        t = str(perf[tag_col].iloc[-1]).upper()
        if "LIVE" in t:
            s.tag = "LIVE"
        elif "SANDBOX" in t or "BACKTEST" in t:
            s.tag = "SANDBOX"
        else:
            s.tag = "HYBRID"

    row = perf.iloc[-1]

    def get_last(colnames):
        c = _pick_col(perf, colnames)
        return _safe_float(row.get(c)) if c else None

    s.intraday_return = get_last(["intraday_return", "return_1d", "ret_1d", "day_return"])
    s.intraday_alpha  = get_last(["intraday_alpha", "alpha_1d", "alpha_capture_1d", "alpha_day"])

    s.r30 = get_last(["return_30d", "ret_30d", "r30", "return30"])
    s.a30 = get_last(["alpha_30d", "a30", "alpha_capture_30d", "alpha30"])

    s.r60 = get_last(["return_60d", "ret_60d", "r60", "return60"])
    s.a60 = get_last(["alpha_60d", "a60", "alpha_capture_60d", "alpha60"])

    s.r1y = get_last(["return_1y", "return_1yr", "ret_1y", "ret_1yr", "return_252d"])
    s.a1y = get_last(["alpha_1y", "alpha_1yr", "a1y", "alpha_capture_1y", "alpha_capture_252d"])

    return s

def apply_sandbox_override(snaps: Dict[str, WaveSnapshot]) -> Dict[str, WaveSnapshot]:
    if not os.path.exists(SANDBOX_OVERRIDE_PATH):
        return snaps
    try:
        sdf = pd.read_csv(SANDBOX_OVERRIDE_PATH)
    except Exception:
        return snaps

    wave_col = None
    for c in ["Wave", "wave", "name", "Name", "portfolio", "Portfolio"]:
        if c in sdf.columns:
            wave_col = c
            break
    if not wave_col:
        return snaps

    for _, r in sdf.iterrows():
        w = normalize_wave_name(str(r[wave_col]))
        if w not in snaps:
            snaps[w] = WaveSnapshot(wave=w)
        s = snaps[w]
        s.tag = "SANDBOX"
        s.updated = "sandbox override"

        def set_if_present(attr, cols):
            for c in cols:
                if c in sdf.columns:
                    v = _safe_float(r.get(c))
                    if v is not None:
                        setattr(s, attr, v)
                    return

        set_if_present("intraday_return", ["intraday_return", "return_1d", "ret_1d"])
        set_if_present("intraday_alpha",  ["intraday_alpha", "alpha_1d", "alpha_capture_1d"])
        set_if_present("r30", ["return_30d", "ret_30d"])
        set_if_present("a30", ["alpha_30d", "alpha_capture_30d"])
        set_if_present("r60", ["return_60d", "ret_60d"])
        set_if_present("a60", ["alpha_60d", "alpha_capture_60d"])
        set_if_present("r1y", ["return_1y", "ret_1y", "return_1yr"])
        set_if_present("a1y", ["alpha_1y", "alpha_capture_1y", "alpha_1yr"])

    return snaps

def top10_from_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker", "Weight", "Value", "Link"])

    tcol = _pick_col(df, ["ticker", "symbol"]) or df.columns[0]
    wcol = _pick_col(df, ["weight", "alloc", "allocation"])
    vcol = _pick_col(df, ["value", "market_value", "dollar_value"])

    out = df.copy()
    out[tcol] = out[tcol].astype(str).str.upper().str.strip()

    if wcol:
        out[wcol] = pd.to_numeric(out[wcol], errors="coerce")
        out = out.sort_values(wcol, ascending=False)
    elif vcol:
        out[vcol] = pd.to_numeric(out[vcol], errors="coerce")
        out = out.sort_values(vcol, ascending=False)

    out = out.head(10).copy()
    out["Link"] = out[tcol].apply(google_quote_link)

    return pd.DataFrame({
        "Ticker": out[tcol].values,
        "Weight": out[wcol].values if wcol else [None]*len(out),
        "Value": out[vcol].values if vcol else [None]*len(out),
        "Link": out["Link"].values
    })

# -----------------------------
# Correlation matrix (from daily returns)
# -----------------------------
def load_daily_return_series(wave: str) -> Optional[pd.Series]:
    perf = load_latest_performance(wave)
    if perf is None or perf.empty:
        return None
    date_col = _pick_col(perf, ["date", "datetime", "timestamp", "asof"])
    ret_col  = _pick_col(perf, ["daily_return", "return", "ret", "return_1d", "ret_1d"])
    if not ret_col:
        return None

    df = perf.copy()
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception:
            pass

    s = pd.to_numeric(df[ret_col], errors="coerce")
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        s.index = df[date_col]
    s = s.dropna()
    if s.empty:
        return None
    return s

def compute_corr_matrix(waves: List[str]) -> pd.DataFrame:
    series = {}
    for w in waves:
        s = load_daily_return_series(w)
        if s is not None and len(s) >= 10:
            series[w] = s
    if len(series) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(series).dropna(how="any")
    if df.shape[0] < 10:
        return pd.DataFrame()
    return df.corr()

# -----------------------------
# WaveScore (optional)
# -----------------------------
def load_latest_wavescore_table() -> Optional[pd.DataFrame]:
    if not os.path.isdir(LOGS_WAVESCORE_DIR):
        return None
    latest = find_latest_file(os.path.join(LOGS_WAVESCORE_DIR, "*.csv"))
    if not latest:
        return None
    try:
        df = pd.read_csv(latest)
        return df
    except Exception:
        return None

# -----------------------------
# Multi-Wave constructor (blend)
# -----------------------------
def blend_portfolio(snaps: Dict[str, WaveSnapshot], weights: Dict[str, float]) -> Dict[str, Optional[float]]:
    # Normalize
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        return {"intraday_return": None, "intraday_alpha": None, "r30": None, "a30": None, "r60": None, "a60": None, "r1y": None, "a1y": None}
    wnorm = {k: max(0.0, float(v))/total for k, v in weights.items()}

    def wavg(attr):
        vals = []
        for w, ww in wnorm.items():
            s = snaps.get(w)
            if not s:
                continue
            x = getattr(s, attr, None)
            if x is None:
                continue
            vals.append(ww * x)
        return sum(vals) if vals else None

    return {
        "intraday_return": wavg("intraday_return"),
        "intraday_alpha":  wavg("intraday_alpha"),
        "r30": wavg("r30"),
        "a30": wavg("a30"),
        "r60": wavg("r60"),
        "a60": wavg("a60"),
        "r1y": wavg("r1y"),
        "a1y": wavg("a1y"),
    }

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.caption(f"Loaded: {_ts_now()}")
    st.write("Weights:", "✅" if os.path.exists(WAVE_WEIGHTS_PATH) else "❌", "wave_weights.csv")
    st.write("Universe:", "✅" if os.path.exists(UNIVERSE_LIST_PATH) else "❌", "list.csv")
    st.write("Positions:", "✅" if os.path.isdir(LOGS_POS_DIR) else "❌", "logs/positions")
    st.write("Performance:", "✅" if os.path.isdir(LOGS_PERF_DIR) else "❌", "logs/performance")

waves = discover_all_waves()
if not waves:
    st.error("No Waves found. Make sure wave_weights.csv exists and/or logs/ have been generated by the engine.")
    st.stop()

# snapshots
snaps: Dict[str, WaveSnapshot] = {}
for w in waves:
    snaps[w] = build_snapshot_from_perf(w, load_latest_performance(w))
snaps = apply_sandbox_override(snaps)

tabs = st.tabs([
    "Overview",
    "Wave Detail",
    "WaveScore Leaderboard",
    "Correlation Matrix",
    "Multi-Wave Constructor",
    "Diagnostics",
])

# -------- Overview
with tabs[0]:
    st.subheader("All Waves — Returns & Alpha Capture")

    rows = []
    for w in waves:
        s = snaps[w]
        rows.append({
            "Wave": s.wave,
            "Tag": s.tag,
            "Updated": s.updated,
            "Intraday Return": s.intraday_return,
            "Intraday Alpha": s.intraday_alpha,
            "30D Return": s.r30,
            "30D Alpha": s.a30,
            "60D Return": s.r60,
            "60D Alpha": s.a60,
            "1Y Return": s.r1y,
            "1Y Alpha": s.a1y,
        })
    df = pd.DataFrame(rows)

    # sort by 30D alpha if present
    if "30D Alpha" in df.columns:
        df = df.sort_values("30D Alpha", ascending=False, na_position="last")

    view = df.copy()
    for c in view.columns:
        if "Return" in c or "Alpha" in c:
            view[c] = view[c].apply(lambda x: _pct(_safe_float(x)))
    st.dataframe(view, use_container_width=True, hide_index=True)

# -------- Wave Detail
with tabs[1]:
    st.subheader("Wave Detail")
    w = st.selectbox("Select a Wave", waves, index=0)
    s = snaps[w]

    c1, c2, c3 = st.columns([1.1, 1.1, 1.8])
    with c1:
        st.markdown(f"**Wave:** {s.wave}")
        st.markdown(f"**Tag:** {s.tag}")
        st.markdown(f"**Updated:** {s.updated}")

    with c2:
        st.metric("Intraday Return", _pct(s.intraday_return), delta=_pct(s.intraday_alpha))
        st.metric("30D Return", _pct(s.r30), delta=_pct(s.a30))
        st.metric("60D Return", _pct(s.r60), delta=_pct(s.a60))
        st.metric("1Y Return", _pct(s.r1y), delta=_pct(s.a1y))
        st.caption("Deltas show Alpha (best-effort from logs).")

    with c3:
        st.markdown("**Top 10 Holdings (latest positions log)**")
        pos = load_latest_positions(w)
        if pos is None or pos.empty:
            st.info("No positions file found yet for this Wave. Run engine to generate logs/positions.")
        else:
            top10 = top10_from_positions(pos)
            # render ticker as link
            top10_render = top10.copy()
            top10_render["Ticker"] = top10_render.apply(lambda r: f"[{r['Ticker']}]({r['Link']})", axis=1)
            if "Weight" in top10_render.columns:
                top10_render["Weight"] = top10_render["Weight"].apply(lambda x: _pct(_safe_float(x)) if _safe_float(x) is not None else "—")
            if "Value" in top10_render.columns:
                top10_render["Value"] = top10_render["Value"].apply(lambda x: f"${_safe_float(x):,.0f}" if _safe_float(x) is not None else "—")
            top10_render = top10_render.drop(columns=["Link"])
            st.markdown(top10_render.to_markdown(index=False), unsafe_allow_html=True)

# -------- WaveScore
with tabs[2]:
    st.subheader("WAVESCORE™ Leaderboard")
    wdf = load_latest_wavescore_table()
    if wdf is None or wdf.empty:
        st.info("No WaveScore table found yet at logs/wavescore/*.csv. If your engine writes it, it will show here automatically.")
        st.caption("Expected columns (flexible): Wave, WaveScore, Grade, ReturnQuality, RiskControl, Consistency, Resilience, Efficiency, Governance")
    else:
        # Try to sort by wavescore
        score_col = _pick_col(wdf, ["wavescore", "WaveScore", "score", "Score"])
        if score_col:
            wdf = wdf.sort_values(score_col, ascending=False)
        st.dataframe(wdf, use_container_width=True, hide_index=True)

# -------- Correlation
with tabs[3]:
    st.subheader("Correlation Matrix (Daily Returns)")
    corr = compute_corr_matrix(waves)
    if corr.empty:
        st.info("Not enough overlapping daily return history in logs/performance to compute correlation yet.")
        st.caption("This tab populates once multiple waves have daily return series logs with overlapping dates.")
    else:
        st.dataframe(corr.round(3), use_container_width=True)

# -------- Constructor
with tabs[4]:
    st.subheader("Multi-Wave Portfolio Constructor")
    st.caption("Select Waves and assign weights to see a blended (weighted) return + alpha snapshot.")

    selected = st.multiselect("Choose Waves", waves, default=waves[:3])
    if not selected:
        st.warning("Select at least one Wave.")
    else:
        weights = {}
        for w in selected:
            weights[w] = st.number_input(f"Weight for {w}", min_value=0.0, max_value=100.0, value=1.0, step=0.5)

        blended = blend_portfolio(snaps, weights)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Intraday Return", _pct(blended["intraday_return"]), delta=_pct(blended["intraday_alpha"]))
        with c2:
            st.metric("30D Return", _pct(blended["r30"]), delta=_pct(blended["a30"]))
        with c3:
            st.metric("60D Return", _pct(blended["r60"]), delta=_pct(blended["a60"]))
        with c4:
            st.metric("1Y Return", _pct(blended["r1y"]), delta=_pct(blended["a1y"]))
        st.caption("Deltas show blended alpha (best-effort).")

# -------- Diagnostics
with tabs[5]:
    st.subheader("Diagnostics")
    st.write("Discovered waves:", waves)
    st.write("Performance files:", len(glob.glob(os.path.join(LOGS_PERF_DIR, '*'))))
    st.write("Positions files:", len(glob.glob(os.path.join(LOGS_POS_DIR, '*'))))
    st.write("Sandbox override present:", "✅" if os.path.exists(SANDBOX_OVERRIDE_PATH) else "❌", SANDBOX_OVERRIDE_PATH)
    st.write("WaveScore dir present:", "✅" if os.path.isdir(LOGS_WAVESCORE_DIR) else "❌", LOGS_WAVESCORE_DIR)