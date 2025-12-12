# app.py — WAVES Intelligence™ Institutional Console (FULL + Engine Trigger)
# Fixes: "Restored but no data" by adding:
#  - Auto-detection of missing logs
#  - "Run Engine Now" button to generate logs on Streamlit Cloud
#  - Better perf-column detection + clearer diagnostics
#
# SAFE: reads files; only writes if your engine writes logs (normal behavior)

import os
import sys
import glob
import math
import time
import subprocess
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

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


# ---------------------------
# Helpers
# ---------------------------

def _ts_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        # strings like "12.3%" -> 0.123
        if isinstance(x, str):
            s = x.strip()
            if s.endswith("%"):
                return float(s.replace("%", "").strip()) / 100.0
        return float(x)
    except Exception:
        return None

def _pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x*100:.{digits}f}%"

def normalize_wave_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    return " ".join(name.strip().split())

def find_latest_file(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def load_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def google_quote_link(ticker: str) -> str:
    q = urllib.parse.quote(str(ticker).strip().upper())
    return f"https://www.google.com/search?q=Google+Finance+{q}"

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def ensure_dirs():
    os.makedirs(LOGS_POS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERF_DIR, exist_ok=True)
    os.makedirs(LOGS_WAVESCORE_DIR, exist_ok=True)
    os.makedirs(LOGS_SANDBOX_DIR, exist_ok=True)

# ---------------------------
# Engine trigger (critical)
# ---------------------------

def run_engine() -> Dict[str, str]:
    """
    Best-effort engine runner:
      1) import waves_engine and call main()/run()/run_engine() if present
      2) fallback: subprocess python waves_engine.py
    Returns dict with status, stdout, stderr.
    """
    status = "unknown"
    out = ""
    err = ""

    # Try import-based execution
    try:
        import importlib
        we = importlib.import_module("waves_engine")

        # common entrypoints
        for fn_name in ["main", "run_engine", "run", "build_all", "update_all"]:
            if hasattr(we, fn_name) and callable(getattr(we, fn_name)):
                status = f"import:{fn_name}"
                try:
                    res = getattr(we, fn_name)()
                    out = f"waves_engine.{fn_name}() executed. Return={res}"
                    return {"status": status, "stdout": out, "stderr": err}
                except Exception as e:
                    status = f"import:{fn_name}:error"
                    err = repr(e)
                    # keep trying other entrypoints
        # If imported but no known function, fall through to subprocess
    except Exception as e:
        err = f"Import error: {repr(e)}"

    # Subprocess fallback
    engine_path = os.path.join(BASE_DIR, "waves_engine.py")
    if os.path.exists(engine_path):
        try:
            status = "subprocess:waves_engine.py"
            p = subprocess.run(
                [sys.executable, engine_path],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=180,
            )
            out = p.stdout[-6000:] if p.stdout else ""
            err2 = p.stderr[-6000:] if p.stderr else ""
            err = (err + "\n" + err2).strip()
            if p.returncode == 0:
                return {"status": status, "stdout": out, "stderr": err}
            else:
                return {"status": f"{status}:rc={p.returncode}", "stdout": out, "stderr": err}
        except Exception as e:
            return {"status": "subprocess:error", "stdout": out, "stderr": (err + "\n" + repr(e)).strip()}

    return {"status": "no_engine_found", "stdout": out, "stderr": err}


# ---------------------------
# Wave discovery
# ---------------------------

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

# ---------------------------
# Load latest logs
# ---------------------------

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

def load_latest_performance_file(wave: str) -> Optional[str]:
    patterns = [
        os.path.join(LOGS_PERF_DIR, f"{wave}_performance_daily.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave}_performance_*.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave.replace(' ', '_')}_performance_*.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave.replace(' ', '')}_performance_*.csv"),
    ]
    for pat in patterns:
        latest = find_latest_file(pat)
        if latest and os.path.exists(latest):
            return latest
    return None

def load_latest_performance(wave: str) -> Optional[pd.DataFrame]:
    f = load_latest_performance_file(wave)
    if not f:
        return None
    try:
        return pd.read_csv(f)
    except Exception:
        return None

# ---------------------------
# Snapshots
# ---------------------------

@dataclass
class WaveSnapshot:
    wave: str
    tag: str = "HYBRID"
    updated: str = "—"
    source_file: str = "—"
    intraday_return: Optional[float] = None
    intraday_alpha: Optional[float] = None
    r30: Optional[float] = None
    a30: Optional[float] = None
    r60: Optional[float] = None
    a60: Optional[float] = None
    r1y: Optional[float] = None
    a1y: Optional[float] = None

def build_snapshot(wave: str) -> WaveSnapshot:
    s = WaveSnapshot(wave=wave)

    perf_file = load_latest_performance_file(wave)
    if not perf_file:
        return s

    s.source_file = os.path.relpath(perf_file, BASE_DIR)
    perf = load_latest_performance(wave)
    if perf is None or perf.empty:
        return s

    # updated
    dt_col = _pick_col(perf, ["timestamp", "datetime", "date", "asof", "time"])
    if dt_col:
        try:
            s.updated = str(perf[dt_col].iloc[-1])
        except Exception:
            s.updated = "latest row"
    else:
        s.updated = "latest row"

    # tag
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

    def get_last(cols):
        c = _pick_col(perf, cols)
        return _safe_float(row.get(c)) if c else None

    # broaden candidates a lot (your engine has changed column names over time)
    s.intraday_return = get_last([
        "intraday_return", "return_intraday", "day_return", "return_1d", "ret_1d", "daily_return", "return"
    ])
    s.intraday_alpha = get_last([
        "intraday_alpha", "alpha_intraday", "alpha_1d", "alpha_day", "alpha_capture_1d", "alpha", "alpha_capture"
    ])

    s.r30 = get_last(["return_30d", "ret_30d", "r30", "return30", "rolling_30d_return"])
    s.a30 = get_last(["alpha_30d", "a30", "alpha30", "alpha_capture_30d", "rolling_30d_alpha", "alpha_capture_30"])

    s.r60 = get_last(["return_60d", "ret_60d", "r60", "return60", "rolling_60d_return"])
    s.a60 = get_last(["alpha_60d", "a60", "alpha60", "alpha_capture_60d", "rolling_60d_alpha", "alpha_capture_60"])

    s.r1y = get_last(["return_1y", "return_1yr", "ret_1y", "ret_1yr", "return_252d", "rolling_1y_return"])
    s.a1y = get_last(["alpha_1y", "alpha_1yr", "a1y", "alpha_capture_1y", "alpha_capture_252d", "rolling_1y_alpha"])

    return s

def apply_sandbox_override(snaps: Dict[str, WaveSnapshot]) -> None:
    if not os.path.exists(SANDBOX_OVERRIDE_PATH):
        return
    try:
        sdf = pd.read_csv(SANDBOX_OVERRIDE_PATH)
    except Exception:
        return

    wave_col = None
    for c in ["Wave", "wave", "name", "Name", "portfolio", "Portfolio"]:
        if c in sdf.columns:
            wave_col = c
            break
    if not wave_col:
        return

    for _, r in sdf.iterrows():
        w = normalize_wave_name(str(r[wave_col]))
        if w not in snaps:
            snaps[w] = WaveSnapshot(wave=w)
        s = snaps[w]
        s.tag = "SANDBOX"
        s.updated = "sandbox override"
        s.source_file = os.path.relpath(SANDBOX_OVERRIDE_PATH, BASE_DIR)

        def set_if_present(attr, cols):
            for c in cols:
                if c in sdf.columns:
                    v = _safe_float(r.get(c))
                    if v is not None:
                        setattr(s, attr, v)
                    return

        set_if_present("intraday_return", ["intraday_return", "return_1d", "ret_1d"])
        set_if_present("intraday_alpha", ["intraday_alpha", "alpha_1d", "alpha_capture_1d"])
        set_if_present("r30", ["return_30d", "ret_30d"])
        set_if_present("a30", ["alpha_30d", "alpha_capture_30d"])
        set_if_present("r60", ["return_60d", "ret_60d"])
        set_if_present("a60", ["alpha_60d", "alpha_capture_60d"])
        set_if_present("r1y", ["return_1y", "ret_1y", "return_1yr"])
        set_if_present("a1y", ["alpha_1y", "alpha_capture_1y", "alpha_1yr"])

def top10_from_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Ticker", "Weight", "Value", "Link"])

    tcol = _pick_col(df, ["ticker", "Ticker", "symbol", "Symbol"]) or df.columns[0]
    wcol = _pick_col(df, ["weight", "Weight", "alloc", "allocation"])
    vcol = _pick_col(df, ["value", "Value", "market_value", "dollar_value"])

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


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

ensure_dirs()

with st.sidebar:
    st.header("Controls")
    st.caption(f"Loaded: {_ts_now()}")

    # Key: run engine button
    if st.button("▶️ Run Engine Now (Generate Logs)"):
        with st.spinner("Running waves_engine..."):
            res = run_engine()
        st.success(f"Engine run finished: {res['status']}")
        if res.get("stdout"):
            st.code(res["stdout"])
        if res.get("stderr"):
            st.code(res["stderr"])
        st.rerun()

    st.divider()
    st.subheader("Data status")
    st.write("wave_weights.csv:", "✅" if os.path.exists(WAVE_WEIGHTS_PATH) else "❌")
    st.write("list.csv:", "✅" if os.path.exists(UNIVERSE_LIST_PATH) else "❌")
    st.write("logs/positions:", "✅" if len(glob.glob(os.path.join(LOGS_POS_DIR, "*.csv"))) else "❌")
    st.write("logs/performance:", "✅" if len(glob.glob(os.path.join(LOGS_PERF_DIR, "*.csv"))) else "❌")
    st.write("sandbox_summary.csv:", "✅" if os.path.exists(SANDBOX_OVERRIDE_PATH) else "❌")

# Auto-run engine ONCE per session if logs are missing
if "engine_autorun_done" not in st.session_state:
    st.session_state.engine_autorun_done = False

perf_files = glob.glob(os.path.join(LOGS_PERF_DIR, "*.csv"))
pos_files = glob.glob(os.path.join(LOGS_POS_DIR, "*.csv"))

if (not perf_files) and (not st.session_state.engine_autorun_done):
    st.session_state.engine_autorun_done = True
    st.warning("No performance logs detected. Auto-running engine once to generate logs...")
    with st.spinner("Auto-running waves_engine..."):
        res = run_engine()
    st.info(f"Auto-run status: {res['status']}")
    if res.get("stderr"):
        st.code(res["stderr"])
    st.rerun()

waves = discover_all_waves()
if not waves:
    st.error(
        "No Waves discovered.\n\n"
        "Fix checklist:\n"
        "• Ensure wave_weights.csv exists and has a Wave column\n"
        "• Or run the engine so logs/performance and logs/positions are generated\n"
    )
    st.stop()

# Build snapshots
snaps: Dict[str, WaveSnapshot] = {w: build_snapshot(w) for w in waves}
apply_sandbox_override(snaps)

tabs = st.tabs(["Overview", "Wave Detail", "Diagnostics"])

# ---- Overview
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

    # Sort by 30D alpha if present
    if "30D Alpha" in df.columns:
        df = df.sort_values("30D Alpha", ascending=False, na_position="last")

    view = df.copy()
    for c in view.columns:
        if "Return" in c or "Alpha" in c:
            view[c] = view[c].apply(lambda x: _pct(_safe_float(x)))

    st.dataframe(view, use_container_width=True, hide_index=True)

    # quick hint if everything is blank
    all_blank = True
    for w in waves:
        s = snaps[w]
        if any(v is not None for v in [s.intraday_return, s.intraday_alpha, s.r30, s.a30, s.r60, s.a60, s.r1y, s.a1y]):
            all_blank = False
            break
    if all_blank:
        st.warning(
            "Waves were discovered, but no return/alpha values were found in the latest performance logs.\n"
            "Use Diagnostics tab to see which files are being read (and column names)."
        )

# ---- Wave Detail
with tabs[1]:
    st.subheader("Wave Detail")
    w = st.selectbox("Select a Wave", waves, index=0)
    s = snaps[w]

    c1, c2 = st.columns([1.2, 2.0])
    with c1:
        st.markdown(f"**Wave:** {s.wave}")
        st.markdown(f"**Tag:** {s.tag}")
        st.markdown(f"**Updated:** {s.updated}")
        st.markdown(f"**Source:** `{s.source_file}`")

        st.metric("Intraday Return", _pct(s.intraday_return), delta=_pct(s.intraday_alpha))
        st.metric("30D Return", _pct(s.r30), delta=_pct(s.a30))
        st.metric("60D Return", _pct(s.r60), delta=_pct(s.a60))
        st.metric("1Y Return", _pct(s.r1y), delta=_pct(s.a1y))
        st.caption("Deltas show Alpha (best-effort from logs).")

    with c2:
        st.markdown("**Top 10 Holdings (latest positions log)**")
        pos = load_latest_positions(w)
        if pos is None or pos.empty:
            st.info("No positions file found yet for this Wave. Run Engine Now to generate logs/positions.")
        else:
            top10 = top10_from_positions(pos)
            top10_render = top10.copy()
            top10_render["Ticker"] = top10_render.apply(lambda r: f"[{r['Ticker']}]({r['Link']})", axis=1)
            if "Weight" in top10_render.columns:
                top10_render["Weight"] = top10_render["Weight"].apply(lambda x: _pct(_safe_float(x)) if _safe_float(x) is not None else "—")
            if "Value" in top10_render.columns:
                top10_render["Value"] = top10_render["Value"].apply(lambda x: f"${_safe_float(x):,.0f}" if _safe_float(x) is not None else "—")
            top10_render = top10_render.drop(columns=["Link"])
            st.markdown(top10_render.to_markdown(index=False), unsafe_allow_html=True)

# ---- Diagnostics
with tabs[2]:
    st.subheader("Diagnostics")

    st.markdown("### Files Found")
    st.write("wave_weights.csv:", "✅" if os.path.exists(WAVE_WEIGHTS_PATH) else "❌", WAVE_WEIGHTS_PATH)
    st.write("list.csv:", "✅" if os.path.exists(UNIVERSE_LIST_PATH) else "❌", UNIVERSE_LIST_PATH)

    perf_list = sorted(glob.glob(os.path.join(LOGS_PERF_DIR, "*.csv")))
    pos_list = sorted(glob.glob(os.path.join(LOGS_POS_DIR, "*.csv")))
    st.write(f"Performance logs: {len(perf_list)}")
    st.write(f"Positions logs: {len(pos_list)}")

    if perf_list:
        st.markdown("**Latest performance files (top 10):**")
        for p in perf_list[-10:]:
            st.code(os.path.relpath(p, BASE_DIR))

    if pos_list:
        st.markdown("**Latest positions files (top 10):**")
        for p in pos_list[-10:]:
            st.code(os.path.relpath(p, BASE_DIR))

    st.markdown("### Wave Snapshots (source file + columns)")
    for w in waves[:25]:
        s = snaps[w]
        st.write(f"• **{w}** → source: `{s.source_file}`")

    st.markdown("### Inspect One Performance File")
    w_pick = st.selectbox("Pick a wave to inspect perf columns", waves, index=0, key="diag_wave")
    f = load_latest_performance_file(w_pick)
    if not f:
        st.info("No perf file found for this wave yet.")
    else:
        st.code(os.path.relpath(f, BASE_DIR))
        pdf = load_latest_performance(w_pick)
        if pdf is None or pdf.empty:
            st.warning("Perf file exists but could not be read or is empty.")
        else:
            st.write("Columns:", list(pdf.columns))
            st.dataframe(pdf.tail(10), use_container_width=True)