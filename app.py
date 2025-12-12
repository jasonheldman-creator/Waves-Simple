# app.py — WAVES Intelligence™ Console (RESTORE + NO-DATA FIX)
# Shows file counts on Overview + adds "Run Engine Now" + prints engine output.
# Goal: eliminate "restored but no data" by ensuring logs are generated and readable.

import os, sys, glob, math, subprocess, urllib.parse
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

WAVE_WEIGHTS_PATH = os.path.join(BASE_DIR, "wave_weights.csv")

def _ts_now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _safe_float(x) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, float) and math.isnan(x): return None
        if isinstance(x, str):
            s = x.strip()
            if s in ["—", "-", ""]: return None
            if s.endswith("%"):
                return float(s.replace("%","").strip())/100.0
        return float(x)
    except Exception:
        return None

def _pct(x: Optional[float], digits: int = 2) -> str:
    return "—" if x is None else f"{x*100:.{digits}f}%"

def normalize_wave_name(name: str) -> str:
    return " ".join(str(name).strip().split())

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def ensure_dirs():
    os.makedirs(LOGS_POS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERF_DIR, exist_ok=True)

def list_waves_from_weights() -> List[str]:
    if not os.path.exists(WAVE_WEIGHTS_PATH):
        return []
    try:
        df = pd.read_csv(WAVE_WEIGHTS_PATH)
    except Exception:
        return []
    if df.empty:
        return []
    for col in ["Wave", "wave", "WAVE", "portfolio", "Portfolio", "name", "Name"]:
        if col in df.columns:
            return sorted({normalize_wave_name(x) for x in df[col].dropna().astype(str).tolist()})
    return sorted({normalize_wave_name(x) for x in df.iloc[:,0].dropna().astype(str).tolist()})

def list_waves_from_logs() -> List[str]:
    waves = set()
    for p in glob.glob(os.path.join(LOGS_PERF_DIR, "*")):
        b = os.path.basename(p)
        if "_performance_" in b:
            waves.add(normalize_wave_name(b.split("_performance_")[0]))
        if b.endswith("_performance_daily.csv"):
            waves.add(normalize_wave_name(b.replace("_performance_daily.csv","")))
    for p in glob.glob(os.path.join(LOGS_POS_DIR, "*_positions_*.csv")):
        b = os.path.basename(p)
        waves.add(normalize_wave_name(b.split("_positions_")[0]))
    return sorted(waves)

def discover_all_waves() -> List[str]:
    return sorted(set(list_waves_from_weights()) | set(list_waves_from_logs()))

def find_latest_file(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

def load_latest_performance_file(wave: str) -> Optional[str]:
    patterns = [
        os.path.join(LOGS_PERF_DIR, f"{wave}_performance_daily.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave}_performance_*.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave.replace(' ','_')}_performance_*.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave.replace(' ','')}_performance_*.csv"),
    ]
    for pat in patterns:
        f = find_latest_file(pat)
        if f and os.path.exists(f):
            return f
    return None

def load_latest_performance(wave: str) -> Optional[pd.DataFrame]:
    f = load_latest_performance_file(wave)
    if not f: return None
    try:
        return pd.read_csv(f)
    except Exception:
        return None

@dataclass
class WaveSnapshot:
    wave: str
    updated: str = "—"
    source: str = "—"
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
    f = load_latest_performance_file(wave)
    if not f:
        return s
    s.source = os.path.relpath(f, BASE_DIR)
    df = load_latest_performance(wave)
    if df is None or df.empty:
        return s

    dt_col = _pick_col(df, ["timestamp","datetime","date","asof","time"])
    if dt_col:
        try: s.updated = str(df[dt_col].iloc[-1])
        except Exception: s.updated = "latest row"
    else:
        s.updated = "latest row"

    row = df.iloc[-1]

    def get(cols):
        c = _pick_col(df, cols)
        return _safe_float(row.get(c)) if c else None

    s.intraday_return = get(["intraday_return","return_1d","ret_1d","daily_return","return"])
    s.intraday_alpha  = get(["intraday_alpha","alpha_1d","alpha_capture_1d","alpha","alpha_capture"])

    s.r30 = get(["return_30d","ret_30d","r30","rolling_30d_return"])
    s.a30 = get(["alpha_30d","alpha_capture_30d","a30","rolling_30d_alpha"])

    s.r60 = get(["return_60d","ret_60d","r60","rolling_60d_return"])
    s.a60 = get(["alpha_60d","alpha_capture_60d","a60","rolling_60d_alpha"])

    s.r1y = get(["return_1y","return_1yr","ret_1y","return_252d"])
    s.a1y = get(["alpha_1y","alpha_1yr","alpha_capture_1y","alpha_capture_252d"])

    return s

def run_engine() -> Dict[str, str]:
    """
    Runs waves_engine on Streamlit Cloud.
    - First tries: python waves_engine.py
    - Prints stdout/stderr so we can see missing keys / import errors.
    """
    engine_path = os.path.join(BASE_DIR, "waves_engine.py")
    if not os.path.exists(engine_path):
        return {"status":"no_engine_file", "stdout":"", "stderr":"waves_engine.py not found in repo root."}

    try:
        p = subprocess.run(
            [sys.executable, "-u", engine_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=180
        )
        return {
            "status": f"rc={p.returncode}",
            "stdout": (p.stdout or "")[-8000:],
            "stderr": (p.stderr or "")[-8000:],
        }
    except Exception as e:
        return {"status":"subprocess_error", "stdout":"", "stderr":repr(e)}

# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
ensure_dirs()

perf_files = sorted(glob.glob(os.path.join(LOGS_PERF_DIR, "*.csv")))
pos_files  = sorted(glob.glob(os.path.join(LOGS_POS_DIR, "*.csv")))

with st.sidebar:
    st.header("Controls")
    st.caption(f"Loaded: {_ts_now()}")
    st.write("Perf files:", len(perf_files))
    st.write("Pos files:", len(pos_files))
    st.write("wave_weights.csv:", "✅" if os.path.exists(WAVE_WEIGHTS_PATH) else "❌")

    if st.button("▶️ Run Engine Now (generate logs)"):
        with st.spinner("Running waves_engine.py…"):
            res = run_engine()
        st.success(f"Engine finished ({res['status']}).")
        if res["stdout"]:
            st.subheader("STDOUT")
            st.code(res["stdout"])
        if res["stderr"]:
            st.subheader("STDERR")
            st.code(res["stderr"])
        st.rerun()

tabs = st.tabs(["Overview", "Diagnostics"])

waves = discover_all_waves()

with tabs[0]:
    st.subheader("All Waves — Returns & Alpha Capture")

    st.caption(f"Perf files found: {len(perf_files)} | Pos files found: {len(pos_files)}")

    if len(perf_files) == 0:
        st.error(
            "No performance logs found in logs/performance.\n\n"
            "Tap ▶️ Run Engine Now in the sidebar.\n"
            "If it errors, open Diagnostics to see the error/output."
        )

    if not waves:
        st.warning("No waves discovered yet. Make sure wave_weights.csv exists OR logs have been generated by the engine.")
        st.stop()

    snaps: Dict[str, WaveSnapshot] = {w: build_snapshot(w) for w in waves}

    rows = []
    for w in waves:
        s = snaps[w]
        rows.append({
            "Wave": s.wave,
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
    if "30D Alpha" in df.columns:
        df = df.sort_values("30D Alpha", ascending=False, na_position="last")

    view = df.copy()
    for c in view.columns:
        if "Return" in c or "Alpha" in c:
            view[c] = view[c].apply(lambda x: _pct(_safe_float(x)))
    st.dataframe(view, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Diagnostics")

    st.write("Repo root:", BASE_DIR)
    st.write("logs/performance:", LOGS_PERF_DIR)
    st.write("logs/positions:", LOGS_POS_DIR)

    st.markdown("### Latest perf files (up to 15)")
    if perf_files:
        for p in perf_files[-15:]:
            st.code(os.path.relpath(p, BASE_DIR))
    else:
        st.info("None found.")

    st.markdown("### Inspect a perf file (shows columns + last rows)")
    if waves:
        pick = st.selectbox("Pick wave", waves, index=0)
        f = load_latest_performance_file(pick)
        if not f:
            st.warning("No perf file for this wave.")
        else:
            st.code(os.path.relpath(f, BASE_DIR))
            pdf = pd.read_csv(f)
            st.write("Columns:", list(pdf.columns))
            st.dataframe(pdf.tail(15), use_container_width=True)