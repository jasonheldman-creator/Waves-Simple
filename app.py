# app.py — WAVES Intelligence™ Console (AUTO-RUN + NO-DATA GUARDRAILS)
# - Auto-runs engine on login (once per session) if logs missing or stale
# - Detects placeholder tickers (REPLACE_) and stops (prevents blank dashboards)
# - Shows perf/pos file counts + latest log times

import os, sys, glob, time, subprocess
from datetime import datetime, timezone
from typing import Optional, Dict, List

import pandas as pd
import streamlit as st

APP_TITLE = "WAVES Intelligence™ — Institutional Console"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOGS_POS_DIR = os.path.join(LOGS_DIR, "positions")
LOGS_PERF_DIR = os.path.join(LOGS_DIR, "performance")

WAVE_WEIGHTS_PATH = os.path.join(BASE_DIR, "wave_weights.csv")

# How “fresh” perf logs must be (seconds). If older, we auto-run engine.
PERF_STALE_SECONDS = 15 * 60  # 15 minutes

def ts_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dirs():
    os.makedirs(LOGS_POS_DIR, exist_ok=True)
    os.makedirs(LOGS_PERF_DIR, exist_ok=True)

def list_files(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern))

def latest_mtime(files: List[str]) -> Optional[float]:
    if not files:
        return None
    try:
        return max(os.path.getmtime(f) for f in files)
    except Exception:
        return None

def run_engine_subprocess() -> Dict[str, str]:
    """
    Runs waves_engine.py in Streamlit Cloud.
    Returns {status, stdout, stderr}
    """
    engine_path = os.path.join(BASE_DIR, "waves_engine.py")
    if not os.path.exists(engine_path):
        return {"status": "no_engine_file", "stdout": "", "stderr": "waves_engine.py not found in repo root."}

    try:
        p = subprocess.run(
            [sys.executable, "-u", engine_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=240,
        )
        return {
            "status": f"rc={p.returncode}",
            "stdout": (p.stdout or "")[-12000:],
            "stderr": (p.stderr or "")[-12000:],
        }
    except Exception as e:
        return {"status": "subprocess_error", "stdout": "", "stderr": repr(e)}

def detect_placeholder_tickers() -> Optional[str]:
    """
    If wave_weights.csv contains REPLACE_ tickers, we stop and instruct to restore real tickers.
    """
    if not os.path.exists(WAVE_WEIGHTS_PATH):
        return "wave_weights.csv is missing."

    try:
        df = pd.read_csv(WAVE_WEIGHTS_PATH)
    except Exception:
        return "wave_weights.csv exists but could not be read."

    # Try to find the ticker column
    ticker_col = None
    for c in ["Ticker", "ticker", "Symbol", "symbol"]:
        if c in df.columns:
            ticker_col = c
            break
    if ticker_col is None:
        return "wave_weights.csv does not have a Ticker/Symbol column."

    tickers = df[ticker_col].astype(str).str.upper().str.strip()
    if tickers.str.contains("REPLACE_").any():
        return (
            "Your wave_weights.csv contains placeholder tickers like REPLACE_01. "
            "That guarantees blank returns/alpha because prices cannot be fetched.\n\n"
            "Fix: restore your real wave_weights.csv (from GitHub History or your saved phone copy), "
            "then run again."
        )
    return None

def pick_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    m = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand.lower() in m:
            return m[cand.lower()]
    return None

def discover_waves_from_weights() -> List[str]:
    if not os.path.exists(WAVE_WEIGHTS_PATH):
        return []
    try:
        df = pd.read_csv(WAVE_WEIGHTS_PATH)
    except Exception:
        return []
    wave_col = pick_col(df, ["Wave", "Portfolio", "Name"])
    if not wave_col:
        # assume first column
        wave_col = df.columns[0]
    return sorted(set(df[wave_col].dropna().astype(str).str.strip().tolist()))

def load_latest_perf_row_for_wave(wave: str) -> Optional[dict]:
    # Look for <Wave>_performance_daily.csv (and variations)
    patterns = [
        os.path.join(LOGS_PERF_DIR, f"{wave}_performance_daily.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave.replace(' ', '_')}_performance_daily.csv"),
        os.path.join(LOGS_PERF_DIR, f"{wave.replace(' ', '')}_performance_daily.csv"),
    ]
    f = None
    for p in patterns:
        if os.path.exists(p):
            f = p
            break
    if not f:
        return None

    try:
        df = pd.read_csv(f)
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[-1].to_dict()
    return row

def as_pct(x) -> str:
    try:
        if x is None:
            return "—"
        # pandas may pass nan
        if isinstance(x, float) and pd.isna(x):
            return "—"
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"


# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
ensure_dirs()

# Guardrail: placeholder tickers
placeholder_msg = detect_placeholder_tickers()
if placeholder_msg:
    st.error(placeholder_msg)
    st.info(
        "Quick restore on iPhone:\n"
        "GitHub → wave_weights.csv → History → open last good version → Copy raw → paste into current → Commit."
    )
    st.stop()

# File counts
perf_files = list_files(os.path.join(LOGS_PERF_DIR, "*.csv"))
pos_files  = list_files(os.path.join(LOGS_POS_DIR, "*.csv"))

perf_mtime = latest_mtime(perf_files)
now = time.time()
perf_is_stale = (perf_mtime is None) or ((now - perf_mtime) > PERF_STALE_SECONDS)

with st.sidebar:
    st.header("Controls")
    st.caption(f"Loaded: {ts_now()}")
    st.write("Perf files:", len(perf_files))
    st.write("Pos files:", len(pos_files))
    st.write("wave_weights.csv:", "✅" if os.path.exists(WAVE_WEIGHTS_PATH) else "❌")

    if st.button("▶️ Run Engine Now"):
        with st.spinner("Running waves_engine.py…"):
            res = run_engine_subprocess()
        st.success(f"Engine finished ({res['status']}).")
        if res.get("stdout"):
            st.subheader("STDOUT")
            st.code(res["stdout"])
        if res.get("stderr"):
            st.subheader("STDERR")
            st.code(res["stderr"])
        st.rerun()

# AUTO-RUN ENGINE ON LOGIN (once per session)
if "autorun_done" not in st.session_state:
    st.session_state.autorun_done = False

if (not st.session_state.autorun_done) and perf_is_stale:
    st.session_state.autorun_done = True
    st.warning("Auto-running engine on login (logs missing or stale)…")
    with st.spinner("Running waves_engine.py…"):
        res = run_engine_subprocess()
    # Show any error immediately
    if res.get("stderr"):
        st.error("Engine reported an error. Open the sidebar to view STDERR.")
        st.code(res["stderr"])
    st.rerun()

# Refresh counts after possible autorun
perf_files = list_files(os.path.join(LOGS_PERF_DIR, "*.csv"))
pos_files  = list_files(os.path.join(LOGS_POS_DIR, "*.csv"))

st.caption(f"Perf files found: {len(perf_files)} | Pos files found: {len(pos_files)}")

waves = discover_waves_from_weights()
if not waves:
    st.error("No waves discovered from wave_weights.csv.")
    st.stop()

# Build Overview table (one row per wave, latest perf row)
rows = []
for w in waves:
    r = load_latest_perf_row_for_wave(w) or {}
    rows.append({
        "Wave": w,
        "Updated": r.get("timestamp", "—"),
        "Intraday Return": r.get("intraday_return", None),
        "Intraday Alpha":  r.get("intraday_alpha", None),
        "30D Return":      r.get("return_30d", None),
        "30D Alpha":       r.get("alpha_30d", None),
        "60D Return":      r.get("return_60d", None),
        "60D Alpha":       r.get("alpha_60d", None),
        "1Y Return":       r.get("return_1y", None),
        "1Y Alpha":        r.get("alpha_1y", None),
    })

df = pd.DataFrame(rows)

# Format %
view = df.copy()
for c in view.columns:
    if "Return" in c or "Alpha" in c:
        view[c] = view[c].apply(as_pct)

st.dataframe(view, use_container_width=True, hide_index=True)

# If everything is still blank, tell exactly why
all_blank = True
for _, r in df.iterrows():
    vals = [r.get("Intraday Return"), r.get("30D Return"), r.get("60D Return"), r.get("1Y Return")]
    if any((v is not None) and (not (isinstance(v, float) and pd.isna(v))) for v in vals):
        all_blank = False
        break

if all_blank:
    st.warning(
        "Performance files exist, but the return/alpha fields are blank.\n\n"
        "Most common causes:\n"
        "1) wave_weights.csv has invalid tickers (or you replaced it accidentally)\n"
        "2) benchmarks/tickers have no price history available via yfinance\n\n"
        "Fix: restore your real tickers in wave_weights.csv, then reload. "
        "The app auto-runs the engine on login now."
    )