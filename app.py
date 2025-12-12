# app.py — WAVES Intelligence™ Console (AUTO-RUN + ROBUST LOG MATCHING)

import os, sys, glob, time, re, subprocess
from datetime import datetime
from typing import Optional, Dict, List

import pandas as pd
import streamlit as st

APP_TITLE = "WAVES Intelligence™ — Institutional Console"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOGS_POS_DIR = os.path.join(LOGS_DIR, "positions")
LOGS_PERF_DIR = os.path.join(LOGS_DIR, "performance")

WAVE_WEIGHTS_PATH = os.path.join(BASE_DIR, "wave_weights.csv")

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
    if not os.path.exists(WAVE_WEIGHTS_PATH):
        return "wave_weights.csv is missing."

    try:
        df = pd.read_csv(WAVE_WEIGHTS_PATH)
    except Exception:
        return "wave_weights.csv exists but could not be read."

    ticker_col = None
    for c in ["ticker", "Ticker", "symbol", "Symbol"]:
        if c in df.columns:
            ticker_col = c
            break
    if ticker_col is None:
        return "wave_weights.csv does not have a ticker/symbol column."

    tickers = df[ticker_col].astype(str).str.upper().str.strip()
    if tickers.str.contains("REPLACE_").any():
        return (
            "Your wave_weights.csv contains placeholder tickers like REPLACE_01. "
            "That guarantees blank returns/alpha because prices cannot be fetched.\n\n"
            "Fix: restore real tickers in wave_weights.csv and reload."
        )
    return None


def slugify(s: str) -> str:
    """
    Make wave name match engine-style filenames:
    - lower
    - replace non-alnum with underscores
    - collapse underscores
    - strip underscores
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def discover_waves_from_weights() -> List[str]:
    if not os.path.exists(WAVE_WEIGHTS_PATH):
        return []
    try:
        df = pd.read_csv(WAVE_WEIGHTS_PATH)
    except Exception:
        return []

    # accept either wave or Wave
    wave_col = "wave" if "wave" in df.columns else ("Wave" if "Wave" in df.columns else df.columns[0])
    return sorted(set(df[wave_col].dropna().astype(str).str.strip().tolist()))


def perf_file_index() -> Dict[str, str]:
    """
    Build mapping: slug(wave_name_from_filename) -> filepath
    Filename expected like:
    <Wave>_performance_daily.csv
    """
    idx = {}
    for f in glob.glob(os.path.join(LOGS_PERF_DIR, "*_performance_daily.csv")):
        base = os.path.basename(f)
        wave_part = base.replace("_performance_daily.csv", "")
        idx[slugify(wave_part)] = f
    return idx


def load_latest_perf_row_for_wave(wave: str, idx: Dict[str, str]) -> Optional[dict]:
    f = idx.get(slugify(wave))
    if not f or not os.path.exists(f):
        return None
    try:
        df = pd.read_csv(f)
    except Exception:
        return None
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def as_pct(x) -> str:
    try:
        if x is None:
            return "—"
        if isinstance(x, float) and pd.isna(x):
            return "—"
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"


# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
ensure_dirs()

placeholder_msg = detect_placeholder_tickers()
if placeholder_msg:
    st.error(placeholder_msg)
    st.stop()

perf_files = list_files(os.path.join(LOGS_PERF_DIR, "*.csv"))
pos_files = list_files(os.path.join(LOGS_POS_DIR, "*.csv"))

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

# Auto-run once per session if stale/missing
if "autorun_done" not in st.session_state:
    st.session_state.autorun_done = False

if (not st.session_state.autorun_done) and perf_is_stale:
    st.session_state.autorun_done = True
    st.warning("Auto-running engine on login (logs missing or stale)…")
    with st.spinner("Running waves_engine.py…"):
        res = run_engine_subprocess()
    if res.get("stderr"):
        st.error("Engine reported an error (see sidebar for details).")
        st.code(res["stderr"])
    st.rerun()

# Re-scan after possible run
perf_files = list_files(os.path.join(LOGS_PERF_DIR, "*.csv"))
pos_files = list_files(os.path.join(LOGS_POS_DIR, "*.csv"))
idx = perf_file_index()

st.caption(f"Perf files found: {len(perf_files)} | Pos files found: {len(pos_files)}")

waves = discover_waves_from_weights()
if not waves:
    st.error("No waves discovered from wave_weights.csv.")
    st.stop()

rows = []
unmatched = []
for w in waves:
    r = load_latest_perf_row_for_wave(w, idx) or {}
    if not r:
        unmatched.append(w)
    rows.append({
        "Wave": w,
        "Updated": r.get("timestamp", r.get("date", r.get("asof", "—"))),
        "Intraday Return": r.get("intraday_return", r.get("intraday_ret", None)),
        "Intraday Alpha":  r.get("intraday_alpha", r.get("intraday_alpha_capture", None)),
        "30D Return":      r.get("return_30d", r.get("ret_30d", None)),
        "30D Alpha":       r.get("alpha_30d", r.get("alpha30d", None)),
        "60D Return":      r.get("return_60d", r.get("ret_60d", None)),
        "60D Alpha":       r.get("alpha_60d", r.get("alpha60d", None)),
        "1Y Return":       r.get("return_1y", r.get("ret_1y", r.get("return_1yr", None))),
        "1Y Alpha":        r.get("alpha_1y", r.get("alpha_1yr", None)),
    })

df = pd.DataFrame(rows)

view = df.copy()
for c in view.columns:
    if "Return" in c or "Alpha" in c:
        view[c] = view[c].apply(as_pct)

st.dataframe(view, use_container_width=True, hide_index=True)

if unmatched:
    st.warning(
        "Some waves have no matching performance file name yet. "
        "This is usually just filename normalization.\n\n"
        "Unmatched waves:\n- " + "\n- ".join(unmatched[:25]) +
        ("\n\n(Showing first 25)" if len(unmatched) > 25 else "")
    )

# Debug section (optional but helpful)
with st.expander("Diagnostics: performance files detected"):
    st.write("Detected perf files (mapped by slug):")
    diag = [{"slug": k, "file": os.path.basename(v)} for k, v in idx.items()]
    st.dataframe(pd.DataFrame(diag), use_container_width=True, hide_index=True)