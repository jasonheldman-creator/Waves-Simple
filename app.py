# app.py — WAVES Intelligence™ Institutional Console (Manifest + Slug Matching)
# Fixes:
# - Uses logs/performance/_wave_manifest.csv to map wave display names to slugs
# - Loads performance logs by slug (stable filenames)
# - Optional Auto-Run Engine on app load (once per session)
# - Clear diagnostics for missing logs / parsing issues

import os
import sys
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PERF_DIR = os.path.join(LOGS_DIR, "performance")
POS_DIR = os.path.join(LOGS_DIR, "positions")

ENGINE_FILE = os.path.join(BASE_DIR, "waves_engine.py")
MANIFEST_FILE = os.path.join(PERF_DIR, "_wave_manifest.csv")
WEIGHTS_FILE = os.path.join(BASE_DIR, "wave_weights.csv")


# -------------------------
# Helpers
# -------------------------
def fmt_pct(x) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"


def fmt_ts(x) -> str:
    if not x or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return str(x)


def list_perf_files() -> List[str]:
    if not os.path.exists(PERF_DIR):
        return []
    return sorted([f for f in os.listdir(PERF_DIR) if f.endswith(".csv") and not f.startswith("_")])


def list_pos_files() -> List[str]:
    if not os.path.exists(POS_DIR):
        return []
    return sorted([f for f in os.listdir(POS_DIR) if f.endswith(".csv")])


def run_engine_capture() -> Tuple[bool, str]:
    """
    Runs waves_engine.py and returns (ok, combined_output).
    Uses subprocess so it works on Streamlit Cloud.
    """
    if not os.path.exists(ENGINE_FILE):
        return False, f"Missing engine file: {ENGINE_FILE}"

    try:
        proc = subprocess.run(
            [sys.executable, ENGINE_FILE],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=240,  # 4 min safety
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        ok = proc.returncode == 0
        return ok, out.strip()
    except Exception as e:
        return False, f"Engine run failed: {repr(e)}"


def load_manifest() -> Optional[pd.DataFrame]:
    if not os.path.exists(MANIFEST_FILE):
        return None
    try:
        m = pd.read_csv(MANIFEST_FILE)
        if "wave" not in m.columns or "slug" not in m.columns:
            return None
        m["wave"] = m["wave"].astype(str)
        m["slug"] = m["slug"].astype(str)
        return m
    except Exception:
        return None


def load_latest_perf_for_slug(slug: str) -> Optional[pd.Series]:
    """
    Loads the latest row from logs/performance/<slug>_performance_daily.csv
    """
    path = os.path.join(PERF_DIR, f"{slug}_performance_daily.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        # take last row
        return df.iloc[-1]
    except Exception:
        return None


def build_overview_table(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns (table_df, missing_slugs)
    """
    rows = []
    missing = []

    for _, r in manifest.iterrows():
        wave = r["wave"]
        slug = r["slug"]

        s = load_latest_perf_for_slug(slug)
        if s is None:
            missing.append(wave)
            rows.append({
                "Wave": wave,
                "Updated": "—",
                "Intraday Return": "—",
                "Intraday Alpha": "—",
                "30D Return": "—",
                "30D Alpha": "—",
                "60D Return": "—",
                "60D Alpha": "—",
                "1Y Return": "—",
                "1Y Alpha": "—",
            })
            continue

        rows.append({
            "Wave": wave,
            "Updated": fmt_ts(s.get("timestamp")),
            "Intraday Return": fmt_pct(s.get("intraday_return")),
            "Intraday Alpha": fmt_pct(s.get("intraday_alpha")),
            "30D Return": fmt_pct(s.get("return_30d")),
            "30D Alpha": fmt_pct(s.get("alpha_30d")),
            "60D Return": fmt_pct(s.get("return_60d")),
            "60D Alpha": fmt_pct(s.get("alpha_60d")),
            "1Y Return": fmt_pct(s.get("return_1y")),
            "1Y Alpha": fmt_pct(s.get("alpha_1y")),
        })

    df = pd.DataFrame(rows)
    return df, missing


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="WAVES Intelligence™ — Institutional Console", layout="wide")
st.title("WAVES Intelligence™ — Institutional Console")
st.caption("All Waves — Returns & Alpha Capture")

tabs = st.tabs(["Overview", "Wave Detail", "Diagnostics"])

# Sidebar Controls
with st.sidebar:
    st.header("Controls")
    st.write(f"Loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    perf_files = list_perf_files()
    pos_files = list_pos_files()
    st.write(f"Perf files: **{len(perf_files)}**")
    st.write(f"Pos files: **{len(pos_files)}**")
    st.write(f"`wave_weights.csv`: {'✅' if os.path.exists(WEIGHTS_FILE) else '❌'}")
    st.write(f"`manifest`: {'✅' if os.path.exists(MANIFEST_FILE) else '❌'}")

    st.divider()

    # Auto-run option
    if "auto_run_enabled" not in st.session_state:
        st.session_state.auto_run_enabled = True

    auto_run = st.checkbox("Auto-run engine on login (once per session)", value=st.session_state.auto_run_enabled)
    st.session_state.auto_run_enabled = auto_run

    run_btn = st.button("▶ Run Engine Now (generate logs)", use_container_width=True)

    # Engine output area (sidebar)
    if "engine_out" not in st.session_state:
        st.session_state.engine_out = ""
    if "engine_ok" not in st.session_state:
        st.session_state.engine_ok = None

    # Manual run
    if run_btn:
        with st.spinner("Running waves_engine.py..."):
            ok, out = run_engine_capture()
            st.session_state.engine_ok = ok
            st.session_state.engine_out = out
        st.success("Engine completed." if st.session_state.engine_ok else "Engine errored — see Diagnostics tab.")

# Auto-run once per session
if st.session_state.get("auto_run_enabled", True) and not st.session_state.get("auto_ran_once", False):
    # Only auto-run if we have no perf files yet OR no manifest yet (first load / after reset)
    if (not os.path.exists(MANIFEST_FILE)) or (len(list_perf_files()) == 0):
        with st.spinner("Auto-running engine (first load)..."):
            ok, out = run_engine_capture()
            st.session_state.engine_ok = ok
            st.session_state.engine_out = out
        st.session_state.auto_ran_once = True
    else:
        st.session_state.auto_ran_once = True


# -------------------------
# Overview Tab
# -------------------------
with tabs[0]:
    manifest = load_manifest()
    if manifest is None:
        st.error(
            "No manifest found in logs/performance/_wave_manifest.csv.\n\n"
            "Fix: run the engine once so it generates the manifest."
        )
    else:
        overview_df, missing_waves = build_overview_table(manifest)

        # Big table
        st.dataframe(overview_df, use_container_width=True, hide_index=True)

        if missing_waves:
            st.warning(
                "Some waves have no matching performance file yet. "
                "This is usually just that the engine has not successfully generated the new slug-based logs.\n\n"
                "Missing waves:\n- " + "\n- ".join(missing_waves)
            )


# -------------------------
# Wave Detail Tab
# -------------------------
with tabs[1]:
    manifest = load_manifest()
    if manifest is None:
        st.info("Run the engine once to generate the manifest, then Wave Detail will populate.")
    else:
        wave_names = sorted(manifest["wave"].tolist())
        wave_choice = st.selectbox("Select a Wave", wave_names)

        slug = manifest.loc[manifest["wave"] == wave_choice, "slug"].iloc[0]
        perf_path = os.path.join(PERF_DIR, f"{slug}_performance_daily.csv")

        st.subheader(wave_choice)
        st.caption(f"Slug: `{slug}` | Perf file: `{os.path.basename(perf_path)}`")

        s = load_latest_perf_for_slug(slug)
        if s is None:
            st.error("No performance file found for this wave yet. Run engine.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Intraday Return", fmt_pct(s.get("intraday_return")))
            c2.metric("Intraday Alpha", fmt_pct(s.get("intraday_alpha")))
            c3.metric("30D Return", fmt_pct(s.get("return_30d")))
            c4.metric("60D Return", fmt_pct(s.get("return_60d")))
            c5.metric("1Y Return", fmt_pct(s.get("return_1y")))

            # Show history table
            try:
                hist = pd.read_csv(perf_path)
                st.write("Performance history (most recent last):")
                st.dataframe(hist.tail(60), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Could not read perf history: {repr(e)}")


# -------------------------
# Diagnostics Tab
# -------------------------
with tabs[2]:
    st.subheader("Diagnostics")

    st.write("### File checks")
    st.code(f"BASE_DIR: {BASE_DIR}")
    st.write(f"- wave_weights.csv: {'✅' if os.path.exists(WEIGHTS_FILE) else '❌'}")
    st.write(f"- waves_engine.py: {'✅' if os.path.exists(ENGINE_FILE) else '❌'}")
    st.write(f"- manifest (_wave_manifest.csv): {'✅' if os.path.exists(MANIFEST_FILE) else '❌'}")

    st.write("### Perf files (slug-based)")
    perf_files = list_perf_files()
    if perf_files:
        st.write(f"Found {len(perf_files)} perf files:")
        st.code("\n".join(perf_files[:200]))
    else:
        st.warning("No perf files found in logs/performance (excluding manifest).")

    st.write("### Engine output (last run)")
    if st.session_state.get("engine_out"):
        st.code(st.session_state.engine_out[:12000])
    else:
        st.info("No engine output captured yet. Use 'Run Engine Now' in the sidebar.")

    st.write("### Guidance")
    st.markdown(
        """
- If you see **unmatched waves**, it almost always means the console couldn't find the slug-based perf files yet.
- Run the engine once and confirm these exist:
  - `logs/performance/_wave_manifest.csv`
  - `logs/performance/<wave_slug>_performance_daily.csv`
- If returns are still blank, check engine output for ticker fetch failures.
        """
    )