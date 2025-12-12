# app.py — WAVES Intelligence™ Institutional Console (FULL FEATURE + Manifest/Slug Matching)
# Restores rich console features while keeping the slug-based log matching fix.

import os
import sys
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st


# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PERF_DIR = os.path.join(LOGS_DIR, "performance")
POS_DIR = os.path.join(LOGS_DIR, "positions")

ENGINE_FILE = os.path.join(BASE_DIR, "waves_engine.py")
MANIFEST_FILE = os.path.join(PERF_DIR, "_wave_manifest.csv")
WEIGHTS_FILE = os.path.join(BASE_DIR, "wave_weights.csv")


# -------------------------
# Formatting helpers
# -------------------------
def fmt_pct(x) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"


def fmt_num(x) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x):.4f}"
    except Exception:
        return "—"


def fmt_ts(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return str(x)


def google_finance_url(ticker: str) -> str:
    # Most US tickers work directly; crypto pairs also work on Google Finance search
    # If you want exchange-specific routing later, we can enhance this.
    t = (ticker or "").strip()
    return f"https://www.google.com/finance/quote/{t}"


# -------------------------
# Canonical normalization (must match engine)
# -------------------------
def normalize_wave_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# -------------------------
# File listing helpers
# -------------------------
def list_perf_files() -> List[str]:
    if not os.path.exists(PERF_DIR):
        return []
    return sorted([f for f in os.listdir(PERF_DIR) if f.endswith(".csv") and not f.startswith("_")])


def list_pos_files() -> List[str]:
    if not os.path.exists(POS_DIR):
        return []
    return sorted([f for f in os.listdir(POS_DIR) if f.endswith(".csv")])


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


def load_perf_history(slug: str) -> Optional[pd.DataFrame]:
    path = os.path.join(PERF_DIR, f"{slug}_performance_daily.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def load_latest_perf(slug: str) -> Optional[pd.Series]:
    df = load_perf_history(slug)
    if df is None or df.empty:
        return None
    return df.iloc[-1]


def latest_positions_file_for_slug(slug: str) -> Optional[str]:
    """
    Finds the latest positions file for a given wave slug.
    Engine writes: <slug>_positions_YYYYMMDD.csv
    """
    if not os.path.exists(POS_DIR):
        return None
    files = [f for f in os.listdir(POS_DIR) if f.startswith(slug + "_positions_") and f.endswith(".csv")]
    if not files:
        return None
    files_sorted = sorted(files)
    return os.path.join(POS_DIR, files_sorted[-1])


def load_top_holdings(slug: str, top_n: int = 10) -> Optional[pd.DataFrame]:
    fpath = latest_positions_file_for_slug(slug)
    if not fpath or not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
        # Expected cols: wave,ticker,weight,price,value_weighted (engine outputs these)
        if "ticker" not in df.columns or "weight" not in df.columns:
            return None
        df = df.copy()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df = df.dropna(subset=["ticker", "weight"])
        df = df.sort_values("weight", ascending=False).head(top_n).reset_index(drop=True)
        df["Google"] = df["ticker"].apply(lambda t: google_finance_url(str(t)))
        return df
    except Exception:
        return None


# -------------------------
# Engine runner
# -------------------------
def run_engine_capture(timeout_sec: int = 240) -> Tuple[bool, str]:
    if not os.path.exists(ENGINE_FILE):
        return False, f"Missing engine file: {ENGINE_FILE}"
    try:
        proc = subprocess.run(
            [sys.executable, ENGINE_FILE],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        ok = proc.returncode == 0
        return ok, out.strip()
    except Exception as e:
        return False, f"Engine run failed: {repr(e)}"


# -------------------------
# Build Overview table
# -------------------------
def build_overview_table(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    missing = []

    for _, r in manifest.iterrows():
        wave = r["wave"]
        slug = r["slug"]
        s = load_latest_perf(slug)

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
# WaveScore (Proxy v0.1)
# -------------------------
def wavescore_proxy(latest: pd.Series) -> Optional[float]:
    """
    A simple proxy score (0-100) so you have a working leaderboard immediately.
    Not the locked WaveScore v1.0 spec — we can wire that later once all ingredients exist.
    """
    try:
        a1y = latest.get("alpha_1y")
        a60 = latest.get("alpha_60d")
        a30 = latest.get("alpha_30d")
        intr = latest.get("intraday_alpha")

        vals = []
        for v, w in [(a1y, 0.55), (a60, 0.25), (a30, 0.15), (intr, 0.05)]:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            vals.append((float(v), w))

        if not vals:
            return None

        # weighted alpha (rough)
        score_raw = sum(v * w for v, w in vals)

        # map: -20% => 0, +20% => 100 (clipped)
        score = (score_raw + 0.20) / 0.40 * 100.0
        score = max(0.0, min(100.0, score))
        return score
    except Exception:
        return None


def build_wavescore_leaderboard(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in manifest.iterrows():
        wave = r["wave"]
        slug = r["slug"]
        latest = load_latest_perf(slug)
        if latest is None:
            continue
        sc = wavescore_proxy(latest)
        rows.append({
            "Wave": wave,
            "WaveScore (proxy)": None if sc is None else round(sc, 1),
            "1Y Alpha": fmt_pct(latest.get("alpha_1y")),
            "60D Alpha": fmt_pct(latest.get("alpha_60d")),
            "30D Alpha": fmt_pct(latest.get("alpha_30d")),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "WaveScore (proxy)" in df.columns:
        df = df.sort_values("WaveScore (proxy)", ascending=False, na_position="last").reset_index(drop=True)
    return df


# -------------------------
# Correlation Matrix
# -------------------------
def build_intraday_correlation(manifest: pd.DataFrame, lookback_rows: int = 30) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Computes correlation matrix using last N rows of intraday_return from each wave perf file.
    Aligns on row order (best available without a unified return series).
    """
    series_map = {}
    missing = []

    for _, r in manifest.iterrows():
        wave = r["wave"]
        slug = r["slug"]
        hist = load_perf_history(slug)
        if hist is None or "intraday_return" not in hist.columns:
            missing.append(wave)
            continue

        s = pd.to_numeric(hist["intraday_return"], errors="coerce").dropna()
        if len(s) < 5:
            missing.append(wave)
            continue

        series_map[wave] = s.tail(lookback_rows).reset_index(drop=True)

    if len(series_map) < 2:
        return None, missing

    df = pd.DataFrame(series_map).dropna(axis=0, how="any")
    if df.empty or df.shape[0] < 5:
        return None, missing

    corr = df.corr()
    return corr, missing


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="WAVES Intelligence™ — Institutional Console", layout="wide")

st.title("WAVES Intelligence™ — Institutional Console")
st.caption("Restored: Overview • Wave Detail • Top-10 Holdings • WaveScore Leaderboard • Correlation Matrix • Diagnostics")

# Tabs (rich)
tabs = st.tabs([
    "Overview",
    "Wave Detail",
    "Top 10 Holdings",
    "WaveScore Leaderboard",
    "Correlation Matrix",
    "Diagnostics",
])

# Sidebar
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

    if "auto_run_enabled" not in st.session_state:
        st.session_state.auto_run_enabled = True
    auto_run = st.checkbox("Auto-run engine on login (once per session)", value=st.session_state.auto_run_enabled)
    st.session_state.auto_run_enabled = auto_run

    run_btn = st.button("▶ Run Engine Now (generate logs)", use_container_width=True)

    if "engine_out" not in st.session_state:
        st.session_state.engine_out = ""
    if "engine_ok" not in st.session_state:
        st.session_state.engine_ok = None

    if run_btn:
        with st.spinner("Running waves_engine.py..."):
            ok, out = run_engine_capture()
            st.session_state.engine_ok = ok
            st.session_state.engine_out = out
        st.success("Engine completed." if ok else "Engine errored — open Diagnostics.")

# Auto-run once per session if needed
if st.session_state.get("auto_run_enabled", True) and not st.session_state.get("auto_ran_once", False):
    if (not os.path.exists(MANIFEST_FILE)) or (len(list_perf_files()) == 0):
        with st.spinner("Auto-running engine (first load)..."):
            ok, out = run_engine_capture()
            st.session_state.engine_ok = ok
            st.session_state.engine_out = out
        st.session_state.auto_ran_once = True
    else:
        st.session_state.auto_ran_once = True

manifest = load_manifest()


# -------------------------
# Overview
# -------------------------
with tabs[0]:
    if manifest is None:
        st.error("No manifest found. Run the engine once to generate logs/performance/_wave_manifest.csv.")
    else:
        overview_df, missing_waves = build_overview_table(manifest)
        st.dataframe(overview_df, use_container_width=True, hide_index=True)

        if missing_waves:
            st.warning(
                "Some waves do not have slug-based performance files yet (or they didn’t generate successfully).\n\n"
                + "\n".join([f"- {w}" for w in missing_waves])
            )


# -------------------------
# Wave Detail
# -------------------------
with tabs[1]:
    if manifest is None:
        st.info("Run the engine once to generate the manifest, then Wave Detail will populate.")
    else:
        wave_names = sorted(manifest["wave"].tolist())
        wave_choice = st.selectbox("Select a Wave", wave_names, key="wave_detail_choice")

        slug = manifest.loc[manifest["wave"] == wave_choice, "slug"].iloc[0]
        latest = load_latest_perf(slug)

        st.subheader(wave_choice)
        st.caption(f"Slug: `{slug}`")

        if latest is None:
            st.error("No performance log found for this wave yet. Run engine.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Intraday Return", fmt_pct(latest.get("intraday_return")))
            c2.metric("Intraday Alpha", fmt_pct(latest.get("intraday_alpha")))
            c3.metric("30D Alpha", fmt_pct(latest.get("alpha_30d")))
            c4.metric("60D Alpha", fmt_pct(latest.get("alpha_60d")))
            c5.metric("1Y Alpha", fmt_pct(latest.get("alpha_1y")))

            st.write("Performance log (latest 60 rows):")
            hist = load_perf_history(slug)
            if hist is not None:
                st.dataframe(hist.tail(60), use_container_width=True, hide_index=True)

        st.divider()
        st.write("Top holdings (latest positions file):")
        holdings = load_top_holdings(slug, top_n=10)
        if holdings is None:
            st.info("No positions file found yet for this wave. Run engine.")
        else:
            # clickable links
            show = holdings[["ticker", "weight"]].copy()
            show["weight"] = show["weight"].apply(lambda x: f"{float(x)*100:.2f}%")
            st.dataframe(show, use_container_width=True, hide_index=True)

            st.write("Google Finance links:")
            for _, r in holdings.iterrows():
                t = str(r["ticker"])
                st.markdown(f"- [{t}]({google_finance_url(t)})")


# -------------------------
# Top 10 Holdings (All Waves)
# -------------------------
with tabs[2]:
    if manifest is None:
        st.info("Run the engine once to generate the manifest, then holdings will populate.")
    else:
        st.subheader("Top 10 Holdings — All Waves")
        st.caption("Pulled from latest `logs/positions/<slug>_positions_YYYYMMDD.csv`")

        for _, r in manifest.iterrows():
            wave = r["wave"]
            slug = r["slug"]

            holdings = load_top_holdings(slug, top_n=10)
            with st.expander(wave, expanded=False):
                if holdings is None:
                    st.info("No positions file found yet for this wave.")
                else:
                    show = holdings[["ticker", "weight"]].copy()
                    show["weight"] = show["weight"].apply(lambda x: f"{float(x)*100:.2f}%")
                    st.dataframe(show, use_container_width=True, hide_index=True)

                    st.write("Links:")
                    for _, rr in holdings.iterrows():
                        t = str(rr["ticker"])
                        st.markdown(f"- [{t}]({google_finance_url(t)})")


# -------------------------
# WaveScore Leaderboard
# -------------------------
with tabs[3]:
    st.subheader("WaveScore Leaderboard")
    st.caption("Proxy leaderboard so you can rank waves immediately. (We’ll wire the locked WaveScore v1.0 next.)")

    if manifest is None:
        st.info("Run the engine once to generate the manifest.")
    else:
        lb = build_wavescore_leaderboard(manifest)
        if lb.empty:
            st.warning("No waves have computed metrics yet. Run the engine.")
        else:
            st.dataframe(lb, use_container_width=True, hide_index=True)


# -------------------------
# Correlation Matrix
# -------------------------
with tabs[4]:
    st.subheader("Correlation Matrix")
    st.caption("Uses last N rows of `intraday_return` from each wave performance log.")

    if manifest is None:
        st.info("Run the engine once to generate the manifest.")
    else:
        lookback = st.slider("Lookback rows", min_value=10, max_value=120, value=30, step=5)
        corr, missing = build_intraday_correlation(manifest, lookback_rows=lookback)

        if corr is None:
            st.warning("Not enough aligned intraday history to compute correlations yet. Run engine a few times.")
        else:
            st.dataframe(corr.round(3), use_container_width=True)

        if missing:
            st.info("Missing or insufficient intraday history for:\n\n" + "\n".join([f"- {w}" for w in missing]))


# -------------------------
# Diagnostics
# -------------------------
with tabs[5]:
    st.subheader("Diagnostics")

    st.write("### File status")
    st.code(f"BASE_DIR: {BASE_DIR}")
    st.write(f"- wave_weights.csv: {'✅' if os.path.exists(WEIGHTS_FILE) else '❌'}")
    st.write(f"- waves_engine.py: {'✅' if os.path.exists(ENGINE_FILE) else '❌'}")
    st.write(f"- manifest: {'✅' if os.path.exists(MANIFEST_FILE) else '❌'}")
    st.write(f"- perf dir: {'✅' if os.path.exists(PERF_DIR) else '❌'}")
    st.write(f"- pos dir: {'✅' if os.path.exists(POS_DIR) else '❌'}")

    st.write("### Perf files")
    pf = list_perf_files()
    st.write(f"Count: {len(pf)}")
    if pf:
        st.code("\n".join(pf[:200]))
    else:
        st.warning("No perf files found (excluding manifest).")

    st.write("### Positions files")
    posf = list_pos_files()
    st.write(f"Count: {len(posf)}")
    if posf:
        st.code("\n".join(posf[:200]))
    else:
        st.warning("No positions files found.")

    st.write("### Engine output (last run)")
    if st.session_state.get("engine_out"):
        st.code(st.session_state.engine_out[:12000])
    else:
        st.info("No engine output captured yet. Use 'Run Engine Now'.")

    st.write("### Notes")
    st.markdown(
        """
- If numbers look “crazy”, it’s almost always **missing price history** for one or more tickers (crypto, microcaps, delisted).
- The engine logs will show which wave failed or returned None values.
- If a wave has a positions file but blank returns, the issue is usually **benchmark history** or **insufficient lookback**.
        """
    )