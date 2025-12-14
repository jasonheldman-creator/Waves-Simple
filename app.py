# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# FULL PRODUCTION LAUNCHER — DIAGNOSTICS+++ (Multi-file, no truncation risk)
#
# This is the MAIN Streamlit entrypoint.
# Heavy diagnostics live in:
#   - diag_core.py
#   - diag_panels.py
#   - whatif_lab.py
#   - ui_blocks.py
#
# Notes:
# • Does NOT modify engine math or baseline results.
# • What-If Lab is “shadow simulation”.
# • Optional yfinance/plotly — app never hard-crashes if missing.

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we

# optional deps
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from ui_blocks import (
    apply_global_css,
    fmt_pct, fmt_num, fmt_score,
    show_df, selectable_table_jump,
    build_alpha_matrix, plot_alpha_heatmap,
    benchmark_source_label,
    compute_spy_nav,
    fetch_prices_daily, fetch_spy_vix, fetch_market_assets,
)

from diag_core import (
    compute_wave_history,
    get_wave_holdings,
    get_benchmark_mix,
    compute_alpha_attribution,
    compute_wavescore_for_all_waves,
    wave_doctor_assess,
)

from diag_panels import (
    panel_mode_separation_proof,
    panel_rolling_diagnostics,
    panel_correlation_matrix,
    panel_data_quality_audit,
    panel_benchmark_truth,
    panel_holdings_top10,
    panel_factor_decomposition,
)

from whatif_lab import (
    panel_whatif_lab,
)

# ============================================================
# Streamlit config
# ============================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_css()

# ============================================================
# Safe engine lists
# ============================================================
def _safe_get_all_waves() -> List[str]:
    try:
        w = we.get_all_waves()
        return list(w) if w is not None else []
    except Exception:
        return []

def _safe_get_modes() -> List[str]:
    try:
        m = we.get_modes()
        if m is None:
            return ["Standard", "Alpha-Minus-Beta", "Private Logic"]
        return list(m)
    except Exception:
        return ["Standard", "Alpha-Minus-Beta", "Private Logic"]

all_waves = _safe_get_all_waves()
all_modes = _safe_get_modes()

if "selected_wave" not in st.session_state:
    st.session_state["selected_wave"] = all_waves[0] if all_waves else ""
if "mode" not in st.session_state:
    st.session_state["mode"] = all_modes[0] if all_modes else "Standard"

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("WAVES Intelligence™")
    st.caption("Mini Bloomberg Console • Vector OS™")

    if all_modes:
        st.selectbox(
            "Mode",
            all_modes,
            index=all_modes.index(st.session_state["mode"]) if st.session_state["mode"] in all_modes else 0,
            key="mode",
        )
    else:
        st.text_input("Mode", value=st.session_state.get("mode", "Standard"), key="mode")

    if all_waves:
        st.selectbox(
            "Select Wave",
            all_waves,
            index=all_waves.index(st.session_state["selected_wave"]) if st.session_state["selected_wave"] in all_waves else 0,
            key="selected_wave",
        )
    else:
        st.text_input("Select Wave", value=st.session_state.get("selected_wave", ""), key="selected_wave")

    st.markdown("---")
    st.markdown("**Display settings**")
    history_days = st.slider("History window (days)", min_value=60, max_value=730, value=365, step=15)

    st.markdown("---")
    st.markdown("**Wave Doctor thresholds**")
    alpha_warn = st.slider("Alpha alert threshold (abs, 30D)", 0.02, 0.20, 0.08, 0.01)
    te_warn = st.slider("Tracking error alert threshold", 0.05, 0.40, 0.20, 0.01)

mode = st.session_state["mode"]
selected_wave = st.session_state["selected_wave"]

if not selected_wave:
    st.error("No Waves available (engine returned empty list). Verify waves_engine.get_all_waves().")
    st.stop()

# ============================================================
# Header
# ============================================================
st.title("WAVES Intelligence™ Institutional Console")
st.caption("Live Alpha Capture • SmartSafe™ • Multi-Asset • Crypto • Gold • Income Ladders")

# ============================================================
# Sticky Summary Bar
# ============================================================
h_bar = compute_wave_history(selected_wave, mode=mode, days=max(365, history_days))

bar_r30 = bar_a30 = bar_r365 = bar_a365 = float("nan")
bar_te = bar_ir = float("nan")

bm_mix_for_src = get_benchmark_mix()
bar_src = benchmark_source_label(selected_wave, bm_mix_for_src)

if not h_bar.empty and len(h_bar) >= 2:
    nav_w = h_bar["wave_nav"]
    nav_b = h_bar["bm_nav"]
    ret_w = h_bar["wave_ret"]
    ret_b = h_bar["bm_ret"]

    # 30D
    r30w = (nav_w.iloc[-1] / nav_w.iloc[-min(30, len(nav_w))] - 1.0) if len(nav_w) >= 2 else float("nan")
    r30b = (nav_b.iloc[-1] / nav_b.iloc[-min(30, len(nav_b))] - 1.0) if len(nav_b) >= 2 else float("nan")
    bar_r30 = float(r30w) if math.isfinite(float(r30w)) else float("nan")
    bar_a30 = float(r30w - r30b) if (math.isfinite(float(r30w)) and math.isfinite(float(r30b))) else float("nan")

    # 365D / full
    r365w = (nav_w.iloc[-1] / nav_w.iloc[0] - 1.0) if len(nav_w) >= 2 else float("nan")
    r365b = (nav_b.iloc[-1] / nav_b.iloc[0] - 1.0) if len(nav_b) >= 2 else float("nan")
    bar_r365 = float(r365w) if math.isfinite(float(r365w)) else float("nan")
    bar_a365 = float(r365w - r365b) if (math.isfinite(float(r365w)) and math.isfinite(float(r365b))) else float("nan")

    # TE/IR
    try:
        bar_te = float((ret_w - ret_b).dropna().std() * np.sqrt(252))
    except Exception:
        bar_te = float("nan")
    try:
        bar_ir = float((bar_r365 - r365b) / bar_te) if (math.isfinite(bar_te) and bar_te > 0) else float("nan")
    except Exception:
        bar_ir = float("nan")

# Regime + VIX
spy_vix = fetch_spy_vix(days=min(365, max(120, history_days)))
reg_now = "—"
vix_last = float("nan")
if not spy_vix.empty and "SPY" in spy_vix.columns and "^VIX" in spy_vix.columns and len(spy_vix) > 60:
    vix_last = float(spy_vix["^VIX"].iloc[-1])
    spy_nav = (1.0 + spy_vix["SPY"].pct_change().fillna(0.0)).cumprod()
    r60 = spy_nav / spy_nav.shift(60) - 1.0
    x = float(r60.iloc[-1]) if math.isfinite(float(r60.iloc[-1])) else 0.0
    if x <= -0.12:
        reg_now = "panic"
    elif x <= -0.04:
        reg_now = "downtrend"
    elif x < 0.06:
        reg_now = "neutral"
    else:
        reg_now = "uptrend"

# WaveScore snapshot
ws_snap = compute_wavescore_for_all_waves(all_waves, mode=mode, days=365)
ws_val = "—"
ws_grade = "—"
if ws_snap is not None and not ws_snap.empty:
    rr = ws_snap[ws_snap["Wave"] == selected_wave]
    if not rr.empty:
        ws_val = fmt_score(float(rr.iloc[0]["WaveScore"]))
        ws_grade = str(rr.iloc[0]["Grade"])

st.markdown(
    f"""
<div class="waves-sticky">
  <div class="waves-hdr">📌 Live Summary</div>
  <span class="waves-chip">Mode: <b>{mode}</b></span>
  <span class="waves-chip">Wave: <b>{selected_wave}</b></span>
  <span class="waves-chip">Benchmark: <b>{bar_src}</b></span>
  <span class="waves-chip">Regime: <b>{reg_now}</b></span>
  <span class="waves-chip">VIX: <b>{fmt_num(vix_last, 1) if math.isfinite(vix_last) else "—"}</b></span>
  <span class="waves-chip">30D α: <b>{fmt_pct(bar_a30)}</b> · 30D r: <b>{fmt_pct(bar_r30)}</b></span>
  <span class="waves-chip">365D α: <b>{fmt_pct(bar_a365)}</b> · 365D r: <b>{fmt_pct(bar_r365)}</b></span>
  <span class="waves-chip">TE: <b>{fmt_pct(bar_te)}</b> · IR: <b>{fmt_num(bar_ir, 2)}</b></span>
  <span class="waves-chip">WaveScore: <b>{ws_val}</b> ({ws_grade})</span>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Tabs
# ============================================================
tab_console, tab_diag, tab_market, tab_factors, tab_vector = st.tabs(
    ["Console Scan", "Diagnostics++", "Market Intel", "Factor Decomp", "Vector OS Insight"]
)

# ============================================================
# TAB: Console Scan
# ============================================================
with tab_console:
    st.subheader("🔥 Alpha Heatmap View (All Waves × Timeframe)")
    alpha_df = build_alpha_matrix(all_waves, mode)
    plot_alpha_heatmap(alpha_df, selected_wave, title=f"Alpha Heatmap — Mode: {mode}")

    st.markdown("### 🧭 One-Click Jump Table")
    jump_df = alpha_df.copy()
    if not jump_df.empty:
        jump_df["RankScore"] = jump_df[[c for c in jump_df.columns if "Alpha" in c]].mean(axis=1, skipna=True)
        jump_df = jump_df.sort_values("RankScore", ascending=False)
        show_df(jump_df, selected_wave, key="jump_table_fmt")
        selectable_table_jump(jump_df, key="jump_table_select")

    st.markdown("---")
    st.subheader("🧾 All Waves Overview (1D / 30D / 60D / 365D Returns + Alpha)")
    rows = []
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=365)
        if h.empty or len(h) < 2:
            rows.append({"Wave": w, "1D Ret": np.nan, "1D Alpha": np.nan, "30D Ret": np.nan, "30D Alpha": np.nan,
                         "60D Ret": np.nan, "60D Alpha": np.nan, "365D Ret": np.nan, "365D Alpha": np.nan})
            continue
        nav_w = h["wave_nav"]
        nav_b = h["bm_nav"]
        # 1D
        r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0) if len(nav_w) >= 2 else np.nan
        r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0) if len(nav_b) >= 2 else np.nan
        a1 = r1w - r1b if (math.isfinite(r1w) and math.isfinite(r1b)) else np.nan
        # 30/60/365
        r30w = float(nav_w.iloc[-1] / nav_w.iloc[-min(30, len(nav_w))] - 1.0)
        r30b = float(nav_b.iloc[-1] / nav_b.iloc[-min(30, len(nav_b))] - 1.0)
        r60w = float(nav_w.iloc[-1] / nav_w.iloc[-min(60, len(nav_w))] - 1.0)
        r60b = float(nav_b.iloc[-1] / nav_b.iloc[-min(60, len(nav_b))] - 1.0)
        r365w = float(nav_w.iloc[-1] / nav_w.iloc[0] - 1.0)
        r365b = float(nav_b.iloc[-1] / nav_b.iloc[0] - 1.0)
        rows.append({
            "Wave": w,
            "1D Ret": r1w, "1D Alpha": a1,
            "30D Ret": r30w, "30D Alpha": (r30w - r30b),
            "60D Ret": r60w, "60D Alpha": (r60w - r60b),
            "365D Ret": r365w, "365D Alpha": (r365w - r365b),
        })
    show_df(pd.DataFrame(rows), selected_wave, key="overview_all_waves")

    st.markdown("---")
    st.subheader(f"📌 Selected Wave — {selected_wave}")
    panel_holdings_top10(selected_wave)

    st.markdown("---")
    st.subheader("🩺 Wave Doctor")
    wd = wave_doctor_assess(selected_wave, mode=mode, days=365, alpha_warn=alpha_warn, te_warn=te_warn)
    if not wd.get("ok", False):
        st.info(wd.get("message", "Wave Doctor unavailable."))
    else:
        show_df(pd.DataFrame(wd["table_rows"]), selected_wave, key="wave_doctor_table")
        if wd.get("flags"):
            st.warning(" | ".join(wd["flags"]))
        st.markdown("**Diagnosis**")
        for line in wd.get("diagnosis", []):
            st.write(f"- {line}")
        if wd.get("recommendations"):
            st.markdown("**Recommendations (shadow controls)**")
            for line in wd["recommendations"]:
                st.write(f"- {line}")

    st.markdown("---")
    st.subheader("🧪 What-If Lab (Shadow Simulation)")
    panel_whatif_lab(selected_wave, mode, history_days)

# ============================================================
# TAB: Diagnostics++
# ============================================================
with tab_diag:
    st.subheader("Diagnostics++ Control Room")
    st.caption("Mode separation proof, rolling metrics, correlation, data quality, benchmark truth, attribution.")

    st.markdown("### ✅ Benchmark Truth + Attribution")
    panel_benchmark_truth(selected_wave, mode)

    st.markdown("---")
    st.markdown("### 🧬 Mode Separation Proof (All Modes)")
    panel_mode_separation_proof(selected_wave, all_modes, days=min(365, max(120, history_days)))

    st.markdown("---")
    st.markdown("### 📈 Rolling Diagnostics (Alpha / TE / Beta / Vol / Persistence)")
    panel_rolling_diagnostics(selected_wave, mode, days=min(365, max(120, history_days)))

    st.markdown("---")
    st.markdown("### 🧩 Correlation Matrix (All Waves)")
    panel_correlation_matrix(all_waves, mode, days=min(365, max(120, history_days)))

    st.markdown("---")
    st.markdown("### 🧾 Data Quality / Coverage Audit")
    panel_data_quality_audit(selected_wave, mode, days=min(365, max(120, history_days)))

# ============================================================
# TAB: Market Intel
# ============================================================
with tab_market:
    st.subheader("🌐 Market Intel")
    mk = fetch_market_assets(days=min(365, max(120, history_days)))
    if mk is None or mk.empty:
        st.warning("Market data unavailable (yfinance missing/blocked).")
    else:
        rets = mk.pct_change().fillna(0.0)
        last = rets.tail(1).T.reset_index()
        last.columns = ["Asset", "1D Return"]
        last["1D Return"] = last["1D Return"].apply(lambda x: fmt_pct(x))
        st.dataframe(last, use_container_width=True)

        if go is not None:
            fig = go.Figure()
            for c in mk.columns:
                s = mk[c] / mk[c].iloc[0] * 100.0
                fig.add_trace(go.Scatter(x=mk.index, y=s, name=c, mode="lines"))
            fig.update_layout(height=420, margin=dict(l=40, r=40, t=40, b=40), title="Indexed Prices (Start=100)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(mk)

# ============================================================
# TAB: Factor Decomposition
# ============================================================
with tab_factors:
    st.subheader("🧩 Factor Decomposition (Simple Regression Betas)")
    panel_factor_decomposition(selected_wave, mode, days=min(365, max(120, history_days)))

# ============================================================
# TAB: Vector OS Insight
# ============================================================
with tab_vector:
    st.subheader("🤖 Vector OS Insight Layer")
    st.caption("Narrative interpretation (non-advice).")

    wd = wave_doctor_assess(selected_wave, mode=mode, days=365, alpha_warn=alpha_warn, te_warn=te_warn)
    attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)

    st.markdown("### Vector Summary")
    st.write(f"**Wave:** {selected_wave}  |  **Mode:** {mode}  |  **Regime:** {reg_now}  |  **Benchmark:** {bar_src}")

    if wd.get("ok", False):
        if wd.get("flags"):
            st.warning("Flags: " + " | ".join(wd["flags"]))
        st.markdown("### Key Readouts")
        for r in wd["table_rows"]:
            st.write(f"- **{r['Metric']}**: {r['Value']}")

    st.markdown("### Attribution Lens")
    if attrib:
        for k in [
            "Engine Return", "Static Basket Return", "Overlay Contribution (Engine - Static)",
            "Alpha vs Benchmark", "Benchmark Difficulty (BM - SPY)",
            "Tracking Error (TE)", "Information Ratio (IR)",
            "β_real (Wave vs BM)", "β_target (if available)", "Alpha Captured (last day)"
        ]:
            if k in attrib:
                v = attrib[k]
                if ("Return" in k) or ("Alpha" in k) or ("TE" in k) or ("Difficulty" in k) or ("Vol" in k) or ("MaxDD" in k):
                    st.write(f"- **{k}:** {fmt_pct(v)}")
                elif ("IR" in k) or ("β" in k):
                    st.write(f"- **{k}:** {fmt_num(v, 2)}")
                else:
                    st.write(f"- **{k}:** {fmt_num(v, 4)}")

    st.markdown("### Vector Guidance (Non-Advice)")
    st.write(
        "Validate benchmark stability first (Benchmark Truth). Then look for alpha that persists across 30D / 60D / 365D. "
        "If 30D alpha is extreme but 365D is not, check benchmark drift and data coverage. "
        "Use Mode Separation Proof to confirm each mode behaves differently."
    )
    # ui_blocks.py — WAVES Intelligence™ Institutional Console
# UI helpers: CSS, formatting, table styling, alpha heatmap + jump tools,
# plus generic market fetchers (yfinance optional).

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ============================================================
# CSS / Global styling
# ============================================================
def apply_global_css() -> None:
    st.markdown(
        """
<style>
.block-container { padding-top: 1rem; padding-bottom: 2.0rem; }

/* Sticky summary container */
.waves-sticky {
  position: sticky;
  top: 0;
  z-index: 999;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 10px 12px 10px 12px;
  margin: 0 0 12px 0;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(10, 15, 28, 0.66);
}

/* Summary chips */
.waves-chip {
  display: inline-block;
  padding: 6px 10px;
  margin: 6px 8px 0 0;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.85rem;
  line-height: 1.0rem;
  white-space: nowrap;
}

/* Section header */
.waves-hdr { font-weight: 800; letter-spacing: 0.2px; margin-bottom: 4px; }

/* Tighter tables */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Reduce huge whitespace for mobile */
@media (max-width: 700px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
}
</style>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# Formatting helpers
# ============================================================
def fmt_pct(x: Any, digits: int = 2) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x * 100:0.{digits}f}%"
    except Exception:
        return "—"


def fmt_num(x: Any, digits: int = 2) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x:.{digits}f}"
    except Exception:
        return "—"


def fmt_score(x: Any) -> str:
    try:
        if x is None:
            return "—"
        x = float(x)
        if math.isnan(x):
            return "—"
        return f"{x:.1f}"
    except Exception:
        return "—"


def safe_series(s: Optional[pd.Series]) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return s.copy()


def google_quote_url(ticker: str) -> str:
    t = str(ticker).replace(" ", "")
    return f"https://www.google.com/finance/quote/{t}"


# ============================================================
# Market fetchers (optional yfinance)
# ============================================================
@st.cache_data(show_spinner=False)
def fetch_prices_daily(tickers: List[str], days: int = 365) -> pd.DataFrame:
    if yf is None or not tickers:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 260)

    data = yf.download(
        tickers=sorted(list(set(tickers))),
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data[data.columns.levels[0][0]]

    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(0, axis=1)

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.sort_index().ffill().bfill()
    if len(data) > days:
        data = data.iloc[-days:]
    return data


@st.cache_data(show_spinner=False)
def fetch_spy_vix(days: int = 365) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    tickers = ["SPY", "^VIX"]
    px = fetch_prices_daily(tickers, days=days)
    return px


@st.cache_data(show_spinner=False)
def fetch_market_assets(days: int = 365) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "BTC-USD", "^VIX", "^TNX"]
    px = fetch_prices_daily(tickers, days=days)
    return px


@st.cache_data(show_spinner=False)
def compute_spy_nav(days: int = 365) -> pd.Series:
    px = fetch_prices_daily(["SPY"], days=days)
    if px.empty or "SPY" not in px.columns:
        return pd.Series(dtype=float)
    nav = (1.0 + px["SPY"].pct_change().fillna(0.0)).cumprod()
    nav.name = "spy_nav"
    return nav


# ============================================================
# Benchmark source label helper (engine-aware is in app; fallback here)
# ============================================================
def benchmark_source_label(wave_name: str, bm_mix_df: pd.DataFrame) -> str:
    # In app.py we pass the engine-specific mix already; this is a fallback.
    if bm_mix_df is not None and not bm_mix_df.empty:
        return "Auto-Composite (Dynamic)"
    return "Unknown"


# ============================================================
# Table formatting + row highlight utilities
# ============================================================
def build_formatter_map(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}

    fmt: Dict[str, Any] = {}
    pct_keywords = [
        " Ret", " Return", " Alpha",
        "Vol", "Volatility",
        "MaxDD", "Max Drawdown",
        "Tracking Error", "TE",
        "Benchmark Difficulty",
        "Captured",
        "Downside", "Upside",
        "Drawdown",
    ]

    for c in df.columns:
        cs = str(c)
        if cs == "WaveScore":
            fmt[c] = lambda v: fmt_score(v)
            continue
        if cs in ["Return Quality", "Risk Control", "Consistency", "Resilience", "Efficiency", "Transparency"]:
            fmt[c] = lambda v: fmt_num(v, 1)
            continue
        if cs.startswith("β") or cs.lower().startswith("beta"):
            fmt[c] = lambda v: fmt_num(v, 2)
            continue
        if ("IR" in cs) and ("Ret" not in cs) and ("Return" not in cs):
            fmt[c] = lambda v: fmt_num(v, 2)
            continue
        if any(k in cs for k in pct_keywords):
            fmt[c] = lambda v: fmt_pct(v, 2)

    return fmt


def style_selected_and_alpha(df: pd.DataFrame, selected_wave: str):
    alpha_cols = [c for c in df.columns if ("Alpha" in str(c)) or ("Captured" in str(c))]
    fmt_map = build_formatter_map(df)

    def row_style(row: pd.Series):
        styles = [""] * len(row)

        if "Wave" in df.columns and str(row.get("Wave", "")) == str(selected_wave):
            styles = ["background-color: rgba(255,255,255,0.07); font-weight: 700;"] * len(row)

        for i, col in enumerate(df.columns):
            if col in alpha_cols:
                try:
                    v = float(row[col])
                except Exception:
                    continue
                if not math.isfinite(v):
                    continue
                if v > 0:
                    styles[i] += "background-color: rgba(0, 255, 140, 0.10);"
                elif v < 0:
                    styles[i] += "background-color: rgba(255, 80, 80, 0.10);"
        return styles

    styler = df.style.apply(row_style, axis=1)
    if fmt_map:
        styler = styler.format(fmt_map)
    return styler


def show_df(df: pd.DataFrame, selected_wave: str, key: str):
    try:
        st.dataframe(style_selected_and_alpha(df, selected_wave), use_container_width=True, key=key)
    except Exception:
        st.dataframe(df, use_container_width=True, key=key)


def selectable_table_jump(df: pd.DataFrame, key: str) -> None:
    if df is None or df.empty or "Wave" not in df.columns:
        st.info("No waves available to jump.")
        return

    # best-effort selection events (may not work on all Streamlit Cloud builds)
    try:
        event = st.dataframe(
            df,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="single-row",
        )
        sel = getattr(event, "selection", None)
        if sel and isinstance(sel, dict):
            rows = sel.get("rows", [])
            if rows:
                idx = int(rows[0])
                wave = str(df.iloc[idx]["Wave"])
                if wave:
                    st.session_state["selected_wave"] = wave
                    st.rerun()
        return
    except Exception:
        pass

    st.dataframe(df, use_container_width=True, key=f"{key}_fallback")
    pick = st.selectbox("Jump to Wave", list(df["Wave"]), key=f"{key}_pick")
    if st.button("Jump", key=f"{key}_btn"):
        st.session_state["selected_wave"] = pick
        st.rerun()


# ============================================================
# Alpha heatmap helpers
# ============================================================
@st.cache_data(show_spinner=False)
def build_alpha_matrix(all_waves: List[str], mode: str) -> pd.DataFrame:
    # Import here to avoid circular dependency (diag_core imports ui_blocks too)
    from diag_core import compute_wave_history, ret_from_nav

    rows: List[Dict[str, Any]] = []
    for wname in all_waves:
        hist = compute_wave_history(wname, mode=mode, days=365)
        if hist.empty or len(hist) < 2:
            rows.append({"Wave": wname, "1D Alpha": np.nan, "30D Alpha": np.nan, "60D Alpha": np.nan, "365D Alpha": np.nan})
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]

        # 1D
        a1 = np.nan
        if len(nav_w) >= 2 and len(nav_b) >= 2:
            r1w = float(nav_w.iloc[-1] / nav_w.iloc[-2] - 1.0)
            r1b = float(nav_b.iloc[-1] / nav_b.iloc[-2] - 1.0)
            if math.isfinite(r1w) and math.isfinite(r1b):
                a1 = r1w - r1b

        r30w = ret_from_nav(nav_w, min(30, len(nav_w)))
        r30b = ret_from_nav(nav_b, min(30, len(nav_b)))
        r60w = ret_from_nav(nav_w, min(60, len(nav_w)))
        r60b = ret_from_nav(nav_b, min(60, len(nav_b)))
        r365w = ret_from_nav(nav_w, len(nav_w))
        r365b = ret_from_nav(nav_b, len(nav_b))

        rows.append({
            "Wave": wname,
            "1D Alpha": a1,
            "30D Alpha": (r30w - r30b),
            "60D Alpha": (r60w - r60b),
            "365D Alpha": (r365w - r365b),
        })

    return pd.DataFrame(rows).sort_values("Wave")


def plot_alpha_heatmap(alpha_df: pd.DataFrame, selected_wave: str, title: str):
    if go is None or alpha_df is None or alpha_df.empty:
        st.info("Heatmap unavailable (Plotly missing or no data).")
        return

    df = alpha_df.copy()
    cols = [c for c in ["1D Alpha", "30D Alpha", "60D Alpha", "365D Alpha"] if c in df.columns]
    if not cols:
        st.info("No alpha columns to plot.")
        return

    # selected wave first
    if "Wave" in df.columns and selected_wave in set(df["Wave"]):
        top = df[df["Wave"] == selected_wave]
        rest = df[df["Wave"] != selected_wave]
        df = pd.concat([top, rest], axis=0)

    z = df[cols].values
    y = df["Wave"].tolist()
    x = cols

    v = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.10
    if not math.isfinite(v) or v <= 0:
        v = 0.10

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            zmin=-v,
            zmax=v,
            colorbar=dict(title="Alpha"),
        )
    )
    fig.update_layout(
        title=title,
        height=min(900, 240 + 22 * max(10, len(y))),
        margin=dict(l=80, r=40, t=60, b=40),
        xaxis_title="Timeframe",
        yaxis_title="Wave",
    )
    st.plotly_chart(fig, use_container_width=True)
    # diag_core.py — WAVES Intelligence™ Institutional Console
# Core diagnostics computations: history, attribution, wavescore proto, wave doctor.
# Engine math is NEVER changed; we only read we.compute_history_nav and holdings.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we

from ui_blocks import fetch_prices_daily, compute_spy_nav, fmt_pct, fmt_num


# ============================================================
# Cached engine accessors
# ============================================================
@st.cache_data(show_spinner=False)
def compute_wave_history(wave_name: str, mode: str, days: int = 365) -> pd.DataFrame:
    try:
        df = we.compute_history_nav(wave_name, mode=mode, days=days)
    except Exception:
        df = pd.DataFrame(columns=["wave_nav", "bm_nav", "wave_ret", "bm_ret"])
    # normalize columns
    for c in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        if c not in df.columns:
            df[c] = np.nan
    return df


@st.cache_data(show_spinner=False)
def get_benchmark_mix() -> pd.DataFrame:
    try:
        return we.get_benchmark_mix_table()
    except Exception:
        return pd.DataFrame(columns=["Wave", "Ticker", "Name", "Weight"])


@st.cache_data(show_spinner=False)
def get_wave_holdings(wave_name: str) -> pd.DataFrame:
    try:
        return we.get_wave_holdings(wave_name)
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name", "Weight"])


# ============================================================
# Math helpers
# ============================================================
def safe_series(s: Optional[pd.Series]) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return s.copy()


def ret_from_nav(nav: pd.Series, window: int) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    if window < 2:
        return float("nan")
    if len(nav) < window:
        window = len(nav)
    sub = nav.iloc[-window:]
    start = float(sub.iloc[0])
    end = float(sub.iloc[-1])
    if start <= 0 or (not math.isfinite(start)) or (not math.isfinite(end)):
        return float("nan")
    return (end / start) - 1.0


def annualized_vol(daily_ret: pd.Series) -> float:
    daily_ret = safe_series(daily_ret)
    if len(daily_ret) < 2:
        return float("nan")
    return float(daily_ret.std() * np.sqrt(252))


def max_drawdown(nav: pd.Series) -> float:
    nav = safe_series(nav)
    if len(nav) < 2:
        return float("nan")
    running_max = nav.cummax()
    dd = (nav / running_max) - 1.0
    return float(dd.min())


def tracking_error(daily_wave: pd.Series, daily_bm: pd.Series) -> float:
    w = safe_series(daily_wave)
    b = safe_series(daily_bm)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 2:
        return float("nan")
    diff = df["w"] - df["b"]
    return float(diff.std() * np.sqrt(252))


def information_ratio(nav_wave: pd.Series, nav_bm: pd.Series, te: float) -> float:
    if te is None or (isinstance(te, float) and (not math.isfinite(te) or te <= 0)):
        return float("nan")
    rw = ret_from_nav(nav_wave, len(nav_wave))
    rb = ret_from_nav(nav_bm, len(nav_bm))
    if not math.isfinite(rw) or not math.isfinite(rb):
        return float("nan")
    return float((rw - rb) / te)


def beta_vs_benchmark(wave_ret: pd.Series, bm_ret: pd.Series) -> float:
    w = safe_series(wave_ret)
    b = safe_series(bm_ret)
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < 20:
        return float("nan")
    var_b = float(df["b"].var())
    if not math.isfinite(var_b) or var_b <= 0:
        return float("nan")
    cov = float(df["w"].cov(df["b"]))
    return float(cov / var_b)


def get_beta_target_if_available(mode: str) -> float:
    candidates = ["MODE_BETA_TARGET", "BETA_TARGET_BY_MODE", "BETA_TARGETS", "BETA_TARGET"]
    for name in candidates:
        try:
            obj = getattr(we, name, None)
            if isinstance(obj, dict) and mode in obj:
                return float(obj[mode])
        except Exception:
            pass
    return float("nan")


def alpha_captured_daily(hist: pd.DataFrame, mode: str) -> pd.Series:
    """
    Daily alpha captured = wave_ret - bm_ret, exposure-scaled if engine exposes
    MODE_BASE_EXPOSURE or similar. (Best-effort; never changes engine math.)
    """
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    if "wave_ret" not in hist.columns or "bm_ret" not in hist.columns:
        return pd.Series(dtype=float)

    w = pd.to_numeric(hist["wave_ret"], errors="coerce")
    b = pd.to_numeric(hist["bm_ret"], errors="coerce")
    a = (w - b).astype(float)

    # exposure scaling best-effort (if available)
    expo = 1.0
    try:
        m = getattr(we, "MODE_BASE_EXPOSURE", None)
        if isinstance(m, dict) and mode in m:
            expo = float(m[mode])
    except Exception:
        expo = 1.0

    try:
        if math.isfinite(expo) and expo > 0:
            a = a / expo
    except Exception:
        pass

    a.name = "alpha_captured"
    return a


# ============================================================
# Attribution: Engine vs Static Basket
# ============================================================
def _weights_from_df(df: pd.DataFrame, ticker_col: str = "Ticker", weight_col: str = "Weight") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    w = df[[ticker_col, weight_col]].copy()
    w[ticker_col] = w[ticker_col].astype(str)
    w[weight_col] = pd.to_numeric(w[weight_col], errors="coerce").fillna(0.0)
    w = w.groupby(ticker_col, as_index=True)[weight_col].sum()
    total = float(w.sum())
    if total <= 0 or not math.isfinite(total):
        return pd.Series(dtype=float)
    return (w / total).sort_index()


@st.cache_data(show_spinner=False)
def compute_static_nav_from_weights(weights: pd.Series, days: int = 365) -> pd.Series:
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    tickers = list(weights.index)
    px = fetch_prices_daily(tickers, days=days)
    if px.empty:
        return pd.Series(dtype=float)

    weights_aligned = weights.reindex(px.columns).fillna(0.0)
    daily_ret = px.pct_change().fillna(0.0)
    port_ret = (daily_ret * weights_aligned).sum(axis=1)

    nav = (1.0 + port_ret).cumprod()
    nav.name = "static_nav"
    return nav


@st.cache_data(show_spinner=False)
def compute_alpha_attribution(wave_name: str, mode: str, days: int = 365) -> Dict[str, float]:
    out: Dict[str, float] = {}

    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist.empty or len(hist) < 2:
        return out

    nav_wave = hist["wave_nav"]
    nav_bm = hist["bm_nav"]
    wave_ret = hist["wave_ret"]
    bm_ret = hist["bm_ret"]

    eng_ret = ret_from_nav(nav_wave, window=len(nav_wave))
    bm_ret_total = ret_from_nav(nav_bm, window=len(nav_bm))
    alpha_vs_bm = eng_ret - bm_ret_total if (math.isfinite(eng_ret) and math.isfinite(bm_ret_total)) else float("nan")

    hold_df = get_wave_holdings(wave_name)
    hold_w = _weights_from_df(hold_df, ticker_col="Ticker", weight_col="Weight")
    static_nav = compute_static_nav_from_weights(hold_w, days=days)
    static_ret = ret_from_nav(static_nav, window=len(static_nav)) if len(static_nav) >= 2 else float("nan")
    overlay_pp = (eng_ret - static_ret) if (math.isfinite(eng_ret) and math.isfinite(static_ret)) else float("nan")

    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, window=len(spy_nav)) if len(spy_nav) >= 2 else float("nan")

    alpha_vs_spy = eng_ret - spy_ret if (math.isfinite(eng_ret) and math.isfinite(spy_ret)) else float("nan")
    benchmark_difficulty = bm_ret_total - spy_ret if (math.isfinite(bm_ret_total) and math.isfinite(spy_ret)) else float("nan")

    te = tracking_error(wave_ret, bm_ret)
    ir = information_ratio(nav_wave, nav_bm, te)

    beta_real = beta_vs_benchmark(wave_ret, bm_ret)
    beta_target = get_beta_target_if_available(mode)

    # alpha captured last day
    ac = alpha_captured_daily(hist, mode=mode)
    alpha_captured_last = float(ac.dropna().iloc[-1]) if not ac.empty and ac.dropna().shape[0] >= 1 else float("nan")

    out["Engine Return"] = float(eng_ret)
    out["Static Basket Return"] = float(static_ret) if math.isfinite(static_ret) else float("nan")
    out["Overlay Contribution (Engine - Static)"] = float(overlay_pp) if math.isfinite(overlay_pp) else float("nan")
    out["Benchmark Return"] = float(bm_ret_total)
    out["Alpha vs Benchmark"] = float(alpha_vs_bm) if math.isfinite(alpha_vs_bm) else float("nan")
    out["SPY Return"] = float(spy_ret) if math.isfinite(spy_ret) else float("nan")
    out["Alpha vs SPY"] = float(alpha_vs_spy) if math.isfinite(alpha_vs_spy) else float("nan")
    out["Benchmark Difficulty (BM - SPY)"] = float(benchmark_difficulty) if math.isfinite(benchmark_difficulty) else float("nan")
    out["Tracking Error (TE)"] = float(te)
    out["Information Ratio (IR)"] = float(ir)
    out["β_real (Wave vs BM)"] = float(beta_real)
    out["β_target (if available)"] = float(beta_target)
    out["Alpha Captured (last day)"] = float(alpha_captured_last)

    out["Wave Vol"] = float(annualized_vol(wave_ret))
    out["Benchmark Vol"] = float(annualized_vol(bm_ret))
    out["Wave MaxDD"] = float(max_drawdown(nav_wave))
    out["Benchmark MaxDD"] = float(max_drawdown(nav_bm))

    return out


# ============================================================
# WaveScore proto helper
# ============================================================
def _grade_from_score(score: float) -> str:
    if score is None or (isinstance(score, float) and (not math.isfinite(score))):
        return "N/A"
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    return "D"


def compute_wavescore_for_all_waves(all_waves: List[str], mode: str, days: int = 365) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for wave in all_waves:
        hist = compute_wave_history(wave, mode=mode, days=days)
        if hist.empty or len(hist) < 20:
            rows.append({
                "Wave": wave,
                "WaveScore": float("nan"),
                "Grade": "N/A",
                "Return Quality": float("nan"),
                "Risk Control": float("nan"),
                "Consistency": float("nan"),
                "Resilience": float("nan"),
                "Efficiency": float("nan"),
                "Transparency": 10.0,
                "IR_365D": float("nan"),
                "Alpha_365D": float("nan"),
            })
            continue

        nav_wave = hist["wave_nav"]
        nav_bm = hist["bm_nav"]
        wave_ret = hist["wave_ret"]
        bm_ret = hist["bm_ret"]

        ret_wave = ret_from_nav(nav_wave, window=len(nav_wave))
        ret_bm = ret_from_nav(nav_bm, window=len(nav_bm))
        alpha_365 = ret_wave - ret_bm if (math.isfinite(ret_wave) and math.isfinite(ret_bm)) else float("nan")

        vol_wave = annualized_vol(wave_ret)
        vol_bm = annualized_vol(bm_ret)

        te = tracking_error(wave_ret, bm_ret)
        ir = information_ratio(nav_wave, nav_bm, te)

        mdd_wave = max_drawdown(nav_wave)
        mdd_bm = max_drawdown(nav_bm)

        hit_rate = float((pd.to_numeric(wave_ret, errors="coerce") >= pd.to_numeric(bm_ret, errors="coerce")).mean())

        # recovery
        trough = float(nav_wave.min())
        peak = float(nav_wave.max())
        last = float(nav_wave.iloc[-1])
        recovery_frac = float((last - trough) / (peak - trough)) if (peak > trough and trough > 0) else float("nan")
        if math.isfinite(recovery_frac):
            recovery_frac = float(np.clip(recovery_frac, 0.0, 1.0))

        vol_ratio = (vol_wave / vol_bm) if (math.isfinite(vol_wave) and math.isfinite(vol_bm) and vol_bm > 0) else float("nan")

        # scoring (proto, display only)
        rq_ir = float(np.clip(ir, 0.0, 1.5) / 1.5 * 15.0) if math.isfinite(ir) else 0.0
        rq_alpha = float(np.clip((alpha_365 + 0.05) / 0.15, 0.0, 1.0) * 10.0) if math.isfinite(alpha_365) else 0.0
        return_quality = float(np.clip(rq_ir + rq_alpha, 0.0, 25.0))

        if not math.isfinite(vol_ratio):
            risk_control = 0.0
        else:
            penalty = max(0.0, abs(vol_ratio - 0.9) - 0.1)
            risk_control = float(np.clip(1.0 - penalty / 0.6, 0.0, 1.0) * 25.0)

        consistency = float(np.clip(hit_rate, 0.0, 1.0) * 15.0) if math.isfinite(hit_rate) else 0.0

        if (not math.isfinite(recovery_frac)) or (not math.isfinite(mdd_wave)) or (not math.isfinite(mdd_bm)):
            resilience = 0.0
        else:
            rec_part = float(np.clip(recovery_frac, 0.0, 1.0) * 6.0)
            dd_edge = (mdd_bm - mdd_wave)
            dd_part = float(np.clip((dd_edge + 0.10) / 0.20, 0.0, 1.0) * 4.0)
            resilience = float(np.clip(rec_part + dd_part, 0.0, 10.0))

        efficiency = float(np.clip((0.25 - te) / 0.20, 0.0, 1.0) * 15.0) if math.isfinite(te) else 0.0
        transparency = 10.0

        total = float(np.clip(return_quality + risk_control + consistency + resilience + efficiency + transparency, 0.0, 100.0))
        grade = _grade_from_score(total)

        rows.append({
            "Wave": wave,
            "WaveScore": total,
            "Grade": grade,
            "Return Quality": return_quality,
            "Risk Control": risk_control,
            "Consistency": consistency,
            "Resilience": resilience,
            "Efficiency": efficiency,
            "Transparency": transparency,
            "IR_365D": ir,
            "Alpha_365D": alpha_365,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df.sort_values("Wave") if not df.empty else df


# ============================================================
# Wave Doctor (structured output for UI)
# ============================================================
def wave_doctor_assess(
    wave_name: str,
    mode: str,
    days: int = 365,
    alpha_warn: float = 0.08,
    te_warn: float = 0.20,
    vol_warn: float = 0.30,
    mdd_warn: float = -0.25,
) -> Dict[str, Any]:

    hist = compute_wave_history(wave_name, mode=mode, days=days)
    if hist.empty or len(hist) < 2:
        return {"ok": False, "message": "Not enough data to run Wave Doctor."}

    nav_w = hist["wave_nav"]
    nav_b = hist["bm_nav"]
    ret_w = pd.to_numeric(hist["wave_ret"], errors="coerce").fillna(0.0)
    ret_b = pd.to_numeric(hist["bm_ret"], errors="coerce").fillna(0.0)

    r365_w = ret_from_nav(nav_w, len(nav_w))
    r365_b = ret_from_nav(nav_b, len(nav_b))
    a365 = (r365_w - r365_b) if (math.isfinite(r365_w) and math.isfinite(r365_b)) else float("nan")

    r30_w = ret_from_nav(nav_w, min(30, len(nav_w)))
    r30_b = ret_from_nav(nav_b, min(30, len(nav_b)))
    a30 = (r30_w - r30_b) if (math.isfinite(r30_w) and math.isfinite(r30_b)) else float("nan")

    vol_w = annualized_vol(ret_w)
    vol_b = annualized_vol(ret_b)
    te = tracking_error(ret_w, ret_b)
    ir = information_ratio(nav_w, nav_b, te)
    mdd_w = max_drawdown(nav_w)
    mdd_b = max_drawdown(nav_b)

    spy_nav = compute_spy_nav(days=days)
    spy_ret = ret_from_nav(spy_nav, len(spy_nav)) if len(spy_nav) >= 2 else float("nan")
    bm_difficulty = (r365_b - spy_ret) if (math.isfinite(r365_b) and math.isfinite(spy_ret)) else float("nan")

    flags: List[str] = []
    diagnosis: List[str] = []
    recs: List[str] = []

    if math.isfinite(a30) and abs(a30) >= alpha_warn:
        flags.append("Large 30D alpha (verify benchmark + coverage)")
        diagnosis.append("30D alpha is unusually large. This can be real, but also benchmark mix drift or data coverage gaps.")
        recs.append("Check Benchmark Truth and Data Quality Audit for drift/gaps. Compare to SPY/QQQ temporarily for validation.")

    if math.isfinite(a365) and a365 < 0:
        diagnosis.append("365D alpha is negative. Could be true underperformance or a tougher benchmark.")
        if math.isfinite(bm_difficulty) and bm_difficulty > 0.03:
            diagnosis.append("Benchmark difficulty is positive vs SPY (benchmark outperformed SPY), so alpha is harder on this window.")
            recs.append("Use Benchmark Difficulty metric + consider peer benchmark for sanity check.")
    else:
        if math.isfinite(a365) and a365 > 0.06:
            diagnosis.append("365D alpha is solidly positive. Freeze benchmark snapshot for reproducibility in demos.")

    if math.isfinite(te) and te > te_warn:
        flags.append("High tracking error (active risk elevated)")
        diagnosis.append("Tracking error is high; the wave behaves very differently than benchmark.")
        recs.append("Use Rolling Diagnostics to confirm TE spikes. Tighten exposure caps (What-If Lab shadow).")

    if math.isfinite(vol_w) and vol_w > vol_warn:
        flags.append("High volatility")
        diagnosis.append("Volatility is elevated; may create large swings in short windows.")
        recs.append("Use Rolling Vol panel and consider lower exposure / higher safe fraction (shadow).")

    if math.isfinite(mdd_w) and mdd_w < mdd_warn:
        flags.append("Deep drawdown")
        diagnosis.append("Drawdown is deep. Consider stronger SmartSafe posture in stress regimes.")
        recs.append("Use What-If to explore regime gating + safe fraction (shadow).")

    if math.isfinite(vol_b) and math.isfinite(vol_w) and vol_b > 0 and (vol_w / vol_b) > 1.6:
        flags.append("Volatility >> benchmark")
        diagnosis.append("Wave volatility is much higher than benchmark; this can inflate wins/losses.")
        recs.append("Confirm beta/TE and exposure settings; validate benchmark composition.")

    if not diagnosis:
        diagnosis.append("No major anomalies detected on selected window. Use Rolling Diagnostics for deeper validation.")

    table_rows = [
        {"Metric": "365D Return", "Value": fmt_pct(r365_w)},
        {"Metric": "365D Alpha", "Value": fmt_pct(a365)},
        {"Metric": "30D Return", "Value": fmt_pct(r30_w)},
        {"Metric": "30D Alpha", "Value": fmt_pct(a30)},
        {"Metric": "Vol (Wave)", "Value": fmt_pct(vol_w)},
        {"Metric": "Vol (Benchmark)", "Value": fmt_pct(vol_b)},
        {"Metric": "Tracking Error (TE)", "Value": fmt_pct(te)},
        {"Metric": "Information Ratio (IR)", "Value": fmt_num(ir, 2)},
        {"Metric": "MaxDD (Wave)", "Value": fmt_pct(mdd_w)},
        {"Metric": "MaxDD (Benchmark)", "Value": fmt_pct(mdd_b)},
        {"Metric": "BM Difficulty (BM - SPY)", "Value": fmt_pct(bm_difficulty)},
    ]

    return {
        "ok": True,
        "flags": flags,
        "diagnosis": diagnosis,
        "recommendations": list(dict.fromkeys(recs)),
        "table_rows": table_rows,
    }
    # diag_panels.py — WAVES Intelligence™ Institutional Console
# Panels for Diagnostics++: mode separation proof, rolling diagnostics,
# correlations, data audit, benchmark truth, holdings, factor decomposition.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from ui_blocks import (
    show_df, fmt_pct, fmt_num, fmt_score,
    fetch_prices_daily, compute_spy_nav, google_quote_url,
)

from diag_core import (
    compute_wave_history,
    get_wave_holdings,
    get_benchmark_mix,
    compute_alpha_attribution,
    ret_from_nav,
    tracking_error,
    information_ratio,
    annualized_vol,
    max_drawdown,
    beta_vs_benchmark,
    get_beta_target_if_available,
    alpha_captured_daily,
)


# ============================================================
# Holdings Panel + Top10
# ============================================================
def panel_holdings_top10(selected_wave: str) -> None:
    hold = get_wave_holdings(selected_wave)
    if hold is None or hold.empty:
        st.warning("No holdings returned by engine for this wave.")
        return

    hold2 = hold.copy()
    if "Weight" in hold2.columns:
        hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce")
    if "Ticker" in hold2.columns:
        hold2["Ticker"] = hold2["Ticker"].astype(str)

    top10 = hold2.sort_values("Weight", ascending=False).head(10) if "Weight" in hold2.columns else hold2.head(10)
    st.markdown("### 🧾 Top 10 Holdings (clickable)")
    for _, r in top10.iterrows():
        t = str(r.get("Ticker", ""))
        wgt = r.get("Weight", np.nan)
        nm = str(r.get("Name", t))
        if t:
            url = google_quote_url(t)
            st.markdown(f"- **[{t}]({url})** — {nm} — **{fmt_pct(wgt)}**")

    st.markdown("### Full Holdings")
    show_df(hold2, selected_wave, key=f"holdings_full_{selected_wave}")


# ============================================================
# Benchmark Truth Panel
# ============================================================
def panel_benchmark_truth(selected_wave: str, mode: str) -> None:
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Benchmark Mix (Engine)")
        bm_mix = get_benchmark_mix()
        if bm_mix is None or bm_mix.empty:
            st.info("No benchmark mix table returned by engine.")
        else:
            if "Wave" in bm_mix.columns:
                show_df(bm_mix[bm_mix["Wave"] == selected_wave], selected_wave, key=f"bm_mix_{selected_wave}")
            else:
                show_df(bm_mix, selected_wave, key=f"bm_mix_{selected_wave}")

    with colB:
        st.markdown("#### Attribution Snapshot (365D)")
        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
        if not attrib:
            st.info("Attribution unavailable.")
        else:
            rows = []
            for k, v in attrib.items():
                if any(x in k for x in ["Return", "Alpha", "Vol", "MaxDD", "TE", "Difficulty", "Captured"]):
                    rows.append({"Metric": k, "Value": fmt_pct(v)})
                elif ("IR" in k) or ("β" in k):
                    rows.append({"Metric": k, "Value": fmt_num(v, 2)})
                else:
                    rows.append({"Metric": k, "Value": fmt_num(v, 4)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ============================================================
# Mode Separation Proof (All Modes)
# ============================================================
def panel_mode_separation_proof(selected_wave: str, all_modes: List[str], days: int = 365) -> None:
    metrics_rows: List[Dict[str, Any]] = []
    nav_lines = {}

    for m in all_modes:
        hist = compute_wave_history(selected_wave, mode=m, days=days)
        if hist.empty or len(hist) < 2:
            metrics_rows.append({
                "Mode": m,
                "365D Return": np.nan,
                "365D Alpha": np.nan,
                "TE": np.nan,
                "IR": np.nan,
                "β_real": np.nan,
                "β_target": get_beta_target_if_available(m),
                "Vol": np.nan,
                "MaxDD": np.nan,
                "Alpha Captured (1D)": np.nan,
            })
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]
        ret_w = pd.to_numeric(hist["wave_ret"], errors="coerce").fillna(0.0)
        ret_b = pd.to_numeric(hist["bm_ret"], errors="coerce").fillna(0.0)

        rw = ret_from_nav(nav_w, len(nav_w))
        rb = ret_from_nav(nav_b, len(nav_b))
        alpha = rw - rb if (math.isfinite(rw) and math.isfinite(rb)) else np.nan

        te = tracking_error(ret_w, ret_b)
        ir = information_ratio(nav_w, nav_b, te)
        beta_real = beta_vs_benchmark(ret_w, ret_b)
        beta_target = get_beta_target_if_available(m)

        vol = annualized_vol(ret_w)
        mdd = max_drawdown(nav_w)

        ac = alpha_captured_daily(hist, mode=m)
        ac1 = float(ac.dropna().iloc[-1]) if not ac.empty and ac.dropna().shape[0] >= 1 else np.nan

        metrics_rows.append({
            "Mode": m,
            "365D Return": rw,
            "365D Alpha": alpha,
            "TE": te,
            "IR": ir,
            "β_real": beta_real,
            "β_target": beta_target,
            "Vol": vol,
            "MaxDD": mdd,
            "Alpha Captured (1D)": ac1,
        })

        nav_lines[m] = nav_w.copy()

    df = pd.DataFrame(metrics_rows)
    st.markdown("#### Mode metrics (side-by-side)")
    st.dataframe(df.style.format({
        "365D Return": lambda v: fmt_pct(v),
        "365D Alpha": lambda v: fmt_pct(v),
        "TE": lambda v: fmt_pct(v),
        "IR": lambda v: fmt_num(v, 2),
        "β_real": lambda v: fmt_num(v, 2),
        "β_target": lambda v: fmt_num(v, 2),
        "Vol": lambda v: fmt_pct(v),
        "MaxDD": lambda v: fmt_pct(v),
        "Alpha Captured (1D)": lambda v: fmt_pct(v),
    }), use_container_width=True)

    st.markdown("#### NAV overlay (Wave NAV per mode)")
    if not nav_lines:
        st.info("No NAV series to plot.")
        return

    nav_df = pd.DataFrame(nav_lines).dropna(how="all")
    if nav_df.empty:
        st.info("No NAV series to plot.")
        return

    # normalize to 100
    nav_df2 = nav_df / nav_df.iloc[0] * 100.0

    if go is not None:
        fig = go.Figure()
        for c in nav_df2.columns:
            fig.add_trace(go.Scatter(x=nav_df2.index, y=nav_df2[c], name=c, mode="lines"))
        fig.update_layout(height=420, margin=dict(l=40, r=40, t=30, b=40), title="Mode Separation Proof — NAV (Indexed=100)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(nav_df2)


# ============================================================
# Rolling Diagnostics (Alpha/TE/Beta/Vol/Persistence)
# ============================================================
def _rolling_beta(w: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < window:
        return pd.Series(dtype=float)

    # rolling cov/var
    cov = df["w"].rolling(window).cov(df["b"])
    var = df["b"].rolling(window).var()
    beta = cov / var
    beta.name = f"beta_{window}"
    return beta


def _rolling_te(w: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < window:
        return pd.Series(dtype=float)
    diff = df["w"] - df["b"]
    te = diff.rolling(window).std() * np.sqrt(252)
    te.name = f"te_{window}"
    return te


def _rolling_vol(w: pd.Series, window: int) -> pd.Series:
    w = pd.to_numeric(w, errors="coerce")
    if w.dropna().shape[0] < window:
        return pd.Series(dtype=float)
    vol = w.rolling(window).std() * np.sqrt(252)
    vol.name = f"vol_{window}"
    return vol


def _rolling_alpha_from_nav(nav_w: pd.Series, nav_b: pd.Series, window: int) -> pd.Series:
    # compute rolling window total return from NAV then alpha
    if nav_w is None or nav_b is None or len(nav_w) < window + 1 or len(nav_b) < window + 1:
        return pd.Series(dtype=float)
    nw = nav_w.astype(float)
    nb = nav_b.astype(float)
    rw = nw / nw.shift(window) - 1.0
    rb = nb / nb.shift(window) - 1.0
    alpha = (rw - rb).rename(f"alpha_{window}")
    return alpha


def _alpha_persistence(alpha_series: pd.Series, window: int) -> Dict[str, float]:
    a = pd.to_numeric(alpha_series, errors="coerce").dropna()
    if a.empty:
        return {"pos_frac": np.nan, "neg_frac": np.nan, "pos_streak": np.nan, "neg_streak": np.nan}

    pos = (a > 0).astype(int)
    neg = (a < 0).astype(int)
    pos_frac = float(pos.mean())
    neg_frac = float(neg.mean())

    # longest streak helper
    def longest_streak(x: pd.Series) -> int:
        best = 0
        cur = 0
        for v in x.values:
            if v == 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)

    pos_streak = float(longest_streak(pos))
    neg_streak = float(longest_streak(neg))

    return {"pos_frac": pos_frac, "neg_frac": neg_frac, "pos_streak": pos_streak, "neg_streak": neg_streak}


def panel_rolling_diagnostics(selected_wave: str, mode: str, days: int = 365) -> None:
    hist = compute_wave_history(selected_wave, mode=mode, days=days)
    if hist is None or hist.empty or len(hist) < 60:
        st.warning("Not enough history for rolling diagnostics (need ~60+ points).")
        return

    nav_w = hist["wave_nav"].astype(float)
    nav_b = hist["bm_nav"].astype(float)
    w = pd.to_numeric(hist["wave_ret"], errors="coerce").fillna(0.0)
    b = pd.to_numeric(hist["bm_ret"], errors="coerce").fillna(0.0)

    # rolling windows
    w_alpha_20 = _rolling_alpha_from_nav(nav_w, nav_b, 20)
    w_alpha_60 = _rolling_alpha_from_nav(nav_w, nav_b, 60)

    w_te_20 = _rolling_te(w, b, 20)
    w_te_60 = _rolling_te(w, b, 60)

    w_beta_60 = _rolling_beta(w, b, 60)
    w_vol_20 = _rolling_vol(w, 20)
    w_vol_60 = _rolling_vol(w, 60)

    beta_target = get_beta_target_if_available(mode)

    # persistence summary
    pers_20 = _alpha_persistence(w_alpha_20, 20)
    pers_60 = _alpha_persistence(w_alpha_60, 60)

    st.markdown("#### Rolling Summary (latest)")
    latest = {
        "Latest Roll Alpha (20D)": float(w_alpha_20.dropna().iloc[-1]) if w_alpha_20.dropna().shape[0] else np.nan,
        "Latest Roll Alpha (60D)": float(w_alpha_60.dropna().iloc[-1]) if w_alpha_60.dropna().shape[0] else np.nan,
        "Latest Roll TE (20D)": float(w_te_20.dropna().iloc[-1]) if w_te_20.dropna().shape[0] else np.nan,
        "Latest Roll TE (60D)": float(w_te_60.dropna().iloc[-1]) if w_te_60.dropna().shape[0] else np.nan,
        "Latest Roll Beta (60D)": float(w_beta_60.dropna().iloc[-1]) if w_beta_60.dropna().shape[0] else np.nan,
        "Beta Target (mode)": float(beta_target) if math.isfinite(beta_target) else np.nan,
        "Latest Roll Vol (20D)": float(w_vol_20.dropna().iloc[-1]) if w_vol_20.dropna().shape[0] else np.nan,
        "Latest Roll Vol (60D)": float(w_vol_60.dropna().iloc[-1]) if w_vol_60.dropna().shape[0] else np.nan,
        "Alpha Persistence 20D (+frac)": pers_20["pos_frac"],
        "Alpha Persistence 60D (+frac)": pers_60["pos_frac"],
        "Longest + Streak (20D alpha>0)": pers_20["pos_streak"],
        "Longest + Streak (60D alpha>0)": pers_60["pos_streak"],
    }
    latest_df = pd.DataFrame([latest]).T.reset_index()
    latest_df.columns = ["Metric", "Value"]

    def _fmt_value(metric: str, v: float) -> str:
        if any(k in metric for k in ["Alpha", "TE"]) and "frac" not in metric and "Streak" not in metric:
            return fmt_pct(v)
        if "Vol" in metric:
            return fmt_pct(v)
        if "Beta" in metric:
            return fmt_num(v, 2)
        if "frac" in metric:
            return fmt_pct(v)
        if "Streak" in metric:
            return fmt_num(v, 0)
        return fmt_num(v, 4)

    latest_df["Value"] = [ _fmt_value(str(m), float(v) if v is not None else np.nan) for m, v in zip(latest_df["Metric"], latest_df["Value"]) ]
    st.dataframe(latest_df, use_container_width=True)

    # beta drift flags
    drift_flag = ""
    if math.isfinite(beta_target) and w_beta_60.dropna().shape[0]:
        beta_last = float(w_beta_60.dropna().iloc[-1])
        drift = abs(beta_last - beta_target)
        if drift > 0.07:
            drift_flag = f"⚠️ Beta Drift: |β_real-β_target|={drift:.2f}"
    if drift_flag:
        st.warning(drift_flag)

    st.markdown("#### Rolling Charts")
    roll_df = pd.DataFrame({
        "Roll Alpha 20D": w_alpha_20,
        "Roll Alpha 60D": w_alpha_60,
        "Roll TE 20D": w_te_20,
        "Roll TE 60D": w_te_60,
        "Roll Beta 60D": w_beta_60,
        "Roll Vol 20D": w_vol_20,
        "Roll Vol 60D": w_vol_60,
    }).dropna(how="all")

    if roll_df.empty:
        st.info("Rolling chart data is empty after alignment.")
        return

    if go is not None:
        fig = go.Figure()
        if "Roll Alpha 20D" in roll_df.columns:
            fig.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll Alpha 20D"], name="Alpha 20D", mode="lines"))
        if "Roll Alpha 60D" in roll_df.columns:
            fig.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll Alpha 60D"], name="Alpha 60D", mode="lines"))
        fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), title="Rolling Alpha")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll TE 20D"], name="TE 20D", mode="lines"))
        fig2.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll TE 60D"], name="TE 60D", mode="lines"))
        fig2.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), title="Rolling Tracking Error")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll Beta 60D"], name="Beta 60D", mode="lines"))
        if math.isfinite(beta_target):
            fig3.add_trace(go.Scatter(x=roll_df.index, y=[beta_target]*len(roll_df.index), name="Beta Target", mode="lines"))
        fig3.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), title="Rolling Beta (Discipline)")
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll Vol 20D"], name="Vol 20D", mode="lines"))
        fig4.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll Vol 60D"], name="Vol 60D", mode="lines"))
        fig4.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), title="Rolling Volatility")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.line_chart(roll_df)


# ============================================================
# Correlation Matrix Panel (All Waves)
# ============================================================
def panel_correlation_matrix(all_waves: List[str], mode: str, days: int = 365) -> None:
    # build returns matrix
    series = {}
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=days)
        if h is None or h.empty or "wave_ret" not in h.columns:
            continue
        r = pd.to_numeric(h["wave_ret"], errors="coerce")
        if r.dropna().shape[0] < 30:
            continue
        series[w] = r

    if not series:
        st.info("Not enough wave return series to compute correlations.")
        return

    df = pd.DataFrame(series).dropna(how="all")
    df = df.fillna(0.0)
    if df.shape[0] < 30 or df.shape[1] < 2:
        st.info("Not enough aligned data for correlation matrix.")
        return

    corr = df.corr()

    st.markdown("#### Correlation Matrix (daily returns)")
    st.dataframe(corr.style.format(lambda v: f"{v:.2f}"), use_container_width=True)

    # optional heatmap
    if go is not None:
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Corr"),
        ))
        fig.update_layout(height=min(900, 300 + 18 * corr.shape[0]), margin=dict(l=60, r=40, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Data Quality / Coverage Audit Panel
# ============================================================
def panel_data_quality_audit(selected_wave: str, mode: str, days: int = 365) -> None:
    hist = compute_wave_history(selected_wave, mode=mode, days=days)
    if hist is None or hist.empty:
        st.warning("No history available for data audit.")
        return

    # audit: length, missingness, nav sanity, return sanity
    nav_w = pd.to_numeric(hist.get("wave_nav", pd.Series(dtype=float)), errors="coerce")
    nav_b = pd.to_numeric(hist.get("bm_nav", pd.Series(dtype=float)), errors="coerce")
    rw = pd.to_numeric(hist.get("wave_ret", pd.Series(dtype=float)), errors="coerce")
    rb = pd.to_numeric(hist.get("bm_ret", pd.Series(dtype=float)), errors="coerce")

    n = int(len(hist))
    miss = {
        "Rows": n,
        "Wave NAV missing %": float(nav_w.isna().mean()) if n else np.nan,
        "BM NAV missing %": float(nav_b.isna().mean()) if n else np.nan,
        "Wave Ret missing %": float(rw.isna().mean()) if n else np.nan,
        "BM Ret missing %": float(rb.isna().mean()) if n else np.nan,
    }

    # nav monotonic sanity not required; but detect zeros/negatives
    nav_bad = {
        "Wave NAV min": float(nav_w.min()) if nav_w.dropna().shape[0] else np.nan,
        "BM NAV min": float(nav_b.min()) if nav_b.dropna().shape[0] else np.nan,
        "Wave NAV nonpositive rows": int((nav_w <= 0).sum()) if nav_w.dropna().shape[0] else 0,
        "BM NAV nonpositive rows": int((nav_b <= 0).sum()) if nav_b.dropna().shape[0] else 0,
    }

    # return spikes
    spikes = {
        "Wave Ret max": float(rw.max()) if rw.dropna().shape[0] else np.nan,
        "Wave Ret min": float(rw.min()) if rw.dropna().shape[0] else np.nan,
        "BM Ret max": float(rb.max()) if rb.dropna().shape[0] else np.nan,
        "BM Ret min": float(rb.min()) if rb.dropna().shape[0] else np.nan,
        "Wave Ret |>|10% count": int((rw.abs() > 0.10).sum()) if rw.dropna().shape[0] else 0,
        "BM Ret |>|10% count": int((rb.abs() > 0.10).sum()) if rb.dropna().shape[0] else 0,
    }

    audit_rows = []
    for k, v in miss.items():
        audit_rows.append({"Category": "Missingness", "Metric": k, "Value": fmt_pct(v) if " %" in k else str(v)})
    for k, v in nav_bad.items():
        audit_rows.append({"Category": "NAV sanity", "Metric": k, "Value": fmt_num(v, 4) if isinstance(v, float) else str(v)})
    for k, v in spikes.items():
        if "max" in k or "min" in k:
            audit_rows.append({"Category": "Return spikes", "Metric": k, "Value": fmt_pct(v)})
        else:
            audit_rows.append({"Category": "Return spikes", "Metric": k, "Value": str(v)})

    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)

    # flags
    flags = []
    if miss["Wave NAV missing %"] and (miss["Wave NAV missing %"] > 0.02):
        flags.append("Wave NAV has missing data >2%")
    if miss["BM NAV missing %"] and (miss["BM NAV missing %"] > 0.02):
        flags.append("Benchmark NAV has missing data >2%")
    if nav_bad["Wave NAV nonpositive rows"] > 0:
        flags.append("Wave NAV contains nonpositive values")
    if nav_bad["BM NAV nonpositive rows"] > 0:
        flags.append("Benchmark NAV contains nonpositive values")
    if spikes["Wave Ret |>|10% count"] > 2:
        flags.append("Wave returns show multiple >10% daily moves (verify data source / splits)")
    if spikes["BM Ret |>|10% count"] > 2:
        flags.append("Benchmark returns show multiple >10% daily moves (verify benchmark construction)")

    if flags:
        st.warning(" | ".join(flags))
    else:
        st.success("No major data quality flags detected in this window.")


# ============================================================
# Factor Decomposition Panel
# ============================================================
def _regress_factors(wave_ret: pd.Series, factor_ret: pd.DataFrame) -> Dict[str, float]:
    wave_ret = pd.to_numeric(wave_ret, errors="coerce")
    if wave_ret.dropna().shape[0] < 60 or factor_ret is None or factor_ret.empty:
        return {c: np.nan for c in (factor_ret.columns if factor_ret is not None else [])}

    df = pd.concat([wave_ret.rename("wave"), factor_ret], axis=1).dropna()
    if df.shape[0] < 60:
        return {c: np.nan for c in factor_ret.columns}

    y = df["wave"].values
    X = df[factor_ret.columns].values
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    except Exception:
        return {c: np.nan for c in factor_ret.columns}

    betas = beta[1:]
    return {col: float(b) for col, b in zip(factor_ret.columns, betas)}


def panel_factor_decomposition(selected_wave: str, mode: str, days: int = 365) -> None:
    hist = compute_wave_history(selected_wave, mode=mode, days=days)
    if hist is None or hist.empty or "wave_ret" not in hist.columns:
        st.warning("Wave history unavailable.")
        return

    factors_px = fetch_prices_daily(["SPY", "QQQ", "IWM", "TLT", "GLD"], days=days)
    if factors_px is None or factors_px.empty:
        st.warning("Factor price data unavailable.")
        return

    factor_ret = factors_px.pct_change().fillna(0.0)
    wave_ret = pd.to_numeric(hist["wave_ret"], errors="coerce").reindex(factor_ret.index).fillna(0.0)

    betas = _regress_factors(wave_ret, factor_ret)
    bdf = pd.DataFrame([{"Factor": k, "Beta": v} for k, v in betas.items()])
    bdf["Beta"] = bdf["Beta"].apply(lambda x: fmt_num(x, 2))
    st.dataframe(bdf, use_container_width=True)
    # diag_panels.py — WAVES Intelligence™ Institutional Console
# Panels for Diagnostics++: mode separation proof, rolling diagnostics,
# correlations, data audit, benchmark truth, holdings, factor decomposition.

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from ui_blocks import (
    show_df, fmt_pct, fmt_num, fmt_score,
    fetch_prices_daily, compute_spy_nav, google_quote_url,
)

from diag_core import (
    compute_wave_history,
    get_wave_holdings,
    get_benchmark_mix,
    compute_alpha_attribution,
    ret_from_nav,
    tracking_error,
    information_ratio,
    annualized_vol,
    max_drawdown,
    beta_vs_benchmark,
    get_beta_target_if_available,
    alpha_captured_daily,
)


# ============================================================
# Holdings Panel + Top10
# ============================================================
def panel_holdings_top10(selected_wave: str) -> None:
    hold = get_wave_holdings(selected_wave)
    if hold is None or hold.empty:
        st.warning("No holdings returned by engine for this wave.")
        return

    hold2 = hold.copy()
    if "Weight" in hold2.columns:
        hold2["Weight"] = pd.to_numeric(hold2["Weight"], errors="coerce")
    if "Ticker" in hold2.columns:
        hold2["Ticker"] = hold2["Ticker"].astype(str)

    top10 = hold2.sort_values("Weight", ascending=False).head(10) if "Weight" in hold2.columns else hold2.head(10)
    st.markdown("### 🧾 Top 10 Holdings (clickable)")
    for _, r in top10.iterrows():
        t = str(r.get("Ticker", ""))
        wgt = r.get("Weight", np.nan)
        nm = str(r.get("Name", t))
        if t:
            url = google_quote_url(t)
            st.markdown(f"- **[{t}]({url})** — {nm} — **{fmt_pct(wgt)}**")

    st.markdown("### Full Holdings")
    show_df(hold2, selected_wave, key=f"holdings_full_{selected_wave}")


# ============================================================
# Benchmark Truth Panel
# ============================================================
def panel_benchmark_truth(selected_wave: str, mode: str) -> None:
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Benchmark Mix (Engine)")
        bm_mix = get_benchmark_mix()
        if bm_mix is None or bm_mix.empty:
            st.info("No benchmark mix table returned by engine.")
        else:
            if "Wave" in bm_mix.columns:
                show_df(bm_mix[bm_mix["Wave"] == selected_wave], selected_wave, key=f"bm_mix_{selected_wave}")
            else:
                show_df(bm_mix, selected_wave, key=f"bm_mix_{selected_wave}")

    with colB:
        st.markdown("#### Attribution Snapshot (365D)")
        attrib = compute_alpha_attribution(selected_wave, mode=mode, days=365)
        if not attrib:
            st.info("Attribution unavailable.")
        else:
            rows = []
            for k, v in attrib.items():
                if any(x in k for x in ["Return", "Alpha", "Vol", "MaxDD", "TE", "Difficulty", "Captured"]):
                    rows.append({"Metric": k, "Value": fmt_pct(v)})
                elif ("IR" in k) or ("β" in k):
                    rows.append({"Metric": k, "Value": fmt_num(v, 2)})
                else:
                    rows.append({"Metric": k, "Value": fmt_num(v, 4)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ============================================================
# Mode Separation Proof (All Modes)
# ============================================================
def panel_mode_separation_proof(selected_wave: str, all_modes: List[str], days: int = 365) -> None:
    metrics_rows: List[Dict[str, Any]] = []
    nav_lines = {}

    for m in all_modes:
        hist = compute_wave_history(selected_wave, mode=m, days=days)
        if hist.empty or len(hist) < 2:
            metrics_rows.append({
                "Mode": m,
                "365D Return": np.nan,
                "365D Alpha": np.nan,
                "TE": np.nan,
                "IR": np.nan,
                "β_real": np.nan,
                "β_target": get_beta_target_if_available(m),
                "Vol": np.nan,
                "MaxDD": np.nan,
                "Alpha Captured (1D)": np.nan,
            })
            continue

        nav_w = hist["wave_nav"]
        nav_b = hist["bm_nav"]
        ret_w = pd.to_numeric(hist["wave_ret"], errors="coerce").fillna(0.0)
        ret_b = pd.to_numeric(hist["bm_ret"], errors="coerce").fillna(0.0)

        rw = ret_from_nav(nav_w, len(nav_w))
        rb = ret_from_nav(nav_b, len(nav_b))
        alpha = rw - rb if (math.isfinite(rw) and math.isfinite(rb)) else np.nan

        te = tracking_error(ret_w, ret_b)
        ir = information_ratio(nav_w, nav_b, te)
        beta_real = beta_vs_benchmark(ret_w, ret_b)
        beta_target = get_beta_target_if_available(m)

        vol = annualized_vol(ret_w)
        mdd = max_drawdown(nav_w)

        ac = alpha_captured_daily(hist, mode=m)
        ac1 = float(ac.dropna().iloc[-1]) if not ac.empty and ac.dropna().shape[0] >= 1 else np.nan

        metrics_rows.append({
            "Mode": m,
            "365D Return": rw,
            "365D Alpha": alpha,
            "TE": te,
            "IR": ir,
            "β_real": beta_real,
            "β_target": beta_target,
            "Vol": vol,
            "MaxDD": mdd,
            "Alpha Captured (1D)": ac1,
        })

        nav_lines[m] = nav_w.copy()

    df = pd.DataFrame(metrics_rows)
    st.markdown("#### Mode metrics (side-by-side)")
    st.dataframe(df.style.format({
        "365D Return": lambda v: fmt_pct(v),
        "365D Alpha": lambda v: fmt_pct(v),
        "TE": lambda v: fmt_pct(v),
        "IR": lambda v: fmt_num(v, 2),
        "β_real": lambda v: fmt_num(v, 2),
        "β_target": lambda v: fmt_num(v, 2),
        "Vol": lambda v: fmt_pct(v),
        "MaxDD": lambda v: fmt_pct(v),
        "Alpha Captured (1D)": lambda v: fmt_pct(v),
    }), use_container_width=True)

    st.markdown("#### NAV overlay (Wave NAV per mode)")
    if not nav_lines:
        st.info("No NAV series to plot.")
        return

    nav_df = pd.DataFrame(nav_lines).dropna(how="all")
    if nav_df.empty:
        st.info("No NAV series to plot.")
        return

    # normalize to 100
    nav_df2 = nav_df / nav_df.iloc[0] * 100.0

    if go is not None:
        fig = go.Figure()
        for c in nav_df2.columns:
            fig.add_trace(go.Scatter(x=nav_df2.index, y=nav_df2[c], name=c, mode="lines"))
        fig.update_layout(height=420, margin=dict(l=40, r=40, t=30, b=40), title="Mode Separation Proof — NAV (Indexed=100)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(nav_df2)


# ============================================================
# Rolling Diagnostics (Alpha/TE/Beta/Vol/Persistence)
# ============================================================
def _rolling_beta(w: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < window:
        return pd.Series(dtype=float)

    # rolling cov/var
    cov = df["w"].rolling(window).cov(df["b"])
    var = df["b"].rolling(window).var()
    beta = cov / var
    beta.name = f"beta_{window}"
    return beta


def _rolling_te(w: pd.Series, b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([w.rename("w"), b.rename("b")], axis=1).dropna()
    if df.shape[0] < window:
        return pd.Series(dtype=float)
    diff = df["w"] - df["b"]
    te = diff.rolling(window).std() * np.sqrt(252)
    te.name = f"te_{window}"
    return te


def _rolling_vol(w: pd.Series, window: int) -> pd.Series:
    w = pd.to_numeric(w, errors="coerce")
    if w.dropna().shape[0] < window:
        return pd.Series(dtype=float)
    vol = w.rolling(window).std() * np.sqrt(252)
    vol.name = f"vol_{window}"
    return vol


def _rolling_alpha_from_nav(nav_w: pd.Series, nav_b: pd.Series, window: int) -> pd.Series:
    # compute rolling window total return from NAV then alpha
    if nav_w is None or nav_b is None or len(nav_w) < window + 1 or len(nav_b) < window + 1:
        return pd.Series(dtype=float)
    nw = nav_w.astype(float)
    nb = nav_b.astype(float)
    rw = nw / nw.shift(window) - 1.0
    rb = nb / nb.shift(window) - 1.0
    alpha = (rw - rb).rename(f"alpha_{window}")
    return alpha


def _alpha_persistence(alpha_series: pd.Series, window: int) -> Dict[str, float]:
    a = pd.to_numeric(alpha_series, errors="coerce").dropna()
    if a.empty:
        return {"pos_frac": np.nan, "neg_frac": np.nan, "pos_streak": np.nan, "neg_streak": np.nan}

    pos = (a > 0).astype(int)
    neg = (a < 0).astype(int)
    pos_frac = float(pos.mean())
    neg_frac = float(neg.mean())

    # longest streak helper
    def longest_streak(x: pd.Series) -> int:
        best = 0
        cur = 0
        for v in x.values:
            if v == 1:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)

    pos_streak = float(longest_streak(pos))
    neg_streak = float(longest_streak(neg))

    return {"pos_frac": pos_frac, "neg_frac": neg_frac, "pos_streak": pos_streak, "neg_streak": neg_streak}


def panel_rolling_diagnostics(selected_wave: str, mode: str, days: int = 365) -> None:
    hist = compute_wave_history(selected_wave, mode=mode, days=days)
    if hist is None or hist.empty or len(hist) < 60:
        st.warning("Not enough history for rolling diagnostics (need ~60+ points).")
        return

    nav_w = hist["wave_nav"].astype(float)
    nav_b = hist["bm_nav"].astype(float)
    w = pd.to_numeric(hist["wave_ret"], errors="coerce").fillna(0.0)
    b = pd.to_numeric(hist["bm_ret"], errors="coerce").fillna(0.0)

    # rolling windows
    w_alpha_20 = _rolling_alpha_from_nav(nav_w, nav_b, 20)
    w_alpha_60 = _rolling_alpha_from_nav(nav_w, nav_b, 60)

    w_te_20 = _rolling_te(w, b, 20)
    w_te_60 = _rolling_te(w, b, 60)

    w_beta_60 = _rolling_beta(w, b, 60)
    w_vol_20 = _rolling_vol(w, 20)
    w_vol_60 = _rolling_vol(w, 60)

    beta_target = get_beta_target_if_available(mode)

    # persistence summary
    pers_20 = _alpha_persistence(w_alpha_20, 20)
    pers_60 = _alpha_persistence(w_alpha_60, 60)

    st.markdown("#### Rolling Summary (latest)")
    latest = {
        "Latest Roll Alpha (20D)": float(w_alpha_20.dropna().iloc[-1]) if w_alpha_20.dropna().shape[0] else np.nan,
        "Latest Roll Alpha (60D)": float(w_alpha_60.dropna().iloc[-1]) if w_alpha_60.dropna().shape[0] else np.nan,
        "Latest Roll TE (20D)": float(w_te_20.dropna().iloc[-1]) if w_te_20.dropna().shape[0] else np.nan,
        "Latest Roll TE (60D)": float(w_te_60.dropna().iloc[-1]) if w_te_60.dropna().shape[0] else np.nan,
        "Latest Roll Beta (60D)": float(w_beta_60.dropna().iloc[-1]) if w_beta_60.dropna().shape[0] else np.nan,
        "Beta Target (mode)": float(beta_target) if math.isfinite(beta_target) else np.nan,
        "Latest Roll Vol (20D)": float(w_vol_20.dropna().iloc[-1]) if w_vol_20.dropna().shape[0] else np.nan,
        "Latest Roll Vol (60D)": float(w_vol_60.dropna().iloc[-1]) if w_vol_60.dropna().shape[0] else np.nan,
        "Alpha Persistence 20D (+frac)": pers_20["pos_frac"],
        "Alpha Persistence 60D (+frac)": pers_60["pos_frac"],
        "Longest + Streak (20D alpha>0)": pers_20["pos_streak"],
        "Longest + Streak (60D alpha>0)": pers_60["pos_streak"],
    }
    latest_df = pd.DataFrame([latest]).T.reset_index()
    latest_df.columns = ["Metric", "Value"]

    def _fmt_value(metric: str, v: float) -> str:
        if any(k in metric for k in ["Alpha", "TE"]) and "frac" not in metric and "Streak" not in metric:
            return fmt_pct(v)
        if "Vol" in metric:
            return fmt_pct(v)
        if "Beta" in metric:
            return fmt_num(v, 2)
        if "frac" in metric:
            return fmt_pct(v)
        if "Streak" in metric:
            return fmt_num(v, 0)
        return fmt_num(v, 4)

    latest_df["Value"] = [ _fmt_value(str(m), float(v) if v is not None else np.nan) for m, v in zip(latest_df["Metric"], latest_df["Value"]) ]
    st.dataframe(latest_df, use_container_width=True)

    # beta drift flags
    drift_flag = ""
    if math.isfinite(beta_target) and w_beta_60.dropna().shape[0]:
        beta_last = float(w_beta_60.dropna().iloc[-1])
        drift = abs(beta_last - beta_target)
        if drift > 0.07:
            drift_flag = f"⚠️ Beta Drift: |β_real-β_target|={drift:.2f}"
    if drift_flag:
        st.warning(drift_flag)

    st.markdown("#### Rolling Charts")
    roll_df = pd.DataFrame({
        "Roll Alpha 20D": w_alpha_20,
        "Roll Alpha 60D": w_alpha_60,
        "Roll TE 20D": w_te_20,
        "Roll TE 60D": w_te_60,
        "Roll Beta 60D": w_beta_60,
        "Roll Vol 20D": w_vol_20,
        "Roll Vol 60D": w_vol_60,
    }).dropna(how="all")

    if roll_df.empty:
        st.info("Rolling chart data is empty after alignment.")
        return

    if go is not None:
        fig = go.Figure()
        if "Roll Alpha 20D" in roll_df.columns:
            fig.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll Alpha 20D"], name="Alpha 20D", mode="lines"))
        if "Roll Alpha 60D" in roll_df.columns:
            fig.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll Alpha 60D"], name="Alpha 60D", mode="lines"))
        fig.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), title="Rolling Alpha")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll TE 20D"], name="TE 20D", mode="lines"))
        fig2.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll TE 60D"], name="TE 60D", mode="lines"))
        fig2.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), title="Rolling Tracking Error")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll Beta 60D"], name="Beta 60D", mode="lines"))
        if math.isfinite(beta_target):
            fig3.add_trace(go.Scatter(x=roll_df.index, y=[beta_target]*len(roll_df.index), name="Beta Target", mode="lines"))
        fig3.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), title="Rolling Beta (Discipline)")
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll Vol 20D"], name="Vol 20D", mode="lines"))
        fig4.add_trace(go.Scatter(x=roll_df.index, y=roll_df["Roll Vol 60D"], name="Vol 60D", mode="lines"))
        fig4.update_layout(height=320, margin=dict(l=40, r=40, t=30, b=40), title="Rolling Volatility")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.line_chart(roll_df)


# ============================================================
# Correlation Matrix Panel (All Waves)
# ============================================================
def panel_correlation_matrix(all_waves: List[str], mode: str, days: int = 365) -> None:
    # build returns matrix
    series = {}
    for w in all_waves:
        h = compute_wave_history(w, mode=mode, days=days)
        if h is None or h.empty or "wave_ret" not in h.columns:
            continue
        r = pd.to_numeric(h["wave_ret"], errors="coerce")
        if r.dropna().shape[0] < 30:
            continue
        series[w] = r

    if not series:
        st.info("Not enough wave return series to compute correlations.")
        return

    df = pd.DataFrame(series).dropna(how="all")
    df = df.fillna(0.0)
    if df.shape[0] < 30 or df.shape[1] < 2:
        st.info("Not enough aligned data for correlation matrix.")
        return

    corr = df.corr()

    st.markdown("#### Correlation Matrix (daily returns)")
    st.dataframe(corr.style.format(lambda v: f"{v:.2f}"), use_container_width=True)

    # optional heatmap
    if go is not None:
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Corr"),
        ))
        fig.update_layout(height=min(900, 300 + 18 * corr.shape[0]), margin=dict(l=60, r=40, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Data Quality / Coverage Audit Panel
# ============================================================
def panel_data_quality_audit(selected_wave: str, mode: str, days: int = 365) -> None:
    hist = compute_wave_history(selected_wave, mode=mode, days=days)
    if hist is None or hist.empty:
        st.warning("No history available for data audit.")
        return

    # audit: length, missingness, nav sanity, return sanity
    nav_w = pd.to_numeric(hist.get("wave_nav", pd.Series(dtype=float)), errors="coerce")
    nav_b = pd.to_numeric(hist.get("bm_nav", pd.Series(dtype=float)), errors="coerce")
    rw = pd.to_numeric(hist.get("wave_ret", pd.Series(dtype=float)), errors="coerce")
    rb = pd.to_numeric(hist.get("bm_ret", pd.Series(dtype=float)), errors="coerce")

    n = int(len(hist))
    miss = {
        "Rows": n,
        "Wave NAV missing %": float(nav_w.isna().mean()) if n else np.nan,
        "BM NAV missing %": float(nav_b.isna().mean()) if n else np.nan,
        "Wave Ret missing %": float(rw.isna().mean()) if n else np.nan,
        "BM Ret missing %": float(rb.isna().mean()) if n else np.nan,
    }

    # nav monotonic sanity not required; but detect zeros/negatives
    nav_bad = {
        "Wave NAV min": float(nav_w.min()) if nav_w.dropna().shape[0] else np.nan,
        "BM NAV min": float(nav_b.min()) if nav_b.dropna().shape[0] else np.nan,
        "Wave NAV nonpositive rows": int((nav_w <= 0).sum()) if nav_w.dropna().shape[0] else 0,
        "BM NAV nonpositive rows": int((nav_b <= 0).sum()) if nav_b.dropna().shape[0] else 0,
    }

    # return spikes
    spikes = {
        "Wave Ret max": float(rw.max()) if rw.dropna().shape[0] else np.nan,
        "Wave Ret min": float(rw.min()) if rw.dropna().shape[0] else np.nan,
        "BM Ret max": float(rb.max()) if rb.dropna().shape[0] else np.nan,
        "BM Ret min": float(rb.min()) if rb.dropna().shape[0] else np.nan,
        "Wave Ret |>|10% count": int((rw.abs() > 0.10).sum()) if rw.dropna().shape[0] else 0,
        "BM Ret |>|10% count": int((rb.abs() > 0.10).sum()) if rb.dropna().shape[0] else 0,
    }

    audit_rows = []
    for k, v in miss.items():
        audit_rows.append({"Category": "Missingness", "Metric": k, "Value": fmt_pct(v) if " %" in k else str(v)})
    for k, v in nav_bad.items():
        audit_rows.append({"Category": "NAV sanity", "Metric": k, "Value": fmt_num(v, 4) if isinstance(v, float) else str(v)})
    for k, v in spikes.items():
        if "max" in k or "min" in k:
            audit_rows.append({"Category": "Return spikes", "Metric": k, "Value": fmt_pct(v)})
        else:
            audit_rows.append({"Category": "Return spikes", "Metric": k, "Value": str(v)})

    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)

    # flags
    flags = []
    if miss["Wave NAV missing %"] and (miss["Wave NAV missing %"] > 0.02):
        flags.append("Wave NAV has missing data >2%")
    if miss["BM NAV missing %"] and (miss["BM NAV missing %"] > 0.02):
        flags.append("Benchmark NAV has missing data >2%")
    if nav_bad["Wave NAV nonpositive rows"] > 0:
        flags.append("Wave NAV contains nonpositive values")
    if nav_bad["BM NAV nonpositive rows"] > 0:
        flags.append("Benchmark NAV contains nonpositive values")
    if spikes["Wave Ret |>|10% count"] > 2:
        flags.append("Wave returns show multiple >10% daily moves (verify data source / splits)")
    if spikes["BM Ret |>|10% count"] > 2:
        flags.append("Benchmark returns show multiple >10% daily moves (verify benchmark construction)")

    if flags:
        st.warning(" | ".join(flags))
    else:
        st.success("No major data quality flags detected in this window.")


# ============================================================
# Factor Decomposition Panel
# ============================================================
def _regress_factors(wave_ret: pd.Series, factor_ret: pd.DataFrame) -> Dict[str, float]:
    wave_ret = pd.to_numeric(wave_ret, errors="coerce")
    if wave_ret.dropna().shape[0] < 60 or factor_ret is None or factor_ret.empty:
        return {c: np.nan for c in (factor_ret.columns if factor_ret is not None else [])}

    df = pd.concat([wave_ret.rename("wave"), factor_ret], axis=1).dropna()
    if df.shape[0] < 60:
        return {c: np.nan for c in factor_ret.columns}

    y = df["wave"].values
    X = df[factor_ret.columns].values
    X_design = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

    try:
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    except Exception:
        return {c: np.nan for c in factor_ret.columns}

    betas = beta[1:]
    return {col: float(b) for col, b in zip(factor_ret.columns, betas)}


def panel_factor_decomposition(selected_wave: str, mode: str, days: int = 365) -> None:
    hist = compute_wave_history(selected_wave, mode=mode, days=days)
    if hist is None or hist.empty or "wave_ret" not in hist.columns:
        st.warning("Wave history unavailable.")
        return

    factors_px = fetch_prices_daily(["SPY", "QQQ", "IWM", "TLT", "GLD"], days=days)
    if factors_px is None or factors_px.empty:
        st.warning("Factor price data unavailable.")
        return

    factor_ret = factors_px.pct_change().fillna(0.0)
    wave_ret = pd.to_numeric(hist["wave_ret"], errors="coerce").reindex(factor_ret.index).fillna(0.0)

    betas = _regress_factors(wave_ret, factor_ret)
    bdf = pd.DataFrame([{"Factor": k, "Beta": v} for k, v in betas.items()])
    bdf["Beta"] = bdf["Beta"].apply(lambda x: fmt_num(x, 2))
    st.dataframe(bdf, use_container_width=True)
    # whatif_lab.py — WAVES Intelligence™ Institutional Console
# Shadow simulation lab for diagnostics only (does NOT change engine math).

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:
    go = None

import waves_engine as we

from ui_blocks import fetch_prices_daily, fmt_pct
from diag_core import get_wave_holdings


def safe_series(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    return s.copy()


def _weights_from_df(df: pd.DataFrame, ticker_col: str = "Ticker", weight_col: str = "Weight") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    w = df[[ticker_col, weight_col]].copy()
    w[ticker_col] = w[ticker_col].astype(str)
    w[weight_col] = pd.to_numeric(w[weight_col], errors="coerce").fillna(0.0)
    w = w.groupby(ticker_col, as_index=True)[weight_col].sum()
    total = float(w.sum())
    if total <= 0 or (not math.isfinite(total)):
        return pd.Series(dtype=float)
    return (w / total).sort_index()


def _regime_from_spy_60d(spy_nav: pd.Series) -> pd.Series:
    spy_nav = safe_series(spy_nav)
    if spy_nav.empty:
        return pd.Series(dtype=str)

    r60 = spy_nav / spy_nav.shift(60) - 1.0

    def label(x: float) -> str:
        if pd.isna(x):
            return "neutral"
        if x <= -0.12:
            return "panic"
        if x <= -0.04:
            return "downtrend"
        if x < 0.06:
            return "neutral"
        return "uptrend"

    return r60.apply(label)


def _vix_exposure_factor_series(vix: pd.Series, mode: str) -> pd.Series:
    vix = safe_series(vix).astype(float)
    if vix.empty:
        return pd.Series(dtype=float)

    def f(v: float) -> float:
        if pd.isna(v) or v <= 0:
            base = 1.0
        elif v < 15:
            base = 1.15
        elif v < 20:
            base = 1.05
        elif v < 25:
            base = 0.95
        elif v < 30:
            base = 0.85
        elif v < 40:
            base = 0.75
        else:
            base = 0.60
        if mode == "Alpha-Minus-Beta":
            base -= 0.05
        elif mode == "Private Logic":
            base += 0.05
        return float(np.clip(base, 0.5, 1.3))

    return vix.apply(f)


def _vix_safe_fraction_series(vix: pd.Series, mode: str) -> pd.Series:
    vix = safe_series(vix).astype(float)
    if vix.empty:
        return pd.Series(dtype=float)

    def g(v: float) -> float:
        if pd.isna(v) or v <= 0:
            base = 0.0
        elif v < 18:
            base = 0.00
        elif v < 24:
            base = 0.05
        elif v < 30:
            base = 0.15
        elif v < 40:
            base = 0.25
        else:
            base = 0.40
        if mode == "Alpha-Minus-Beta":
            base *= 1.5
        elif mode == "Private Logic":
            base *= 0.7
        return float(np.clip(base, 0.0, 0.8))

    return vix.apply(g)


@st.cache_data(show_spinner=False)
def simulate_whatif_nav(
    wave_name: str,
    mode: str,
    days: int,
    tilt_strength: float,
    vol_target: float,
    extra_safe_boost: float,
    exp_min: float,
    exp_max: float,
    freeze_benchmark: bool,
) -> pd.DataFrame:

    hold_df = get_wave_holdings(wave_name)
    weights = _weights_from_df(hold_df, "Ticker", "Weight")
    if weights.empty:
        return pd.DataFrame()

    tickers = list(weights.index)
    needed = set(tickers + ["SPY", "^VIX", "SGOV", "BIL", "SHY"])
    px = fetch_prices_daily(list(needed), days=days)
    if px.empty or "SPY" not in px.columns or "^VIX" not in px.columns:
        return pd.DataFrame()

    px = px.sort_index().ffill().bfill()
    if len(px) > days:
        px = px.iloc[-days:]

    rets = px.pct_change().fillna(0.0)
    w = weights.reindex(px.columns).fillna(0.0)

    spy_nav = (1.0 + rets["SPY"]).cumprod()
    regime = _regime_from_spy_60d(spy_nav)
    vix = px["^VIX"]

    vix_exposure = _vix_exposure_factor_series(vix, mode)
    vix_safe = _vix_safe_fraction_series(vix, mode)

    base_expo = 1.0
    try:
        base_map = getattr(we, "MODE_BASE_EXPOSURE", None)
        if isinstance(base_map, dict) and mode in base_map:
            base_expo = float(base_map[mode])
    except Exception:
        pass

    regime_exposure_map = {"panic": 0.80, "downtrend": 0.90, "neutral": 1.00, "uptrend": 1.10}
    try:
        rm = getattr(we, "REGIME_EXPOSURE", None)
        if isinstance(rm, dict):
            regime_exposure_map = {k: float(v) for k, v in rm.items()}
    except Exception:
        pass

    def regime_gate(mode_in: str, reg: str) -> float:
        try:
            rg = getattr(we, "REGIME_GATING", None)
            if isinstance(rg, dict) and mode_in in rg and reg in rg[mode_in]:
                return float(rg[mode_in][reg])
        except Exception:
            pass
        fallback = {
            "Standard": {"panic": 0.50, "downtrend": 0.30, "neutral": 0.10, "uptrend": 0.00},
            "Alpha-Minus-Beta": {"panic": 0.75, "downtrend": 0.50, "neutral": 0.25, "uptrend": 0.05},
            "Private Logic": {"panic": 0.40, "downtrend": 0.25, "neutral": 0.05, "uptrend": 0.00},
        }
        return float(fallback.get(mode_in, fallback["Standard"]).get(reg, 0.10))

    safe_ticker = "SGOV" if "SGOV" in rets.columns else ("BIL" if "BIL" in rets.columns else ("SHY" if "SHY" in rets.columns else "SPY"))
    safe_ret = rets[safe_ticker]
    mom60 = px / px.shift(60) - 1.0

    wave_ret: List[float] = []
    dates: List[pd.Timestamp] = []

    for dtt in rets.index:
        r = rets.loc[dtt]

        mom_row = mom60.loc[dtt] if dtt in mom60.index else None
        if mom_row is not None:
            mom_clipped = mom_row.reindex(px.columns).fillna(0.0).clip(lower=-0.30, upper=0.30)
            tilt = 1.0 + tilt_strength * mom_clipped
            ew = (w * tilt).clip(lower=0.0)
        else:
            ew = w.copy()

        ew_hold = ew.reindex(tickers).fillna(0.0)
        s = float(ew_hold.sum())
        if s > 0:
            rw = ew_hold / s
        else:
            rw = w.reindex(tickers).fillna(0.0)
            s2 = float(rw.sum())
            rw = (rw / s2) if s2 > 0 else rw

        port_risk_ret = float((r.reindex(tickers).fillna(0.0) * rw).sum())

        if len(wave_ret) >= 20:
            recent = np.array(wave_ret[-20:], dtype=float)
            realized = float(recent.std() * np.sqrt(252))
        else:
            realized = float(vol_target)

        vol_adj = 1.0
        if realized > 0 and math.isfinite(realized):
            vol_adj = float(np.clip(vol_target / realized, 0.7, 1.3))

        reg = str(regime.get(dtt, "neutral"))
        reg_expo = float(regime_exposure_map.get(reg, 1.0))
        vix_expo = float(vix_exposure.get(dtt, 1.0))
        vix_gate = float(vix_safe.get(dtt, 0.0))

        expo = float(np.clip(base_expo * reg_expo * vol_adj * vix_expo, exp_min, exp_max))

        sf = float(np.clip(regime_gate(mode, reg) + vix_gate + extra_safe_boost, 0.0, 0.95))
        rf = 1.0 - sf

        total = sf * float(safe_ret.get(dtt, 0.0)) + rf * expo * port_risk_ret

        # private logic shock shaping (display-only sandbox)
        if mode == "Private Logic" and len(wave_ret) >= 20:
            recent = np.array(wave_ret[-20:], dtype=float)
            daily_vol = float(recent.std())
            if daily_vol > 0 and math.isfinite(daily_vol):
                shock = 2.0 * daily_vol
                if total <= -shock:
                    total = total * 1.30
                elif total >= shock:
                    total = total * 0.70

        wave_ret.append(float(total))
        dates.append(pd.Timestamp(dtt))

    wave_ret_s = pd.Series(wave_ret, index=pd.Index(dates, name="Date"), name="whatif_ret")
    wave_nav = (1.0 + wave_ret_s).cumprod().rename("whatif_nav")

    out = pd.DataFrame({"whatif_nav": wave_nav, "whatif_ret": wave_ret_s})

    if freeze_benchmark:
        try:
            hist_eng = we.compute_history_nav(wave_name, mode=mode, days=days)
        except Exception:
            hist_eng = pd.DataFrame()
        if not hist_eng.empty and "bm_nav" in hist_eng.columns:
            bm_nav = pd.to_numeric(hist_eng["bm_nav"], errors="coerce").reindex(out.index).ffill().bfill()
            bm_ret = pd.to_numeric(hist_eng.get("bm_ret", pd.Series(index=out.index, dtype=float)), errors="coerce").reindex(out.index).fillna(0.0)
            out["bm_nav"] = bm_nav
            out["bm_ret"] = bm_ret
    else:
        # fallback benchmark = SPY
        spy_ret = rets["SPY"].reindex(out.index).fillna(0.0)
        spy_nav2 = (1.0 + spy_ret).cumprod()
        out["bm_nav"] = spy_nav2
        out["bm_ret"] = spy_ret

    return out


def panel_whatif_lab(selected_wave: str, mode: str, history_days: int) -> None:
    st.caption("This does NOT change engine math. It is a sandbox overlay simulation for diagnostics.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tilt_strength = st.slider("Tilt strength", 0.0, 1.0, 0.30, 0.05)
    with c2:
        vol_target = st.slider("Vol target (annual)", 0.05, 0.50, 0.20, 0.01)
    with c3:
        extra_safe = st.slider("Extra safe boost", 0.0, 0.40, 0.00, 0.01)
    with c4:
        freeze_bm = st.checkbox("Freeze benchmark (use engine BM)", value=True)

    c5, c6 = st.columns(2)
    with c5:
        exp_min = st.slider("Exposure min", 0.0, 1.5, 0.60, 0.05)
    with c6:
        exp_max = st.slider("Exposure max", 0.2, 2.0, 1.20, 0.05)

    if st.button("Run What-If Shadow Sim"):
        sim = simulate_whatif_nav(
            selected_wave,
            mode=mode,
            days=min(365, max(120, history_days)),
            tilt_strength=tilt_strength,
            vol_target=vol_target,
            extra_safe_boost=extra_safe,
            exp_min=exp_min,
            exp_max=exp_max,
            freeze_benchmark=freeze_bm,
        )

        if sim is None or sim.empty:
            st.warning("Simulation failed (insufficient prices).")
            return

        nav = sim["whatif_nav"]
        bm_nav = sim["bm_nav"] if "bm_nav" in sim.columns else None

        ret_total = float(nav.iloc[-1] / nav.iloc[0] - 1.0) if len(nav) >= 2 else float("nan")
        bm_total = float(bm_nav.iloc[-1] / bm_nav.iloc[0] - 1.0) if (bm_nav is not None and len(bm_nav) >= 2) else 0.0
        alpha_total = ret_total - bm_total

        st.markdown(f"**What-If Return:** {fmt_pct(ret_total)}   |   **What-If Alpha:** {fmt_pct(alpha_total)}")

        if go is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sim.index, y=sim["whatif_nav"], name="What-If NAV", mode="lines"))
            if "bm_nav" in sim.columns:
                fig.add_trace(go.Scatter(x=sim.index, y=sim["bm_nav"], name="Benchmark NAV", mode="lines"))
            fig.update_layout(height=380, margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            cols = ["whatif_nav"] + (["bm_nav"] if "bm_nav" in sim.columns else [])
            st.line_chart(sim[cols])