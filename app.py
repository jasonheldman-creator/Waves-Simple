# app.py  — WAVES Institutional Console (Internal Alpha Version)

import glob
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# PATHS / CONSTANTS
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
PERF_DIR = LOGS_DIR / "performance"
POS_DIR = LOGS_DIR / "positions"
WEIGHTS_FILE = BASE_DIR / "wave_weights.csv"

TRADING_DAYS_1M = 21
TRADING_DAYS_2M = 42
TRADING_DAYS_6M = 126
TRADING_DAYS_1Y = 252

# Internal alpha settings (can be tuned per acquirer)
BETA_TARGET_DEFAULT = 0.90
EXPECTED_DRIFT_ANNUAL = 0.07  # 7% p.a. long-run drift assumption


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def pct(x, digits: int = 2):
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


def load_wave_names_from_logs() -> list[str]:
    waves = set()
    if PERF_DIR.exists():
        for path in PERF_DIR.glob("*_performance_daily.csv"):
            name = path.name.replace("_performance_daily.csv", "")
            if name:
                waves.add(name)
    return sorted(waves)


def load_wave_names_from_weights() -> list[str]:
    if not WEIGHTS_FILE.exists():
        return []
    df = pd.read_csv(WEIGHTS_FILE)
    # Expecting a column like 'Wave' or 'Wave_Name' or 'Wave_Ticker_Weight'
    for col in ["Wave", "Wave_Name", "wave", "wave_name"]:
        if col in df.columns:
            return sorted(df[col].dropna().unique().tolist())

    # Fallback: parse from a combined column
    if "Wave_Ticker_Weight" in df.columns:
        waves = df["Wave_Ticker_Weight"].astype(str).str.split(",", expand=True)[0]
        return sorted(waves.dropna().unique().tolist())

    return []


def load_performance(wave_name: str) -> pd.DataFrame | None:
    """Load daily performance log for a given Wave."""
    if not PERF_DIR.exists():
        return None

    path = PERF_DIR / f"{wave_name}_performance_daily.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Try to normalize date column
    date_col = None
    for c in ["date", "Date", "as_of_date", "timestamp"]:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        return None

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "date"})

    # Try to locate daily return column
    return_col = None
    for c in ["daily_return", "return", "ret", "day_return"]:
        if c in df.columns:
            return_col = c
            break

    if return_col is None:
        # Fallback: derive from 'total_return' if present
        if "total_return" in df.columns:
            tr = df["total_return"].astype(float)
            df["daily_return"] = tr.pct_change().fillna(0.0)
        else:
            return None
    else:
        df["daily_return"] = df[return_col].astype(float)

    return df


def load_positions(wave_name: str) -> pd.DataFrame | None:
    """Load latest positions log for a given Wave."""
    if not POS_DIR.exists():
        return None

    pattern = str(POS_DIR / f"{wave_name}_positions_*.csv")
    paths = glob.glob(pattern)
    if not paths:
        return None

    latest_path = sorted(paths)[-1]
    df = pd.read_csv(latest_path)

    # Try to enforce expected columns
    # e.g. Ticker, Name, Weight, Value, Sector
    # If names differ, we just display what exists.
    return df


def compute_internal_alpha(perf_df: pd.DataFrame,
                           beta_target: float = BETA_TARGET_DEFAULT,
                           expected_drift_annual: float = EXPECTED_DRIFT_ANNUAL):
    """
    Compute internal alpha vs β-adjusted long-run drift.
    All alpha metrics are relative to a constant expected drift, not an index benchmark.
    """
    if perf_df is None or perf_df.empty:
        return None

    df = perf_df.copy()

    # Long-run daily drift from annual
    expected_drift_daily = (1.0 + expected_drift_annual) ** (1.0 / 252.0) - 1.0
    beta_adjusted_daily = expected_drift_daily * beta_target

    df["expected_drift_daily"] = beta_adjusted_daily
    df["alpha_daily_internal"] = df["daily_return"] - beta_adjusted_daily

    # Build cumulative series
    df["cum_return"] = (1.0 + df["daily_return"]).cumprod() - 1.0
    df["cum_expected"] = (1.0 + df["expected_drift_daily"]).cumprod() - 1.0
    df["cum_alpha_internal"] = df["cum_return"] - df["cum_expected"]

    return df, beta_adjusted_daily


def _window_mask(df: pd.DataFrame, days: int):
    if df is None or df.empty:
        return df.iloc[0:0]
    end = df["date"].max()
    start = end - timedelta(days=days)
    return df[df["date"] >= start]


def summarize_alpha_windows(df: pd.DataFrame,
                            beta_adjusted_daily: float,
                            days: dict[str, int]):
    """
    days: dict of label -> calendar day span
    Returns dict with return & alpha metrics.
    """
    out = {}
    if df is None or df.empty:
        for k in days.keys():
            out[f"{k}_ret"] = None
            out[f"{k}_alpha"] = None
        out["today_ret"] = None
        out["today_alpha"] = None
        out["total_ret"] = None
        out["total_alpha"] = None
        out["max_dd"] = None
        return out

    # Today (last row)
    last = df.iloc[-1]
    out["today_ret"] = float(last["daily_return"])
    out["today_alpha"] = float(last["alpha_daily_internal"])

    # Since inception
    out["total_ret"] = float(df["cum_return"].iloc[-1])
    out["total_alpha"] = float(df["cum_alpha_internal"].iloc[-1])

    # Max drawdown on total return series
    cum = 1.0 + df["cum_return"]
    roll_max = cum.cummax()
    dd = (cum / roll_max) - 1.0
    out["max_dd"] = float(dd.min())

    # Rolling windows
    for label, span in days.items():
        window = _window_mask(df, span)
        if window.empty:
            out[f"{label}_ret"] = None
            out[f"{label}_alpha"] = None
            continue

        # Cumulative realized return over the window
        realized = (1.0 + window["daily_return"]).prod() - 1.0
        # Expected drift over same length
        n = len(window)
        expected = (1.0 + beta_adjusted_daily) ** n - 1.0
        alpha = realized - expected

        out[f"{label}_ret"] = float(realized)
        out[f"{label}_alpha"] = float(alpha)

    return out


# -------------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------------

st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
)

st.markdown(
    """
    <style>
    .waves-hero {
        padding: 24px 32px;
        border-radius: 12px;
        background: radial-gradient(circle at 0% 0%, #00ff99 0, #00101f 40%, #000814 100%);
        color: #ffffff;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .waves-hero-title {
        font-size: 30px;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .waves-hero-sub {
        font-size: 14px;
        opacity: 0.9;
        max-width: 520px;
    }
    .hero-badge-row {
        margin-top: 12px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .hero-badge {
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.25);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.95;
    }
    .index-tile {
        background: rgba(0,0,0,0.4);
        padding: 10px 14px;
        border-radius: 8px;
        margin-top: 12px;
        font-size: 12px;
        width: 180px;
    }
    .index-label {
        opacity: 0.7;
        font-size: 11px;
    }
    .index-value {
        font-size: 15px;
        font-weight: 600;
        margin-top: 2px;
    }
    .index-pct.pos { color: #00ff99; }
    .index-pct.neg { color: #ff4d4f; }

    .waves-subtitle {
        font-size: 11px;
        opacity: 0.75;
        margin-top: 10px;
    }

    .metric-card > div {
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# SIDEBAR – ENGINE CONTROLS
# -------------------------------------------------------------------

st.sidebar.title("Engine Controls")

wave_names = load_wave_names_from_logs()
using_weights_list = False

if not wave_names:
    wave_names = load_wave_names_from_weights()
    using_weights_list = True

if not wave_names:
    st.sidebar.error("No Waves discovered yet. Run waves_engine.py to generate logs.")
    selected_wave = None
else:
    selected_wave = st.sidebar.selectbox("Select Wave", wave_names)

# Risk mode display only (logic handled in engine)
st.sidebar.markdown("### Mode")
mode = st.sidebar.radio(
    "Risk Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
    label_visibility="collapsed",
)

equity_exposure = st.sidebar.slider("Equity Exposure", 0, 100, 90, step=5)
st.sidebar.caption(f"Target β: {BETA_TARGET_DEFAULT:.2f} • Cash buffer: {100 - equity_exposure}%")

if using_weights_list:
    st.sidebar.warning("Using Wave list from wave_weights.csv (no performance logs found yet).")


# -------------------------------------------------------------------
# HERO
# -------------------------------------------------------------------

st.markdown(
    """
    <div class="waves-hero">
      <div class="waves-hero-title">WAVES INSTITUTIONAL CONSOLE</div>
      <div class="waves-hero-sub">
        Live engine view for <b>WAVES Intelligence™</b> — Adaptive Index Waves.
        This sample session is powered by the current WAVES internal engine; any
        acquirer can plug in their own trading models into the same rails.
      </div>
      <div class="hero-badge-row">
        <div class="hero-badge">Live Engine</div>
        <div class="hero-badge">Multi-Wave</div>
        <div class="hero-badge">Internal Alpha (β-Adjusted)</div>
      </div>

      <div style="position:absolute; top:18px; right:28px;">
        <div class="index-tile">
          <div class="index-label">S&P 500*</div>
          <div class="index-value">6870.40</div>
          <div class="index-pct pos">0.19%</div>
        </div>
        <div class="index-tile">
          <div class="index-label">VIX (Risk Pulse)*</div>
          <div class="index-value">15.41</div>
          <div class="index-pct pos">2.34%</div>
        </div>
        <div class="waves-subtitle">
          *Index values shown for layout only (demo); live feeds connect via acquirer data pipes.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# MAIN BODY
# -------------------------------------------------------------------

st.markdown("## WAVES Engine Dashboard")
st.markdown(
    """
    Live console for <b>WAVES Intelligence™</b> — Adaptive Index Waves.<br>
    All alpha metrics shown here are <b>internal, live-only, β-adjusted vs long-run drift</b>
    (no index benchmark alpha).
    """,
    unsafe_allow_html=True,
)

if selected_wave is None:
    st.info("Select a Wave on the left to view engine metrics.")
    st.stop()

perf_df_raw = load_performance(selected_wave)
pos_df = load_positions(selected_wave)

if perf_df_raw is None:
    st.warning(
        f"No performance log found yet for <b>{selected_wave}</b> in <code>logs/performance</code>.",
        icon="⚠️",
    )
    st.markdown(
        """
        The dashboard is running in <b>structure/demo mode</b>.  
        Run <code>waves_engine.py</code> locally to generate performance logs, then redeploy or
        upload the resulting CSVs to see full live charts and alpha.
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Compute internal alpha
perf_df, beta_adj_daily = compute_internal_alpha(perf_df_raw)
alpha_windows = summarize_alpha_windows(
    perf_df,
    beta_adj_daily,
    days={
        "30d": TRADING_DAYS_1M,
        "60d": TRADING_DAYS_2M,
        "6m": TRADING_DAYS_6M,
        "1y": TRADING_DAYS_1Y,
    },
)

# -------------------------------------------------------------------
# TOP METRIC CARDS
# -------------------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Total Return (Since Inception)",
        pct(alpha_windows["total_ret"]),
    )

with m2:
    st.metric(
        "Intraday Return (Today)",
        pct(alpha_windows["today_ret"]),
    )

with m3:
    st.metric(
        "Max Drawdown",
        pct(alpha_windows["max_dd"]),
    )

with m4:
    st.metric(
        "1-Day Internal Alpha vs Drift",
        pct(alpha_windows["today_alpha"]),
    )

# Second row: horizon alpha tiles
m5, m6, m7, m8 = st.columns(4)

with m5:
    st.metric(
        "30-Day Internal Alpha",
        pct(alpha_windows["30d_alpha"]),
        help="Cumulative return over last ~30 calendar days minus β-adjusted expected drift.",
    )

with m6:
    st.metric(
        "60-Day Internal Alpha",
        pct(alpha_windows["60d_alpha"]),
    )

with m7:
    st.metric(
        "6-Month Internal Alpha",
        pct(alpha_windows["6m_alpha"]),
    )

with m8:
    st.metric(
        "1-Year Internal Alpha",
        pct(alpha_windows["1y_alpha"]),
    )

st.info(
    "This view uses WAVES’ internal engine as a sample black box. "
    "In an acquisition, the acquirer’s own trading logic can be plugged into the "
    "same performance and alpha rails with no changes to the console.",
    icon="🔌",
)

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------

tab_overview, tab_alpha, tab_logs = st.tabs(
    ["Overview", "Alpha Dashboard", "Engine Logs"]
)

# === TAB 1: OVERVIEW =========================================================
with tab_overview:
    c1, c2 = st.columns([2, 1.3])

    with c1:
        st.markdown("### Performance Curve (Live Engine)")
        chart_df = perf_df[["date", "cum_return"]].set_index("date")
        chart_df = chart_df.rename(columns={"cum_return": "Wave Total Return"})
        st.line_chart(chart_df)

    with c2:
        st.markdown("### Exposure & Risk")
        st.metric("Equity Exposure", f"{equity_exposure} %")
        st.metric("Cash Buffer", f"{100 - equity_exposure} %")
        st.metric("Target β", f"{BETA_TARGET_DEFAULT:.2f}")

    st.markdown("### Top 10 Positions — Google Finance Links")
    if pos_df is None or pos_df.empty:
        st.info("No positions file found yet for this Wave in `logs/positions`.")
    else:
        # Try to normalize column names
        cols = {c.lower(): c for c in pos_df.columns}
        ticker_col = cols.get("ticker") or cols.get("symbol") or list(pos_df.columns)[0]
        weight_col = cols.get("weight") or cols.get("target_weight") or None
        name_col = cols.get("name") or cols.get("security") or None

        top = pos_df.copy()
        if weight_col:
            top = top.sort_values(weight_col, ascending=False)
        top = top.head(10)

        # Build Google Finance links
        display_rows = []
        for _, row in top.iterrows():
            ticker = str(row[ticker_col])
            name = str(row[name_col]) if name_col else ""
            weight = float(row[weight_col]) if weight_col else np.nan
            link = f"https://www.google.com/finance/quote/{ticker}:NYSE"
            display_rows.append(
                {
                    "Ticker": f"[{ticker}]({link})",
                    "Name": name,
                    "Weight": weight,
                }
            )
        display_df = pd.DataFrame(display_rows)
        st.markdown(
            display_df.to_markdown(index=False),
            unsafe_allow_html=True,
        )

# === TAB 2: ALPHA DASHBOARD ==================================================
with tab_alpha:
    st.markdown("### Alpha Capture Timeline (Internal)")
    alpha_curve = perf_df[["date", "cum_alpha_internal"]].set_index("date")
    alpha_curve = alpha_curve.rename(
        columns={"cum_alpha_internal": "Cumulative Internal Alpha"}
    )
    st.line_chart(alpha_curve)

    st.markdown("### Rolling Daily Internal Alpha")
    st.line_chart(
        perf_df.set_index("date")[["alpha_daily_internal"]].rename(
            columns={"alpha_daily_internal": "Daily Internal Alpha"}
        )
    )

# === TAB 3: ENGINE LOGS ======================================================
with tab_logs:
    st.markdown("### Raw Engine Feeds")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Performance Feed (tail)**")
        st.dataframe(perf_df.tail(50), use_container_width=True, height=420)

    with c2:
        st.markdown("**Positions Feed (latest)**")
        if pos_df is not None and not pos_df.empty:
            st.dataframe(pos_df, use_container_width=True, height=420)
        else:
            st.info("No positions feed available for this Wave yet.")

    st.markdown("---")
    st.caption(
        "All data shown here is sourced from WAVES Engine logs. "
        "Status (LIVE/SANDBOX/HYBRID) and regime tags are inferred from the underlying data regime."
    )
