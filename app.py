# app.py – WAVES Institutional Console (Hybrid Mode, Internal Alpha Windows)
#
# Features:
# - Hybrid: uses real logs if available, otherwise generates demo data
# - Internal, beta-adjusted alpha vs long-run drift (no benchmark index alpha)
# - Intraday (1-Day), 30-Day, 60-Day internal alpha captures
# - Performance curve, alpha charts, top-10 with Google links
# - SPX / VIX tiles and engine logs view

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf  # optional; console still works without it
except Exception:
    yf = None

# ---------------------------------------------------------------------
# PATHS / CONSTANTS
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
PERF_DIR = LOGS_DIR / "performance"
POS_DIR = LOGS_DIR / "positions"
WEIGHTS_PATH = BASE_DIR / "wave_weights.csv"

# Internal alpha settings
MU_ANNUAL = 0.07      # long-run annual drift
TRADING_DAYS = 252
MU_DAILY = (1.0 + MU_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0  # ≈ 0.00027
BETA_TARGET_DEFAULT = 0.90

# ---------------------------------------------------------------------
# STYLES
# ---------------------------------------------------------------------
st.set_page_config(page_title="WAVES Institutional Console", layout="wide")

st.markdown(
    """
    <style>
    body { background-color: #020617; color: #e5e7eb; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #020617 100%);
    }

    .waves-hero {
        background: radial-gradient(circle at 0% 0%, #0ea5e933, transparent 60%),
                    radial-gradient(circle at 100% 100%, #22c55e33, transparent 60%),
                    linear-gradient(90deg, #020617, #020617);
        border-radius: 1rem;
        padding: 1.35rem 1.6rem;
        border: 1px solid #111827;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        gap: 1.25rem;
    }
    .waves-hero-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }
    .waves-hero-sub {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-top: 0.35rem;
    }
    .pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        font-size: 0.7rem;
        border-radius: 999px;
        margin-right: 0.25rem;
        border: 1px solid #1f2937;
        background-color: rgba(15,23,42,0.8);
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }
    .pill-live { border-color: #22c55e; color: #22c55e; }
    .pill-demo { border-color: #eab308; color: #facc15; }

    .index-strip {
        margin-top: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.9rem;
    }
    .index-tile {
        background: #020617;
        border-radius: 0.7rem;
        padding: 0.55rem 0.75rem;
        border: 1px solid #111827;
        min-width: 130px;
    }
    .index-label {
        font-size: 0.72rem;
        color: #9ca3af;
    }
    .index-value {
        font-size: 0.95rem;
        font-weight: 600;
    }
    .index-pct {
        font-size: 0.8rem;
    }
    .index-pct.pos { color: #22c55e; }
    .index-pct.neg { color: #f97373; }

    .waves-metric {
        padding: 0.7rem 0.8rem;
        border-radius: 0.75rem;
        background: #020617;
        border: 1px solid #111827;
        margin-bottom: 0.6rem;
    }
    .waves-metric-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
    }
    .waves-metric-value {
        font-size: 1.25rem;
        font-weight: 600;
    }
    .waves-pos { color: #22c55e; }
    .waves-neg { color: #f97373; }

    .small-caption {
        font-size: 0.72rem;
        color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# SMALL UTILITIES
# ---------------------------------------------------------------------
def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:0.2f}%"


def fmt_pct_signed(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x*100:+0.2f}%"


def fmt_val(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.2f}"


def ticker_to_google_link(ticker: str) -> str:
    ticker = str(ticker).strip()
    if not ticker:
        return ""
    url = f"https://www.google.com/finance/quote/{ticker}"
    return f"[{ticker}]({url})"


# ---------------------------------------------------------------------
# DISCOVERY & LOADING
# ---------------------------------------------------------------------
def discover_waves():
    """Discover Waves from logs/weights; default to 9 Waves if nothing found."""
    waves = set()

    # from logs
    if PERF_DIR.exists():
        for p in PERF_DIR.glob("*_performance_daily*.csv"):
            name = p.name.replace("_performance_daily", "").replace(".csv", "")
            if name:
                waves.add(name)

    # from wave_weights
    if WEIGHTS_PATH.exists():
        try:
            ww = pd.read_csv(WEIGHTS_PATH)
            if "Wave" in ww.columns:
                for w in ww["Wave"].dropna().unique():
                    waves.add(str(w))
        except Exception:
            pass

    # fallback: nine Waves (for demo / structure mode)
    if not waves:
        waves = {
            "SP500_Wave",
            "Growth_Wave",
            "Income_Wave",
            "Future_Power_Energy_Wave",
            "Crypto_Wave",
            "Crypto_Income_Wave",
            "Quantum_Computing_Wave",
            "Small_Cap_Growth_Wave",
            "Clean_Transit_Infrastructure_Wave",
        }

    return sorted(waves)


def load_performance_from_logs(wave_name: str):
    if not PERF_DIR.exists():
        return None
    candidates = sorted(PERF_DIR.glob(f"{wave_name}_performance_daily*.csv"))
    if not candidates:
        return None
    path = candidates[-1]
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    # normalize date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"])
    else:
        df["date"] = pd.to_datetime(df.index)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def generate_demo_performance(wave_name: str, days: int = 260):
    rng = np.random.default_rng(abs(hash(wave_name)) % (2**32))
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")

    # mild drift, realistic vol
    drift = 0.00035
    vol = 0.012
    shocks = rng.normal(loc=drift, scale=vol, size=len(dates))
    daily = shocks
    total = (1 + daily).cumprod() - 1

    return pd.DataFrame(
        {
            "date": dates,
            "daily_return": daily,
            "total_return": total,
        }
    )


def ensure_return_columns(df: pd.DataFrame):
    df = df.copy()
    # daily_return
    if "daily_return" not in df.columns:
        for c in ["return", "strategy_return", "day_return"]:
            if c in df.columns:
                df["daily_return"] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                break
    if "daily_return" not in df.columns:
        # derive from value if possible
        for c in ["nav", "portfolio_value", "total_value", "equity_curve"]:
            if c in df.columns:
                val = pd.to_numeric(df[c], errors="coerce")
                df["daily_return"] = val.pct_change().fillna(0.0)
                break
    if "daily_return" not in df.columns:
        df["daily_return"] = 0.0

    # total_return
    if "total_return" not in df.columns:
        df = df.sort_values("date")
        df["total_return"] = (1 + df["daily_return"]).cumprod() - 1

    return df.sort_values("date").reset_index(drop=True)


def load_positions(wave_name: str):
    # from logs
    if POS_DIR.exists():
        candidates = sorted(POS_DIR.glob(f"{wave_name}_positions_*.csv"))
        if candidates:
            path = candidates[-1]
            try:
                return pd.read_csv(path)
            except Exception:
                pass
    # from wave_weights as fallback
    if WEIGHTS_PATH.exists():
        try:
            ww = pd.read_csv(WEIGHTS_PATH)
            if "Wave" in ww.columns and "Ticker" in ww.columns:
                sub = ww[ww["Wave"] == wave_name].copy()
                if sub.empty:
                    return None
                if "Weight" not in sub.columns:
                    sub["Weight"] = 1.0 / len(sub)
                sub["Name"] = ""
                sub["Sector"] = ""
                return sub[["Ticker", "Name", "Weight", "Sector"]]
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------
# INTERNAL ALPHA (β-ADJUSTED vs DRIFT)
# ---------------------------------------------------------------------
def compute_internal_alpha(df: pd.DataFrame, equity_exposure: float, beta_target: float):
    """
    Internal alpha vs β-adjusted drift:
        expected_daily = MU_DAILY * beta_target * equity_exposure
        alpha_daily = daily_return - expected_daily
    """
    df = ensure_return_columns(df)
    expected_daily = MU_DAILY * beta_target * equity_exposure

    df["expected_daily"] = expected_daily
    df["expected_total"] = (1 + df["expected_daily"]).cumprod() - 1
    df["alpha_daily"] = df["daily_return"] - df["expected_daily"]
    df["alpha_total"] = df["total_return"] - df["expected_total"]

    return df, expected_daily


def compute_window_alpha(df: pd.DataFrame, days: int, expected_daily: float):
    if df.empty:
        return None
    df = df.sort_values("date")
    tail = df.tail(days)
    if tail.empty:
        tail = df
    realized = (1 + tail["daily_return"]).prod() - 1
    expected = (1 + expected_daily) ** len(tail) - 1
    return float(realized - expected)


def compute_max_drawdown(total_return_series: pd.Series):
    if total_return_series is None or total_return_series.empty:
        return None
    cum_curve = 1 + total_return_series
    roll_max = cum_curve.cummax()
    dd = (cum_curve / roll_max) - 1.0
    return float(dd.min())


# ---------------------------------------------------------------------
# INDEX SNAPSHOT (SPX / VIX) WITH FALLBACK
# ---------------------------------------------------------------------
def get_index_snapshot():
    out = {
        "spx_value": 6870.40,
        "spx_pct": 0.0019,
        "vix_value": 15.41,
        "vix_pct": 0.0234,
    }
    if yf is None:
        return out
    try:
        spx = yf.Ticker("^GSPC").history(period="2d", interval="1d")
        vix = yf.Ticker("^VIX").history(period="2d", interval="1d")
        if not spx.empty:
            last = float(spx["Close"].iloc[-1])
            prev = float(spx["Close"].iloc[-2]) if len(spx) > 1 else last
            out["spx_value"] = last
            out["spx_pct"] = (last / prev) - 1
        if not vix.empty:
            last = float(vix["Close"].iloc[-1])
            prev = float(vix["Close"].iloc[-2]) if len(vix) > 1 else last
            out["vix_value"] = last
            out["vix_pct"] = (last / prev) - 1
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------
def main():
    # Sidebar
    st.sidebar.header("⚙️ Engine Controls")
    waves = discover_waves()
    selected_wave = st.sidebar.selectbox("Select Wave", waves, index=0)

    st.sidebar.markdown("**Mode (label only)**")
    mode = st.sidebar.radio(
        "Risk Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        label_visibility="collapsed",
        index=0,
    )

    equity_exposure_pct = st.sidebar.slider("Equity Exposure", 0, 100, 90, step=5)
    equity_exposure = equity_exposure_pct / 100.0
    st.sidebar.caption(f"Target β ≈ {BETA_TARGET_DEFAULT:.2f} · Cash buffer: {100 - equity_exposure_pct}%")

    # Data: Hybrid
    perf_logs = load_performance_from_logs(selected_wave)
    using_logs = perf_logs is not None and not perf_logs.empty
    if using_logs:
        perf_df = perf_logs
    else:
        perf_df = generate_demo_performance(selected_wave)

    perf_df, expected_daily = compute_internal_alpha(
        perf_df,
        equity_exposure=equity_exposure,
        beta_target=BETA_TARGET_DEFAULT,
    )

    perf_df = perf_df.sort_values("date").reset_index(drop=True)
    latest = perf_df.iloc[-1]

    total_return = float(latest["total_return"])
    intraday_return = float(latest["daily_return"])
    alpha_1d = float(latest["alpha_daily"])
    max_dd = compute_max_drawdown(perf_df["total_return"])

    alpha_30d = compute_window_alpha(perf_df, 30, expected_daily)
    alpha_60d = compute_window_alpha(perf_df, 60, expected_daily)

    pos_df = load_positions(selected_wave)
    idx = get_index_snapshot()

    # HERO
    spx_pct = idx["spx_pct"]
    vix_pct = idx["vix_pct"]
    hero_html = f"""
    <div class="waves-hero">
      <div>
        <div class="waves-hero-title">WAVES INSTITUTIONAL CONSOLE</div>
        <div style="margin-top:0.35rem;">
          <span class="pill pill-live">LIVE ENGINE</span>
          <span class="pill">MULTI-WAVE</span>
          <span class="pill">Internal Alpha · β-Adjusted Drift</span>
        </div>
        <div class="waves-hero-sub">
          Live console for WAVES Intelligence™ — Adaptive Index Waves.<br/>
          All alpha metrics shown here are internal, live-only, and β-adjusted vs long-run drift
          (no external benchmark alpha). Current view uses the WAVES internal engine as a sample
          black box; any acquirer can plug in their own models into the same rails.
        </div>
        <div class="index-strip">
          <div class="index-tile">
            <div class="index-label">Selected Wave 1D Internal Alpha</div>
            <div class="index-value {'waves-pos' if (alpha_1d or 0) >= 0 else 'waves-neg'}">
              {fmt_pct_signed(alpha_1d)}
            </div>
          </div>
          <div class="index-tile">
            <div class="index-label">S&P 500</div>
            <div class="index-value">{fmt_val(idx['spx_value'])}</div>
            <div class="index-pct {'pos' if spx_pct >= 0 else 'neg'}">{fmt_pct_signed(spx_pct)}</div>
          </div>
          <div class="index-tile">
            <div class="index-label">VIX (Risk Pulse)</div>
            <div class="index-value">{fmt_val(idx['vix_value'])}</div>
            <div class="index-pct {'pos' if vix_pct >= 0 else 'neg'}">{fmt_pct_signed(vix_pct)}</div>
          </div>
        </div>
      </div>
      <div style="min-width:230px;">
        <div class="waves-metric">
          <div class="waves-metric-label">Data Regime</div>
          <div class="waves-metric-value">{'LIVE LOGS' if using_logs else 'STRUCTURE / DEMO'}</div>
          <div class="small-caption">
            {'Sourced from logs/performance & logs/positions.'
             if using_logs
             else 'Engine in demo mode — run waves_engine.py & upload CSVs to go fully live.'}
          </div>
        </div>
        <div class="waves-metric">
          <div class="waves-metric-label">Last Console Refresh (UTC)</div>
          <div class="waves-metric-value">{datetime.utcnow().strftime("%Y-%m-%d %H:%M")}</div>
          <div class="small-caption">Internal clock reference only</div>
        </div>
      </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # TOP METRIC STRIP
    st.markdown("### WAVES Engine Dashboard")
    st.caption(
        "All alpha captures below are internal, β-adjusted vs long-run market drift. "
        "This represents the performance of the current WAVES internal engine as a sample black box."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="waves-metric">
              <div class="waves-metric-label">Total Return (Since Inception)</div>
              <div class="waves-metric-value {'waves-pos' if total_return >= 0 else 'waves-neg'}">
                {fmt_pct_signed(total_return)}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="waves-metric">
              <div class="waves-metric-label">Intraday Return (Today)</div>
              <div class="waves-metric-value {'waves-pos' if intraday_return >= 0 else 'waves-neg'}">
                {fmt_pct_signed(intraday_return)}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="waves-metric">
              <div class="waves-metric-label">Max Drawdown</div>
              <div class="waves-metric-value waves-neg">
                {fmt_pct(max_dd)}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="waves-metric">
              <div class="waves-metric-label">1-Day Internal Alpha</div>
              <div class="waves-metric-value {'waves-pos' if (alpha_1d or 0) >= 0 else 'waves-neg'}">
                {fmt_pct_signed(alpha_1d)}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"""
            <div class="waves-metric">
              <div class="waves-metric-label">β-Adjusted Expected Drift (Daily)</div>
              <div class="waves-metric-value">
                {fmt_pct(expected_daily)}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # SECOND ROW: WINDOW ALPHAS (30D / 60D)
    c6, c7 = st.columns(2)
    with c6:
        st.markdown(
            f"""
            <div class="waves-metric">
              <div class="waves-metric-label">30-Day Internal Alpha</div>
              <div class="waves-metric-value {'waves-pos' if (alpha_30d or 0) >= 0 else 'waves-neg'}">
                {fmt_pct_signed(alpha_30d)}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c7:
        st.markdown(
            f"""
            <div class="waves-metric">
              <div class="waves-metric-label">60-Day Internal Alpha</div>
              <div class="waves-metric-value {'waves-pos' if (alpha_60d or 0) >= 0 else 'waves-neg'}">
                {fmt_pct_signed(alpha_60d)}
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    if not using_logs:
        st.info(
            "No performance log found yet for this Wave in `logs/performance`.\n\n"
            "The console is currently running in **structure/demo mode** using synthetic data. "
            "Run `waves_engine.py` locally and sync its CSV logs to go fully live.",
            icon="ℹ️",
        )

    # TABS
    tab_overview, tab_alpha, tab_logs = st.tabs(["Overview", "Alpha Dashboard", "Engine Logs"])

    # OVERVIEW
    with tab_overview:
        left, right = st.columns([2.4, 1.6])

        with left:
            st.markdown("#### Performance Curve (Live Engine / Demo)")
            perf_plot = perf_df.set_index("date")[["total_return"]]
            perf_plot.columns = ["Total Return"]
            st.line_chart(perf_plot)

            st.markdown("#### Top 10 Positions — Clickable Google Links")
            if pos_df is not None and not pos_df.empty:
                dfp = pos_df.copy()
                cols_lower = {c.lower(): c for c in dfp.columns}
                ticker_col = cols_lower.get("ticker", list(dfp.columns)[0])
                weight_col = cols_lower.get("weight", None)
                name_col = cols_lower.get("name", None)
                sector_col = cols_lower.get("sector", None)
                value_col = cols_lower.get("value", None)

                keep_cols = [ticker_col]
                if name_col:
                    keep_cols.append(name_col)
                if weight_col:
                    keep_cols.append(weight_col)
                if value_col:
                    keep_cols.append(value_col)
                if sector_col:
                    keep_cols.append(sector_col)

                dfp = dfp[keep_cols].copy()

                if weight_col and weight_col in dfp.columns:
                    dfp = dfp.sort_values(weight_col, ascending=False)

                dfp = dfp.head(10).reset_index(drop=True)

                # Create link column
                dfp["Ticker"] = dfp[ticker_col].apply(ticker_to_google_link)

                rename = {ticker_col: "Ticker"}
                if name_col:
                    rename[name_col] = "Name"
                if weight_col:
                    rename[weight_col] = "Weight"
                if value_col:
                    rename[value_col] = "Value"
                if sector_col:
                    rename[sector_col] = "Sector"
                dfp.rename(columns=rename, inplace=True)

                table_md = dfp.to_markdown(index=False)
                st.markdown(table_md)
            else:
                st.info("No positions file found yet in `logs/positions` for this Wave.")

        with right:
            st.markdown("#### Exposure & Risk Profile")
            risk_df = pd.DataFrame(
                {
                    "Metric": ["Equity Exposure", "Cash Buffer", "Target β"],
                    "Value": [f"{equity_exposure_pct}%", f"{100 - equity_exposure_pct}%", f"{BETA_TARGET_DEFAULT:.2f}"],
                }
            ).set_index("Metric")
            st.table(risk_df)

            st.markdown("#### Total vs Expected Drift")
            drift_plot = perf_df.set_index("date")[["total_return", "expected_total"]]
            drift_plot.columns = ["Total Return", "Expected Drift"]
            st.line_chart(drift_plot)

    # ALPHA DASHBOARD
    with tab_alpha:
        st.markdown("#### Internal Alpha Windows (β-Adjusted · Drift-Relative)")
        alpha_rows = [
            ["1-Day", fmt_pct_signed(alpha_1d)],
            ["30-Day", fmt_pct_signed(alpha_30d)],
            ["60-Day", fmt_pct_signed(alpha_60d)],
        ]
        alpha_table = pd.DataFrame(alpha_rows, columns=["Window", "Internal Alpha"])
        st.table(alpha_table.set_index("Window"))

        st.markdown("#### Cumulative Internal Alpha Timeline")
        alpha_cum = perf_df.set_index("date")[["alpha_total"]]
        alpha_cum.columns = ["Cumulative Internal Alpha"]
        st.line_chart(alpha_cum)

        st.markdown("#### Rolling Daily Internal Alpha")
        alpha_daily = perf_df.set_index("date")[["alpha_daily"]]
        alpha_daily.columns = ["Alpha (Daily)"]
        st.line_chart(alpha_daily)

        st.caption(
            "Internal alpha is computed as realized Wave return minus β-adjusted long-run drift "
            f"({MU_ANNUAL*100:0.1f}% annual assumption, {TRADING_DAYS} trading days/year, β={BETA_TARGET_DEFAULT:.2f}, "
            "scaled by Wave equity exposure)."
        )

    # LOGS
    with tab_logs:
        st.markdown("#### Engine Feeds")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Performance Feed (tail)**")
            st.dataframe(perf_df.tail(50), use_container_width=True)

        with c2:
            st.markdown("**Positions Feed (latest)**")
            if pos_df is not None and not pos_df.empty:
                st.dataframe(pos_df.tail(50), use_container_width=True)
            else:
                st.info("No positions feed found for this Wave.")

        st.markdown("---")
        st.caption(
            "All views above are powered by the WAVES Engine logs when present. "
            "If logs are missing, the console switches to structure/demo mode with synthetic data, "
            "keeping the full experience live for presentations."
        )


if __name__ == "__main__":
    main()
