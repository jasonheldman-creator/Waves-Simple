import os
import glob
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ================================
# High-level configuration
# ================================
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple dark-mode theming via CSS injection.
st.markdown(
    """
    <style>
    .stApp {
        background-color: #050816;
        color: #f5f6fa;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9fa6b2 !important;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #e5f9ff !important;
    }
    .section-card {
        background: #0b1020;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #1f2937;
        box-shadow: 0 0 25px rgba(0, 0, 0, 0.45);
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 600;
        color: #e5f9ff;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #9fa6b2;
    }
    .wave-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border: 1px solid #1f2937;
        background: rgba(15,23,42,0.9);
        color: #a5b4fc;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border: 1px solid #1f2937;
        margin-left: 0.4rem;
        color: #6ee7b7;
    }
    .alpha-positive {
        color: #22c55e !important;
        font-weight: 600;
    }
    .alpha-negative {
        color: #f97373 !important;
        font-weight: 600;
    }
    table.top10-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    table.top10-table thead tr {
        background-color: #111827;
    }
    table.top10-table th, table.top10-table td {
        padding: 0.35rem 0.55rem;
        border-bottom: 1px solid #1f2937;
        text-align: left;
    }
    table.top10-table th {
        font-weight: 600;
        color: #e5f9ff;
    }
    table.top10-table td {
        color: #f9fafb;
    }
    table.top10-table a {
        color: #38bdf8;
        text-decoration: none;
        font-weight: 500;
    }
    table.top10-table a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================================
# Paths & constants
# ================================
BASE_DIR = Path(".")
LOGS_PERF_DIR = BASE_DIR / "logs" / "performance"
LOGS_POS_DIR = BASE_DIR / "logs" / "positions"

MODES = {
    "Standard": {"beta_target": 0.90, "drift_annual": 0.07},
    "Alpha-Minus-Beta": {"beta_target": 0.80, "drift_annual": 0.06},
    "Private Logic™": {"beta_target": 1.05, "drift_annual": 0.09},
}


# ================================
# Utility helpers
# ================================

def normalize_key(s: str) -> str:
    """Loose normalizer so 'AI Wave' and 'AI_Wave' and 'aiwave' all match."""
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def discover_waves_from_logs() -> list:
    """
    Discover wave names based on performance logs; fall back to fixed lineup
    so the console never comes up blank.
    """
    if LOGS_PERF_DIR.exists():
        pattern = str(LOGS_PERF_DIR / "*_performance_daily.csv")
        files = glob.glob(pattern)
    else:
        files = []

    wave_names = sorted(
        {
            os.path.basename(f).replace("_performance_daily.csv", "")
            for f in files
        }
    )

    if wave_names:
        return wave_names

    # Fallback (only used if no logs yet)
    return [
        "S&P 500 Wave",
        "Growth Wave",
        "Infinity Wave",
        "Income Wave",
        "Future Power & Energy Wave",
        "Crypto Income Wave",
        "Quantum Computing Wave",
        "Clean Transit-Infrastructure Wave",
        "Small/Mid Growth Wave",
    ]


def get_latest_positions_file_for_wave(wave_name: str) -> Path | None:
    """
    Find the latest positions file for a Wave, using a *loose* name match.

    This fixes the common problem where display name = 'AI Wave'
    but file prefix = 'AI_Wave'.
    """
    if not LOGS_POS_DIR.exists():
        return None

    pattern = str(LOGS_POS_DIR / "*_positions_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None

    target_key = normalize_key(wave_name)
    candidates = []

    for f in files:
        base = os.path.basename(f)
        # <prefix>_positions_YYYYMMDD.csv
        if "_positions_" not in base:
            continue
        prefix = base.split("_positions_")[0]
        if normalize_key(prefix) == target_key:
            # Parse date for ordering
            try:
                date_str = base.split("_positions_")[1].split(".csv")[0]
                dt = datetime.strptime(date_str, "%Y%m%d")
            except Exception:
                dt = datetime.min
            candidates.append((dt, f))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return Path(candidates[-1][1])


def load_positions_top10(wave_name: str) -> pd.DataFrame | None:
    """
    Load the latest positions file for a Wave and return the Top 10 holdings
    with clickable Google Finance ticker links.

    Columns returned:
      Ticker (plain)
      Name
      Sector
      Weight% (string)
      TickerLink (HTML anchor, used for rendering)
    """
    latest_file = get_latest_positions_file_for_wave(wave_name)
    if latest_file is None or not latest_file.exists():
        return None

    try:
        df = pd.read_csv(latest_file)
    except Exception:
        return None

    if df.empty:
        return None

    cols = {c.lower(): c for c in df.columns}

    ticker_col = cols.get("ticker") or cols.get("symbol") or list(df.columns)[0]
    weight_candidates = [
        "weight",
        "portfolio_weight",
        "target_weight",
        "actual_weight",
        "weight_pct",
        "weight_percent",
    ]
    weight_col = None
    for cand in weight_candidates:
        if cand in cols:
            weight_col = cols[cand]
            break

    if weight_col is None:
        df["__weight__"] = 1.0 / len(df)
        weight_col = "__weight__"

    name_col = cols.get("name") or cols.get("security_name")
    sector_col = cols.get("sector")

    df = df.copy()
    df = df.sort_values(by=weight_col, ascending=False).head(10)

    def google_url(ticker: str) -> str:
        t = str(ticker).strip().upper()
        if not t or t == "NAN":
            return ""
        return f"https://www.google.com/finance/quote/{t}"

    out = pd.DataFrame()
    out["Ticker"] = df[ticker_col].astype(str).str.upper()
    out["Name"] = df[name_col].astype(str) if name_col else ""
    out["Sector"] = df[sector_col].astype(str) if sector_col else ""
    out["Weight%"] = (df[weight_col].astype(float) * 100.0).round(2).astype(str) + "%"

    # HTML anchor for clickable ticker
    out["TickerLink"] = out["Ticker"].apply(
        lambda t: f'<a href="{google_url(t)}" target="_blank">{t}</a>' if t else ""
    )

    return out


def infer_return_and_benchmark_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lower = {c.lower(): c for c in df.columns}

    return_candidates = [
        "daily_return",
        "return",
        "portfolio_return",
        "pct_return",
        "r",
    ]
    bench_candidates = [
        "benchmark_return",
        "spx_return",
        "index_return",
        "benchmark_pct_return",
        "b",
    ]

    ret_col = None
    bench_col = None

    for cand in return_candidates:
        if cand in lower:
            ret_col = lower[cand]
            break

    for cand in bench_candidates:
        if cand in lower:
            bench_col = lower[cand]
            break

    return ret_col, bench_col


def generate_demo_performance(
    wave_name: str,
    mode_label: str,
    n_days: int = 120,
) -> pd.DataFrame:
    mode_cfg = MODES.get(mode_label, MODES["Standard"])
    beta_target = mode_cfg["beta_target"]
    drift_annual = mode_cfg["drift_annual"]

    seed = abs(hash((wave_name, mode_label))) % (2**32)
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(end=datetime.now(), periods=n_days)

    mu_bench_annual = 0.08
    sigma_annual = 0.16

    mu_bench_daily = mu_bench_annual / 252.0
    sigma_daily = sigma_annual / np.sqrt(252.0)

    bench_rets = rng.normal(loc=mu_bench_daily, scale=sigma_daily, size=len(dates))

    alpha_drift_daily = drift_annual / 252.0
    idio_vol = 0.07 / np.sqrt(252.0)
    idio_noise = rng.normal(loc=0.0, scale=idio_vol, size=len(dates))

    wave_rets = beta_target * bench_rets + alpha_drift_daily + idio_noise

    df = pd.DataFrame(
        {
            "date": dates,
            "benchmark_return": bench_rets,
            "portfolio_return": wave_rets,
        }
    )

    return df


def load_wave_performance(
    wave_name: str,
    mode_label: str,
) -> tuple[pd.DataFrame, bool]:
    is_live = False
    perf_file = LOGS_PERF_DIR / f"{wave_name}_performance_daily.csv"

    if perf_file.exists():
        try:
            df = pd.read_csv(perf_file)
            if df.empty:
                raise ValueError("Empty performance log")

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            elif "Date" in df.columns:
                df["date"] = pd.to_datetime(df["Date"])
            else:
                start_date = datetime.now() - timedelta(days=len(df))
                df["date"] = pd.bdate_range(start=start_date, periods=len(df))

            ret_col, bench_col = infer_return_and_benchmark_cols(df)
            if ret_col is None or bench_col is None:
                raise ValueError("Could not infer return/benchmark columns")

            df = df.sort_values("date").reset_index(drop=True)
            df["portfolio_return"] = df[ret_col].astype(float)
            df["benchmark_return"] = df[bench_col].astype(float)

            is_live = True
            return df[["date", "portfolio_return", "benchmark_return"]], is_live
        except Exception:
            pass

    df_demo = generate_demo_performance(wave_name, mode_label)
    return df_demo, is_live


def compute_alpha_windows(
    df: pd.DataFrame,
    mode_label: str,
) -> pd.DataFrame:
    mode_cfg = MODES.get(mode_label, MODES["Standard"])
    beta_target = mode_cfg["beta_target"]

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df["alpha_1d"] = df["portfolio_return"] - beta_target * df["benchmark_return"]

    df["alpha_30d"] = df["alpha_1d"].rolling(window=30, min_periods=1).sum()
    df["alpha_60d"] = df["alpha_1d"].rolling(window=60, min_periods=1).sum()

    df["ret_30d"] = (1.0 + df["portfolio_return"]).rolling(30, min_periods=1).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )
    df["ret_60d"] = (1.0 + df["portfolio_return"]).rolling(60, min_periods=1).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )

    df["bench_30d"] = (1.0 + df["benchmark_return"]).rolling(30, min_periods=1).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )
    df["bench_60d"] = (1.0 + df["benchmark_return"]).rolling(60, min_periods=1).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )

    return df


def compute_drawdown(df: pd.DataFrame) -> tuple[float, pd.Series]:
    df = df.copy().sort_values("date")
    cum = (1.0 + df["portfolio_return"]).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    max_dd = dd.min() * 100.0
    return float(max_dd), dd


@st.cache_data(show_spinner=False)
def get_spx_vix_tiles() -> dict:
    data = {
        "SPX": {"label": "S&P 500", "value": None, "change": None},
        "VIX": {"label": "VIX", "value": None, "change": None},
    }

    try:
        import yfinance as yf

        for symbol, key in [("^GSPC", "SPX"), ("^VIX", "VIX")]:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if hist.empty:
                continue
            last_close = float(hist["Close"].iloc[-1])
            if len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
                change_pct = (last_close / prev_close - 1.0) * 100.0
            else:
                change_pct = 0.0
            data[key]["value"] = last_close
            data[key]["change"] = change_pct

    except Exception:
        pass

    return data


def format_pct(x: float | None, decimals: int = 2) -> str:
    if x is None or np.isnan(x):
        return "—"
    return f"{x:.{decimals}f}%"


def compute_alpha_summary_for_wave(
    wave_name: str,
    mode_label: str,
) -> dict:
    perf_df_raw, _live_flag = load_wave_performance(wave_name, mode_label)
    perf_df = compute_alpha_windows(perf_df_raw, mode_label)

    if perf_df.empty:
        return {
            "Wave": wave_name,
            "Intraday α": np.nan,
            "1D α": np.nan,
            "30D α": np.nan,
            "60D α": np.nan,
            "1Y α": np.nan,
        }

    perf_df = perf_df.sort_values("date").reset_index(drop=True)

    latest = perf_df.iloc[-1]
    intraday_alpha = latest["alpha_1d"]

    if len(perf_df) >= 2:
        one_day_alpha = perf_df["alpha_1d"].iloc[-2]
    else:
        one_day_alpha = intraday_alpha

    alpha_30d = latest["alpha_30d"]
    alpha_60d = latest["alpha_60d"]

    window = perf_df["alpha_1d"].tail(252)
    alpha_1y = window.sum()

    return {
        "Wave": wave_name,
        "Intraday α": intraday_alpha * 100.0,
        "1D α": one_day_alpha * 100.0,
        "30D α": alpha_30d * 100.0,
        "60D α": alpha_60d * 100.0,
        "1Y α": alpha_1y * 100.0,
    }


# ================================
# Sidebar controls
# ================================
waves = discover_waves_from_logs()

with st.sidebar:
    st.markdown("### WAVES Intelligence™")
    st.markdown("#### Institutional Console")

    selected_wave = st.selectbox(
        "Select Wave",
        options=waves,
        index=0,
    )

    selected_mode = st.radio(
        "Mode",
        options=list(MODES.keys()),
        index=0,
        help="Display logic adapts to Standard, Alpha-Minus-Beta, and Private Logic™ targets.",
    )

    st.markdown("---")
    st.markdown(
        "This console is running in **Hybrid Mode**:\n"
        "- Uses real engine logs when available\n"
        "- Auto-generates synthetic demo performance when logs are missing\n"
        "- Never renders a blank screen"
    )


# ================================
# Load data for selected Wave/mode
# ================================
perf_df_raw, is_live = load_wave_performance(selected_wave, selected_mode)
perf_df = compute_alpha_windows(perf_df_raw, selected_mode)
max_dd_pct, dd_series = compute_drawdown(perf_df)

latest = perf_df.iloc[-1]

mode_cfg = MODES[selected_mode]
beta_target = mode_cfg["beta_target"]
drift_annual = mode_cfg["drift_annual"]
drift_daily = drift_annual / 252.0 * 100.0

one_day_ret = latest["portfolio_return"] * 100.0
one_day_alpha = latest["alpha_1d"] * 100.0
alpha_30d = latest["alpha_30d"] * 100.0
alpha_60d = latest["alpha_60d"] * 100.0
ret_30d = latest["ret_30d"] * 100.0
ret_60d = latest["ret_60d"] * 100.0
bench_30d = latest["bench_30d"] * 100.0
bench_60d = latest["bench_60d"] * 100.0

cum_port = (1.0 + perf_df["portfolio_return"]).cumprod()
cum_bench = (1.0 + perf_df["benchmark_return"]).cumprod()
since_inc_ret = (cum_port.iloc[-1] - 1.0) * 100.0
since_inc_bench = (cum_bench.iloc[-1] - 1.0) * 100.0


# ================================
# Hero header & market tiles
# ================================
spx_vix = get_spx_vix_tiles()

st.markdown(
    f"""
    <div class="section-card" style="margin-bottom: 0.8rem;">
        <div class="hero-title">
            WAVES Intelligence™ Institutional Console
        </div>
        <div class="hero-subtitle">
            Live multi-wave, mode-aware monitoring &mdash; engine output visualized for institutional use.
        </div>
        <div style="margin-top: 0.6rem;">
            <span class="wave-badge">{selected_wave}</span>
            <span class="mode-badge">{selected_mode}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_spx, col_vix, col_mode = st.columns([1, 1, 1.2])

with col_spx:
    tile = spx_vix["SPX"]
    label = tile["label"]
    val = tile["value"]
    chg = tile["change"]
    chg_str = format_pct(chg) if chg is not None else "—"
    chg_class = "alpha-positive" if (chg or 0) >= 0 else "alpha-negative"

    st.markdown(
        f"""
        <div class="section-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">
                {f"{val:,.0f}" if val is not None else "—"}
            </div>
            <div class="{chg_class}" style="font-size:0.85rem;margin-top:0.15rem;">
                {chg_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_vix:
    tile = spx_vix["VIX"]
    label = tile["label"]
    val = tile["value"]
    chg = tile["change"]
    chg_str = format_pct(chg) if chg is not None else "—"
    chg_class = "alpha-positive" if (chg or 0) <= 0 else "alpha-negative"

    st.markdown(
        f"""
        <div class="section-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">
                {f"{val:,.2f}" if val is not None else "—"}
            </div>
            <div class="{chg_class}" style="font-size:0.85rem;margin-top:0.15rem;">
                {chg_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_mode:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="metric-label">Mode Parameters</div>
            <div style="display:flex;gap:1.5rem;margin-top:0.3rem;">
                <div>
                    <div style="font-size:0.75rem;color:#9fa6b2;">β Target</div>
                    <div class="metric-value">{beta_target:.2f}</div>
                </div>
                <div>
                    <div style="font-size:0.75rem;color:#9fa6b2;">Drift (Annual)</div>
                    <div class="metric-value">{drift_annual*100:.1f}%</div>
                </div>
                <div>
                    <div style="font-size:0.75rem;color:#9fa6b2;">Expected Daily Drift</div>
                    <div class="metric-value">{drift_daily:.2f}%</div>
                </div>
            </div>
            <div style="font-size:0.75rem;color:#6b7280;margin-top:0.4rem;">
                Internal targets only &mdash; used for beta-adjusted alpha and drift framing.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================
# Tabs
# ================================
tab_overview, tab_alpha, tab_top_perf, tab_top10, tab_logs = st.tabs(
    ["Overview", "Alpha Dashboard", "Top Performers", "Top 10 Holdings", "Engine Logs"]
)


# ---------------- Overview tab ----------------
with tab_overview:
    col_left, col_right = st.columns([1.6, 1.4])

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Return & Alpha Strip</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**1D Return**")
            st.markdown(format_pct(one_day_ret))
        with c2:
            st.markdown("**1D Alpha (β-adjusted)**")
            cls = "alpha-positive" if one_day_alpha >= 0 else "alpha-negative"
            st.markdown(f"<span class='{cls}'>{format_pct(one_day_alpha)}</span>", unsafe_allow_html=True)
        with c3:
            st.markdown("**30D Alpha**")
            cls = "alpha-positive" if alpha_30d >= 0 else "alpha-negative"
            st.markdown(f"<span class='{cls}'>{format_pct(alpha_30d)}</span>", unsafe_allow_html=True)
        with c4:
            st.markdown("**60D Alpha**")
            cls = "alpha-positive" if alpha_60d >= 0 else "alpha-negative"
            st.markdown(f"<span class='{cls}'>{format_pct(alpha_60d)}</span>", unsafe_allow_html=True)

        st.markdown("---", unsafe_allow_html=True)

        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown("**30D Return vs Benchmark**")
            st.markdown(
                f"{format_pct(ret_30d)} | <span style='color:#9fa6b2;'>{format_pct(bench_30d)}</span>",
                unsafe_allow_html=True,
            )
        with c6:
            st.markdown("**60D Return vs Benchmark**")
            st.markdown(
                f"{format_pct(ret_60d)} | <span style='color:#9fa6b2;'>{format_pct(bench_60d)}</span>",
                unsafe_allow_html=True,
            )
        with c7:
            st.markdown("**Since Inception vs Benchmark**")
            st.markdown(
                f"{format_pct(since_inc_ret)} | <span style='color:#9fa6b2;'>{format_pct(since_inc_bench)}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card" style="margin-top:0.85rem;">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Performance Curve</div>', unsafe_allow_html=True)

        perf_chart_df = pd.DataFrame(
            {
                "Date": perf_df["date"],
                "Wave": (1.0 + perf_df["portfolio_return"]).cumprod(),
                "Benchmark": (1.0 + perf_df["benchmark_return"]).cumprod(),
            }
        ).set_index("Date")

        st.line_chart(perf_chart_df)
        st.markdown(
            "<span style='font-size:0.75rem;color:#9fa6b2;'>"
            "Cumulative performance of the Wave vs its benchmark using daily returns."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Risk & Drawdown</div>', unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**Max Drawdown**")
            st.markdown(format_pct(max_dd_pct))
        with col_r2:
            realized_beta = np.cov(
                perf_df["portfolio_return"], perf_df["benchmark_return"]
            )[0, 1] / np.var(perf_df["benchmark_return"])
            st.markdown("**Realized β**")
            st.markdown(f"{realized_beta:.2f}")

        st.markdown("---", unsafe_allow_html=True)

        dd_chart_df = pd.DataFrame(
            {
                "Date": perf_df["date"],
                "Drawdown": dd_series.values,
            }
        ).set_index("Date")
        st.area_chart(dd_chart_df)

        st.markdown(
            "<span style='font-size:0.75rem;color:#9fa6b2;'>"
            "Drawdown is computed from cumulative portfolio value; minimum value is shown as Max DD."
            "</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card" style="margin-top:0.85rem;">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Data Regime</div>', unsafe_allow_html=True)
        regime = "LIVE (engine logs)" if is_live else "SANDBOX (synthetic demo)"
        st.markdown(f"**Regime:** {regime}")
        st.markdown(
            "- Live logs: `logs/performance/<Wave>_performance_daily.csv`\n"
            "- If missing or malformed, synthetic but internally consistent series is generated.\n"
            "- β targets and drift assumptions always come from the selected mode."
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Alpha Dashboard tab ----------------
with tab_alpha:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Alpha Timelines</div>', unsafe_allow_html=True)

    alpha_chart_df = pd.DataFrame(
        {
            "Date": perf_df["date"],
            "Alpha 1D": perf_df["alpha_1d"],
            "Alpha 30D": perf_df["alpha_30d"],
            "Alpha 60D": perf_df["alpha_60d"],
        }
    ).set_index("Date")

    st.line_chart(alpha_chart_df)

    st.markdown(
        "<span style='font-size:0.8rem;color:#9fa6b2;'>"
        "All alphas are β-adjusted excess returns using the mode's β target. "
        "30D and 60D are rolling sums of daily alpha, making them easy to audit."
        "</span>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Top Performers tab ----------------
with tab_top_perf:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Top Performers & Alpha Capture (All Waves)</div>', unsafe_allow_html=True)

    summaries = [compute_alpha_summary_for_wave(w, selected_mode) for w in waves]
    summary_df = pd.DataFrame(summaries)

    if not summary_df.empty:
        summary_df = summary_df.sort_values("60D α", ascending=False).reset_index(drop=True)
        display_df = summary_df.copy()
        for col in ["Intraday α", "1D α", "30D α", "60D α", "1Y α"]:
            display_df[col] = display_df[col].apply(lambda x: format_pct(x))

        st.markdown(
            "<span style='font-size:0.8rem;color:#9fa6b2;'>"
            "All values are β-adjusted alpha captures in % terms for the selected mode "
            f"(**{selected_mode}**). Intraday = latest; 1D = prior day when available; "
            "30D/60D = rolling sums; 1Y = sum of last 252 trading days."
            "</span>",
            unsafe_allow_html=True,
        )
        st.dataframe(display_df, use_container_width=True)
    else:
        st.markdown(
            "No data available across Waves yet. Once the engine writes performance logs "
            "for each Wave, this panel will populate automatically."
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Top 10 Holdings tab ----------------
with tab_top10:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Top 10 Holdings</div>', unsafe_allow_html=True)

    top10_df = load_positions_top10(selected_wave)

    if top10_df is None or top10_df.empty:
        st.markdown(
            "No positions logs found for this Wave yet.\n\n"
            "- Expected pattern: `logs/positions/<Wave>_positions_YYYYMMDD.csv`\n"
            "- The console will automatically populate once the engine writes at least one positions file."
        )
    else:
        st.markdown(
            "<span style='font-size:0.8rem;color:#9fa6b2;'>"
            "Tickers link directly to Google Finance (new tab)."
            "</span>",
            unsafe_allow_html=True,
        )

        # Build HTML table with clickable ticker links
        html_rows = []
        html_rows.append(
            "<tr><th>Ticker</th><th>Name</th><th>Sector</th><th>Weight</th></tr>"
        )
        for _, row in top10_df.iterrows():
            html_rows.append(
                "<tr>"
                f"<td>{row['TickerLink']}</td>"
                f"<td>{row['Name']}</td>"
                f"<td>{row['Sector']}</td>"
                f"<td>{row['Weight%']}</td>"
                "</tr>"
            )

        html_table = "<table class='top10-table'><thead>{head}</thead><tbody>{body}</tbody></table>".format(
            head=html_rows[0], body="".join(html_rows[1:])
        )

        st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Engine Logs tab ----------------
with tab_logs:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Engine Performance Log</div>', unsafe_allow_html=True)

    perf_file = LOGS_PERF_DIR / f"{selected_wave}_performance_daily.csv"

    if perf_file.exists():
        st.markdown(
            f"Reading from: `{perf_file}`  "
            f"({'LIVE' if is_live else 'SANDBOX mirror from this file'})"
        )

        display_cols = ["date", "portfolio_return", "benchmark_return", "alpha_1d", "alpha_30d", "alpha_60d"]
        display_cols_existing = [c for c in display_cols if c in perf_df.columns]
        st.dataframe(perf_df[display_cols_existing].tail(75), use_container_width=True)
    else:
        st.markdown(
            "No live performance log found for this Wave.\n\n"
            "The console is currently running in **demo/sandbox** mode for this Wave, "
            "using internally generated performance that respects the mode's β and drift assumptions."
        )

    st.markdown("</div>", unsafe_allow_html=True)
