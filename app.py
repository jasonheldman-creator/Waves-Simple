import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from datetime import datetime, timedelta
from pathlib import Path


# ======================================================
# STREAMLIT CONFIG & STYLING
# ======================================================
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# ======================================================
# CONSTANTS / PATHS
# ======================================================
BASE_DIR = Path(".")
WEIGHTS_FILE = BASE_DIR / "wave_weights.csv"

MODES_BASE = {
    "Standard": {"beta_target": 0.90, "drift_annual": 0.07},
    "Alpha-Minus-Beta": {"beta_target": 0.80, "drift_annual": 0.06},
    "Private Logic™": {"beta_target": 1.05, "drift_annual": 0.09},
}

BENCHMARK_MAP = {
    "S&P 500 Wave": "^GSPC",
    "Growth Wave": "QQQ",
    "Infinity Wave": "^GSPC",
    "Income Wave": "^GSPC",
    "Future Power & Energy Wave": "XLE",
    "Crypto Income Wave": "BTC-USD",
    "Quantum Computing Wave": "QQQ",
    "Clean Transit-Infrastructure Wave": "IGE",
    "Small/Mid Growth Wave": "IWM",
}

HISTORY_DAYS = 365  # calendar days


# ======================================================
# UTILS
# ======================================================
def format_pct(x, decimals=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{decimals}f}%"


def google_url(ticker):
    t = str(ticker).strip().upper()
    if not t or t == "NAN":
        return ""
    return f"https://www.google.com/finance/quote/{t}"


def get_spx_vix_tiles():
    data = {
        "SPX": {"label": "S&P 500", "value": None, "change": None},
        "VIX": {"label": "VIX", "value": None, "change": None},
    }
    try:
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


# ======================================================
# HUMAN OVERRIDES (GLOBAL STATE)
# ======================================================
if "overrides" not in st.session_state:
    st.session_state["overrides"] = {
        "global": {
            "equity_tilt": "Neutral",  # Defensive / Neutral / Aggressive
            "max_position_pct": 10.0,  # cap on any single position
            "smartsafe_floor_pct": 0.0,
            "trading_freeze": False,
        },
        "per_wave": {},  # {wave_name: {"equity_tilt": "..."}}
    }


def get_effective_tilt(wave):
    per_wave = st.session_state["overrides"]["per_wave"]
    if wave in per_wave and per_wave[wave].get("equity_tilt"):
        return per_wave[wave]["equity_tilt"]
    return st.session_state["overrides"]["global"]["equity_tilt"]


def get_effective_mode_config(wave, mode_label):
    """
    Start from base mode config, then adjust β & drift based on Human Overrides.
    """
    base = MODES_BASE.get(mode_label, MODES_BASE["Standard"]).copy()
    tilt = get_effective_tilt(wave)

    beta = base["beta_target"]
    drift = base["drift_annual"]

    # Simple mapping: Defensive <= Neutral <= Aggressive
    if tilt == "Defensive":
        beta *= 0.9
        drift *= 0.8
    elif tilt == "Aggressive":
        beta *= 1.1
        drift *= 1.2

    # safety caps
    beta = max(0.5, min(1.3, beta))
    drift = max(0.0, min(0.20, drift))

    base["beta_target"] = beta
    base["drift_annual"] = drift
    return base


def apply_max_position_cap(weights_df):
    """
    Apply global max_position_pct override per Wave.
    This is a simple 'portfolio construction' override.
    """
    cap = st.session_state["overrides"]["global"]["max_position_pct"]
    cap_frac = cap / 100.0

    df = weights_df.copy()

    def _cap_and_norm(group):
        w = group["weight"].values
        w = np.minimum(w, cap_frac)
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        group["weight"] = w
        return group

    df = df.groupby("wave", group_keys=False).apply(_cap_and_norm)
    return df


# ======================================================
# ENGINE: LOAD WEIGHTS & BUILD PERFORMANCE
# ======================================================
def load_weights_raw():
    """
    Load wave_weights.csv and normalize within each Wave.
    """
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(f"Missing weights file: {WEIGHTS_FILE}")

    df = pd.read_csv(WEIGHTS_FILE)
    if df.empty:
        raise ValueError("wave_weights.csv is empty")

    cols = {c.lower(): c for c in df.columns}

    wave_col = cols.get("wave") or cols.get("portfolio")
    ticker_col = cols.get("ticker") or cols.get("symbol")

    if wave_col is None or ticker_col is None:
        raise ValueError("wave_weights.csv must have 'wave' and 'ticker' columns")

    weight_candidates = [
        "weight",
        "weight_pct",
        "weight_percent",
        "target_weight",
        "portfolio_weight",
    ]
    weight_col = None
    for cand in weight_candidates:
        if cand in cols:
            weight_col = cols[cand]
            break

    if weight_col is None:
        df["__weight__"] = 1.0
        weight_col = "__weight__"

    df = df.rename(
        columns={wave_col: "wave", ticker_col: "ticker", weight_col: "raw_weight"}
    )
    df["wave"] = df["wave"].astype(str)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["raw_weight"] = df["raw_weight"].astype(float)

    df["weight"] = df.groupby("wave")["raw_weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else 1.0 / len(x)
    )

    return df[["wave", "ticker", "weight"]]


def load_weights_effective():
    """
    Apply Human Override caps to base weights.
    (Future: SmartSafe floor / sector tilts can be added here.)
    """
    raw = load_weights_raw()
    capped = apply_max_position_cap(raw)
    return capped


def get_benchmark_for_wave(wave_name):
    return BENCHMARK_MAP.get(wave_name, "^GSPC")


def fetch_prices(tickers, start, end):
    """
    Fetch adjusted close prices for tickers between start & end.
    """
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=start.date(),
        end=(end + timedelta(days=1)).date(),
        auto_adjust=True,
        group_by="ticker",
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        cols = []
        for t in tickers:
            if (t, "Close") in data.columns:
                cols.append(data[(t, "Close")].rename(t))
        if not cols:
            return pd.DataFrame()
        prices = pd.concat(cols, axis=1)
    else:
        prices = data["Close"].to_frame()
        prices.columns = [tickers[0]]

    prices = prices.dropna(how="all")
    return prices


def compute_returns_from_prices(prices):
    return prices.pct_change().dropna(how="all")


def generate_synthetic_performance(wave_name, mode_label, history_days):
    """
    Generate a synthetic benchmark + portfolio series consistent with
    the mode's β target and drift assumptions.
    """
    mode_cfg = get_effective_mode_config(wave_name, mode_label)
    beta_target = mode_cfg["beta_target"]
    drift_annual = mode_cfg["drift_annual"]

    n_days = min(history_days, 252)
    end_date = datetime.utcnow().date()
    dates = pd.bdate_range(end=end_date, periods=n_days)

    # Assume benchmark ~7% annual drift, 16% vol
    mu_bench = 0.07 / 252.0
    sigma_bench = 0.16 / np.sqrt(252.0)

    # Sample benchmark returns
    bench = np.random.normal(mu_bench, sigma_bench, size=n_days)

    # Portfolio: drift per mode, correlated to benchmark via beta
    mu_port = drift_annual / 252.0
    sigma_idio = 0.10 / np.sqrt(252.0)

    idio = np.random.normal(0.0, sigma_idio, size=n_days)
    port = beta_target * bench + idio + (mu_port - beta_target * mu_bench)

    df = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": port,
            "benchmark_return": bench,
        }
    ).reset_index(drop=True)

    return df, "SANDBOX"


def compute_alpha_windows(df, wave_name, mode_label):
    """
    Given a DataFrame with date, portfolio_return, benchmark_return,
    attach alpha and window metrics.
    """
    mode_cfg = get_effective_mode_config(wave_name, mode_label)
    beta_target = mode_cfg["beta_target"]

    df = df.sort_values("date").reset_index(drop=True)

    df["alpha_1d"] = df["portfolio_return"] - beta_target * df["benchmark_return"]
    df["alpha_30d"] = df["alpha_1d"].rolling(30, min_periods=1).sum()
    df["alpha_60d"] = df["alpha_1d"].rolling(60, min_periods=1).sum()

    df["ret_30d"] = (1.0 + df["portfolio_return"]).rolling(30, min_periods=1).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )
    df["ret_60d"] = (1.0 + df["portfolio_return"]).rolling(60, min_periods=1).apply(
        lambda x: float(np.prod(x) - 1.0),
        raw=True,
    )
    df["bench_30d"] = (1.0 + df["benchmark_return"]).rolling(
        30, min_periods=1
    ).apply(lambda x: float(np.prod(x) - 1.0), raw=True)
    df["bench_60d"] = (1.0 + df["benchmark_return"]).rolling(
        60, min_periods=1
    ).apply(lambda x: float(np.prod(x) - 1.0), raw=True)

    return df


def build_wave_performance(wave_name, mode_label, weights_df, history_days=HISTORY_DAYS):
    """
    Hybrid engine:

    1) Try to build performance from LIVE prices (yfinance).
    2) If that fails or is incomplete, fall back to synthetic series.
    """
    wave_weights = weights_df[weights_df["wave"] == wave_name].copy()
    if wave_weights.empty:
        # No weights at all -> synthetic
        df, regime = generate_synthetic_performance(wave_name, mode_label, history_days)
        df = compute_alpha_windows(df, wave_name, mode_label)
        return df, regime

    tickers = sorted(wave_weights["ticker"].unique().tolist())
    bench_ticker = get_benchmark_for_wave(wave_name)

    end = datetime.utcnow()
    start = end - timedelta(days=history_days)

    regime = "LIVE"
    try:
        basket_prices = fetch_prices(tickers, start, end)
        bench_prices = fetch_prices([bench_ticker], start, end)

        if basket_prices.empty or bench_prices.empty:
            raise ValueError("empty price history")
        basket_rets = compute_returns_from_prices(basket_prices)
        bench_rets = compute_returns_from_prices(bench_prices)

        common_dates = basket_rets.index.intersection(bench_rets.index)
        if len(common_dates) < 30:  # not enough history
            raise ValueError("insufficient overlap")

        basket_rets = basket_rets.loc[common_dates]
        bench_rets = bench_rets.loc[common_dates]

        weight_map = {row["ticker"]: row["weight"] for _, row in wave_weights.iterrows()}
        aligned_weights = np.array([weight_map[t] for t in basket_rets.columns])

        port_ret = basket_rets.values @ aligned_weights

        df = pd.DataFrame(
            {
                "date": common_dates,
                "portfolio_return": port_ret,
                "benchmark_return": bench_rets.iloc[:, 0].values,
            }
        )
        df = compute_alpha_windows(df, wave_name, mode_label)
        return df, regime
    except Exception:
        # Synthetic fallback
        df, regime = generate_synthetic_performance(wave_name, mode_label, history_days)
        df = compute_alpha_windows(df, wave_name, mode_label)
        return df, regime


def compute_drawdown(df):
    cum = (1.0 + df["portfolio_return"]).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    return float(dd.min() * 100.0), dd


def compute_standard_matrix_row(wave_name, weights_df):
    """
    Standard mode snapshot (uses overrides for caps, but Standard mode β).
    """
    perf_df, regime = build_wave_performance(
        wave_name, "Standard", weights_df, HISTORY_DAYS
    )
    if perf_df.empty:
        return {k: np.nan for k in ["Wave"]}

    perf_df = perf_df.sort_values("date").reset_index(drop=True)
    latest = perf_df.iloc[-1]

    one_day_ret = latest["portfolio_return"] * 100.0
    one_day_alpha = latest["alpha_1d"] * 100.0
    alpha_30d = latest["alpha_30d"] * 100.0
    alpha_60d = latest["alpha_60d"] * 100.0

    if np.var(perf_df["benchmark_return"]) > 0:
        realized_beta = np.cov(
            perf_df["portfolio_return"], perf_df["benchmark_return"]
        )[0, 1] / np.var(perf_df["benchmark_return"])
    else:
        realized_beta = np.nan

    max_dd, _ = compute_drawdown(perf_df)

    cum_port = (1.0 + perf_df["portfolio_return"]).cumprod()
    cum_bench = (1.0 + perf_df["benchmark_return"]).cumprod()
    si_ret = (cum_port.iloc[-1] - 1.0) * 100.0
    si_bench = (cum_bench.iloc[-1] - 1.0) * 100.0
    si_excess = si_ret - si_bench

    return {
        "Wave": wave_name,
        "Regime": regime,
        "1D Return": one_day_ret,
        "1D Alpha": one_day_alpha,
        "30D Alpha": alpha_30d,
        "60D Alpha": alpha_60d,
        "Realized β": realized_beta,
        "Max Drawdown": max_dd,
        "SI Return": si_ret,
        "SI Benchmark": si_bench,
        "SI Excess": si_excess,
    }


def compute_alpha_summary_for_wave(wave_name, mode_label, weights_df):
    perf_df, regime = build_wave_performance(
        wave_name, mode_label, weights_df, HISTORY_DAYS
    )
    if perf_df.empty:
        return {"Wave": wave_name, "Regime": regime}

    perf_df = perf_df.sort_values("date").reset_index(drop=True)
    latest = perf_df.iloc[-1]

    intraday_alpha = latest["alpha_1d"]
    one_day_alpha = (
        perf_df["alpha_1d"].iloc[-2] if len(perf_df) >= 2 else intraday_alpha
    )
    alpha_30d = latest["alpha_30d"]
    alpha_60d = latest["alpha_60d"]
    alpha_1y = perf_df["alpha_1d"].tail(252).sum()

    cum_port = (1.0 + perf_df["portfolio_return"]).cumprod()
    cum_bench = (1.0 + perf_df["benchmark_return"]).cumprod()
    si_wave_ret = (cum_port.iloc[-1] - 1.0) * 100.0
    si_bench_ret = (cum_bench.iloc[-1] - 1.0) * 100.0
    si_alpha = perf_df["alpha_1d"].sum() * 100.0

    return {
        "Wave": wave_name,
        "Regime": regime,
        "Intraday α": intraday_alpha * 100.0,
        "1D α": one_day_alpha * 100.0,
        "30D α": alpha_30d * 100.0,
        "60D α": alpha_60d * 100.0,
        "1Y α": alpha_1y * 100.0,
        "SI Alpha": si_alpha,
        "SI Wave Return": si_wave_ret,
        "SI Benchmark Return": si_bench_ret,
    }


def load_top10_holdings(wave_name, weights_df):
    wave_weights = weights_df[weights_df["wave"] == wave_name].copy()
    if wave_weights.empty:
        return pd.DataFrame()

    wave_weights["weight"] = wave_weights["weight"] / wave_weights["weight"].sum()
    wave_weights["weight_pct"] = wave_weights["weight"] * 100.0

    tickers = sorted(wave_weights["ticker"].unique().tolist())
    names = {}
    sectors = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            names[t] = info.get("shortName") or info.get("longName") or ""
            sectors[t] = info.get("sector") or ""
        except Exception:
            names[t] = ""
            sectors[t] = ""

    wave_weights["Name"] = wave_weights["ticker"].map(names)
    wave_weights["Sector"] = wave_weights["ticker"].map(sectors)

    out = wave_weights.sort_values("weight_pct", ascending=False).head(10)
    out["Ticker"] = out["ticker"].astype(str).str.upper()
    out["Weight%"] = out["weight_pct"].round(2).astype(str) + "%"
    out["TickerLink"] = out["Ticker"].apply(
        lambda t: f'<a href="{google_url(t)}" target="_blank">{t}</a>'
    )
    return out[["Ticker", "TickerLink", "Name", "Sector", "Weight%"]]


# ======================================================
# SIDEBAR: WAVE & MODE
# ======================================================
weights_effective = load_weights_effective()
waves = sorted(weights_effective["wave"].unique().tolist())

with st.sidebar:
    st.markdown("### WAVES Intelligence™")
    st.markdown("#### Institutional Console")

    selected_wave = st.selectbox("Select Wave", options=waves, index=0)

    selected_mode = st.radio(
        "Mode",
        options=list(MODES_BASE.keys()),
        index=0,
        help="Display logic adapts to Standard, Alpha-Minus-Beta, and Private Logic™ targets.",
    )

    st.markdown("---")
    st.markdown(
        "This console runs the engine **inside the app**:\n"
        "- Full basket + secondary basket from `wave_weights.csv`\n"
        "- Prices pulled live via yfinance when available\n"
        "- Synthetic series generated when prices are missing\n"
        "- Human overrides applied (tilt, caps) before performance is computed."
    )


# ======================================================
# RUN ENGINE FOR SELECTED WAVE/MODE
# ======================================================
perf_df, perf_regime = build_wave_performance(
    selected_wave, selected_mode, weights_effective
)
perf_df = perf_df.sort_values("date").reset_index(drop=True)
latest = perf_df.iloc[-1]
mode_cfg_eff = get_effective_mode_config(selected_wave, selected_mode)

beta_target = mode_cfg_eff["beta_target"]
drift_annual = mode_cfg_eff["drift_annual"]
drift_daily = drift_annual / 252.0 * 100.0

max_dd_pct, dd_series = compute_drawdown(perf_df)

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


# ======================================================
# HERO HEADER & TILES
# ======================================================
spx_vix = get_spx_vix_tiles()

st.markdown(
    f"""
    <div class="section-card" style="margin-bottom: 0.8rem;">
        <div class="hero-title">WAVES Intelligence™ Institutional Console</div>
        <div class="hero-subtitle">
            Live multi-wave, mode-aware monitoring — engine output visualized for institutional use.
        </div>
        <div style="margin-top: 0.6rem;">
            <span class="wave-badge">{selected_wave}</span>
            <span class="mode-badge">{selected_mode}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_spx, col_vix, col_mode = st.columns([1, 1, 1.5])

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
            <div class="metric-value">{f"{val:,.0f}" if val is not None else "—"}</div>
            <div class="{chg_class}" style="font-size:0.85rem;margin-top:0.15rem;">{chg_str}</div>
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
            <div class="metric-value">{f"{val:,.2f}" if val is not None else "—"}</div>
            <div class="{chg_class}" style="font-size:0.85rem;margin-top:0.15rem;">{chg_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_mode:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="metric-label">Mode Parameters (Effective)</div>
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
                Effective parameters after Human Overrides (equity tilt, caps) are applied.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ======================================================
# TABS
# ======================================================
(
    tab_overview,
    tab_alpha,
    tab_std_matrix,
    tab_alpha_matrix,
    tab_top10,
    tab_logs,
    tab_override,
) = st.tabs(
    [
        "Overview",
        "Alpha Dashboard",
        "Standard Mode Matrix",
        "Alpha Capture Matrix (All Waves)",
        "Top 10 Holdings",
        "Engine Logs",
        "Human Override",
    ]
)


# -------- Overview --------
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
            st.markdown(
                f"<span class='{cls}'>{format_pct(one_day_alpha)}</span>",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown("**30D Alpha**")
            cls = "alpha-positive" if alpha_30d >= 0 else "alpha-negative"
            st.markdown(
                f"<span class='{cls}'>{format_pct(alpha_30d)}</span>",
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown("**60D Alpha**")
            cls = "alpha-positive" if alpha_60d >= 0 else "alpha-negative"
            st.markdown(
                f"<span class='{cls}'>{format_pct(alpha_60d)}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---", unsafe_allow_html=True)

        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown("**30D Return vs Benchmark**")
            st.markdown(
                f"{format_pct(ret_30d)} | "
                f"<span style='color:#9fa6b2;'>{format_pct(bench_30d)}</span>",
                unsafe_allow_html=True,
            )
        with c6:
            st.markdown("**60D Return vs Benchmark**")
            st.markdown(
                f"{format_pct(ret_60d)} | "
                f"<span style='color:#9fa6b2;'>{format_pct(bench_60d)}</span>",
                unsafe_allow_html=True,
            )
        with c7:
            st.markdown("**Since Inception vs Benchmark**")
            st.markdown(
                f"{format_pct(since_inc_ret)} | "
                f"<span style='color:#9fa6b2;'>{format_pct(since_inc_bench)}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card" style="margin-top:0.85rem;">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Performance Curve</div>', unsafe_allow_html=True)

        perf_chart_df = pd.DataFrame(
            {
                "Date": perf_df["date"],
                "Wave": cum_port.values,
                "Benchmark": cum_bench.values,
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
            if np.var(perf_df["benchmark_return"]) > 0:
                realized_beta_sel = np.cov(
                    perf_df["portfolio_return"], perf_df["benchmark_return"]
                )[0, 1] / np.var(perf_df["benchmark_return"])
            else:
                realized_beta_sel = np.nan
            st.markdown("**Realized β**")
            st.markdown(
                f"{realized_beta_sel:.2f}" if not np.isnan(realized_beta_sel) else "—"
            )

        st.markdown("---", unsafe_allow_html=True)

        dd_chart_df = pd.DataFrame(
            {"Date": perf_df["date"], "Drawdown": dd_series.values}
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
        if perf_regime == "LIVE":
            st.markdown("**Regime:** LIVE (real prices via yfinance)")
            st.markdown(
                "- Full basket + secondary basket from `wave_weights.csv`\n"
                "- Prices pulled live via yfinance\n"
                "- Human overrides (tilt, caps) applied before computing returns."
            )
        else:
            st.markdown("**Regime:** SANDBOX (synthetic demo)")
            st.markdown(
                "- Synthetic benchmark & portfolio series generated\n"
                "- Parameters respect this mode's β target and drift assumptions\n"
                "- Human overrides still applied to weights and β."
            )
        st.markdown("</div>", unsafe_allow_html=True)


# -------- Alpha Dashboard --------
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
        "All alphas are β-adjusted excess returns using the effective mode β. "
        "30D and 60D are rolling sums of daily alpha."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -------- Standard Mode Matrix --------
with tab_std_matrix:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-label">Standard Mode Matrix — All Waves</div>',
        unsafe_allow_html=True,
    )

    rows = [compute_standard_matrix_row(w, weights_effective) for w in waves]
    matrix_df = pd.DataFrame(rows)

    if not matrix_df.empty:
        matrix_df = matrix_df.sort_values("60D Alpha", ascending=False).reset_index(
            drop=True
        )
        display_df = matrix_df.copy()

        for col in [
            "1D Return",
            "1D Alpha",
            "30D Alpha",
            "60D Alpha",
            "Max Drawdown",
            "SI Return",
            "SI Benchmark",
            "SI Excess",
        ]:
            display_df[col] = display_df[col].apply(lambda x: format_pct(x))

        display_df["Realized β"] = display_df["Realized β"].apply(
            lambda x: f"{x:.2f}" if not np.isnan(x) else "—"
        )

        st.markdown(
            "<span style='font-size:0.8rem;color:#9fa6b2;'>"
            "All metrics computed in Standard mode with Human Overrides applied "
            "to weights (position caps). Regime column shows LIVE vs SANDBOX."
            "</span>",
            unsafe_allow_html=True,
        )
        st.dataframe(display_df, use_container_width=True)

        bar_df = matrix_df[["Wave", "60D Alpha"]].set_index("Wave")
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            "<span class='metric-label'>60D Alpha (Standard Mode, All Waves)</span>",
            unsafe_allow_html=True,
        )
        st.bar_chart(bar_df)
    else:
        st.markdown("No data available yet.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------- Alpha Capture Matrix --------
with tab_alpha_matrix:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-label">Alpha Capture Matrix — All Waves</div>',
        unsafe_allow_html=True,
    )

    summaries = [
        compute_alpha_summary_for_wave(w, selected_mode, weights_effective)
        for w in waves
    ]
    summary_df = pd.DataFrame(summaries)

    if not summary_df.empty:
        summary_df = summary_df.sort_values("60D α", ascending=False).reset_index(
            drop=True
        )

        display_df = summary_df.copy()
        for col in [
            "Intraday α",
            "1D α",
            "30D α",
            "60D α",
            "1Y α",
            "SI Alpha",
            "SI Wave Return",
            "SI Benchmark Return",
        ]:
            display_df[col] = display_df[col].apply(lambda x: format_pct(x))

        st.markdown(
            f"<span style='font-size:0.8rem;color:#9fa6b2;'>"
            f"All values are β-adjusted alpha captures and cumulative returns for "
            f"the selected mode (<b>{selected_mode}</b>). Regime shows LIVE vs SANDBOX."
            f"</span>",
            unsafe_allow_html=True,
        )
        st.dataframe(display_df, use_container_width=True)

        bar_df = summary_df[["Wave", "SI Alpha"]].set_index("Wave")
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            "<span class='metric-label'>Since-Inception Alpha (All Waves)</span>",
            unsafe_allow_html=True,
        )
        st.bar_chart(bar_df)
    else:
        st.markdown("No data available yet.")

    st.markdown("</div>", unsafe_allow_html=True)


# -------- Top 10 Holdings --------
with tab_top10:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Top 10 Holdings</div>', unsafe_allow_html=True)

    top10_df = load_top10_holdings(selected_wave, weights_effective)

    if top10_df.empty:
        st.markdown("No holdings data found for this Wave in wave_weights.csv.")
    else:
        st.markdown(
            "<span style='font-size:0.8rem;color:#9fa6b2;'>"
            "Source: wave_weights.csv (full basket, with caps applied). "
            "Tickers link directly to Google Finance."
            "</span>",
            unsafe_allow_html=True,
        )
        html_rows = [
            "<tr><th>Ticker</th><th>Name</th><th>Sector</th><th>Weight</th></tr>"
        ]
        for _, row in top10_df.iterrows():
            html_rows.append(
                "<tr>"
                f"<td>{row['TickerLink']}</td>"
                f"<td>{row['Name']}</td>"
                f"<td>{row['Sector']}</td>"
                f"<td>{row['Weight%']}</td>"
                "</tr>"
            )
        html_table = (
            "<table class='top10-table'><thead>{head}</thead>"
            "<tbody>{body}</tbody></table>"
        ).format(head=html_rows[0], body="".join(html_rows[1:]))
        st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -------- Engine Logs (in-memory) --------
with tab_logs:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Engine Performance Log</div>', unsafe_allow_html=True)

    st.markdown(
        "This view shows the **computed daily performance** for the selected Wave "
        "based on full-basket prices from yfinance (LIVE) or synthetic series (SANDBOX) "
        "after applying Human Overrides."
    )

    display_cols = [
        "date",
        "portfolio_return",
        "benchmark_return",
        "alpha_1d",
        "alpha_30d",
        "alpha_60d",
    ]
    existing = [c for c in display_cols if c in perf_df.columns]
    st.dataframe(perf_df[existing].tail(75), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# -------- Human Override Tab --------
with tab_override:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-label">Human Override — Risk & Tilt Controls</div>',
        unsafe_allow_html=True,
    )

    overrides = st.session_state["overrides"]
    global_cfg = overrides["global"]

    col_g1, col_g2, col_g3 = st.columns(3)

    with col_g1:
        equity_tilt = st.radio(
            "Global Equity Tilt",
            options=["Defensive", "Neutral", "Aggressive"],
            index=["Defensive", "Neutral", "Aggressive"].index(
                global_cfg["equity_tilt"]
            ),
            help="Defensive tilts β & drift down; Aggressive tilts them up.",
        )

    with col_g2:
        max_pos = st.slider(
            "Max Position Weight (%)",
            min_value=1.0,
            max_value=25.0,
            value=float(global_cfg["max_position_pct"]),
            step=0.5,
            help="Hard cap on any single position's target weight; engine renormalizes the rest.",
        )

    with col_g3:
        smartsafe_floor = st.slider(
            "SmartSafe Floor (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(global_cfg["smartsafe_floor_pct"]),
            step=1.0,
            help="Reserved for future SmartSafe allocation logic (not yet affecting weights).",
        )
        freeze = st.checkbox(
            "Freeze Trading (Preview)",
            value=bool(global_cfg["trading_freeze"]),
            help="Flag only, for now; future versions can use this to halt rebalancing.",
        )

    overrides["global"]["equity_tilt"] = equity_tilt
    overrides["global"]["max_position_pct"] = float(max_pos)
    overrides["global"]["smartsafe_floor_pct"] = float(smartsafe_floor)
    overrides["global"]["trading_freeze"] = bool(freeze)

    st.markdown("---")

    st.markdown("##### Per-Wave Tilt Overrides")

    with st.expander("Override tilt for a specific Wave"):
        ow_wave = st.selectbox("Select Wave to Override", options=["(None)"] + waves)
        if ow_wave != "(None)":
            current_tilt = get_effective_tilt(ow_wave)
            tilt_options = ["Inherit Global", "Defensive", "Neutral", "Aggressive"]
            if ow_wave not in overrides["per_wave"]:
                default_idx = tilt_options.index("Inherit Global")
            else:
                if current_tilt in tilt_options:
                    default_idx = tilt_options.index(current_tilt)
                else:
                    default_idx = tilt_options.index("Inherit Global")

            wave_tilt_choice = st.radio(
                f"{ow_wave} Tilt",
                options=tilt_options,
                index=default_idx,
                help="Set a custom tilt for this Wave, or inherit the global tilt.",
            )

            if wave_tilt_choice == "Inherit Global":
                overrides["per_wave"].pop(ow_wave, None)
            else:
                overrides["per_wave"].setdefault(ow_wave, {})
                overrides["per_wave"][ow_wave]["equity_tilt"] = wave_tilt_choice

    st.session_state["overrides"] = overrides

    st.markdown(
        "<br><span style='font-size:0.8rem;color:#9fa6b2;'>"
        "Human overrides are applied immediately on refresh: caps adjust weights, "
        "and tilt adjusts effective β & drift for all engine calculations across all Waves."
        "</span>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)