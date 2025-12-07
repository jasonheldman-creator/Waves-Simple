import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
)

BENCHMARK_TICKER = "SPY"
VIX_TICKER = "^VIX"
BETA_TARGET = 0.90  # target beta vs SPY
BUSINESS_DAYS_PER_YEAR = 252

# ---------------------------------------------------------
# LOCKED-IN WAVES (CODE-MANAGED DEFAULT WEIGHTS)
# ---------------------------------------------------------
# These are code-level defaults so the engine is 100% stable
# and independent of wave_weights.csv.
# You can rename wave keys (e.g. "AI_Wave") to your exact
# 9 locked wave names without touching any other code.
# Each inner dict should sum to ~1.0 (weights per wave).
# ---------------------------------------------------------

DEFAULT_WEIGHTS = {
    "AI_Wave": {
        "NVDA": 0.18,
        "MSFT": 0.18,
        "META": 0.16,
        "GOOGL": 0.14,
        "AMZN": 0.12,
        "AVGO": 0.11,
        "PLTR": 0.11,
    },
    "Growth_Wave": {
        "AAPL": 0.16,
        "AMZN": 0.16,
        "TSLA": 0.16,
        "NFLX": 0.16,
        "CRM": 0.18,
        "ADBE": 0.18,
    },
    "SmallCapGrowth_Wave": {
        "SMCI": 0.20,
        "CELH": 0.20,
        "UPST": 0.20,
        "SOFI": 0.20,
        "ONON": 0.20,
    },
    "SmallMidGrowth_Wave": {
        "SHOP": 0.20,
        "DDOG": 0.20,
        "CRWD": 0.20,
        "NET": 0.20,
        "MDB": 0.20,
    },
    "FuturePower_Wave": {
        "ENPH": 0.20,
        "FSLR": 0.20,
        "SEDG": 0.20,
        "NEE": 0.20,
        "TSLA": 0.20,
    },
    "CleanTransitInfra_Wave": {
        "TSLA": 0.20,
        "NIO": 0.20,
        "BLNK": 0.20,
        "CHPT": 0.20,
        "NEE": 0.20,
    },
    "CryptoIncome_Wave": {
        "COIN": 0.40,
        "MSTR": 0.30,
        "BITO": 0.30,
    },
    "QuantumComputing_Wave": {
        "NVDA": 0.25,
        "AMD": 0.25,
        "IBM": 0.25,
        "QCOM": 0.25,
    },
    "SP500_Wave": {
        "SPY": 1.0,
    },
}

# Preferred display order
PREFERRED_WAVE_ORDER = [
    "AI_Wave",
    "Growth_Wave",
    "SmallCapGrowth_Wave",
    "SmallMidGrowth_Wave",
    "FuturePower_Wave",
    "CleanTransitInfra_Wave",
    "CryptoIncome_Wave",
    "QuantumComputing_Wave",
    "SP500_Wave",
]

MODE_LABELS = {
    "Standard": "Standard",
    "AlphaMinusBeta": "Alpha-Minus-Beta",
    "PrivateLogic": "Private Logic™",
}

# =========================================================
# DATA HELPERS
# =========================================================

@st.cache_data
def load_weights_from_defaults() -> pd.DataFrame:
    """
    Build a weights DataFrame purely from DEFAULT_WEIGHTS,
    ignoring any CSVs. This is our stable fallback engine.
    """
    rows = []
    for wave_name, positions in DEFAULT_WEIGHTS.items():
        total = float(sum(positions.values()))
        if total <= 0:
            continue
        for ticker, w in positions.items():
            rows.append(
                {
                    "wave": wave_name,
                    "ticker": str(ticker).strip().upper(),
                    "weight": float(w) / total,
                }
            )
    df = pd.DataFrame(rows, columns=["wave", "ticker", "weight"])
    return df


@st.cache_data
def fetch_price_history(tickers, lookback_days: int):
    """
    Fetch price history for all tickers for the given lookback window.
    Returns (prices_df, engine_status) where engine_status is 'LIVE' or 'SANDBOX'.
    """
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days + 40)  # buffer

    tickers = sorted(set(tickers))
    frames = []
    live_ok = False

    for t in tickers:
        try:
            hist = yf.download(
                t,
                start=start,
                end=end + timedelta(days=1),
                auto_adjust=True,
                progress=False,
            )
            if hist is not None and not hist.empty:
                s = hist["Adj Close"].rename(t)
                frames.append(s)
                live_ok = True
        except Exception:
            continue

    if frames:
        prices = pd.concat(frames, axis=1).sort_index()
        prices = prices.loc[~prices.index.duplicated(keep="last")]
        prices = prices.dropna(how="all")
        if not prices.empty:
            return prices, ("LIVE" if live_ok else "SANDBOX")

    # Synthetic SANDBOX data if live pull fails
    dates = pd.date_range(end - timedelta(days=lookback_days), end, freq="B")
    rng = np.random.default_rng(42)
    data = {}
    for t in tickers:
        rets = rng.normal(loc=0.0004, scale=0.01, size=len(dates))
        curve = (1 + pd.Series(rets, index=dates)).cumprod()
        data[t] = curve
    prices = pd.DataFrame(data, index=dates)
    return prices, "SANDBOX"


@st.cache_data
def fetch_vix_series(lookback_days: int):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days + 40)

    try:
        hist = yf.download(
            VIX_TICKER,
            start=start,
            end=end + timedelta(days=1),
            auto_adjust=False,
            progress=False,
        )
        if hist is not None and not hist.empty:
            s = hist["Adj Close"].rename("VIX")
            s.index = s.index.tz_convert(None)
            return s
    except Exception:
        pass

    # Synthetic VIX if needed
    dates = pd.date_range(start, end, freq="B")
    base = 18.0
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=3.0, size=len(dates))
    vix = pd.Series(np.clip(base + noise, 10, 50), index=dates, name="VIX")
    return vix


def compute_wave_timeseries(weights_df, prices: pd.DataFrame, wave_name: str):
    """
    Compute wave equity curve and benchmark curve.
    Returns dict with keys: 'dates', 'wave_curve', 'bench_curve',
    'wave_rets', 'bench_rets', 'spy_price'.
    """
    w = weights_df[weights_df["wave"] == wave_name].copy()
    if w.empty:
        return None

    tickers = w["ticker"].tolist()
    w_vec = w.set_index("ticker")["weight"]

    needed = set(tickers) | {BENCHMARK_TICKER}
    missing = [t for t in needed if t not in prices.columns]
    if len(missing) == len(needed):
        return None

    avail_tickers = [t for t in tickers if t in prices.columns]
    if not avail_tickers or BENCHMARK_TICKER not in prices.columns:
        return None

    sub = prices[avail_tickers].copy()
    sub = sub.dropna(how="all")
    bench = prices[BENCHMARK_TICKER].dropna()

    # Align
    sub, bench = sub.align(bench, join="inner", axis=0)
    if sub.empty or bench.empty:
        return None

    w_vec = w_vec.reindex(avail_tickers).fillna(0)
    w_vec = w_vec / w_vec.sum()

    rets_sub = sub.pct_change().fillna(0.0)
    bench_rets = bench.pct_change().fillna(0.0)

    wave_rets = (rets_sub * w_vec).sum(axis=1)

    wave_curve = (1 + wave_rets).cumprod()
    bench_curve = (1 + bench_rets).cumprod()

    return {
        "dates": wave_curve.index,
        "wave_curve": wave_curve,
        "bench_curve": bench_curve,
        "wave_rets": wave_rets,
        "bench_rets": bench_rets,
        "spy_price": bench,
    }


def estimate_beta(wave_rets: pd.Series, bench_rets: pd.Series) -> float:
    """Simple daily-return beta estimate."""
    x, y = bench_rets.align(wave_rets, join="inner")
    if len(x) < 10:
        return np.nan
    xr = x.values
    yr = y.values
    var = np.var(xr)
    if var <= 0:
        return np.nan
    cov = np.cov(xr, yr)[0, 1]
    return float(cov / var)


# =========================================================
# REGIMES, EXPOSURE, SMARTSAFE
# =========================================================

def compute_regimes(spy_series: pd.Series, vix_series: pd.Series) -> pd.Series:
    """
    Classify each date into a simple regime based on SPY vs its 100D MA and VIX.
    """
    spy = spy_series.copy().dropna()
    vix = vix_series.copy().dropna()
    spy, vix = spy.align(vix, join="inner")

    ma = spy.rolling(100, min_periods=20).mean()
    regimes = []

    for date in spy.index:
        price = spy.loc[date]
        ma_val = ma.loc[date]
        vix_val = vix.loc[date]

        if pd.isna(ma_val) or pd.isna(vix_val):
            regimes.append("Unknown")
            continue

        if price > ma_val and vix_val < 18:
            regimes.append("Calm Bull")
        elif price > ma_val and 18 <= vix_val <= 25:
            regimes.append("Choppy Bull")
        elif price <= ma_val and vix_val < 25:
            regimes.append("Correction")
        else:
            regimes.append("Bear / High Vol")

    regime_series = pd.Series(regimes, index=spy.index, name="Regime")
    return regime_series


def base_exposure_from_vix(vix_value: float) -> float:
    """
    Core exposure ladder driven by VIX alone.
    """
    if np.isnan(vix_value):
        return 0.90
    if vix_value <= 15:
        return 0.95
    if vix_value <= 20:
        return 0.85
    if vix_value <= 25:
        return 0.75
    if vix_value <= 35:
        return 0.60
    return 0.45


def adjust_exposure_for_mode_and_beta(base: float, mode: str, est_beta: float) -> float:
    """
    Adjust exposure based on mode and realized beta vs target.
    """
    exp = base

    if mode == "Standard":
        # Mild adjustment if beta drifts too high
        if not np.isnan(est_beta) and est_beta > BETA_TARGET + 0.10:
            exp *= 0.9

    elif mode == "AlphaMinusBeta":
        # More conservative, cap and faster cut if beta high
        exp = min(exp, 0.80)
        if not np.isnan(est_beta) and est_beta > BETA_TARGET + 0.05:
            exp *= 0.85

    elif mode == "PrivateLogic":
        # Can lean in a bit more when VIX is calm
        if base >= 0.85:
            exp = min(base * 1.1, 1.10)
        # If beta already high, don't overdo it
        if not np.isnan(est_beta) and est_beta > BETA_TARGET + 0.15:
            exp = min(exp, base)

    return float(np.clip(exp, 0.0, 1.20))


def compute_exposure_series(vix_series: pd.Series, mode: str, est_beta: float) -> pd.Series:
    """
    Build a time-varying exposure series based on VIX and mode.
    Beta is used as a mild governor.
    """
    base_series = vix_series.apply(base_exposure_from_vix)
    adjusted = base_series.apply(lambda x: adjust_exposure_for_mode_and_beta(x, mode, est_beta))
    return adjusted.rename("Exposure")


def build_smartsafe_series(dates: pd.DatetimeIndex, annual_yield: float = 0.04) -> pd.Series:
    """
    Synthetic SmartSafe™ series: low-vol, money-market-like.
    """
    daily_rate = (1.0 + annual_yield) ** (1.0 / BUSINESS_DAYS_PER_YEAR) - 1.0
    rng = np.random.default_rng(21)
    noise = rng.normal(loc=0.0, scale=0.0003, size=len(dates))  # tiny wiggle
    daily_rets = daily_rate + noise
    curve = (1 + pd.Series(daily_rets, index=dates)).cumprod()
    return curve.rename("SmartSafe")


def max_drawdown(series: pd.Series) -> float:
    """
    Compute max drawdown as a fraction (0.20 = -20%).
    """
    if series is None or series.empty:
        return np.nan
    running_max = series.cummax()
    dd = (running_max - series) / running_max
    return float(dd.max())


# =========================================================
# ALPHA / HOLDINGS HELPERS
# =========================================================

def compute_alpha_series(wave_rets: pd.Series, bench_rets: pd.Series, exposure_series: pd.Series):
    """
    Compute both simple alpha and 'Alpha Captured' (exposure-adjusted alpha).

    simple_alpha_daily   = wave - bench
    captured_alpha_daily = wave - exposure_t * bench
    """
    # Align everything
    wave_rets, bench_rets, exposure_series = wave_rets.align(
        bench_rets, join="inner"
    )[0].align(exposure_series, join="inner")[0], bench_rets.align(exposure_series, join="inner")[0], exposure_series.align(wave_rets, join="inner")[0]

    # Simpler/safer realignment:
    idx = wave_rets.index.intersection(bench_rets.index).intersection(exposure_series.index)
    wave_rets = wave_rets.loc[idx]
    bench_rets = bench_rets.loc[idx]
    exposure_series = exposure_series.loc[idx]

    simple_daily = wave_rets - bench_rets
    simple_cum = (1 + simple_daily).cumprod() - 1

    captured_daily = wave_rets - exposure_series * bench_rets
    captured_cum = (1 + captured_daily).cumprod() - 1

    return {
        "simple_daily": simple_daily,
        "simple_cum": simple_cum,
        "captured_daily": captured_daily,
        "captured_cum": captured_cum,
    }


def get_top_holdings(weights_df, prices: pd.DataFrame, wave_name: str):
    w = weights_df[weights_df["wave"] == wave_name].copy()
    if w.empty:
        return pd.DataFrame(columns=["Ticker", "Weight", "Today%"])

    last_two = prices.tail(2)
    if last_two.shape[0] < 2:
        today_rets = pd.Series(0.0, index=prices.columns)
    else:
        today_rets = last_two.iloc[-1] / last_two.iloc[-2] - 1.0

    w["Weight"] = w["weight"] * 100.0
    w["Today%"] = w["ticker"].map(today_rets) * 100.0
    w = w.sort_values("weight", ascending=False)

    df = w[["ticker", "Weight", "Today%"]].rename(columns={"ticker": "Ticker"})
    df["Ticker"] = df["Ticker"].astype(str)

    return df


# =========================================================
# UI HELPERS
# =========================================================

def google_quote_url(ticker: str) -> str:
    return f"https://www.google.com/finance/quote/{ticker}"


def metric_color(value: float) -> str:
    if np.isnan(value):
        return "white"
    if value > 0:
        return "rgb(0, 200, 120)"
    if value < 0:
        return "rgb(255, 80, 80)"
    return "white"


# =========================================================
# MAIN APP
# =========================================================

def main():
    # ---------- SIDEBAR ----------
    st.sidebar.title("Wave & Mode")

    weights_df = load_weights_from_defaults()
    available_waves = sorted(weights_df["wave"].unique().tolist())

    ordered_waves = [w for w in PREFERRED_WAVE_ORDER if w in available_waves]
    for w in available_waves:
        if w not in ordered_waves:
            ordered_waves.append(w)

    selected_wave = st.sidebar.selectbox("Select Wave", ordered_waves)

    mode_key = st.sidebar.selectbox(
        "Mode",
        options=["Standard", "AlphaMinusBeta", "PrivateLogic"],
        format_func=lambda k: MODE_LABELS[k],
    )

    lookback_days = st.sidebar.slider(
        "Lookback (trading days)",
        min_value=60,
        max_value=365,
        value=365,
        step=5,
    )

    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    # ---------- DATA PULL ----------
    needed_tickers = weights_df["ticker"].unique().tolist()
    needed_tickers.append(BENCHMARK_TICKER)

    prices, engine_status = fetch_price_history(needed_tickers, lookback_days)
    vix_series = fetch_vix_series(lookback_days)

    wave_data = compute_wave_timeseries(weights_df, prices, selected_wave)
    if wave_data is None:
        st.error(
            f"Could not compute returns for {selected_wave}. "
            "Please verify tickers and weights."
        )
        if show_debug:
            st.write("Debug: Missing tickers or price history.", prices.head())
        return

    dates = wave_data["dates"]
    wave_curve = wave_data["wave_curve"]
    bench_curve = wave_data["bench_curve"]
    wave_rets = wave_data["wave_rets"]
    bench_rets = wave_data["bench_rets"]
    spy_price = wave_data["spy_price"]

    cutoff = dates[-1] - timedelta(days=lookback_days)
    mask = dates >= cutoff
    wave_curve = wave_curve[mask]
    bench_curve = bench_curve[mask]
    wave_rets = wave_rets[wave_rets.index >= cutoff]
    bench_rets = bench_rets[bench_rets.index >= cutoff]
    spy_price = spy_price[spy_price.index >= cutoff]

    # Align VIX to SPY
    vix_series = vix_series.loc[spy_price.index.min():spy_price.index.max()]
    vix_last = float(vix_series.iloc[-1]) if not vix_series.empty else np.nan

    # ---------- REGIME + EXPOSURE + SMARTSAFE ----------
    regime_series = compute_regimes(spy_price, vix_series)
    current_regime = regime_series.iloc[-1] if not regime_series.empty else "Unknown"

    est_beta = estimate_beta(wave_rets, bench_rets)

    # Time-varying exposure series
    exposure_series = compute_exposure_series(vix_series, mode_key, est_beta)
    # Align to wave rets
    exposure_series = exposure_series.reindex(wave_rets.index).ffill().bfill()
    # Current exposure = last value
    current_exposure = float(exposure_series.iloc[-1]) if len(exposure_series) else np.nan
    current_smartsafe = float(max(0.0, 1.0 - current_exposure))

    # SmartSafe series and blended wave curve
    smartsafe_curve = build_smartsafe_series(wave_curve.index)
    smartsafe_rets = smartsafe_curve.pct_change().fillna(0.0)

    # Align for blending
    idx_blend = wave_rets.index.intersection(smartsafe_rets.index).intersection(exposure_series.index)
    wave_rets_blend = wave_rets.loc[idx_blend]
    smartsafe_rets_blend = smartsafe_rets.loc[idx_blend]
    exposure_blend = exposure_series.loc[idx_blend]

    blended_rets = exposure_blend * wave_rets_blend + (1.0 - exposure_blend) * smartsafe_rets_blend
    blended_curve = (1 + blended_rets).cumprod()
    blended_curve.name = "Wave+SmartSafe"

    # ---------- ADVANCED ALPHA ----------
    alpha_info = compute_alpha_series(wave_rets, bench_rets, exposure_series)
    simple_alpha_daily = alpha_info["simple_daily"]
    captured_alpha_daily = alpha_info["captured_daily"]
    captured_cum = alpha_info["captured_cum"]

    if len(wave_curve) >= 2 and len(bench_curve) >= 2:
        wave_today = float(wave_curve.iloc[-1] / wave_curve.iloc[-2] - 1.0)
        bench_today = float(bench_curve.iloc[-1] / bench_curve.iloc[-2] - 1.0)
    else:
        wave_today = np.nan
        bench_today = np.nan

    today_captured_alpha = (
        captured_alpha_daily.iloc[-1] if len(captured_alpha_daily) > 0 else np.nan
    )

    # Discipline metrics
    realized_beta = est_beta
    beta_diff = realized_beta - BETA_TARGET if not np.isnan(realized_beta) else np.nan

    wave_mdd = max_drawdown(wave_curve)
    bench_mdd = max_drawdown(bench_curve)

    # Regime compliance: on high-vol days (VIX > 25), % of days exposure <= 0.75
    high_vol_mask = vix_series > 25
    if high_vol_mask.any():
        exp_on_high_vol = exposure_series[high_vol_mask]
        if len(exp_on_high_vol) > 0:
            compliant_days = (exp_on_high_vol <= 0.75).sum()
            regime_compliance = compliant_days / len(exp_on_high_vol)
        else:
            regime_compliance = np.nan
    else:
        regime_compliance = np.nan

    # ---------- HEADER ----------
    col_header_left, col_header_right = st.columns([4, 2])
    with col_header_left:
        st.markdown(
            "<h1 style='margin-bottom:0.1rem;'>WAVES Institutional Console</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='opacity:0.8;'>Adaptive Portfolio Waves • AIWs/APWs • "
            "SmartSafe™ • VIX-gated risk • Alpha-Minus-Beta & Private Logic™</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='opacity:0.9;'><b>Current Regime:</b> {current_regime}</p>",
            unsafe_allow_html=True,
        )

    with col_header_right:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        status_color = (
            "rgb(0, 200, 120)" if engine_status == "LIVE" else "rgb(255, 200, 0)"
        )
        st.markdown(
            f"""
            <div style='text-align:right;'>
                <div style='font-size:0.9rem;'>
                    <span style='color:{status_color};font-weight:600;'>
                        Engine Status: {engine_status}
                    </span><br/>
                    <span style='font-size:0.8rem;opacity:0.8;'>
                        Last refresh: {now_utc} UTC
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---------- METRIC STRIP ----------
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.markdown("**Wave Today**")
        st.markdown(
            f"<span style='font-size:1.6rem;color:{metric_color(wave_today)};'>"
            f"{'' if np.isnan(wave_today) else f'{wave_today*100:.2f}%'}"
            "</span>",
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown("**Benchmark Today (SPY)**")
        st.markdown(
            f"<span style='font-size:1.6rem;color:{metric_color(bench_today)};'>"
            f"{'' if np.isnan(bench_today) else f'{bench_today*100:.2f}%'}"
            "</span>",
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown("**Today Alpha Captured**")
        st.markdown(
            f"<span style='font-size:1.6rem;color:{metric_color(today_captured_alpha)};'>"
            f"{'' if np.isnan(today_captured_alpha) else f'{today_captured_alpha*100:.2f}%'}"
            "</span>",
            unsafe_allow_html=True,
        )
        st.caption("Wave − Exposure(t) × SPY")

    with m4:
        st.markdown("**Realized Beta vs SPY**")
        st.markdown(
            f"<span style='font-size:1.6rem;'>"
            f"{'' if np.isnan(realized_beta) else f'{realized_beta:.2f}'}"
            "</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Target β = {BETA_TARGET:.2f}")

    with m5:
        st.markdown("**Current Exposure**")
        st.markdown(
            f"<span style='font-size:1.6rem;'>{'' if np.isnan(current_exposure) else f'{current_exposure*100:.1f}%'}"
            "</span>",
            unsafe_allow_html=True,
        )
        st.caption(MODE_LABELS[mode_key])

    with m6:
        st.markdown("**SmartSafe™ Allocation Now**")
        st.markdown(
            f"<span style='font-size:1.6rem;'>{'' if np.isnan(current_smartsafe) else f'{current_smartsafe*100:.1f}%'}"
            "</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ---------- MAIN CHART + TOP HOLDINGS ----------
    left_main, right_main = st.columns([3, 2])

    with left_main:
        st.subheader(f"{selected_wave} vs Benchmark (Equity Curves)")
        chart_df = pd.DataFrame(
            {
                "Wave (Equity Only)": wave_curve,
                "Benchmark (SPY)": bench_curve,
                "Wave+SmartSafe": blended_curve,
            }
        )
        st.line_chart(chart_df)

    with right_main:
        st.subheader("Top Holdings (Live)")
        top_df = get_top_holdings(weights_df, prices, selected_wave)
        if top_df.empty:
            st.info("No holdings found for this wave.")
        else:
            rows_html = []
            for _, row in top_df.iterrows():
                ticker = row["Ticker"]
                weight_val = row["Weight"]
                today_val = row["Today%"]
                url = google_quote_url(ticker)

                today_color = metric_color(today_val / 100.0)
                today_str = "" if np.isnan(today_val) else f"{today_val:+.2f}%"

                rows_html.append(
                    f"<tr>"
                    f"<td><a href='{url}' target='_blank'>{ticker}</a></td>"
                    f"<td style='text-align:right;'>{weight_val:.2f}%</td>"
                    f"<td style='text-align:right;color:{today_color};'>{today_str}</td>"
                    f"</tr>"
                )

            table_html = (
                "<table style='width:100%;font-size:0.9rem;'>"
                "<thead><tr><th>Ticker</th>"
                "<th style='text-align:right;'>Weight</th>"
                "<th style='text-align:right;'>Today%</th></tr></thead>"
                "<tbody>"
                + "".join(rows_html)
                + "</tbody></table>"
            )

            st.markdown(table_html, unsafe_allow_html=True)

    # ---------- LOWER CHARTS & DISCIPLINE BOX ----------
    st.markdown("---")
    c1, c2, c3 = st.columns([3, 3, 2])

    with c1:
        st.subheader("Cumulative Alpha Captured (Wave − Exposure(t) × SPY)")
        st.line_chart(captured_cum.rename("Alpha Captured"))

    with c2:
        st.subheader("SPY & VIX (Regime Context)")
        combo_df = pd.DataFrame({"SPY": spy_price, "VIX": vix_series})
        st.line_chart(combo_df)

    with c3:
        st.subheader("Discipline Metrics")
        dd_wave_str = "" if np.isnan(wave_mdd) else f"{wave_mdd*100:.1f}%"
        dd_bench_str = "" if np.isnan(bench_mdd) else f"{bench_mdd*100:.1f}%"
        beta_diff_str = "" if np.isnan(beta_diff) else f"{beta_diff:+.2f}"
        regime_comp_str = (
            "" if np.isnan(regime_compliance) else f"{regime_compliance*100:.1f}%"
        )

        st.markdown(
            f"""
            • Max Drawdown (Wave): **{dd_wave_str}**  
            • Max Drawdown (SPY): **{dd_bench_str}**  
            • Beta Deviation (β − {BETA_TARGET:.2f}): **{beta_diff_str}**  
            • High-Vol Regime Compliance: **{regime_comp_str}**  
            
            """,
            unsafe_allow_html=True,
        )

    # ---------- DEBUG ----------
    if show_debug:
        st.markdown("---")
        st.subheader("Debug Info")
        st.write("Weights DataFrame", weights_df)
        st.write("Prices (tail)", prices.tail())
        st.write("Wave returns (tail)", wave_rets.tail())
        st.write("Benchmark returns (tail)", bench_rets.tail())
        st.write("Exposure series (tail)", exposure_series.tail())
        st.write("Regimes (tail)", regime_series.tail())
        st.write("Engine status:", engine_status)


if __name__ == "__main__":
    main()
