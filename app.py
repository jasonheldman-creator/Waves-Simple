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
BETA_TARGET = 0.90

# ---------------------------------------------------------
# LOCKED-IN WAVES (CODE-MANAGED DEFAULT WEIGHTS)
# ---------------------------------------------------------
# IMPORTANT:
# - These are code-level defaults so the engine is 100% stable
#   and independent of wave_weights.csv.
# - You can rename wave keys (e.g. "AI_Wave") to your exact
#   9 locked wave names without touching any other code.
# - Each inner dict must sum to 1.0 (weights per wave).
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

# Preferred display order (you can reorder or rename these)
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
    'wave_rets', 'bench_rets'.
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


def compute_exposure_and_smartsafe(vix_last: float, mode: str):
    """
    Simple VIX-gated exposure logic.
    """
    if np.isnan(vix_last):
        base = 0.9
    elif vix_last <= 15:
        base = 0.9
    elif vix_last <= 25:
        base = 0.75
    elif vix_last <= 35:
        base = 0.6
    else:
        base = 0.45

    if mode == "AlphaMinusBeta":
        base *= 0.9
    elif mode == "PrivateLogic":
        base *= 1.1

    base = float(np.clip(base, 0.0, 1.2))
    smartsafe = float(max(0.0, 1.0 - base))

    return base, smartsafe


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
    # Google usually resolves without the exchange suffix for majors
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

    show_alpha_curve = st.sidebar.checkbox("Show alpha curve", value=True)
    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    # ---------- DATA PULL ----------
    needed_tickers = weights_df["ticker"].unique().tolist()
    needed_tickers.append(BENCHMARK_TICKER)

    prices, engine_status = fetch_price_history(needed_tickers, lookback_days)
    vix_series = fetch_vix_series(lookback_days)
    vix_last = float(vix_series.iloc[-1]) if not vix_series.empty else np.nan

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

    cutoff = dates[-1] - timedelta(days=lookback_days)
    wave_curve = wave_curve[wave_curve.index >= cutoff]
    bench_curve = bench_curve[bench_curve.index >= cutoff]
    wave_rets = wave_rets[wave_rets.index >= cutoff]
    bench_rets = bench_rets[bench_rets.index >= cutoff]

    # ---------- TOP METRICS ----------
    if len(wave_curve) >= 2 and len(bench_curve) >= 2:
        wave_today = float(wave_curve.iloc[-1] / wave_curve.iloc[-2] - 1.0)
        bench_today = float(bench_curve.iloc[-1] / bench_curve.iloc[-2] - 1.0)
    else:
        wave_today = np.nan
        bench_today = np.nan

    today_alpha = (
        wave_today - bench_today
        if not (np.isnan(wave_today) or np.isnan(bench_today))
        else np.nan
    )
    est_beta = estimate_beta(wave_rets, bench_rets)
    exposure, smartsafe = compute_exposure_and_smartsafe(vix_last, mode_key)

    # ---------- HEADER ----------
    col_header_left, col_header_right = st.columns([4, 1])
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
        st.markdown("**Today Alpha**")
        st.markdown(
            f"<span style='font-size:1.6rem;color:{metric_color(today_alpha)};'>"
            f"{'' if np.isnan(today_alpha) else f'{today_alpha*100:.2f}%'}"
            "</span>",
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown("**Exposure**")
        st.markdown(
            f"<span style='font-size:1.6rem;'>{exposure*100:.1f}%</span>",
            unsafe_allow_html=True,
        )

    with m5:
        st.markdown("**SmartSafe™ Allocation**")
        st.markdown(
            f"<span style='font-size:1.6rem;'>{smartsafe*100:.1f}%</span>",
            unsafe_allow_html=True,
        )

    with m6:
        st.markdown("**VIX (latest)**")
        st.markdown(
            f"<span style='font-size:1.6rem;'>{'' if np.isnan(vix_last) else f'{vix_last:.1f}'}</span>",
            unsafe_allow_html=True,
        )
        st.caption(MODE_LABELS[mode_key])

    st.markdown("---")

    # ---------- MAIN CHART + TOP HOLDINGS ----------
    left_main, right_main = st.columns([3, 2])

    with left_main:
        st.subheader(f"{selected_wave} vs Benchmark")
        chart_df = pd.DataFrame(
            {
                "Wave": wave_curve,
                "Benchmark": bench_curve,
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

    # ---------- LOWER CHARTS ----------
    st.markdown("---")
    c1, c2, c3 = st.columns([3, 3, 2])

    with c1:
        st.subheader("Cumulative Alpha (Wave – Scaled Benchmark)")
        alpha_curve = (1 + (wave_rets - bench_rets)).cumprod() - 1
        st.line_chart(alpha_curve.rename("Alpha"))

    with c2:
        st.subheader("SPY (Benchmark) – Price")
        st.line_chart(prices[BENCHMARK_TICKER].rename("SPY"))

    with c3:
        st.subheader("VIX – Level")
        st.line_chart(vix_series)

    if show_debug:
        st.markdown("---")
        st.subheader("Debug Info")
        st.write("Weights DataFrame", weights_df)
        st.write("Prices (tail)", prices.tail())
        st.write("Wave returns (tail)", wave_rets.tail())
        st.write("Benchmark returns (tail)", bench_rets.tail())
        st.write("Engine status:", engine_status)


if __name__ == "__main__":
    main()
