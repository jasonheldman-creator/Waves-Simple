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

# ---------- CODE-MANAGED DEFAULT WAVE WEIGHTS ----------
# If wave_weights.csv is missing or malformed, the app will fall back to this.
# You can expand this to ALL 15 WAVES and full baskets – this is just a starter.
DEFAULT_WAVE_WEIGHTS = [
    # AI_Wave – example 10-name basket (weights sum to 1.0)
    {"wave": "AI_Wave", "ticker": "NVDA", "weight": 0.1722},
    {"wave": "AI_Wave", "ticker": "MSFT", "weight": 0.1581},
    {"wave": "AI_Wave", "ticker": "META", "weight": 0.1427},
    {"wave": "AI_Wave", "ticker": "GOOGL", "weight": 0.1195},
    {"wave": "AI_Wave", "ticker": "AMZN", "weight": 0.1025},
    {"wave": "AI_Wave", "ticker": "AVGO", "weight": 0.0842},
    {"wave": "AI_Wave", "ticker": "CRM",  "weight": 0.0707},
    {"wave": "AI_Wave", "ticker": "PLTR", "weight": 0.0598},
    {"wave": "AI_Wave", "ticker": "AMD",  "weight": 0.0488},
    {"wave": "AI_Wave", "ticker": "TSLA", "weight": 0.0415},

    # CleanTransitInfra_Wave – example basket (edit freely)
    {"wave": "CleanTransitInfra_Wave", "ticker": "TSLA", "weight": 0.20},
    {"wave": "CleanTransitInfra_Wave", "ticker": "NIO",  "weight": 0.20},
    {"wave": "CleanTransitInfra_Wave", "ticker": "NEE",  "weight": 0.20},
    {"wave": "CleanTransitInfra_Wave", "ticker": "PLUG", "weight": 0.20},
    {"wave": "CleanTransitInfra_Wave", "ticker": "BE",   "weight": 0.20},
]

DEFAULT_WAVES_ORDER = [
    "AI_Wave",
    "Growth_Wave",
    "Income_Wave",
    "SmallCap_Wave",
    "SmallMidGrowth_Wave",
    "FuturePowerEnergy_Wave",
    "CryptoIncome_Wave",
    "Quantum_Wave",
    "CleanTransitInfra_Wave",
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
def load_weights(path: str) -> pd.DataFrame:
    """
    Load wave weights from CSV, clean them, and if anything goes wrong,
    fall back to the code-managed DEFAULT_WAVE_WEIGHTS.
    """
    # Helper: default DataFrame built from code
    default_df = pd.DataFrame(DEFAULT_WAVE_WEIGHTS)[["wave", "ticker", "weight"]]

    # Try to read the CSV file safely
    try:
        try:
            # Newer pandas
            raw = pd.read_csv(path, on_bad_lines="skip")
        except TypeError:
            # Older pandas
            raw = pd.read_csv(path, error_bad_lines=False)
    except FileNotFoundError:
        st.warning(
            f"wave_weights.csv not found at '{path}'. "
            "Using code-managed default weights."
        )
        return default_df
    except Exception as e:
        st.warning(
            f"Could not parse wave_weights.csv ({e}). "
            "Using code-managed default weights."
        )
        return default_df

    # If we got nothing usable, fall back
    if raw is None or raw.empty:
        st.warning(
            "wave_weights.csv is empty or malformed. "
            "Using code-managed default weights."
        )
        return default_df

    # ---- Normalize columns (case-insensitive, flexible order) ----
    col_map = {c.strip().lower(): c for c in raw.columns}
    required = ["wave", "ticker", "weight"]
    missing = [r for r in required if r not in col_map]

    if missing:
        st.warning(
            "wave_weights.csv is missing required columns "
            f"{missing}. Using code-managed default weights."
        )
        return default_df

    wave_col = col_map["wave"]
    ticker_col = col_map["ticker"]
    weight_col = col_map["weight"]

    df = raw.copy()

    # ---- Clean up data types / whitespace ----
    df[wave_col] = df[wave_col].astype(str).str.strip()
    df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")

    # Drop rows where weight is NaN
    df = df.dropna(subset=[weight_col])

    if df.empty:
        st.warning(
            "No valid weights found in wave_weights.csv "
            "(after cleaning). Using code-managed default weights."
        )
        return default_df

    # ---- Normalize weights to sum to 1.0 within each wave ----
    def _normalize_group(g: pd.DataFrame) -> pd.DataFrame:
        total = g[weight_col].sum()
        if total <= 0:
            # If bad group, just return as-is; it will be dropped later
            return g
        g = g.copy()
        g[weight_col] = g[weight_col] / total
        return g

    df = df.groupby(wave_col, group_keys=False).apply(_normalize_group)

    # Drop any rows where weight is still invalid
    df = df.dropna(subset=[weight_col])
    df = df[df[weight_col] > 0]

    if df.empty:
        st.warning(
            "All rows in wave_weights.csv had invalid or zero weights "
            "after normalization. Using code-managed default weights."
        )
        return default_df

    # ---- Final standardized DataFrame ----
    cleaned = df.rename(
        columns={
            wave_col: "wave",
            ticker_col: "ticker",
            weight_col: "weight",
        }
    )[["wave", "ticker", "weight"]]

    # Try to overwrite a clean version of the CSV for future runs
    try:
        cleaned.to_csv(path, index=False)
    except Exception:
        # Non-fatal: if we can't write, just keep going
        pass

    return cleaned


@st.cache_data
def fetch_price_history(tickers, lookback_days: int):
    """
    Fetch price history for all tickers for the given lookback window.
    Returns (prices_df, engine_status) where engine_status is 'LIVE' or 'SANDBOX'.
    """
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days + 40)  # small buffer

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
            # Keep going; we'll handle total failure later
            continue

    if frames:
        prices = pd.concat(frames, axis=1).sort_index()
        prices = prices.loc[~prices.index.duplicated(keep="last")]
        prices = prices.dropna(how="all")
        if not prices.empty:
            return prices, ("LIVE" if live_ok else "SANDBOX")

    # --- If we reach here, build a synthetic sandbox dataset ---
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

    # synthetic VIX if needed
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

    # Align
    sub, bench = sub.align(bench, join="inner", axis=0)
    if sub.empty or bench.empty:
        return None

    w_vec = w_vec.reindex(avail_tickers).fillna(0)
    w_vec = w_vec / w_vec.sum()  # ensure normalized

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

    w = w.copy()
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
    return f"https://www.google.com/finance/quote/{ticker}:NASDAQ"


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

    weights_df = load_weights("wave_weights.csv")
    available_waves = sorted(weights_df["wave"].unique().tolist())

    # Keep default order where possible
    ordered_waves = [w for w in DEFAULT_WAVES_ORDER if w in available_waves]
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
    show_drawdown = st.sidebar.checkbox("Show drawdown", value=False)
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

    # Trim to lookback window
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

    today_alpha = wave_today - bench_today if not (np.isnan(wave_today) or np.isnan(bench_today)) else np.nan
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
            "<p style='opacity:0.8;'>Adaptive Portfolio Waves • AIWs/APWs • SmartSafe™ • "
            "VIX-gated risk • Alpha-Minus-Beta & Private Logic™</p>",
            unsafe_allow_html=True,
        )
    with col_header_right:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        status_color = "rgb(0, 200, 120)" if engine_status == "LIVE" else "rgb(255, 200, 0)"
        st.markdown(
            f"""
            <div style='text-align:right;'>
                <div style='font-size:0.9rem;'>
                    <span style='color:{status_color};font-weight:600;'>Engine Status: {engine_status}</span><br/>
                    <span style='font-size:0.8rem;opacity:0.8;'>Last refresh: {now_utc}</span>
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
            # Build HTML table with Google links and colorized returns
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
                "<thead><tr><th>Ticker</th><th style='text-align:right;'>Weight</th>"
                "<th style='text-align:right;'>Today%</th></tr></thead>"
                "<tbody>"
                + "".join(rows_html)
                + "</tbody></table>"
            )

            st.markdown(table_html, unsafe_allow_html=True)

    # ---------- LOWER CHARTS ----------
    st.markdown("---")
    c1, c2, c3 = st.columns([3, 3, 2])

    # Cumulative alpha
    with c1:
        st.subheader("Cumulative Alpha (Wave – Scaled Benchmark)")
        alpha_curve = (1 + (wave_rets - bench_rets)).cumprod() - 1
        st.line_chart(alpha_curve.rename("Alpha"))

    # SPY Price
    with c2:
        st.subheader("SPY (Benchmark) – Price")
        st.line_chart(prices[BENCHMARK_TICKER].rename("SPY"))

    # VIX Level
    with c3:
        st.subheader("VIX – Level")
        st.line_chart(vix_series)

    # ---------- OPTIONAL DEBUG ----------
    if show_debug:
        st.markdown("---")
        st.subheader("Debug Info")
        st.write("Weights DataFrame", weights_df.head())
        st.write("Prices (tail)", prices.tail())
        st.write("Wave returns (tail)", wave_rets.tail())
        st.write("Benchmark returns (tail)", bench_rets.tail())
        st.write("Engine status:", engine_status)


if __name__ == "__main__":
    main()
