import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------
# 1. WAVE DEFINITIONS (9-WAVE LINEUP, CODE-MANAGED WEIGHTS)
# ---------------------------------------------------------

WAVE_TICKERS = {
    # 1) AI_Wave – core AI mega-cap + platforms
    "AI_Wave": [
        "NVDA", "MSFT", "META", "GOOGL", "AMZN",
        "AVGO", "ADBE", "CRM", "SMCI", "TSLA",
    ],

    # 2) Growth_Wave – secular large-cap growth compounders
    "Growth_Wave": [
        "AAPL", "MSFT", "V", "MA", "COST",
        "LIN", "ASML", "ADBE", "NFLX", "NOW",
    ],

    # 3) SmallCapGrowth_Wave – U.S. small-cap growth bias
    "SmallCapGrowth_Wave": [
        "ZS", "DDOG", "NET", "APP", "MDB",
        "SMAR", "OKTA", "ESTC", "FSLY", "BILL",
    ],

    # 4) MidCapGrowth_Wave – mid-cap innovators
    "MidCapGrowth_Wave": [
        "PLTR", "SHOP", "TWLO", "ROKU", "TEAM",
        "U", "ON", "ENPH", "FSLR", "ALGN",
    ],

    # 5) FuturePowerEnergy_Wave – future power, grid & energy tech
    "FuturePowerEnergy_Wave": [
        "NEE", "ENPH", "FSLR", "PLUG", "SEDG",
        "RUN", "DQ", "ORA", "AES", "ED",
    ],

    # 6) QuantumComputing_Wave – quantum/HPC stack
    "QuantumComputing_Wave": [
        "IBM", "NVDA", "AMD", "TSM", "ASML",
        "AMAT", "LRCX", "QUBT", "INTC", "ACLX",
    ],

    # 7) CleanTransitInfra_Wave – EV + transit infrastructure
    "CleanTransitInfra_Wave": [
        "TSLA", "RIVN", "LCID", "BYDDF", "NIO",
        "CHPT", "BLNK", "ALB", "F", "GM",
    ],

    # 8) CryptoIncome_Wave – ETF-wrapped crypto / blockchain
    "CryptoIncome_Wave": [
        "IBIT", "BITB", "GBTC", "BLOK", "ARKW",
        "COIN", "MARA", "RIOT", "HUT", "BITX",
    ],

    # 9) SP500_Wave – S&P 500 benchmark wave (SPY only for now)
    "SP500_Wave": [
        "SPY",
    ],
}

BENCHMARK_TICKER = "SPY"
VIX_TICKER = "^VIX"


def build_wave_weights() -> pd.DataFrame:
    """Build equal-weight mapping from WAVE_TICKERS."""
    rows = []
    for wave_name, tickers in WAVE_TICKERS.items():
        if not tickers:
            continue
        w = 1.0 / len(tickers)
        for t in tickers:
            rows.append({"wave": wave_name, "ticker": t.upper(), "weight": w})
    return pd.DataFrame(rows, columns=["wave", "ticker", "weight"])


DEFAULT_WAVE_WEIGHTS = build_wave_weights()

# ---------------------------------------------------------
# 2. DATA FETCHING
# ---------------------------------------------------------

@st.cache_data(show_spinner=True, ttl=60 * 30)
def fetch_price_history(all_tickers, start, end):
    """
    Fetch daily adjusted close prices for given tickers between start and end.
    Returns a DataFrame with one column per ticker.
    """
    all_tickers = sorted({t.upper() for t in all_tickers})
    if not all_tickers:
        return pd.DataFrame()

    try:
        data = yf.download(
            all_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        st.error(f"Error fetching price history from Yahoo Finance: {e}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    # If multi-index columns, grab Adj Close
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.levels[0]:
            prices = data["Adj Close"].copy()
        elif "Close" in data.columns.levels[0]:
            prices = data["Close"].copy()
        else:
            prices = data.xs(data.columns.levels[-1][0], axis=1, level=0)
    else:
        prices = data.copy()

    # Normalize column labels to just ticker symbols
    prices.columns = [str(c).upper().split()[0] for c in prices.columns]
    prices = prices.dropna(axis=1, how="all")
    return prices


@st.cache_data(show_spinner=True, ttl=60 * 30)
def fetch_vix_history(start, end):
    try:
        vix = yf.download(
            VIX_TICKER,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return pd.Series(dtype=float)

    if vix.empty:
        return pd.Series(dtype=float)

    if "Adj Close" in vix.columns:
        series = vix["Adj Close"]
    else:
        series = vix["Close"]
    return series.dropna()


# ---------------------------------------------------------
# 3. PORTFOLIO & ALPHA ENGINE
# ---------------------------------------------------------

def compute_wave_returns(
    prices: pd.DataFrame,
    wave_name: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute daily returns for a wave and benchmark (SPY).
    Returns (wave_ret, spy_ret). Any missing tickers are dropped.
    """
    if prices.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    if BENCHMARK_TICKER not in prices.columns:
        st.warning(f"Benchmark {BENCHMARK_TICKER} not in price data.")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    wdf = DEFAULT_WAVE_WEIGHTS[
        DEFAULT_WAVE_WEIGHTS["wave"] == wave_name
    ].copy()

    if wdf.empty:
        st.warning(f"No tickers configured for {wave_name}.")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    tickers = [t.upper() for t in wdf["ticker"]]
    weights = wdf["weight"].values

    # Keep only tickers that exist in prices
    existing = [t for t in tickers if t in prices.columns]
    if not existing:
        st.error(f"No price data available for any tickers in {wave_name}.")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    if len(existing) != len(tickers):
        missing = sorted(set(tickers) - set(existing))
        st.warning(
            f"Missing price data for tickers (dropped): {', '.join(missing)}"
        )
        mask = [t in existing for t in tickers]
        weights = np.array(weights)[mask]
        weights = weights / weights.sum()
        tickers = existing

    ret = prices.pct_change().dropna(how="all")
    spy_ret = ret[BENCHMARK_TICKER].dropna()

    wave_cols = [t for t in tickers if t in ret.columns]
    if not wave_cols:
        st.error(f"No return data available for {wave_name}.")
        return pd.Series(dtype=float), spy_ret

    col_idx = [tickers.index(t) for t in wave_cols]
    w_vec = np.array(weights)[col_idx]

    wave_ret = (ret[wave_cols] * w_vec).sum(axis=1)

    common_index = wave_ret.index.intersection(spy_ret.index)
    wave_ret = wave_ret.loc[common_index]
    spy_ret = spy_ret.loc[common_index]

    return wave_ret, spy_ret


def realized_beta(wave_ret: pd.Series, spy_ret: pd.Series) -> float:
    if wave_ret.empty or spy_ret.empty:
        return np.nan
    x = spy_ret.values
    y = wave_ret.values
    if len(x) < 2:
        return np.nan
    cov = np.cov(y, x)[0, 1]
    var = np.var(x)
    if var <= 0:
        return np.nan
    return float(cov / var)


def alpha_window(wave_ret: pd.Series, spy_ret: pd.Series, days: int) -> float:
    """
    Total alpha (wave - SPY) over the last `days` of returns, in percent.
    """
    if len(wave_ret) <= days or len(spy_ret) <= days:
        return np.nan
    wr = (1 + wave_ret.iloc[-days:]).prod() - 1
    sr = (1 + spy_ret.iloc[-days:]).prod() - 1
    return float((wr - sr) * 100.0)


def cumulative_equity_curve(ret: pd.Series) -> pd.Series:
    if ret.empty:
        return pd.Series(dtype=float)
    return (1 + ret).cumprod()


# ---------------------------------------------------------
# 4. VIX-GATED EXPOSURE ENGINE
# ---------------------------------------------------------

def classify_vix_regime(latest_vix: float) -> tuple[str, float]:
    """
    Map VIX level to regime + base equity exposure (before modes).
    Returns: (regime_label, base_equity_exposure_0_to_1)
    """
    if np.isnan(latest_vix):
        return "Unknown", 0.75

    if latest_vix < 15:
        # Calm: offense
        return "Calm", 0.90
    elif 15 <= latest_vix < 25:
        # Normal: balanced offense/defense
        return "Normal", 0.75
    elif 25 <= latest_vix < 35:
        # Volatile / correction: more defense
        return "Volatile", 0.55
    else:
        # Stress / panic: big SmartSafe
        return "Stress", 0.35


def mode_exposure_adjustment(mode: str) -> float:
    """
    Mode-level adjustment. >1.0 = more aggressive, <1.0 = more defensive.
    """
    if mode == "Alpha-Minus-Beta":
        # Slightly more defensive
        return 0.9
    elif mode == "Private Logic™":
        # Slightly more aggressive (but still VIX-gated)
        return 1.1
    return 1.0  # Standard


def combine_exposure(
    base_exposure: float,
    mode: str,
    realized_beta_val: float,
    target_beta: float = 0.90,
) -> float:
    """
    Combine:
      - VIX-based base_exposure
      - Mode adjust (Standard / Alpha-Minus-Beta / Private Logic™)
      - Mild beta clamp toward target_beta
    """
    # Apply mode multiplier
    exposure = base_exposure * mode_exposure_adjustment(mode)

    # Gentle beta clamp: if beta too high, trim exposure a bit; if low, add a bit.
    if not np.isnan(realized_beta_val):
        beta_diff = realized_beta_val - target_beta  # + if too aggressive
        # Cap influence
        beta_diff = max(min(beta_diff, 0.3), -0.3)
        # Turn 0.3 beta difference into ~ +/- 10% exposure tweak
        exposure *= (1.0 - 0.35 * beta_diff)

    # Bound within [0,1]
    exposure = max(0.0, min(1.0, exposure))
    return float(exposure)


# ---------------------------------------------------------
# 5. STREAMLIT UI
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="WAVES Institutional Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ----- Sidebar -----
    st.sidebar.title("Wave & Mode")

    wave_names = list(WAVE_TICKERS.keys())
    default_wave = "AI_Wave" if "AI_Wave" in wave_names else wave_names[0]
    selected_wave = st.sidebar.selectbox("Select Wave", wave_names, index=wave_names.index(default_wave))

    mode = st.sidebar.selectbox(
        "Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    lookback_days = st.sidebar.slider(
        "Lookback (trading days)",
        min_value=60,
        max_value=730,
        value=365,
        step=5,
    )

    show_debug = st.sidebar.checkbox("Show debug info", value=False)

    # ----- Date range -----
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=int(lookback_days * 1.4))

    # ----- Fetch prices -----
    all_tickers = set()
    for ts in WAVE_TICKERS.values():
        all_tickers.update([t.upper() for t in ts])
    all_tickers.add(BENCHMARK_TICKER)

    prices = fetch_price_history(all_tickers, start_date, end_date)
    vix_series = fetch_vix_history(start_date, end_date)

    latest_vix = float(vix_series.iloc[-1]) if not vix_series.empty else np.nan
    vix_regime, vix_base_exposure = classify_vix_regime(latest_vix)

    # ----- Wave vs SPY returns -----
    wave_ret, spy_ret = compute_wave_returns(prices, selected_wave)

    if not wave_ret.empty and not spy_ret.empty:
        today_wave = wave_ret.iloc[-1] * 100.0
        today_spy = spy_ret.iloc[-1] * 100.0
        today_alpha = (wave_ret.iloc[-1] - spy_ret.iloc[-1]) * 100.0
        beta = realized_beta(wave_ret, spy_ret)
    else:
        today_wave = today_spy = today_alpha = np.nan
        beta = np.nan

    # ----- VIX-gated exposure + SmartSafe -----
    # Combine VIX-based exposure, mode effects, and beta clamp
    current_exposure = combine_exposure(
        base_exposure=vix_base_exposure,
        mode=mode,
        realized_beta_val=beta,
        target_beta=0.90,
    )
    smartsafe_alloc = 1.0 - current_exposure

    # ----- Alpha capture windows -----
    alpha_30d = alpha_window(wave_ret, spy_ret, 30)
    alpha_60d = alpha_window(wave_ret, spy_ret, 60)
    alpha_6m = alpha_window(wave_ret, spy_ret, 126)
    alpha_1y = alpha_window(wave_ret, spy_ret, 252)

    # ----- Equity curves -----
    wave_eq = cumulative_equity_curve(wave_ret)
    spy_eq = cumulative_equity_curve(spy_ret)

    # ----- Header -----
    st.markdown(
        """
        <h1 style="margin-bottom:0;">WAVES Institutional Console</h1>
        <p style="margin-top:4px; color:#999;">
        Adaptive Portfolio Waves • AIWs/APWs™ • SmartSafe™ • VIX-gated risk • Alpha-Minus-Beta & Private Logic™
        </p>
        """,
        unsafe_allow_html=True,
    )

    def fmt_pct(x):
        return "—" if np.isnan(x) else f"{x:+.2f}%"

    # ----- Top metrics row -----
    cols_top = st.columns(7)

    with cols_top[0]:
        st.caption("Wave Today")
        st.markdown(f"### {fmt_pct(today_wave)}")

    with cols_top[1]:
        st.caption("Benchmark Today (SPY)")
        st.markdown(f"### {fmt_pct(today_spy)}")

    with cols_top[2]:
        st.caption("Today Alpha Captured")
        st.markdown(f"### {fmt_pct(today_alpha)}")

    with cols_top[3]:
        st.caption("Realized Beta vs SPY")
        st.markdown("### " + ("—" if np.isnan(beta) else f"{beta:.2f}"))

    with cols_top[4]:
        st.caption("Current Exposure (VIX + Mode + Beta)")
        st.markdown(f"### {current_exposure * 100:.1f}%")

    with cols_top[5]:
        st.caption("SmartSafe™ Allocation Now")
        st.markdown(f"### {smartsafe_alloc * 100:.1f}%")

    with cols_top[6]:
        st.caption("Regime / Engine")
        engine_html = (
            f"<div><b>Regime:</b> {vix_regime} "
            f"(VIX {latest_vix:.1f} if not np.isnan(latest_vix) else '—')</div>"
        )
        engine_html += (
            "<div style='color:#00ff00; font-size:0.8rem;'>"
            f"Engine: SANDBOX • Last refresh: {dt.datetime.utcnow():%Y-%m-%d %H:%M UTC}"
            "</div>"
        )
        # Quick fix: format VIX safely
        if np.isnan(latest_vix):
            vix_display = "—"
        else:
            vix_display = f"{latest_vix:.1f}"
        engine_html = (
            f"<div><b>Regime:</b> {vix_regime} (VIX {vix_display})</div>"
            "<div style='color:#00ff00; font-size:0.8rem;'>"
            f"Engine: SANDBOX • Last refresh: {dt.datetime.utcnow():%Y-%m-%d %H:%M UTC}"
            "</div>"
        )
        st.markdown(engine_html, unsafe_allow_html=True)

    st.markdown("---")

    # ----- Alpha Captured Windows -----
    st.subheader("Alpha Captured Windows (This Wave Only)")
    cols_alpha = st.columns(4)
    for col, label, value in zip(
        cols_alpha,
        ["30D", "60D", "6M", "1Y"],
        [alpha_30d, alpha_60d, alpha_6m, alpha_1y],
    ):
        with col:
            st.caption(label)
            st.markdown("### " + (fmt_pct(value)))

    st.markdown("---")

    # ----- Equity Curves + Top Holdings -----
    left, right = st.columns([2.2, 1.3])

    with left:
        st.subheader(f"{selected_wave} vs Benchmark (Equity Curves)")
        if wave_eq.empty or spy_eq.empty:
            st.info("No price data available for this wave / benchmark.")
        else:
            eq_df = pd.DataFrame(
                {
                    "Wave": wave_eq,
                    "Benchmark": spy_eq,
                }
            )
            st.line_chart(eq_df)

    with right:
        st.subheader("Top Holdings (Static Weights)")
        wdf = DEFAULT_WAVE_WEIGHTS[
            DEFAULT_WAVE_WEIGHTS["wave"] == selected_wave
        ].copy()
        if wdf.empty:
            st.info("No holdings configured for this wave.")
        else:
            wdf = wdf.sort_values("weight", ascending=False)
            wdf["Weight %"] = wdf["weight"] * 100.0
            wdf["Google Quote"] = wdf["ticker"].apply(
                lambda t: f"https://www.google.com/finance/quote/{t}:NASDAQ"
            )
            display_df = wdf[["ticker", "Weight %", "Google Quote"]].rename(
                columns={"ticker": "Ticker"}
            )
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    # ----- SPY Price & VIX Level -----
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("SPY (Benchmark) – Price (Indexed)")
        if spy_eq.empty:
            st.info("No SPY price data available.")
        else:
            spy_df = pd.DataFrame({"SPY": spy_eq})
            st.line_chart(spy_df)

    with c2:
        st.subheader("VIX – Level")
        if vix_series.empty:
            st.info("No VIX data available.")
        else:
            st.line_chart(vix_series)

    # ----- Debug Info -----
    if show_debug:
        st.markdown("---")
        st.subheader("Debug Info")
        st.write("Selected wave:", selected_wave)
        st.write("Mode:", mode)
        st.write("Lookback days:", lookback_days)
        st.write("Available price columns:", list(prices.columns))
        st.write("Wave tickers:", WAVE_TICKERS[selected_wave])
        st.write(
            "Wave weights DF:",
            DEFAULT_WAVE_WEIGHTS[
                DEFAULT_WAVE_WEIGHTS["wave"] == selected_wave
            ].reset_index(drop=True),
        )
        st.write("Latest VIX:", latest_vix)
        st.write("VIX regime:", vix_regime)
        st.write("VIX base exposure:", vix_base_exposure)
        st.write("Realized beta:", beta)
        st.write("Final exposure:", current_exposure)


if __name__ == "__main__":
    main()
