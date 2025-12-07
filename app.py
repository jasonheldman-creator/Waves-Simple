# app.py  — WAVES Institutional Console (Cloud Engine Edition)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# =========================
#  BASIC SETTINGS
# =========================

st.set_page_config(
    page_title="WAVES Institutional Console",
    layout="wide",
)

LOOKBACK_DAYS = 540          # ~18 months
BENCHMARK_SYMBOL = "SPY"     # main equity benchmark
VIX_SYMBOL = "^VIX"

# Nice wave labels for the dropdown; fall back to auto formatting if missing
WAVE_DISPLAY_NAMES = {
    "AI_Wave": "AI Leaders Wave",
    "Growth_Wave": "Growth Leaders Wave",
    "Quantum_Wave": "Quantum Computing Wave",
    "Income_Wave": "Income Wave",
    "SmallCap_Wave": "Small Cap Growth Wave",
    "SMID_Wave": "Small–Mid Cap Growth Wave",
    "FuturePower_Wave": "Future Power & Energy Wave",
    "CryptoIncome_Wave": "Crypto Income Wave",
    "CleanTransit_Wave": "Clean Transit & Infrastructure Wave",
    "SP500_Wave": "S&P 500 Core Wave",
}

# =========================
#  DATA HELPERS
# =========================

@st.cache_data(show_spinner=False)
def load_wave_weights(path: str = "wave_weights.csv") -> pd.DataFrame:
    """
    Load the master wave weights file.
    Expected columns (case-insensitive):
        - Ticker
        - Wave
        - Weight or Weight %
    """
    df = pd.read_csv(path)

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    # ticker
    if "ticker" not in cols:
        raise ValueError("wave_weights.csv must have a 'Ticker' column.")
    ticker_col = cols["ticker"]
    # wave
    if "wave" not in cols:
        raise ValueError("wave_weights.csv must have a 'Wave' column.")
    wave_col = cols["wave"]

    # weight
    weight_col = None
    for candidate in ["weight %", "weight_pct", "weight", "w"]:
        if candidate.lower() in cols:
            weight_col = cols[candidate.lower()]
            break
    if weight_col is None:
        raise ValueError(
            "wave_weights.csv must have a weight column "
            "(e.g. 'Weight %' or 'Weight')."
        )

    out = pd.DataFrame({
        "Ticker": df[ticker_col].astype(str).str.upper().str.strip(),
        "Wave": df[wave_col].astype(str).str.strip(),
        "Weight": pd.to_numeric(df[weight_col], errors="coerce")
    }).dropna(subset=["Weight"])

    # Normalize weights within each wave
    out["Weight"] = out.groupby("Wave")["Weight"].transform(
        lambda x: x / x.sum()
    )
    return out


@st.cache_data(show_spinner=False)
def bulletproof_benchmark(symbol: str, lookback_days: int = LOOKBACK_DAYS) -> pd.Series:
    """
    Bulletproof benchmark loader:
      1) Try Yahoo Finance (Adj Close)
      2) Try Stooq fallback (Close)
      3) Try local CSV in benchmarks/{symbol}_fallback.csv (Adj Close)

    Returns a pandas Series indexed by date.
    """
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    # 1) Yahoo Finance
    try:
        data = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )
        if data is not None and not data.empty:
            if "Adj Close" in data.columns:
                s = data["Adj Close"].dropna()
                if not s.empty:
                    return s
            elif "Close" in data.columns:
                s = data["Close"].dropna()
                if not s.empty:
                    return s
    except Exception:
        pass

    # 2) Stooq fallback (works well in cloud)
    try:
        stooq_symbol = symbol.lower()
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        df = pd.read_csv(url)
        if not df.empty and "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            if "Close" in df.columns:
                s = df["Close"].dropna()
                # Restrict to lookback window
                s = s[s.index >= start]
                if not s.empty:
                    return s
    except Exception:
        pass

    # 3) Local fallback CSV
    try:
        fallback_path = f"benchmarks/{symbol}_fallback.csv"
        df = pd.read_csv(fallback_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        col = "Adj Close" if "Adj Close" in df.columns else df.columns[-1]
        s = df[col].dropna()
        s = s[s.index >= start]
        if not s.empty:
            return s
    except Exception:
        pass

    raise ValueError(f"Benchmark {symbol} could not be loaded from any source.")


@st.cache_data(show_spinner=False)
def load_history_for_wave(
    wave_name: str,
    weights_df: pd.DataFrame,
    lookback_days: int = LOOKBACK_DAYS,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Load historic price data for all tickers in the given wave and compute:
      - Wave equity curve (100 = start)
      - SPY benchmark equity curve (100 = start)
      - Daily returns per ticker (for red/green holdings)
    """
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    wave_weights = weights_df[weights_df["Wave"] == wave_name].copy()
    if wave_weights.empty:
        raise ValueError(f"No weights found for wave '{wave_name}'.")

    tickers = wave_weights["Ticker"].unique().tolist()

    # Download prices for component tickers
    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )

    # Build a DataFrame of Adj Close (columns = tickers)
    price_frames = []
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                s = data[ticker]["Adj Close"]
            else:
                s = data["Adj Close"][ticker]
        except Exception:
            continue
        s = s.dropna()
        if s.empty:
            continue
        s.name = ticker
        price_frames.append(s)

    if not price_frames:
        raise ValueError("Could not download price data for any tickers in this Wave.")

    prices = pd.concat(price_frames, axis=1).sort_index()

    # Restrict to latest lookback window
    prices = prices[prices.index >= start]

    # Align weights to actual columns we have
    in_universe = [t for t in tickers if t in prices.columns]
    wave_weights = wave_weights[wave_weights["Ticker"].isin(in_universe)].copy()
    wave_weights["Weight"] = wave_weights["Weight"] / wave_weights["Weight"].sum()

    # Daily % returns per ticker
    rets = prices.pct_change().dropna(how="all")

    # Wave returns (weighted sum)
    w = wave_weights.set_index("Ticker").loc[prices.columns, "Weight"]
    wave_ret = (rets * w).sum(axis=1)

    # Build equity curve (100 = start)
    wave_equity = (1.0 + wave_ret).cumprod()
    wave_equity = 100.0 * wave_equity / wave_equity.iloc[0]

    # Benchmark curve
    spy = bulletproof_benchmark(BENCHMARK_SYMBOL, lookback_days=lookback_days)
    spy = spy[spy.index.isin(wave_equity.index)]
    spy = spy.sort_index()
    spy_ret = spy.pct_change().dropna()
    spy_equity = (1.0 + spy_ret).cumprod()
    spy_equity = 100.0 * spy_equity / spy_equity.iloc[0]

    # Re-align
    idx = wave_equity.index.intersection(spy_equity.index)
    wave_equity = wave_equity.loc[idx]
    spy_equity = spy_equity.loc[idx]
    rets = rets.loc[idx]

    return wave_equity, spy_equity, rets


@st.cache_data(show_spinner=False)
def load_vix_and_spy_snapshots(lookback_days: int = 90) -> tuple[pd.Series, pd.Series]:
    vix = bulletproof_benchmark(VIX_SYMBOL, lookback_days=lookback_days)
    spy = bulletproof_benchmark(BENCHMARK_SYMBOL, lookback_days=lookback_days)

    # Align
    idx = vix.index.intersection(spy.index)
    return vix.loc[idx], spy.loc[idx]


def pretty_wave_label(raw_name: str) -> str:
    if raw_name in WAVE_DISPLAY_NAMES:
        return WAVE_DISPLAY_NAMES[raw_name]
    # Fallback: convert "AI_Wave" -> "AI Wave"
    name = raw_name.replace("_", " ").strip()
    # Capitalize nicely
    return " ".join(w.capitalize() for w in name.split())


def compute_drawdown(equity_curve: pd.Series) -> float:
    """
    Max drawdown in percent (negative).
    """
    running_max = equity_curve.cummax()
    dd = (equity_curve / running_max) - 1.0
    return float(dd.min() * 100.0)


# =========================
#  UI LAYOUT
# =========================

weights_df = load_wave_weights()

all_waves = sorted(weights_df["Wave"].unique().tolist())
wave_label_map = {w: pretty_wave_label(w) for w in all_waves}
label_to_wave = {v: k for k, v in wave_label_map.items()}

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 🌊 WAVES Intelligence™")
    st.markdown("**DESKTOP ENGINE + CLOUD SNAPSHOT**")

    st.markdown("##### Select Wave")
    selected_label = st.selectbox(
        "Wave",
        options=list(label_to_wave.keys()),
        index=0,
        label_visibility="collapsed",
    )
    selected_wave = label_to_wave[selected_label]

    st.markdown("##### Risk Mode (label only)")
    risk_mode = st.radio(
        "Risk Mode",
        options=["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
        label_visibility="collapsed",
    )

    equity_exposure = st.slider(
        "Equity Exposure (target)",
        min_value=0,
        max_value=100,
        value=90,
        step=5,
    )
    st.caption(
        f"Target β ≈ {equity_exposure/100:.2f} in Standard mode. "
        "SmartSafe™ buffer = 100% − equity exposure."
    )

    st.divider()
    st.markdown(
        "Keep this console open while the cloud engine runs in the background. "
        "All returns & metrics are based on public market data via Yahoo Finance & fallbacks."
    )

# ---- Main body ----

st.title("WAVES Institutional Console")
st.caption(
    f"Live / demo console for WAVES Intelligence™ — showing **{selected_label}**.  "
    f"Risk Mode (label): **{risk_mode}**  |  Benchmark: **{BENCHMARK_SYMBOL}**"
)

# Load main data
try:
    wave_curve, spy_curve, returns = load_history_for_wave(
        selected_wave, weights_df, lookback_days=LOOKBACK_DAYS
    )
except Exception as e:
    st.error(f"Problem loading data for {selected_label}: {e}")
    st.stop()

# ---- Metrics ----

# Total return over lookback
wave_total_ret = (wave_curve.iloc[-1] / wave_curve.iloc[0] - 1.0) * 100.0
spy_total_ret = (spy_curve.iloc[-1] / spy_curve.iloc[0] - 1.0) * 100.0
alpha_total = wave_total_ret - spy_total_ret

# Today's return (last day return)
if len(wave_curve) > 1:
    wave_today_ret = (wave_curve.iloc[-1] / wave_curve.iloc[-2] - 1.0) * 100.0
else:
    wave_today_ret = 0.0

max_dd = compute_drawdown(wave_curve)

# Get VIX snapshot
try:
    vix_series, spy_snapshot = load_vix_and_spy_snapshots(lookback_days=90)
    vix_last = float(vix_series.iloc[-1])
    vix_change = ((vix_series.iloc[-1] / vix_series.iloc[-2]) - 1.0) * 100.0
except Exception:
    vix_last = np.nan
    vix_change = np.nan

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Return (lookback)", f"{wave_total_ret:0.2f}%")
with col2:
    st.metric("Today", f"{wave_today_ret:0.2f}%")
with col3:
    st.metric("Max Drawdown", f"{max_dd:0.2f}%")
with col4:
    alpha_display = f"{alpha_total:.2f}%"
st.metric("Alpha vs SPY", alpha_display)

st.caption(
    f"SPY (benchmark) over this window: {spy_total_ret:0.2f}%  "
    f"| VIX (spot): {vix_last:0.2f} ({vix_change:+0.2f}%)  "
    f"| Equity exposure in engine (label only): {equity_exposure}% / SmartSafe™ cash: {100-equity_exposure}%."
)

st.divider()

# ---- Performance Curve & Holdings ----

left, right = st.columns([2.0, 1.4])

with left:
    st.subheader("Performance Curve")

    perf_df = pd.DataFrame({
        selected_label: wave_curve,
        "SPY": spy_curve,
    })

    st.line_chart(perf_df)

    st.caption(
        "Curve is normalized to 100 at start of the lookback window. "
        "Source: Yahoo Finance (Adj Close) with fallbacks."
    )

with right:
    st.subheader("Holdings, Weights & Risk")

    # Top 10 by weight
    wave_weights = weights_df[weights_df["Wave"] == selected_wave].copy()
    wave_weights = wave_weights.sort_values("Weight", ascending=False).head(10)

    # Today returns per ticker (for red/green)
    today_rets = returns.iloc[-1].to_dict() if not returns.empty else {}

    # Build HTML table with Google Finance links & colored daily change
    rows_html = []
    header_html = """
        <tr>
            <th style="text-align:left;padding:4px 8px;">Ticker</th>
            <th style="text-align:right;padding:4px 8px;">Weight %</th>
            <th style="text-align:right;padding:4px 8px;">Today</th>
        </tr>
    """

    for _, row in wave_weights.iterrows():
        ticker = row["Ticker"]
        weight_pct = float(row["Weight"]) * 100.0
        today = float(today_rets.get(ticker, np.nan) * 100.0) if ticker in today_rets else np.nan

        # Today color
        if np.isnan(today):
            today_str = "—"
            color = "#cbd5f5"
        else:
            color = "#22c55e" if today >= 0 else "#ef4444"
            today_str = f"{today:+0.2f}%"

        url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
        ticker_link = f'<a href="{url}" target="_blank" style="text-decoration:none;color:#38bdf8;">{ticker}</a>'

        rows_html.append(
            f"""
            <tr>
                <td style="text-align:left;padding:4px 8px;">{ticker_link}</td>
                <td style="text-align:right;padding:4px 8px;">{weight_pct:0.2f}%</td>
                <td style="text-align:right;padding:4px 8px;color:{color};">{today_str}</td>
            </tr>
            """
        )

    holdings_html = f"""
        <div style="border-radius:6px;border:1px solid #4b5563;padding:8px 8px 4px 8px;">
            <div style="font-weight:600;margin-bottom:4px;">
                Top 10 Positions — Google Finance Links (Bloomberg-style red/green)
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
                {header_html}
                {''.join(rows_html)}
            </table>
        </div>
    """
    st.markdown(holdings_html, unsafe_allow_html=True)

    st.button("Full Wave universe table", disabled=True, help="Coming next: full universe detail view")

st.divider()

# ---- Mini SPY & VIX charts ----

st.subheader("Market Context — SPY & VIX (last 90 days)")

try:
    vix_series, spy_series = load_vix_and_spy_snapshots(lookback_days=90)
    mc_df = pd.DataFrame({
        "SPY": spy_series / spy_series.iloc[0] * 100.0,
        "VIX": vix_series,
    })
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**SPY (normalized to 100)**")
        st.line_chart(mc_df[["SPY"]])
    with c2:
        st.markdown("**VIX level**")
        st.line_chart(mc_df[["VIX"]])
except Exception as e:
    st.warning(f"Could not load SPY/VIX mini charts: {e}")

st.divider()

# ---- Disclaimers ----

st.caption(
    "WAVES Institutional Console — demo view only. Returns & metrics are based on "
    "public market data via Yahoo Finance and do not represent live trading or an offer of advisory services."
)
