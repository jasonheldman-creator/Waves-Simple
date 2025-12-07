import os
import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
PERF_DIR_CANDIDATES = ["logs/performance", "performance"]
POS_DIR_CANDIDATES = ["logs/positions", "positions"]
WEIGHTS_FILE = "wave_weights.csv"

# -----------------------------
# UTILS: FILE DISCOVERY
# -----------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def find_existing_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            return d
    return None


def discover_waves_from_weights():
    if not os.path.isfile(WEIGHTS_FILE):
        return []

    df = pd.read_csv(WEIGHTS_FILE)
    # Assume any column that is not obviously metadata is a Wave
    meta_like = {"ticker", "tickers", "symbol", "symbols", "name", "names",
                 "sector", "industry", "asset_class", "weight"}
    wave_cols = [c for c in df.columns if c.lower() not in meta_like]
    return wave_cols


def discover_waves_from_performance(perf_dir):
    waves = set()
    if perf_dir and os.path.isdir(perf_dir):
        for f in os.listdir(perf_dir):
            if f.endswith("_performance_daily.csv"):
                wave_name = f.replace("_performance_daily.csv", "")
                waves.add(wave_name)
    return sorted(waves)


def discover_waves():
    # Prefer explicit performance logs, else fall back to weights file
    perf_dir = find_existing_dir(PERF_DIR_CANDIDATES)
    waves_from_perf = discover_waves_from_performance(perf_dir) if perf_dir else []
    if waves_from_perf:
        return waves_from_perf

    waves_from_weights = discover_waves_from_weights()
    return sorted(waves_from_weights)


# -----------------------------
# DATA LOADING
# -----------------------------
def load_performance(wave_name: str) -> (pd.DataFrame, str):
    """
    Returns (df, mode):
      - df: performance dataframe (date indexed)
      - mode: "LIVE LOGS" or "STRUCTURE / DEMO"
    """
    perf_dir = find_existing_dir(PERF_DIR_CANDIDATES)
    if perf_dir:
        path = os.path.join(perf_dir, f"{wave_name}_performance_daily.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            # Try to standardize date column
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                df = df.set_index("date")
            elif "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").reset_index(drop=True)
                df = df.set_index("Date")
            else:
                # fabricate a date index if needed
                df = df.reset_index(drop=True)
                df.index = pd.date_range(end=datetime.today(), periods=len(df))

            return df, "LIVE LOGS"

    # Fallback: synthetic data
    df = generate_synthetic_performance(wave_name)
    return df, "STRUCTURE / DEMO"


def generate_synthetic_performance(wave_name: str, days: int = 180) -> pd.DataFrame:
    np.random.seed(abs(hash(wave_name)) % (2**32 - 1))
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
    mu = 0.0005  # ~12% annualized drift
    sigma = 0.01
    daily_rets = np.random.normal(mu, sigma, size=len(dates))
    values = 100 * (1 + pd.Series(daily_rets, index=dates)).cumprod()

    df = pd.DataFrame(
        {
            "portfolio_value": values,
            "daily_return": daily_rets,
            "expected_drift_daily": np.full(len(dates), mu),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def get_latest_positions_file(wave_name: str):
    for base_dir in POS_DIR_CANDIDATES:
        if not os.path.isdir(base_dir):
            continue
        pattern = os.path.join(base_dir, f"{wave_name}_positions_*.csv")
        files = glob.glob(pattern)
        if files:
            # pick the latest by date in filename if possible
            try:
                files_sorted = sorted(
                    files,
                    key=lambda x: datetime.strptime(
                        os.path.basename(x)
                        .replace(f"{wave_name}_positions_", "")
                        .replace(".csv", ""),
                        "%Y%m%d",
                    ),
                )
            except Exception:
                files_sorted = sorted(files, key=os.path.getmtime)
            return files_sorted[-1]
    return None


def load_positions(wave_name: str) -> pd.DataFrame | None:
    path = get_latest_positions_file(wave_name)
    if path is None or not os.path.isfile(path):
        return None

    df = pd.read_csv(path)
    # Normalize common column names
    if "Ticker" in df.columns:
        df.rename(columns={"Ticker": "ticker"}, inplace=True)
    if "Symbol" in df.columns:
        df.rename(columns={"Symbol": "ticker"}, inplace=True)
    if "Weight" in df.columns:
        df.rename(columns={"Weight": "weight"}, inplace=True)
    if "Name" in df.columns:
        df.rename(columns={"Name": "name"}, inplace=True)

    return df


# -----------------------------
# METRICS & ALPHA
# -----------------------------
def compute_returns_and_alpha(perf_df: pd.DataFrame):
    df = perf_df.copy()

    # Daily returns
    if "daily_return" in df.columns:
        daily_ret = df["daily_return"].astype(float)
    elif "portfolio_value" in df.columns:
        daily_ret = df["portfolio_value"].pct_change().fillna(0.0)
    elif "total_return" in df.columns:
        # assume total_return is cumulative since inception
        tr = df["total_return"].astype(float)
        daily_ret = tr.diff().fillna(tr.iloc[0])
    else:
        daily_ret = pd.Series(0.0, index=df.index)

    # Expected drift per day
    if "expected_drift_daily" in df.columns:
        drift = df["expected_drift_daily"].astype(float)
    else:
        drift = pd.Series(daily_ret.mean(), index=df.index)

    # Internal daily alpha = realized - expected drift
    daily_alpha = daily_ret - drift

    # Cumulative portfolio return since inception
    total_return = (1 + daily_ret).prod() - 1

    # 1-Day return (last observation)
    one_day_return = daily_ret.iloc[-1] if len(daily_ret) > 0 else 0.0

    # Max drawdown based on portfolio_value if available; else cumulated returns
    if "portfolio_value" in df.columns:
        series = df["portfolio_value"].astype(float)
    else:
        series = (1 + daily_ret).cumprod()

    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    max_drawdown = drawdown.min()

    # Internal alpha windows (ONLY 1-Day & 30-Day now)
    alpha_1d = daily_alpha.iloc[-1] if len(daily_alpha) > 0 else 0.0

    window_30 = min(30, len(daily_alpha))
    if window_30 > 0:
        alpha_30d = daily_alpha.iloc[-window_30:].sum()
    else:
        alpha_30d = 0.0

    # Expected daily drift (latest)
    expected_drift_daily = drift.iloc[-1] if len(drift) > 0 else 0.0

    # Volatility & basic “risk” metrics
    window_vol = min(30, len(daily_ret))
    if window_vol > 1:
        vol_30d = daily_ret.iloc[-window_vol:].std() * np.sqrt(252)
    else:
        vol_30d = 0.0

    metrics = {
        "total_return": total_return,
        "one_day_return": one_day_return,
        "max_drawdown": max_drawdown,
        "alpha_1d": alpha_1d,
        "alpha_30d": alpha_30d,
        "expected_drift_daily": expected_drift_daily,
        "vol_30d": vol_30d,
        "daily_return": daily_ret,
        "daily_alpha": daily_alpha,
        "drift": drift,
        "equity_curve": series,
    }
    return metrics


def fmt_pct(x, decimals=2):
    return f"{x*100:.{decimals}f}%"


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-small .stMetric-label {
        font-size: 0.85rem;
    }
    .metric-small .stMetric-value {
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("WAVES Intelligence™")
st.sidebar.caption("Institutional Console — Hybrid Mode")

waves = discover_waves()
if not waves:
    st.error(
        "No Waves discovered. Please ensure `wave_weights.csv` or performance logs exist."
    )
    st.stop()

selected_wave = st.sidebar.selectbox("Select Wave", waves)

perf_df, mode_label = load_performance(selected_wave)
metrics = compute_returns_and_alpha(perf_df)
positions_df = load_positions(selected_wave)

st.sidebar.markdown("---")
st.sidebar.subheader("Engine Mode")
if mode_label == "LIVE LOGS":
    st.sidebar.success("LIVE LOGS")
else:
    st.sidebar.warning("STRUCTURE / DEMO (synthetic)")

st.sidebar.markdown(
    f"**Wave:** `{selected_wave}`  \n"
    f"**Sessions:** {len(perf_df):,} days"
)

# -----------------------------
# MAIN HEADER & METRICS
# -----------------------------
st.title("WAVES Intelligence™ Institutional Console")
st.subheader(f"Wave: {selected_wave}")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Return (Since Inception)", fmt_pct(metrics["total_return"]))
with col2:
    st.metric("Intraday / 1-Day Return", fmt_pct(metrics["one_day_return"]))
with col3:
    st.metric("Max Drawdown", fmt_pct(metrics["max_drawdown"]))
with col4:
    st.metric("Internal Alpha (1-Day)", fmt_pct(metrics["alpha_1d"]))
with col5:
    st.metric("Internal Alpha (30-Day)", fmt_pct(metrics["alpha_30d"]))

st.markdown(
    f"**β-Adjusted Expected Daily Drift:** {fmt_pct(metrics['expected_drift_daily'], 4)}"
)

tabs = st.tabs(["Overview", "Alpha", "Engine Logs"])

# -----------------------------
# OVERVIEW TAB
# -----------------------------
with tabs[0]:
    st.markdown("### Performance Overview")

    # Performance curve
    st.markdown("#### Equity Curve")
    perf_chart_df = pd.DataFrame(
        {
            "Equity Curve": metrics["equity_curve"],
        }
    )
    st.line_chart(perf_chart_df)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### Top 10 Holdings")

        if positions_df is not None and not positions_df.empty:
            df_pos = positions_df.copy()

            if "weight" not in df_pos.columns:
                # Derive weights if not present
                if "position_value" in df_pos.columns:
                    total_val = df_pos["position_value"].sum()
                    if total_val != 0:
                        df_pos["weight"] = df_pos["position_value"] / total_val
                    else:
                        df_pos["weight"] = 1.0 / len(df_pos)
                else:
                    df_pos["weight"] = 1.0 / len(df_pos)

            # Sort by weight desc and take top 10
            df_pos = df_pos.sort_values("weight", ascending=False).reset_index(drop=True)
            top10 = df_pos.head(10).copy()

            # Normalize ticker/name columns
            if "ticker" not in top10.columns:
                # Try best-effort
                for c in top10.columns:
                    if c.lower() in ["ticker", "symbol"]:
                        top10.rename(columns={c: "ticker"}, inplace=True)
                        break
            if "name" not in top10.columns:
                for c in top10.columns:
                    if c.lower() == "name":
                        top10.rename(columns={c: "name"}, inplace=True)
                        break

            # Google Finance links (as URLs)
            def google_finance_url(ticker: str) -> str:
                # Generic Google Finance search that redirects to the right quote page
                return f"https://www.google.com/finance?q={ticker}"

            top10["Google Finance"] = top10["ticker"].astype(str).apply(google_finance_url)

            display_cols = []
            for c in ["ticker", "name", "weight", "Google Finance"]:
                if c in top10.columns:
                    display_cols.append(c)

            st.dataframe(
                top10[display_cols],
                use_container_width=True,
            )

            st.markdown("##### Quick Links (Google Finance)")
            for _, row in top10.iterrows():
                tkr = str(row.get("ticker", "")).strip()
                nm = row.get("name", "")
                url = row.get("Google Finance", "")
                if tkr and url:
                    label = f"{tkr}" if not nm else f"{tkr} — {nm}"
                    st.markdown(f"- [{label}]({url})")
        else:
            st.info(
                "No positions file found for this Wave yet. Once positions logs exist, the top 10 holdings and Google Finance links will appear here."
            )

    with col_right:
        st.markdown("#### Exposure & Risk")

        # Simple placeholder exposure metrics based on volatility
        vol_30d = metrics["vol_30d"]
        st.metric("30-Day Realized Volatility (Ann.)", f"{vol_30d*100:.2f}%")

        st.caption(
            "Exposure & risk metrics are computed from the Wave’s realized return stream. "
            "Additional factor/sector breakdowns can be layered here."
        )

        # Total vs expected drift chart (last 60 days)
        st.markdown("##### Total vs Expected Drift (Last 60 Days)")
        dr = metrics["daily_return"]
        drift = metrics["drift"]

        window = min(60, len(dr))
        if window > 0:
            drift_df = pd.DataFrame(
                {
                    "Realized Daily Return": dr.iloc[-window:],
                    "Expected Drift": drift.iloc[-window:],
                }
            )
            st.line_chart(drift_df)
        else:
            st.info("Not enough data to show drift comparison yet.")


# -----------------------------
# ALPHA TAB
# -----------------------------
with tabs[1]:
    st.markdown("### Internal Alpha")

    # Alpha table (ONLY 1-Day & 30-Day now)
    alpha_table = pd.DataFrame(
        {
            "Window": ["1-Day", "30-Day"],
            "Internal Alpha": [
                metrics["alpha_1d"],
                metrics["alpha_30d"],
            ],
            "Internal Alpha (pct)": [
                fmt_pct(metrics["alpha_1d"]),
                fmt_pct(metrics["alpha_30d"]),
            ],
        }
    )

    st.markdown("#### Alpha Summary (Internal, β-Adjusted)")
    st.table(alpha_table)

    daily_alpha = metrics["daily_alpha"]

    if len(daily_alpha) > 0:
        # Cumulative internal alpha chart
        st.markdown("#### Cumulative Internal Alpha (Since Inception)")
        cum_alpha = daily_alpha.cumsum()
        cum_df = pd.DataFrame({"Cumulative Alpha": cum_alpha})
        st.line_chart(cum_df)

        # Daily alpha chart
        st.markdown("#### Daily Internal Alpha")
        alpha_df = pd.DataFrame({"Daily Alpha": daily_alpha})
        st.bar_chart(alpha_df)
    else:
        st.info("No alpha series available yet for this Wave.")


# -----------------------------
# ENGINE LOGS TAB
# -----------------------------
with tabs[2]:
    st.markdown("### Engine Logs")

    st.markdown("#### Performance Data (Tail)")
    if perf_df is not None and not perf_df.empty:
        st.dataframe(perf_df.tail(50), use_container_width=True)
    else:
        st.info("No performance records available for this Wave yet.")

    st.markdown("#### Latest Positions Snapshot")
    if positions_df is not None and not positions_df.empty:
        st.dataframe(positions_df.head(100), use_container_width=True)
    else:
        st.info("No positions snapshot available for this Wave yet.")

    st.caption(
        "Hybrid Mode:\n"
        "- **LIVE LOGS**: Using real WAVES Engine logs from `logs/performance` & `logs/positions`.\n"
        "- **STRUCTURE / DEMO**: Synthetic data fills in so the console stays fully populated."
    )
