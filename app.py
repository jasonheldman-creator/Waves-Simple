import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ======================================================================
# CONFIG
# ======================================================================

TARGET_BETA = 0.90  # Alpha-Minus-Beta discipline anchor

# Built-in default Waves + holdings (used if CSVs are missing or invalid)
WAVE_TICKERS_DEFAULT = {
    "S&P Wave": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "BRK.B", "JPM", "JNJ", "V", "PG",
    ],
    "Growth Wave": [
        "TSLA", "NVDA", "AMD", "AVGO", "ADBE",
        "CRM", "SHOP", "NOW", "NFLX", "INTU",
    ],
    "Future Power & Energy Wave": [
        "XOM", "CVX", "NEE", "DUK", "ENPH",
        "FSLR", "LNG", "SLB", "PSX", "PLUG",
    ],
    "Small Cap Growth Wave": [
        "TTD", "PLTR", "TWLO", "OKTA", "ZS",
        "DDOG", "APPF", "MDB", "ESTC", "SMAR",
    ],
    "Small-Mid Cap Growth Wave": [
        "CRWD", "ZS", "DDOG", "OKTA", "TEAM",
        "SNOW", "NET", "BILL", "DOCU", "PATH",
    ],
    "Clean Transit-Infrastructure Wave": [
        "TSLA", "NIO", "GM", "F", "BLDP",
        "PLUG", "CHPT", "ABB", "DE", "URI",
    ],
    "Quantum Computing Wave": [
        "NVDA", "AMD", "IBM", "MSFT", "GOOGL",
        "AMZN", "IONQ", "QUBT", "BRKS", "INTC",
    ],
    "Crypto Income Wave": [
        "COIN", "MSTR", "RIOT", "MARA", "HUT",
        "CLSK", "BITO", "IBIT", "FBTC", "DEFI",
    ],
    "Income Wave": [
        "TLT", "LQD", "HYG", "JNJ", "PG",
        "KO", "PEP", "XLU", "O", "VZ",
    ],
}

# ======================================================================
# DATA: LOAD OR BUILD (AUTO LIVE/DEMO)
# ======================================================================

@st.cache_data
def load_or_build_wave_weights():
    """
    Try to load wave_weights.csv (wave,ticker,weight).
    If missing/invalid, build a full default table from WAVE_TICKERS_DEFAULT.
    Returns: (df, is_demo)
    """
    required = {"wave", "ticker", "weight"}
    try:
        df = pd.read_csv("wave_weights.csv")
        if not required.issubset(df.columns):
            raise ValueError(f"wave_weights.csv missing {required - set(df.columns)}")
        return df, False
    except Exception as e:
        st.warning(f"Using built-in demo wave_weights (reason: {e})")
        rows = []
        for wave, tickers in WAVE_TICKERS_DEFAULT.items():
            w = 1.0 / len(tickers)
            for t in tickers:
                rows.append({"wave": wave, "ticker": t, "weight": w})
        return pd.DataFrame(rows), True


@st.cache_data
def load_or_build_market_history():
    """
    Try to load market_history.csv (date,symbol,close).
    If missing/invalid, build synthetic SPY + VIX series for 180 days.
    Returns: (df, is_demo)
    """
    required = {"date", "symbol", "close"}
    try:
        df = pd.read_csv("market_history.csv", parse_dates=["date"])
        if not required.issubset(df.columns):
            raise ValueError(f"market_history.csv missing {required - set(df.columns)}")
        df = df.sort_values(["symbol", "date"])
        return df, False
    except Exception as e:
        st.warning(f"Using built-in demo market_history (reason: {e})")

        num_days = 180
        end_date = datetime.today().date()
        dates = [end_date - timedelta(days=i) for i in range(num_days)]
        dates = sorted(dates)

        rows = []
        spy_price = 450.0
        vix_level = 18.0
        np.random.seed(42)

        for d in dates:
            # SPY random walk
            dr = np.random.normal(0.0005, 0.01)
            spy_price *= (1 + dr)
            rows.append({"date": d, "symbol": "SPY", "close": spy_price})

            # VIX drift
            dv = np.random.normal(0.0, 0.4)
            vix_level = max(10.0, vix_level + dv)
            rows.append({"date": d, "symbol": "VIX", "close": vix_level})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["symbol", "date"]), True


@st.cache_data
def load_or_build_wave_history():
    """
    Try to load wave_history.csv (date,wave,portfolio_return,benchmark_return).
    If missing/invalid, generate a full 252-day history for each Wave using
    WAVE_TICKERS_DEFAULT.
    Returns: (df, is_demo)
    """
    required = {"date", "wave", "portfolio_return", "benchmark_return"}
    try:
        df = pd.read_csv("wave_history.csv", parse_dates=["date"])
        if not required.issubset(df.columns):
            raise ValueError(f"wave_history.csv missing {required - set(df.columns)}")
        return df.sort_values(["wave", "date"]), False
    except Exception as e:
        st.warning(f"Using built-in demo wave_history (reason: {e})")

        waves = sorted(WAVE_TICKERS_DEFAULT.keys())
        num_days = 252
        end_date = datetime.today().date()
        dates = [end_date - timedelta(days=i) for i in range(num_days)]
        dates = sorted(dates)

        rows = []
        np.random.seed(43)

        for wave in waves:
            name = wave.lower()
            if "crypto" in name:
                port_mu, port_sigma = 0.0012, 0.03
                bench_mu, bench_sigma = 0.0009, 0.025
            elif "income" in name:
                port_mu, port_sigma = 0.0003, 0.005
                bench_mu, bench_sigma = 0.00025, 0.004
            elif "s&p" in name:
                port_mu, port_sigma = 0.00045, 0.011
                bench_mu, bench_sigma = 0.00040, 0.010
            else:
                port_mu, port_sigma = 0.0006, 0.012
                bench_mu, bench_sigma = 0.00045, 0.010

            for d in dates:
                pr = float(np.random.normal(port_mu, port_sigma))
                br = float(np.random.normal(bench_mu, bench_sigma))
                rows.append(
                    {
                        "date": d,
                        "wave": wave,
                        "portfolio_return": pr,
                        "benchmark_return": br,
                    }
                )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["wave", "date"]), True


# ======================================================================
# METRICS ENGINE
# ======================================================================

def compute_wave_metrics(wave_history: pd.DataFrame) -> pd.DataFrame:
    metrics_rows = []

    windows = {
        "alpha_30d": 30,
        "alpha_60d": 60,
        "alpha_6m": 126,
        "alpha_1y": 252,
    }

    last_date = wave_history["date"].max()

    for wave, g in wave_history.groupby("wave"):
        g = g.sort_values("date")
        row = {"wave": wave}

        # Alpha windows (cumulative excess vs benchmark)
        for name, lookback in windows.items():
            window = g.iloc[-lookback:] if len(g) >= lookback else g
            pr = window["portfolio_return"].astype(float)
            br = window["benchmark_return"].astype(float)

            port_cum = (1 + pr).prod() - 1
            bench_cum = (1 + br).prod() - 1
            row[name] = port_cum - bench_cum

        # Realized beta (252d)
        win = g.iloc[-252:] if len(g) >= 252 else g
        pr = win["portfolio_return"].astype(float)
        br = win["benchmark_return"].astype(float)

        if br.var() > 0 and len(br) > 5:
            cov = np.cov(br, pr)[0, 1]
            beta = cov / br.var()
        else:
            beta = np.nan

        row["beta_252d"] = beta
        row["beta_drift"] = beta - TARGET_BETA if not np.isnan(beta) else np.nan

        # Explicit 1Y alpha (absolute)
        port_cum_1y = (1 + pr).prod() - 1
        bench_cum_1y = (1 + br).prod() - 1
        row["alpha_1y_abs"] = port_cum_1y - bench_cum_1y

        row["last_date"] = last_date
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows)

    # Z-scored leadership composite
    for col in ["alpha_30d", "alpha_60d", "alpha_6m", "alpha_1y"]:
        mean = metrics[col].mean()
        std = metrics[col].std()
        if std == 0 or np.isnan(std):
            metrics[col + "_z"] = 0
        else:
            metrics[col + "_z"] = (metrics[col] - mean) / std

    metrics["leadership_score"] = (
        0.4 * metrics["alpha_30d_z"]
        + 0.3 * metrics["alpha_60d_z"]
        + 0.2 * metrics["alpha_6m_z"]
        + 0.1 * metrics["alpha_1y_z"]
    )

    metrics = metrics.sort_values("leadership_score", ascending=False)
    n = len(metrics)
    tiers = []
    for i in range(n):
        pct = (i + 1) / n
        if pct <= 0.33:
            tiers.append("Tier 1 — Leader")
        elif pct <= 0.66:
            tiers.append("Tier 2 — Supporting")
        else:
            tiers.append("Tier 3 — Lagging")
    metrics["tier"] = tiers

    return metrics


# ======================================================================
# ALLOCATION & POSITION ENGINE
# ======================================================================

def vix_to_equity_pct(vix_level: float) -> float:
    if np.isnan(vix_level):
        return 0.6
    if vix_level < 12:
        return 1.0
    elif vix_level < 15:
        return 0.9
    elif vix_level < 18:
        return 0.8
    elif vix_level < 22:
        return 0.65
    elif vix_level < 26:
        return 0.5
    elif vix_level < 32:
        return 0.35
    else:
        return 0.2


def build_suggested_allocations(metrics: pd.DataFrame, vix_level: float):
    equity_pct = vix_to_equity_pct(vix_level)
    df = metrics.copy()

    # make strictly positive weights
    min_score = df["leadership_score"]..min()
    shift = -min_score if min_score < 0 else 0
    df["adj_score"] = df["leadership_score"] + shift + 1e-6

    total = df["adj_score"].sum()
    if total <= 0:
        df["wave_equity_weight"] = 1.0 / len(df)
    else:
        df["wave_equity_weight"] = df["adj_score"] / total

    floor, cap = 0.03, 0.35
    df["wave_equity_weight"] = df["wave_equity_weight"].clip(lower=floor, upper=cap)
    df["wave_equity_weight"] = df["wave_equity_weight"] / df["wave_equity_weight"].sum()

    df["suggested_portfolio_weight"] = df["wave_equity_weight"] * equity_pct
    return equity_pct, df


def build_position_allocations(
    wave_weights: pd.DataFrame,
    wave_allocs: pd.DataFrame,
) -> pd.DataFrame:
    merged = wave_weights.merge(
        wave_allocs[["wave", "suggested_portfolio_weight"]],
        on="wave",
        how="inner",
        validate="many_to_one",
    )
    merged["position_weight"] = merged["weight"] * merged["suggested_portfolio_weight"]
    return merged.sort_values("position_weight", ascending=False)


# ======================================================================
# UI HELPERS
# ======================================================================

def google_finance_link(ticker: str) -> str:
    safe = ticker.strip().upper()
    return f"https://www.google.com/finance/quote/{safe}"


def style_wave_table(df: pd.DataFrame):
    def highlight_tier(row):
        tier = row.get("tier", "")
        if isinstance(tier, str) and tier.startswith("Tier 1"):
            return ["background-color: rgba(0, 255, 0, 0.15)"] * len(row)
        elif isinstance(tier, str) and tier.startswith("Tier 2"):
            return ["background-color: rgba(255, 255, 0, 0.10)"] * len(row)
        elif isinstance(tier, str) and tier.startswith("Tier 3"):
            return ["background-color: rgba(255, 0, 0, 0.10)"] * len(row)
        else:
            return [""] * len(row)

    fmt_cols = {
        "alpha_30d": "{:.2%}",
        "alpha_60d": "{:.2%}",
        "alpha_6m": "{:.2%}",
        "alpha_1y": "{:.2%}",
        "alpha_1y_abs": "{:.2%}",
        "beta_252d": "{:.2f}",
        "beta_drift": "{:+.2f}",
        "leadership_score": "{:+.2f}",
    }
    existing_fmt = {k: v for k, v in fmt_cols.items() if k in df.columns}
    return df.style.format(existing_fmt).apply(highlight_tier, axis=1)


# ======================================================================
# MAIN APP
# ======================================================================

def main():
    st.set_page_config(
        page_title="WAVES Institutional Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load data (LIVE or DEMO)
    wave_weights, ww_demo = load_or_build_wave_weights()
    wave_history, wh_demo = load_or_build_wave_history()
    market_history, mh_demo = load_or_build_market_history()

    metrics = compute_wave_metrics(wave_history)

    vix_df = market_history[market_history["symbol"].str.upper().isin(["^VIX", "VIX"])]
    spy_df = market_history[market_history["symbol"].str.upper().isin(["SPY"])]
    vix_latest = float(vix_df.sort_values("date")["close"].iloc[-1]) if not vix_df.empty else np.nan

    equity_pct, wave_allocs = build_suggested_allocations(metrics, vix_latest)
    position_allocs = build_position_allocations(wave_weights, wave_allocs)

    # Determine DATA MODE badge
    demo_flags = [ww_demo, wh_demo, mh_demo]
    if all(flag is False for flag in demo_flags):
        data_mode = "LIVE"
        badge_color = "#16c784"  # green
    elif all(flag is True for flag in demo_flags):
        data_mode = "DEMO"
        badge_color = "#ff9800"  # orange
    else:
        data_mode = "MIXED"
        badge_color = "#f4c842"  # yellow

    # HEADER
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div>
            <h1 style="margin-bottom:0">WAVES Institutional Console</h1>
            <p style="color:#8f9bb3;margin-top:0">
              Adaptive Portfolio Waves™ • WAVES Intelligence™ • Alpha-Minus-Beta Discipline
            </p>
          </div>
          <div>
            <span style="
                padding:6px 14px;
                border-radius:999px;
                background-color:{badge_color};
                color:#0b1020;
                font-weight:600;
                font-size:13px;
                letter-spacing:0.08em;
                text-transform:uppercase;
            ">
              Data Mode: {data_mode}
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # TOP PANEL — MARKET REGIME + SNAPSHOT
    # ------------------------------------------------------------------
    top_left, top_right = st.columns([1.5, 2])

    with top_left:
        st.subheader("Market Regime — VIX & SPY")

        if not vix_df.empty and not spy_df.empty:
            chart_df = pd.DataFrame(
                {"date": spy_df["date"], "SPY": spy_df["close"].values}
            ).merge(
                vix_df[["date", "close"]].rename(columns={"close": "VIX"}),
                on="date",
                how="left",
            )
            chart_df = chart_df.set_index("date")
            st.line_chart(chart_df)
        else:
            st.info("VIX/SPY chart using demo data (SPY/VIX series).")

        st.metric("Latest VIX", f"{vix_latest:.2f}" if not np.isnan(vix_latest) else "N/A")
        st.metric("Equity Sleeve %", f"{equity_pct:.0%}")
        st.metric("SmartSafe %", f"{1 - equity_pct:.0%}")

    with top_right:
        st.subheader("Wave Leadership & Risk Snapshot")
        display_cols = [
            "wave",
            "alpha_30d",
            "alpha_60d",
            "alpha_6m",
            "alpha_1y",
            "beta_252d",
            "beta_drift",
            "leadership_score",
            "tier",
        ]
        snapshot_df = metrics[display_cols].copy()
        st.dataframe(style_wave_table(snapshot_df), use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # MIDDLE PANEL — ALLOCATIONS + ALERTS
    # ------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Suggested Wave Allocation (Equity Sleeve)")
        alloc_view = wave_allocs[
            ["wave", "tier", "leadership_score", "wave_equity_weight", "suggested_portfolio_weight"]
        ].copy()
        alloc_view["wave_equity_weight"] = alloc_view["wave_equity_weight"].map(lambda x: f"{x:.1%}")
        alloc_view["suggested_portfolio_weight"] = alloc_view["suggested_portfolio_weight"].map(lambda x: f"{x:.1%}")
        alloc_view = alloc_view.sort_values("suggested_portfolio_weight", ascending=False)
        st.dataframe(alloc_view, use_container_width=True)

    with col2:
        st.subheader("Risk Alerts")
        alerts = []

        for _, row in metrics.iterrows():
            beta = row["beta_252d"]
            if pd.isna(beta):
                continue
            drift = row["beta_drift"]
            if abs(drift) > 0.07:
                alerts.append(f"⚠️ {row['wave']}: Beta drift {drift:+.2f} vs target {TARGET_BETA:.2f}")
            if beta > 1.10:
                alerts.append(f"🚨 {row['wave']}: High beta {beta:.2f}")

        if not np.isnan(vix_latest):
            if vix_latest >= 26:
                alerts.append(f"🚨 High VIX regime ({vix_latest:.2f}) — SmartSafe overlay strongly engaged.")
            elif vix_latest >= 20:
                alerts.append(f"⚠️ Elevated VIX ({vix_latest:.2f}) — risk throttling partially engaged.")

        if not alerts:
            st.success("No critical risk alerts. Data within parameters.")
        else:
            for a in alerts:
                st.write(a)

    st.markdown("---")

    # ------------------------------------------------------------------
    # POSITION-LEVEL VIEW
    # ------------------------------------------------------------------
    st.subheader("Position-Level Allocation (Equity Sleeve)")
    pos_view = position_allocs.copy()
    pos_view["Google Finance"] = pos_view["ticker"].apply(google_finance_link)
    pos_view["position_weight"] = pos_view["position_weight"].map(lambda x: f"{x:.2%}")
    st.dataframe(
        pos_view[["wave", "ticker", "weight", "position_weight", "Google Finance"]],
        use_container_width=True,
    )

    # ------------------------------------------------------------------
    # TOP 10 HOLDINGS PER WAVE
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Top 10 Holdings per Wave")
    waves = sorted(wave_weights["wave"].unique())
    selected_wave = st.selectbox("Select Wave", waves)
    ww = wave_weights[wave_weights["wave"] == selected_wave].copy()
    ww = ww.sort_values("weight", ascending=False).head(10)
    ww["Google Finance"] = ww["ticker"].apply(google_finance_link)
    ww["weight"] = ww["weight"].map(lambda x: f"{x:.2%}")
    st.dataframe(ww[["ticker", "weight", "Google Finance"]], use_container_width=True)

    # ------------------------------------------------------------------
    # CSV EXPORTS
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Download CSVs")

    st.download_button(
        "Download Wave Metrics (Alpha/Beta/Leadership)",
        data=metrics.to_csv(index=False),
        file_name="wave_metrics.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download Suggested Wave Allocations",
        data=wave_allocs.to_csv(index=False),
        file_name="wave_allocations.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download Position-Level Allocations",
        data=position_allocs.to_csv(index=False),
        file_name="position_allocations.csv",
        mime="text/csv",
    )


# ======================================================================
# ENTRYPOINT
# ======================================================================

if __name__ == "__main__":
    main()
