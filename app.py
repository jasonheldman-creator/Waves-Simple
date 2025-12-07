import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ======================================================================
# CONFIG / CONSTANTS
# ======================================================================

TARGET_BETA = 0.90  # Alpha-Minus-Beta discipline anchor


# ======================================================================
# DATA LOADERS (CACHED)
# ======================================================================

@st.cache_data
def load_wave_history(path: str = "wave_history.csv") -> pd.DataFrame:
    """
    Expected schema (long format, one row per wave per day):
        date              (YYYY-MM-DD or ISO)
        wave              (e.g., 'S&P 500 Wave')
        portfolio_return  (daily pct return, e.g., 0.004 for +0.4%)
        benchmark_return  (daily pct return of its benchmark)
    """
    df = pd.read_csv(path, parse_dates=["date"])
    expected_cols = {"date", "wave", "portfolio_return", "benchmark_return"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"wave_history.csv is missing columns: {missing}")
    df = df.sort_values(["wave", "date"])
    return df


@st.cache_data
def load_market_history(path: str = "market_history.csv") -> pd.DataFrame:
    """
    Expected schema:
        date    (date)
        symbol  (e.g. 'SPY', '^VIX', 'VIX')
        close   (closing price)
    """
    df = pd.read_csv(path, parse_dates=["date"])
    expected_cols = {"date", "symbol", "close"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"market_history.csv is missing columns: {missing}")
    df = df.sort_values(["symbol", "date"])
    return df


@st.cache_data
def load_wave_weights(path: str = "wave_weights.csv") -> pd.DataFrame:
    """
    Expected schema (position-level):
        wave    (Wave name)
        ticker  (e.g. 'AAPL')
        weight  (position weight inside that Wave, must sum ~1.0 per Wave)
    """
    df = pd.read_csv(path)
    expected_cols = {"wave", "ticker", "weight"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"wave_weights.csv is missing columns: {missing}")
    return df


# ======================================================================
# CORE METRICS ENGINE
# ======================================================================

def compute_wave_metrics(wave_history: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the full wave-level analytics table:
    - alpha_30d / 60d / 6m / 1y (excess vs benchmark)
    - realized beta over 252d
    - beta drift vs TARGET_BETA
    - leadership_score using z-scored alpha windows
    - tier classification (Tier 1/2/3)
    """
    metrics_rows = []

    # trading-day approximations
    windows = {
        "alpha_30d": 30,
        "alpha_60d": 60,
        "alpha_6m": 126,   # ~6 months
        "alpha_1y": 252,   # ~1 year
    }

    last_date = wave_history["date"].max()

    for wave, g in wave_history.groupby("wave"):
        g = g.sort_values("date")
        row = {"wave": wave}

        # Rolling alpha windows (cumulative excess return vs benchmark)
        for name, lookback in windows.items():
            window = g.iloc[-lookback:] if len(g) >= lookback else g
            pr = window["portfolio_return"].astype(float)
            br = window["benchmark_return"].astype(float)

            port_cum = (1 + pr).prod() - 1
            bench_cum = (1 + br).prod() - 1
            row[name] = port_cum - bench_cum  # excess over benchmark

        # Realized beta and 1Y alpha on 252 days
        win = g.iloc[-252:] if len(g) >= 252 else g
        pr = win["portfolio_return"].astype(float)
        br = win["benchmark_return"].astype(float)

        # Avoid zero variance / tiny samples
        if br.var() > 0 and len(br) > 5:
            cov = np.cov(br, pr)[0, 1]
            beta = cov / br.var()
        else:
            beta = np.nan

        row["beta_252d"] = beta
        row["beta_drift"] = beta - TARGET_BETA if not np.isnan(beta) else np.nan

        # Simple 1Y alpha again, explicit
        port_cum_1y = (1 + pr).prod() - 1
        bench_cum_1y = (1 + br).prod() - 1
        row["alpha_1y_abs"] = port_cum_1y - bench_cum_1y

        row["last_date"] = last_date
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows)

    # Leadership score via z-scored windows
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

    # Tiering by rank (not absolute)
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
# VIX-GATED RISK LADDER + ALLOCATION ENGINE
# ======================================================================

def vix_to_equity_pct(vix_level: float) -> float:
    """
    Maps the latest VIX to an equity sleeve percentage according to
    the SmartSafe ladder.
    """
    if np.isnan(vix_level):
        # Fallback neutral setting
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


def build_suggested_allocations(metrics: pd.DataFrame, vix_level: float) -> (float, pd.DataFrame):
    """
    Combines VIX regime + leadership scores into:
        - equity_pct (for total portfolio)
        - wave_equity_weight (within equity sleeve)
        - suggested_portfolio_weight (final suggested weight per Wave)
    """
    equity_pct = vix_to_equity_pct(vix_level)
    df = metrics.copy()

    # Make leadership scores strictly positive for proportional allocation
    min_score = df["leadership_score"].min()
    shift = 0
    if min_score < 0:
        shift = -min_score
    df["adj_score"] = df["leadership_score"] + shift + 1e-6

    total = df["adj_score"].sum()
    if total <= 0:
        df["wave_equity_weight"] = 1.0 / len(df)
    else:
        df["wave_equity_weight"] = df["adj_score"] / total

    # Apply floor/cap so nothing is absurdly tiny or dominant
    floor = 0.03
    cap = 0.35
    df["wave_equity_weight"] = df["wave_equity_weight"].clip(lower=floor, upper=cap)
    df["wave_equity_weight"] = df["wave_equity_weight"] / df["wave_equity_weight"].sum()

    # Convert into full-portfolio weight (equity sleeve)
    df["suggested_portfolio_weight"] = df["wave_equity_weight"] * equity_pct

    return equity_pct, df


def build_position_allocations(
    wave_weights: pd.DataFrame,
    wave_allocs: pd.DataFrame,
    equity_pct: float,
) -> pd.DataFrame:
    """
    Builds position-level allocations from:
        - Wave weights within each Wave (wave_weights.csv)
        - Suggested Wave allocations (wave_allocs)
    """
    merged = wave_weights.merge(
        wave_allocs[["wave", "suggested_portfolio_weight"]],
        on="wave",
        how="inner",
        validate="many_to_one",
    )
    merged["position_weight"] = merged["weight"] * merged["suggested_portfolio_weight"]
    merged = merged.sort_values("position_weight", ascending=False)
    return merged


# ======================================================================
# UI HELPERS
# ======================================================================

def google_finance_link(ticker: str) -> str:
    safe = ticker.strip().upper()
    # Let Google figure out the exchange – this keeps it robust.
    return f"https://www.google.com/finance/quote/{safe}"


def style_wave_table(df: pd.DataFrame):
    """
    Safe styling function (fixes the earlier style_wave_table bug).
    Applies tier-based shading + numeric formatting.
    """
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
# MAIN STREAMLIT APP
# ======================================================================

def main():
    st.set_page_config(
        page_title="WAVES Institutional Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # HEADER
    st.markdown(
        """
        <h1 style="margin-bottom:0">WAVES Institutional Console</h1>
        <p style="color:#8f9bb3;margin-top:0">
        Adaptive Portfolio Waves™ • WAVES Intelligence™ • Alpha-Minus-Beta Discipline
        </p>
        """,
        unsafe_allow_html=True,
    )

    # DATA LOAD & CORE CALCULATIONS
    # ---------------------------------------------------------------
    try:
        wave_history = load_wave_history()
        metrics = compute_wave_metrics(wave_history)
    except Exception as e:
        st.error(f"Error loading or computing wave history: {e}")
        st.stop()

    try:
        market_history = load_market_history()
        vix_df = market_history[market_history["symbol"].str.upper().isin(["^VIX", "VIX"])]
        spy_df = market_history[market_history["symbol"].str.upper().isin(["SPY"])]
        vix_latest = float(vix_df.sort_values("date")["close"].iloc[-1]) if not vix_df.empty else np.nan
    except Exception as e:
        st.warning(f"Market history not fully available: {e}")
        vix_df = pd.DataFrame()
        spy_df = pd.DataFrame()
        vix_latest = np.nan

    try:
        wave_weights = load_wave_weights()
    except Exception as e:
        st.error(f"Error loading wave weights: {e}")
        st.stop()

    # Build suggested allocations and positions
    equity_pct, wave_allocs = build_suggested_allocations(metrics, vix_latest)
    position_allocs = build_position_allocations(wave_weights, wave_allocs, equity_pct)

    # ==================================================================
    # TOP PANEL — MARKET REGIME + WAVE SNAPSHOT
    # ==================================================================
    top_left, top_right = st.columns([1.5, 2])

    with top_left:
        st.subheader("Market Regime — VIX & SPY")

        if not vix_df.empty and not spy_df.empty:
            chart_df = pd.DataFrame(
                {
                    "date": spy_df["date"],
                    "SPY": spy_df["close"].values,
                }
            ).merge(
                vix_df[["date", "close"]].rename(columns={"close": "VIX"}),
                on="date",
                how="left",
            )
            chart_df = chart_df.set_index("date")
            st.line_chart(chart_df)
        else:
            st.info("VIX/SPY chart unavailable — check market_history.csv for SPY and ^VIX/VIX.")

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

    # ==================================================================
    # MIDDLE PANEL — SUGGESTED ALLOCATION + RISK ALERTS
    # ==================================================================
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

        # Beta drift + high beta alerts
        for _, row in metrics.iterrows():
            beta = row["beta_252d"]
            if pd.isna(beta):
                continue
            drift = row["beta_drift"]
            if abs(drift) > 0.07:
                alerts.append(f"⚠️ {row['wave']}: Beta drift {drift:+.2f} vs target {TARGET_BETA:.2f}")
            if beta > 1.10:
                alerts.append(f"🚨 {row['wave']}: High beta {beta:.2f}")

        # VIX regime alerts
        if not np.isnan(vix_latest):
            if vix_latest >= 26:
                alerts.append(f"🚨 High VIX regime ({vix_latest:.2f}) — SmartSafe overlay strongly engaged.")
            elif vix_latest >= 20:
                alerts.append(f"⚠️ Elevated VIX ({vix_latest:.2f}) — risk throttling partially engaged.")

        if not alerts:
            st.success("No critical risk alerts. System operating within parameters.")
        else:
            for a in alerts:
                st.write(a)

    st.markdown("---")

    # ==================================================================
    # POSITION-LEVEL ENGINE
    # ==================================================================
    st.subheader("Position-Level Allocation (Equity Sleeve)")

    pos_view = position_allocs.copy()
    pos_view["Google Finance"] = pos_view["ticker"].apply(google_finance_link)
    pos_view["position_weight"] = pos_view["position_weight"].map(lambda x: f"{x:.2%}")

    st.dataframe(
        pos_view[["wave", "ticker", "weight", "position_weight", "Google Finance"]],
        use_container_width=True,
    )

    # ==================================================================
    # TOP 10 HOLDINGS PER WAVE (WITH LINKS)
    # ==================================================================
    st.markdown("---")
    st.subheader("Top 10 Holdings per Wave")

    waves = sorted(wave_weights["wave"].unique())
    selected_wave = st.selectbox("Select Wave", waves)

    ww = wave_weights[wave_weights["wave"] == selected_wave].copy()
    ww = ww.sort_values("weight", ascending=False).head(10)
    ww["Google Finance"] = ww["ticker"].apply(google_finance_link)
    ww["weight"] = ww["weight"].map(lambda x: f"{x:.2%}")

    st.dataframe(ww[["ticker", "weight", "Google Finance"]], use_container_width=True)

    # ==================================================================
    # CSV EXPORTS
    # ==================================================================
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
