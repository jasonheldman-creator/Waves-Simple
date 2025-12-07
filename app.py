import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st


# ============================
# Data loading helpers
# ============================

@st.cache_data
def load_market_history():
    """
    Load SPY/VIX history from market_history.csv.
    Falls back to a synthetic 1Y series if the file is missing.
    Expected columns: date, symbol, close
    """
    try:
        df = pd.read_csv("market_history.csv", parse_dates=["date"])
        return df, None
    except Exception as e:
        # Build a simple demo SPY/VIX series so the app still runs
        today = datetime.today().date()
        dates = pd.date_range(end=today, periods=365, freq="B")

        base_spy = 400
        base_vix = 18
        spy = base_spy + np.cumsum(np.random.normal(0, 1.0, len(dates)))
        vix = np.clip(base_vix + np.cumsum(np.random.normal(0, 0.2, len(dates))), 10, 35)

        demo = pd.DataFrame({
            "date": list(dates) + list(dates),
            "symbol": ["SPY"] * len(dates) + ["VIX"] * len(dates),
            "close": list(spy) + list(vix),
        })
        return demo, e


@st.cache_data
def load_wave_history():
    """
    Load wave return history from wave_history.csv.
    Expected columns: date, wave, portfolio_return, benchmark_return
    """
    try:
        df = pd.read_csv("wave_history.csv", parse_dates=["date"])
        return df, None
    except Exception as e:
        # Tiny demo history so UI still works
        dates = pd.date_range(end=datetime.today().date(), periods=260, freq="B")
        waves = [
            "S&P Wave",
            "Growth Wave",
            "Quantum Computing Wave",
            "Crypto Income Wave",
        ]
        records = []
        rng = np.random.default_rng(42)
        for wave in waves:
            port_rets = rng.normal(0.0004, 0.01, len(dates))
            bench_rets = rng.normal(0.0003, 0.009, len(dates))
            for d, pr, br in zip(dates, port_rets, bench_rets):
                records.append(
                    {
                        "date": d,
                        "wave": wave,
                        "portfolio_return": pr,
                        "benchmark_return": br,
                    }
                )
        df_demo = pd.DataFrame.from_records(records)
        return df_demo, e


@st.cache_data
def load_wave_weights():
    """
    Load static holdings from wave_weights.csv.
    Expected columns: wave, ticker, weight
    """
    try:
        df = pd.read_csv("wave_weights.csv")
        return df, None
    except Exception as e:
        return pd.DataFrame(columns=["wave", "ticker", "weight"]), e


# ============================
# Metrics & leadership
# ============================

WINDOWS = {
    30: "alpha_30d",
    60: "alpha_60d",
    126: "alpha_6m",
    252: "alpha_1y",
}


def annualize_alpha(alpha_daily: pd.Series) -> float:
    """
    Convert average daily excess return to annualized % alpha (≈ 252 trading days).
    """
    if alpha_daily.empty or alpha_daily.isna().all():
        return np.nan
    mu = alpha_daily.mean()
    return float(252 * mu * 100.0)


def compute_wave_metrics(wave_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling annualized alpha windows + beta + a mode-independent leadership score.
    """
    rows = []

    for wave, g in wave_hist.groupby("wave"):
        g = g.sort_values("date")
        record = {"wave": wave}

        # rolling alpha windows
        for window, col_name in WINDOWS.items():
            if len(g) >= 5:
                sub = g.tail(window)
                alpha_daily = sub["portfolio_return"] - sub["benchmark_return"]
                record[col_name] = annualize_alpha(alpha_daily)
            else:
                record[col_name] = np.nan

        # 1Y alpha daily (252 trading days or all available if shorter)
        last = g.tail(252)
        alpha_daily_y = last["portfolio_return"] - last["benchmark_return"]
        record["alpha_1y"] = annualize_alpha(alpha_daily_y)

        # beta over last 252d
        if len(last) > 10 and last["benchmark_return"].var() > 0:
            cov = np.cov(last["portfolio_return"], last["benchmark_return"])[0, 1]
            var_b = np.var(last["benchmark_return"])
            record["beta_252d"] = float(cov / var_b)
        else:
            record["beta_252d"] = np.nan

        rows.append(record)

    df = pd.DataFrame(rows)

    # Ensure columns exist
    for col in ["alpha_30d", "alpha_60d", "alpha_6m", "alpha_1y"]:
        if col not in df:
            df[col] = np.nan

    # leadership score (mode-independent)
    def norm(series: pd.Series) -> pd.Series:
        s = series.copy()
        if s.dropna().empty:
            return pd.Series([0.5] * len(s), index=s.index)
        # clip to [-20%, +40%] for scoring band
        s = s.clip(-20, 40)
        if s.max() == s.min():
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    a30 = norm(df["alpha_30d"])
    a60 = norm(df["alpha_60d"])
    a6m = norm(df["alpha_6m"])
    a1y = norm(df["alpha_1y"])

    beta = df["beta_252d"].fillna(1.0)
    beta_penalty = np.clip(np.abs(beta - 1.0), 0, 0.5)

    df["leadership_score"] = (
        0.15 * a30 + 0.20 * a60 + 0.25 * a6m + 0.40 * a1y - 0.20 * beta_penalty
    )
    df["leadership_score"] = df["leadership_score"].clip(0, 1)

    return df


# ============================
# Mode overlays
# ============================

def compute_equity_target(mode: str, latest_vix):
    """
    Convert VIX + mode into a target equity % (0–1).
    """
    base_equity = 0.75
    if latest_vix is not None:
        base_equity = 0.8 - 0.025 * max(latest_vix - 15, 0)
        base_equity = float(np.clip(base_equity, 0.35, 0.85))

    if mode == "Standard":
        return base_equity
    elif mode == "Alpha-Minus-Beta":
        return float(np.clip(base_equity - 0.15, 0.25, 0.7))
    elif mode == "Private Logic™":
        return float(np.clip(base_equity + 0.10, 0.45, 0.95))
    else:
        return base_equity


def build_mode_allocation(metrics_df: pd.DataFrame, mode: str, latest_vix):
    """
    Apply mode-specific risk logic on top of leadership scores.
    Returns:
        alloc_df, eq_target, ss_target, portfolio_beta, beta_target, mode_score
    """
    df = metrics_df.copy()

    beta = df["beta_252d"].fillna(1.0)
    ls = df["leadership_score"].fillna(0.0)

    if mode == "Standard":
        # Reward strong leaders, mild discipline around β=1
        risk_score = ls * (1.0 - 0.3 * np.abs(beta - 1.0))

    elif mode == "Alpha-Minus-Beta":
        # Stronger tilt to lower-beta Waves, especially when VIX is elevated
        vix_level = latest_vix if latest_vix is not None else 18.0
        vix_factor = 1.0 + 0.01 * max(vix_level - 18.0, 0)

        low_beta_bonus = 1.0 + 0.5 * np.clip(1.0 - beta, 0, 0.5)
        high_beta_penalty = 1.0 - 0.8 * np.clip(beta - 0.9, 0, 0.6) * vix_factor

        risk_score = ls * low_beta_bonus * high_beta_penalty

    elif mode == "Private Logic™":
        # Slight tilt toward higher-beta leaders but still anchored in leadership
        high_beta_bonus = 1.0 + 0.4 * np.clip(beta - 1.0, 0, 0.8)
        risk_score = ls * high_beta_bonus

    else:
        risk_score = ls

    risk_score = np.where(risk_score < 0, 0, risk_score)

    if np.all(risk_score == 0):
        weights = np.ones_like(risk_score) / max(len(risk_score), 1)
    else:
        weights = risk_score / risk_score.sum()

    df["equity_weight"] = weights

    equity_target = compute_equity_target(mode, latest_vix)
    df["target_equity_pct"] = equity_target * 100.0
    df["target_smartsafe_pct"] = (1.0 - equity_target) * 100.0

    # Portfolio beta vs. target
    portfolio_beta = float(np.nansum(df["equity_weight"] * beta))

    if mode == "Standard":
        beta_target = 1.0
    elif mode == "Alpha-Minus-Beta":
        beta_target = 0.8
    elif mode == "Private Logic™":
        beta_target = 1.1
    else:
        beta_target = 1.0

    beta_error = abs(portfolio_beta - beta_target)
    mode_score = max(0.0, 100.0 - 120.0 * beta_error)  # simple 0–100

    df["portfolio_beta"] = portfolio_beta
    df["beta_target"] = beta_target
    df["mode_score"] = mode_score

    # Alpha contribution (1Y alpha * equity weight)
    df["alpha_1y_contrib"] = df["alpha_1y"] * df["equity_weight"]

    return df, equity_target, 1.0 - equity_target, portfolio_beta, beta_target, mode_score


# ============================
# Layout helpers
# ============================

def render_market_regime(market_df: pd.DataFrame):
    """
    Show SPY & VIX over the last year and return latest values.
    """
    import altair as alt

    one_year_ago = datetime.today() - timedelta(days=365)
    m = market_df[market_df["date"] >= one_year_ago]

    spy = m[m["symbol"] == "SPY"].copy()
    vix = m[m["symbol"] == "VIX"].copy()

    if spy.empty or vix.empty:
        st.info("Not enough SPY/VIX data to display the regime chart.")
        return None, None

    spy = spy.sort_values("date")
    vix = vix.sort_values("date")

    chart_spy = (
        alt.Chart(spy)
        .mark_line()
        .encode(x="date:T", y=alt.Y("close:Q", title="SPY"))
        .properties(height=260)
    )
    chart_vix = (
        alt.Chart(vix)
        .mark_line()
        .encode(x="date:T", y=alt.Y("close:Q", title="VIX"))
        .properties(height=260)
    )

    st.altair_chart(chart_spy, use_container_width=True)
    st.altair_chart(chart_vix, use_container_width=True)

    latest_spy = spy.iloc[-1]["close"]
    latest_vix = vix.iloc[-1]["close"]

    return float(latest_spy), float(latest_vix)


def build_risk_alerts(mode: str, latest_vix, portfolio_beta, beta_target, eq_target):
    alerts = []

    if latest_vix is not None:
        if latest_vix >= 28:
            alerts.append("VIX elevated — engine tilted to risk-off / SmartSafe heavy.")
        elif latest_vix <= 14:
            alerts.append("VIX subdued — engine tilted slightly pro-risk.")

    beta_diff = abs(portfolio_beta - beta_target)
    if beta_diff > 0.10:
        alerts.append(
            f"Portfolio β {portfolio_beta:.2f} is off target ({beta_target:.2f}) by {beta_diff:.2f}."
        )

    if eq_target < 0.40:
        alerts.append("Equity allocation < 40% — SmartSafe™ is dominant.")
    elif eq_target > 0.80:
        alerts.append("Equity allocation > 80% — engine is running hot; monitor drawdowns.")

    if mode == "Private Logic™" and latest_vix is not None and latest_vix > 25:
        alerts.append("Private Logic™ + high VIX — expect more aggressive alpha hunting with guardrails.")

    if not alerts:
        alerts.append("No critical risk alerts — engine operating within normal parameters.")

    return alerts


# ============================
# Main app
# ============================

def main():
    st.set_page_config(
        page_title="WAVES Institutional Console",
        page_icon="🌊",
        layout="wide",
    )

    st.markdown(
        "<h1 style='color:#f9f9fb;'>WAVES Institutional Console</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Adaptive Portfolio Waves™ • Single Truth Engine • Mode-Aware Risk Overlays")

    # Sidebar — Modes
    st.sidebar.markdown("### WAVES Mode")
    mode = st.sidebar.radio(
        "",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Leadership & alpha are mode-independent. Modes act as risk overlays: "
        "equity vs. SmartSafe™, β discipline, and Wave weighting."
    )

    # Load data
    market_df, market_err = load_market_history()
    wave_hist, wave_hist_err = load_wave_history()
    wave_weights, wave_weights_err = load_wave_weights()

    if market_err is not None:
        st.warning(
            f"Using demo SPY/VIX series (reason: {market_err}). "
            "Add 'market_history.csv' for full LIVE data."
        )
    if wave_hist_err is not None:
        st.warning(
            f"Using demo wave_history (reason: {wave_hist_err}). "
            "Run build_wave_history_from_prices.py to generate full history."
        )
    if wave_weights_err is not None:
        st.info("No wave_weights.csv found — holdings view will be empty until uploaded.")

    col_left, col_right = st.columns([1.15, 1])

    # -------- Left: Market regime --------
    with col_left:
        st.subheader("Market Regime — SPY & VIX")
        latest_spy, latest_vix = render_market_regime(market_df)

        st.markdown("#### Latest Regime Snapshot")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Latest VIX", f"{latest_vix:.2f}" if latest_vix is not None else "—")
        # placeholders for equity / SmartSafe that we fill after allocation
        eq_placeholder = m2
        ss_placeholder = m3

    # -------- Right: Leadership & alpha --------
    with col_right:
        st.subheader("Wave Leadership & Alpha Snapshot")

        metrics_df = compute_wave_metrics(wave_hist)
        (
            alloc_df,
            eq_target,
            ss_target,
            portfolio_beta,
            beta_target,
            mode_score,
        ) = build_mode_allocation(metrics_df, mode, latest_vix)

        # Update regime equity / SmartSafe metrics
        with eq_placeholder:
            st.metric("Equity Sleeve %", f"{eq_target*100:.0f}%")
        with ss_placeholder:
            st.metric("SmartSafe™ %", f"{ss_target*100:.0f}%")

        # Mode banner
        st.markdown(
            f"<div style='padding:6px 10px;border-radius:4px;background-color:#333333;'>"
            f"<strong>DATA MODE:</strong> MIXED &nbsp;•&nbsp; "
            f"<strong>Mode:</strong> {mode} &nbsp;•&nbsp; "
            f"<strong>β target:</strong> {beta_target:.2f} &nbsp;•&nbsp; "
            f"<strong>Mode score:</strong> {mode_score:.1f}/100"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Annualized alpha windows are computed on LIVE/SANDBOX history. "
            "Modes tilt risk around the same underlying performance engine."
        )

        # Display leadership table
        display_cols = [
            "wave",
            "alpha_30d",
            "alpha_60d",
            "alpha_6m",
            "alpha_1y",
            "alpha_1y_contrib",
            "beta_252d",
            "equity_weight",
        ]
        display_df = alloc_df[display_cols].copy()

        display_df["equity_weight"] = (display_df["equity_weight"] * 100.0).round(1)
        display_df.rename(
            columns={
                "alpha_30d": "alpha_30d (%)",
                "alpha_60d": "alpha_60d (%)",
                "alpha_6m": "alpha_6m (%)",
                "alpha_1y": "alpha_1y (%)",
                "alpha_1y_contrib": "alpha_1y_contrib (pct-pts)",
                "beta_252d": "beta_252d",
                "equity_weight": "Wave equity weight %",
            },
            inplace=True,
        )

        st.dataframe(
            display_df.sort_values("alpha_1y (%)", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

    # -------- Risk alerts row --------
    st.markdown("---")
    st.subheader("Mode-Aware Risk Alerts")

    alerts = build_risk_alerts(mode, latest_vix, portfolio_beta, beta_target, eq_target)
    for a in alerts:
        st.write(f"- {a}")

    # -------- Holdings / positions --------
    st.markdown("---")
    st.subheader("Wave Holdings & Effective Portfolio Weights")

    if wave_weights.empty:
        st.info("Upload wave_weights.csv to see holdings and effective portfolio weights.")
    else:
        ww = wave_weights.copy()

        # Normalize weights within each wave (in case they are not perfectly to 1.0)
        ww["weight"] = ww["weight"].astype(float)
        ww["norm_weight"] = ww.groupby("wave")["weight"].transform(
            lambda x: x / x.abs().sum() if x.abs().sum() != 0 else x
        )

        # Join Wave-level equity weights from current mode
        ww = ww.merge(
            alloc_df[["wave", "equity_weight"]],
            on="wave",
            how="left",
        )
        ww["equity_weight"] = ww["equity_weight"].fillna(0.0)

        # Effective portfolio weight = Wave equity weight * stock weight
        ww["effective_portfolio_weight"] = ww["equity_weight"] * ww["norm_weight"]

        ww_display = ww.copy()
        ww_display["Wave equity %"] = (ww_display["equity_weight"] * 100.0).round(2)
        ww_display["Stock in-Wave %"] = (ww_display["norm_weight"] * 100.0).round(2)
        ww_display["Total portfolio %"] = (ww_display["effective_portfolio_weight"] * 100.0).round(2)

        # Google Finance links
        def make_link(ticker: str) -> str:
            url = f"https://www.google.com/finance/quote/{ticker}:NYSE"
            return f'<a href="{url}" target="_blank">{ticker}</a>'

        ww_display["Ticker"] = ww_display["ticker"].astype(str)
        ww_display["Google Quote"] = ww_display["Ticker"].apply(make_link)

        cols_show = [
            "wave",
            "Ticker",
            "Google Quote",
            "Wave equity %",
            "Stock in-Wave %",
            "Total portfolio %",
        ]
        ww_display = ww_display[cols_show]

        # Limit to top 50 by total portfolio weight for readability
        ww_display = ww_display.sort_values("Total portfolio %", ascending=False).head(50)

        st.write(
            ww_display.to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )

    # -------- Downloads --------
    st.markdown("---")
    st.subheader("Downloads")

    metrics_csv = alloc_df.to_csv(index=False)
    st.download_button(
        label=f"Download Wave metrics & mode weights ({mode})",
        data=metrics_csv,
        file_name=f"wave_metrics_{mode.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    if not wave_hist.empty:
        wave_hist_csv = wave_hist.to_csv(index=False)
        st.download_button(
            label="Download raw wave_history.csv",
            data=wave_hist_csv,
            file_name="wave_history.csv",
            mime="text/csv",
        )

    st.caption("WAVES Intelligence™ • Single Truth Engine • For institutional demo use only.")


if __name__ == "__main__":
    main()
