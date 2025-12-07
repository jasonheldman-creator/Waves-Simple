import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ======================================================================
# CONFIG
# ======================================================================

WAVE_TICKERS_DEFAULT = {
    "S&P Wave": ["AAPL", "MSFT", "AMZN", "GOOGL", "META",
                 "BRK.B", "JPM", "JNJ", "V", "PG"],
    "Growth Wave": ["TSLA", "NVDA", "AMD", "AVGO", "ADBE",
                    "CRM", "SHOP", "NOW", "NFLX", "INTU"],
    "Future Power & Energy Wave": ["XOM", "CVX", "NEE", "DUK", "ENPH",
                                   "FSLR", "LNG", "SLB", "PSX", "PLUG"],
    "Small Cap Growth Wave": ["TTD", "PLTR", "TWLO", "OKTA", "ZS",
                              "DDOG", "APPF", "MDB", "ESTC", "SMAR"],
    "Small-Mid Cap Growth Wave": ["CRWD", "ZS", "DDOG", "OKTA", "TEAM",
                                  "SNOW", "NET", "BILL", "DOCU", "PATH"],
    "Clean Transit-Infrastructure Wave": ["TSLA", "NIO", "GM", "F", "BLDP",
                                          "PLUG", "CHPT", "ABB", "DE", "URI"],
    "Quantum Computing Wave": ["NVDA", "AMD", "IBM", "MSFT", "GOOGL",
                               "AMZN", "IONQ", "QUBT", "BRKS", "INTC"],
    "Crypto Income Wave": ["COIN", "MSTR", "RIOT", "MARA", "HUT",
                           "CLSK", "BITO", "IBIT", "FBTC", "DEFI"],
    "Income Wave": ["TLT", "LQD", "HYG", "JNJ", "PG",
                    "KO", "PEP", "XLU", "O", "VZ"],
}


# ======================================================================
# DATA LOADERS
# ======================================================================

@st.cache_data
def load_wave_weights():
    required = {"wave", "ticker", "weight"}
    try:
        df = pd.read_csv("wave_weights.csv")
        if not required.issubset(df.columns):
            raise ValueError("wave_weights.csv missing required columns")
        df["wave"] = df["wave"].astype(str)
        df["ticker"] = df["ticker"].astype(str).str.upper()
        return df, False
    except Exception as e:
        st.warning(f"Using built-in demo wave_weights (reason: {e})")
        rows = []
        for wave, tickers in WAVE_TICKERS_DEFAULT.items():
            w = 1.0 / len(tickers)
            for t in tickers:
                rows.append({"wave": wave, "ticker": t, "weight": w})
        demo_df = pd.DataFrame(rows)
        demo_df["wave"] = demo_df["wave"].astype(str)
        demo_df["ticker"] = demo_df["ticker"].astype(str).str.upper()
        return demo_df, True


@st.cache_data
def load_market_history():
    """
    market_history.csv: date,symbol,close
    Used for SPY & VIX chart. Falls back to synthetic if missing.
    """
    required = {"date", "symbol", "close"}
    try:
        df = pd.read_csv("market_history.csv", parse_dates=["date"])
        if not required.issubset(df.columns):
            raise ValueError("market_history.csv missing required columns")
        df["symbol"] = df["symbol"].astype(str).str.upper()
        return df.sort_values(["symbol", "date"]), False
    except Exception as e:
        st.warning(f"Using demo SPY/VIX series (reason: {e})")
        num_days = 180
        end_date = datetime.today().date()
        dates = sorted([end_date - timedelta(days=i) for i in range(num_days)])

        rows = []
        spy_price = 450.0
        vix_level = 18.0
        np.random.seed(42)
        for d in dates:
            dr = np.random.normal(0.0005, 0.01)
            spy_price *= (1 + dr)
            rows.append({"date": d, "symbol": "SPY", "close": spy_price})

            dv = np.random.normal(0.0, 0.4)
            vix_level = max(10.0, vix_level + dv)
            rows.append({"date": d, "symbol": "VIX", "close": vix_level})

        demo_df = pd.DataFrame(rows)
        demo_df["date"] = pd.to_datetime(demo_df["date"])
        demo_df["symbol"] = demo_df["symbol"].astype(str).str.upper()
        return demo_df.sort_values(["symbol", "date"]), True


@st.cache_data
def load_wave_history():
    """
    Prefers real wave_history.csv (built from prices), falls back to 1-year synthetic.
    wave_history.csv: date,wave,portfolio_return,benchmark_return
    """
    required = {"date", "wave", "portfolio_return", "benchmark_return"}
    try:
        df = pd.read_csv("wave_history.csv", parse_dates=["date"])
        if not required.issubset(df.columns):
            raise ValueError("wave_history.csv missing required columns")
        df["wave"] = df["wave"].astype(str)
        return df.sort_values(["wave", "date"]), False
    except Exception as e:
        st.warning(f"Using synthetic demo wave_history (reason: {e})")
        waves = sorted(WAVE_TICKERS_DEFAULT.keys())
        num_days = 252  # ~1 year demo
        end_date = datetime.today().date()
        dates = sorted([end_date - timedelta(days=i) for i in range(num_days)])

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

        demo_df = pd.DataFrame(rows)
        demo_df["date"] = pd.to_datetime(demo_df["date"])
        demo_df["wave"] = demo_df["wave"].astype(str)
        return demo_df.sort_values(["wave", "date"]), True


# ======================================================================
# METRICS ENGINE — SMOOTHED ALPHA, ONE TRUTH
# ======================================================================

def annualized_excess_alpha(window: pd.DataFrame) -> float:
    """
    Take a window of daily returns, compute average daily excess, annualize.
    This gives smoother, more stable alpha while staying fully "real".
    """
    if window.empty:
        return np.nan
    excess = window["portfolio_return"].astype(float) - window["benchmark_return"].astype(float)
    if excess.empty:
        return np.nan
    mean_excess = excess.mean()
    return float(mean_excess * 252.0)  # annualized excess return


def compute_wave_metrics(wave_history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    last_date = wave_history["date"].max()

    for wave, g in wave_history.groupby("wave"):
        g = g.sort_values("date")
        n = len(g)

        def window_ann_alpha(days: int) -> float:
            if n < days:
                return np.nan
            window = g.iloc[-days:]
            return annualized_excess_alpha(window)

        alpha_30d = window_ann_alpha(30)
        alpha_60d = window_ann_alpha(60)
        alpha_6m = window_ann_alpha(126)
        alpha_1y = window_ann_alpha(252)
        alpha_3y = window_ann_alpha(756)  # will be NaN if not enough history

        # Beta over last 252 days if available
        if n >= 252:
            pr = g["portfolio_return"].astype(float).iloc[-252:]
            br = g["benchmark_return"].astype(float).iloc[-252:]
        else:
            pr = g["portfolio_return"].astype(float)
            br = g["benchmark_return"].astype(float)

        if len(pr) > 5 and br.var() > 0:
            cov = np.cov(br, pr)[0, 1]
            beta_252d = cov / br.var()
        else:
            beta_252d = np.nan

        # 1Y absolute alpha using same annualized logic
        alpha_1y_abs = alpha_1y

        rows.append(
            {
                "wave": wave,
                "alpha_30d": alpha_30d,
                "alpha_60d": alpha_60d,
                "alpha_6m": alpha_6m,
                "alpha_1y": alpha_1y,
                "alpha_3y": alpha_3y,
                "alpha_1y_abs": alpha_1y_abs,
                "beta_252d": beta_252d,
                "last_date": last_date,
            }
        )

    m = pd.DataFrame(rows)

    # Z-scores for leadership (mode-independent)
    for col in ["alpha_30d", "alpha_60d", "alpha_6m", "alpha_1y"]:
        mean = m[col].mean(skipna=True)
        std = m[col].std(skipna=True)
        if std and not np.isnan(std):
            m[col + "_z"] = (m[col] - mean) / std
        else:
            m[col + "_z"] = 0.0

    # Locked leadership formula
    m["leadership_score"] = (
        0.40 * m["alpha_30d_z"]
        + 0.30 * m["alpha_60d_z"]
        + 0.20 * m["alpha_6m_z"]
        + 0.10 * m["alpha_1y_z"]
    )

    m = m.sort_values("leadership_score", ascending=False)
    n = len(m)
    tiers = []
    for i in range(n):
        pct = (i + 1) / n
        if pct <= 0.33:
            tiers.append("Tier 1 — Leader")
        elif pct <= 0.66:
            tiers.append("Tier 2 — Supporting")
        else:
            tiers.append("Tier 3 — Lagging")
    m["tier"] = tiers

    return m


# ======================================================================
# MODES AS REAL OVERLAYS (DIFFERENT LOGIC)
# ======================================================================

def vix_to_equity_pct(vix_level: float, mode: str) -> float:
    # Base ladder
    if np.isnan(vix_level):
        base = 0.60
    elif vix_level < 12:
        base = 1.00
    elif vix_level < 15:
        base = 0.90
    elif vix_level < 18:
        base = 0.80
    elif vix_level < 22:
        base = 0.65
    elif vix_level < 26:
        base = 0.50
    elif vix_level < 32:
        base = 0.35
    else:
        base = 0.20

    if mode == "Standard":
        equity_pct = base
    elif mode == "Alpha-Minus-Beta":
        equity_pct = max(0.20, min(base * 0.80, 0.75))  # more SmartSafe
    else:  # Private Logic™
        equity_pct = max(0.35, min(base * 1.20, 1.00))  # more equity

    return equity_pct


def build_suggested_allocations(metrics: pd.DataFrame, vix_level: float, mode: str):
    """
    This is where modes actually diverge:

    - Standard: uses pure leadership_score.
    - Alpha-Minus-Beta: penalizes high beta vs 0.80 target.
    - Private Logic™: rewards higher beta & alpha slightly.
    """
    df = metrics.copy()

    # Target betas for penalty/bonus
    if mode == "Standard":
        beta_target = 0.90
    elif mode == "Alpha-Minus-Beta":
        beta_target = 0.80
    else:
        beta_target = 1.05

    beta = df["beta_252d"].fillna(beta_target)
    base_score = df["leadership_score"]

    if mode == "Standard":
        adj_score = base_score
    elif mode == "Alpha-Minus-Beta":
        # Heavier penalty for high beta
        penalty = 0.75 * np.abs(beta - beta_target)
        adj_score = base_score - penalty
    else:  # Private Logic™
        # Slight bonus for higher beta above target
        bonus = 0.50 * (beta - beta_target)
        adj_score = base_score + bonus

    df["mode_score"] = adj_score

    # Turn into positive weights
    min_score = df["mode_score"].min()
    shift = -min_score if min_score < 0 else 0
    df["adj_score"] = df["mode_score"] + shift + 1e-6
    total = df["adj_score"].sum()
    if total <= 0:
        df["wave_equity_weight"] = 1.0 / len(df)
    else:
        df["wave_equity_weight"] = df["adj_score"] / total

    # Mode-specific floor/cap
    if mode == "Standard":
        floor, cap = 0.03, 0.35
    elif mode == "Alpha-Minus-Beta":
        floor, cap = 0.03, 0.30  # more compressed
    else:
        floor, cap = 0.04, 0.40  # more aggressive

    df["wave_equity_weight"] = df["wave_equity_weight"].clip(lower=floor, upper=cap)
    df["wave_equity_weight"] /= df["wave_equity_weight"].sum()

    equity_pct = vix_to_equity_pct(vix_level, mode)
    df["suggested_portfolio_weight"] = df["wave_equity_weight"] * equity_pct

    return equity_pct, beta_target, df


def build_position_allocations(wave_weights: pd.DataFrame,
                               wave_allocs: pd.DataFrame) -> pd.DataFrame:
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
    t = ticker.strip().upper()
    return f"https://www.google.com/finance/quote/{t}"


def style_wave_table(df: pd.DataFrame):
    def highlight_tier(row):
        tier = str(row.get("tier", ""))
        if tier.startswith("Tier 1"):
            return ["background-color: rgba(0,255,0,0.12)"] * len(row)
        elif tier.startswith("Tier 2"):
            return ["background-color: rgba(255,255,0,0.10)"] * len(row)
        elif tier.startswith("Tier 3"):
            return ["background-color: rgba(255,0,0,0.08)"] * len(row)
        return [""] * len(row)

    fmt_cols = {
        "alpha_30d": "{:.2%}",
        "alpha_60d": "{:.2%}",
        "alpha_6m": "{:.2%}",
        "alpha_1y": "{:.2%}",
        "alpha_3y": "{:.2%}",
        "alpha_1y_abs": "{:.2%}",
        "alpha_1y_contrib": "{:.2%}",
        "beta_252d": "{:.2f}",
        "leadership_score": "{:+.2f}",
        "mode_score": "{:+.2f}",
    }
    fmt = {k: v for k, v in fmt_cols.items() if k in df.columns}
    return df.style.format(fmt).apply(highlight_tier, axis=1)


# ======================================================================
# MAIN APP
# ======================================================================

def main():
    st.set_page_config(
        page_title="WAVES Institutional Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    mode = st.sidebar.radio(
        "WAVES Mode",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
        help="Modes change risk posture and tilts (beta-aware), not the underlying return history.",
    )

    wave_weights, ww_demo = load_wave_weights()
    wave_history, wh_demo = load_wave_history()
    market_history, mh_demo = load_market_history()

    metrics = compute_wave_metrics(wave_history)

    # VIX / SPY
    vix_df = market_history[market_history["symbol"].isin(["VIX", "^VIX"])]
    spy_df = market_history[market_history["symbol"] == "SPY"]
    vix_latest = float(vix_df.sort_values("date")["close"].iloc[-1]) if not vix_df.empty else np.nan

    equity_pct, beta_target, wave_allocs = build_suggested_allocations(metrics, vix_latest, mode)
    position_allocs = build_position_allocations(wave_weights, wave_allocs)

    # Merge allocs for alpha contribution
    metrics = metrics.merge(
        wave_allocs[["wave", "suggested_portfolio_weight", "mode_score"]],
        on="wave",
        how="left",
    )
    metrics["suggested_portfolio_weight"] = metrics["suggested_portfolio_weight"].fillna(0.0)
    metrics["alpha_1y_contrib"] = metrics["alpha_1y_abs"] * metrics["suggested_portfolio_weight"]
    total_alpha_1y = metrics["alpha_1y_contrib"].sum()

    # Data mode badge
    flags = [ww_demo, wh_demo, mh_demo]
    if all(not f for f in flags):
        data_mode = "LIVE"
        badge_color = "#16c784"
    elif all(flags):
        data_mode = "DEMO"
        badge_color = "#ff9800"
    else:
        data_mode = "MIXED"
        badge_color = "#f4c842"

    # HEADER
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div>
            <h1 style="margin-bottom:0">WAVES Institutional Console</h1>
            <p style="color:#8f9bb3;margin-top:0">
              Adaptive Portfolio Waves™ • Single Truth Engine • Mode-Aware Risk Overlays
            </p>
            <p style="color:#65708f;font-size:12px;margin-top:4px;">
              Performance metrics (alpha, beta, tiers) are mode-independent. Modes change risk posture,
              equity vs SmartSafe, and how aggressively we tilt toward leadership & beta.
            </p>
          </div>
          <div style="display:flex;gap:8px;align-items:center;">
            <span style="padding:6px 14px;border-radius:999px;
                         background-color:{badge_color};color:#0b1020;
                         font-weight:600;font-size:13px;letter-spacing:0.08em;
                         text-transform:uppercase;">
              Data Mode: {data_mode}
            </span>
            <span style="padding:6px 14px;border-radius:999px;
                         background-color:#1f2a3c;color:#e4e9ff;
                         font-weight:500;font-size:13px;">
              Mode: {mode} • Target β ≈ {beta_target:.2f}
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # TOP PANEL
    left, right = st.columns([1.5, 2])

    with left:
        st.subheader("Market Regime — SPY & VIX")

        if not spy_df.empty and not vix_df.empty:
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
            st.info("Demo SPY/VIX regime view (attach market_history.csv for LIVE).")

        st.metric("Latest VIX", f"{vix_latest:.2f}" if not np.isnan(vix_latest) else "N/A")
        st.metric("Equity Sleeve %", f"{equity_pct:.0%}")
        st.metric("SmartSafe %", f"{1 - equity_pct:.0%}")

    with right:
        st.subheader("Wave Leadership & Alpha Snapshot")

        display_cols = [
            "wave",
            "alpha_30d",
            "alpha_60d",
            "alpha_6m",
            "alpha_1y",
            "alpha_3y",
            "alpha_1y_contrib",
            "beta_252d",
            "leadership_score",
            "mode_score",
            "tier",
        ]
        snap = metrics[display_cols].copy()
        st.dataframe(style_wave_table(snap), use_container_width=True)
        st.caption(
            f"Estimated annualized 1-Year excess alpha from this mode's allocation: "
            f"**{total_alpha_1y:.2%}** vs benchmark."
        )

    st.markdown("---")

    # MIDDLE
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Suggested Wave Allocation (Equity Sleeve)")
        view = wave_allocs[
            ["wave", "tier", "mode_score", "wave_equity_weight", "suggested_portfolio_weight"]
        ].copy()
        view["wave_equity_weight"] = view["wave_equity_weight"].map(lambda x: f"{x:.1%}")
        view["suggested_portfolio_weight"] = view["suggested_portfolio_weight"].map(lambda x: f"{x:.1%}")
        view = view.sort_values("suggested_portfolio_weight", ascending=False)
        st.dataframe(view, use_container_width=True)

    with c2:
        st.subheader("Risk Alerts")
        alerts = []

        for _, row in metrics.iterrows():
            beta = row["beta_252d"]
            if pd.isna(beta):
                continue
            drift = beta - beta_target
            if abs(drift) > 0.07:
                alerts.append(
                    f"⚠️ {row['wave']}: β drift {drift:+.2f} vs target {beta_target:.2f}"
                )
            if beta > beta_target + 0.20:
                alerts.append(f"🚨 {row['wave']}: Elevated β {beta:.2f}")

        if not np.isnan(vix_latest):
            if vix_latest >= 26:
                alerts.append(f"🚨 High VIX regime ({vix_latest:.2f}) — SmartSafe heavily engaged.")
            elif vix_latest >= 20:
                alerts.append(f"⚠️ Elevated VIX ({vix_latest:.2f}) — partial risk throttling.")

        if not alerts:
            st.success("No critical risk alerts for this mode.")
        else:
            for a in alerts:
                st.write(a)

    st.markdown("---")

    # POSITION-LEVEL
    st.subheader("Position-Level Allocation (Equity Sleeve)")
    pos = position_allocs.copy()
    pos["Google Quote"] = pos["ticker"].apply(google_finance_link)
    pos["position_weight"] = pos["position_weight"].map(lambda x: f"{x:.2%}")
    st.dataframe(
        pos[["wave", "ticker", "weight", "position_weight", "Google Quote"]],
        use_container_width=True,
    )

    st.markdown("---")

    # TOP 10
    st.subheader("Top 10 Holdings per Wave")
    waves = sorted(wave_weights["wave"].unique())
    selected_wave = st.selectbox("Select Wave", waves)
    ww = wave_weights[wave_weights["wave"] == selected_wave].copy()
    ww = ww.sort_values("weight", ascending=False).head(10)
    ww["Google Quote"] = ww["ticker"].apply(google_finance_link)
    ww["weight"] = ww["weight"].map(lambda x: f"{x:.2%}")
    st.dataframe(ww[["ticker", "weight", "Google Quote"]], use_container_width=True)

    st.markdown("---")

    # DOWNLOADS
    st.subheader("Download CSVs")
    st.download_button(
        "Download Wave Metrics",
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


if __name__ == "__main__":
    main()
