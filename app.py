import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yfinance as yf

# Optional auto-refresh component
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ------------------------------------------------------------
# PAGE CONFIG & BRANDING
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES Intelligence – Wave Engine Console",
    layout="wide"
)

BRAND_BG = "#07072C"       # deep navy
BRAND_CARD = "#111133"     # card background
BRAND_ACCENT = "#30F2A0"   # neon green/teal
BRAND_TEXT = "#F5F7FF"     # light text

st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(circle at top left, #101030 0, {BRAND_BG} 45%, #02020a 100%);
        color: {BRAND_TEXT};
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif;
    }}
    section[data-testid="stSidebar"] > div {{
        background: linear-gradient(180deg, #050522 0, #02020f 100%);
        border-right: 1px solid #1a1a40;
    }}
    section[data-testid="stSidebar"] label {{
        color: #E0E4FF !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
        font-size: 0.8rem;
        text-transform: uppercase;
    }}
    .stMetric {{
        background-color: {BRAND_CARD};
        padding: 0.6rem 0.8rem;
        border-radius: 0.6rem;
        border: 1px solid #262654;
    }}
    .stMetric > div > div:nth-child(1) {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #B6B9FF;
    }}
    .stMetric > div > div:nth-child(2) {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {BRAND_ACCENT};
    }}
    .stDataFrame, .stDataFrame table {{
        color: {BRAND_TEXT} !important;
        background-color: {BRAND_CARD} !important;
        border-radius: 0.6rem;
    }}
    .stDataFrame [data-testid="stTable"] th {{
        background-color: #181842 !important;
        color: #C0C4FF !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .streamlit-expanderHeader {{
        background-color: #11112E !important;
        color: #C7CBFF !important;
    }}
    button[kind="primary"] {{
        background: linear-gradient(90deg, {BRAND_ACCENT}, #4DF2D2);
        color: #02020a !important;
        border-radius: 999px;
        border: none;
        font-weight: 600;
        padding: 0.4rem 1.1rem;
    }}
    button[kind="primary"]:hover {{
        filter: brightness(1.08);
    }}
    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div style="display:flex;align-items:baseline;gap:0.6rem;margin-bottom:0.2rem;">
        <div style="font-size:1.5rem;font-weight:650;color:#FFFFFF;">
            WAVES Intelligence&trade; – Wave Engine Console
        </div>
        <div style="font-size:0.9rem;color:#7F84FF;">
            Live multi-wave portfolio engine • Demo view
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# LIVE CONTROLS (AUTO + MANUAL REFRESH)
# ------------------------------------------------------------
st.sidebar.markdown("### Live Engine")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s (demo)", value=False)

if auto_refresh and HAS_AUTOREFRESH:
    st_autorefresh(interval=30_000, limit=None, key="auto_refresh")
elif auto_refresh and not HAS_AUTOREFRESH:
    st.sidebar.info("Auto-refresh component not installed; using manual refresh only.")

top_left, top_right = st.columns([1, 3])
with top_left:
    if st.button("🔄 Manual refresh now"):
        st.experimental_rerun()
with top_right:
    st.caption(
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  •  "
        "Prices via Yahoo Finance (demo only, not for trading)."
    )

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    """Forgiving CSV reader that won't crash on bad lines."""
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception as e:
        st.error(f"Error reading {path.name}: {e}")
        return pd.DataFrame()


def load_universe(path: Path = Path("Master_Stock_Sheet.csv")) -> pd.DataFrame:
    """Load and normalize the master stock universe."""
    df = safe_read_csv(path)
    if df.empty:
        return df

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    rename_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            rename_map[col] = "Ticker"
        elif col in ["company", "company_name", "name", "security"]:
            rename_map[col] = "Company"
        elif col in ["sector", "gics_sector"]:
            rename_map[col] = "Sector"
        elif col in ["weight", "index_weight", "wgt"]:
            rename_map[col] = "Weight_universe"
        elif col in ["market_value", "marketvalue", "mv"]:
            rename_map[col] = "MarketValue"
        elif col in ["price", "last_price"]:
            rename_map[col] = "Price"

    df = df.rename(columns=rename_map)

    if "Ticker" not in df.columns:
        df["Ticker"] = np.nan
    if "Company" not in df.columns:
        df["Company"] = df["Ticker"]
    if "Sector" not in df.columns:
        df["Sector"] = ""
    if "Weight_universe" not in df.columns:
        df["Weight_universe"] = np.nan
    if "MarketValue" not in df.columns:
        df["MarketValue"] = np.nan
    if "Price" not in df.columns:
        df["Price"] = np.nan

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Company"] = df["Company"].astype(str).str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()
    df["Weight_universe"] = pd.to_numeric(df["Weight_universe"], errors="coerce")
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df[df["Ticker"] != ""]
    df = df.dropna(subset=["Ticker"])

    return df[["Ticker", "Company", "Sector", "Weight_universe", "MarketValue", "Price"]]


def load_wave_weights(path: Path = Path("wave_weights.csv")) -> pd.DataFrame:
    """Load and normalize wave weight definitions."""
    df = safe_read_csv(path)
    if df.empty:
        return df

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    rename_map = {}
    for col in df.columns:
        if col in ["ticker", "symbol"]:
            rename_map[col] = "Ticker"
        elif col in ["wave", "portfolio", "strategy"]:
            rename_map[col] = "Wave"
        elif col in ["weight", "wgt", "alloc"]:
            rename_map[col] = "Weight_wave"

    df = df.rename(columns=rename_map)

    if "Ticker" not in df.columns:
        df["Ticker"] = np.nan
    if "Wave" not in df.columns:
        df["Wave"] = "SP500_Wave"
    if "Weight_wave" not in df.columns:
        df["Weight_wave"] = 1.0

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Wave"] = df["Wave"].astype(str).str.strip()
    df["Weight_wave"] = pd.to_numeric(df["Weight_wave"], errors="coerce")

    df = df[(df["Ticker"] != "") & df["Ticker"].notna()]
    df = df.dropna(subset=["Weight_wave"])

    # Normalize within each wave so they sum to 1.0
    df["Weight_wave"] = df.groupby("Wave")["Weight_wave"].transform(
        lambda x: x / x.sum() if x.sum() else x
    )

    return df[["Ticker", "Wave", "Weight_wave"]]


@st.cache_data(ttl=900)
def fetch_live_prices(tickers) -> dict:
    """Fetch live prices from Yahoo Finance. Cached for 15 minutes."""
    prices = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="1d", interval="1m")
            if not hist.empty:
                prices[t] = float(hist["Close"].iloc[-1])
        except Exception:
            continue
    return prices


@st.cache_data(ttl=3600)
def get_sector_from_yf(ticker: str) -> str:
    """Fetch sector from Yahoo Finance for a single ticker, cached for 1 hour."""
    try:
        t = yf.Ticker(ticker)
        try:
            info = t.get_info()
        except Exception:
            info = getattr(t, "info", {}) or {}
        sector = info.get("sector") or info.get("industry") or ""
        if not sector:
            return "Unknown"
        return str(sector)
    except Exception:
        return "Unknown"


def simulate_nav_series(wave_name: str, days: int = 60) -> pd.Series:
    """Simulate a NAV history (for demo charts)."""
    seed = abs(hash(wave_name)) % (2**32)
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.0004, scale=0.012, size=days)
    nav = 100 * np.cumprod(1 + rets)
    idx = pd.date_range(end=datetime.now(), periods=days, freq="B")
    return pd.Series(nav, index=idx)


def generate_commentary(wave_name: str, df: pd.DataFrame, risk_mode: str) -> str:
    """AI-style narrative based on concentration, sectors, and drift."""
    if df.empty:
        return "No holdings available for this Wave yet."

    top = df.sort_values("Weight_wave", ascending=False).head(1).iloc[0]
    top_name = f"{top['Ticker']} ({top['Company']})"
    top_weight = float(top["Weight_wave"]) * 100

    sector_weights = (
        df.groupby("Sector")["Weight_wave"]
        .sum()
        .sort_values(ascending=False)
    )
    top_sector = sector_weights.index[0]
    top_sector_w = float(sector_weights.iloc[0]) * 100
    concentration5 = float(df["Weight_wave"].nlargest(5).sum()) * 100
    drift_mag = float(df["Drift"].abs().sum()) * 100

    if risk_mode == "Standard":
        mode_line = "running in **Standard** mode, targeting full beta with disciplined rebalancing."
    elif risk_mode == "Alpha-Minus-Beta":
        mode_line = (
            "running in **Alpha-Minus-Beta** mode, dialing back beta while "
            "preserving as much stock-selection alpha as possible."
        )
    else:
        mode_line = (
            "running in **Private Logic** mode, allowing more adaptive tilts "
            "within tight risk guardrails."
        )

    return (
        f"The **{wave_name}** is {mode_line} The top position is **{top_name}** at "
        f"~{top_weight:.1f}% of the Wave. The portfolio is currently most exposed to "
        f"**{top_sector}** (~{top_sector_w:.1f}% of weight), with top-5 holdings "
        f"concentration around **{concentration5:.1f}%**. Aggregate drift versus "
        f"targets is about **{drift_mag:.2f}%** of notional, which the engine translates "
        f"into clean, human-readable trade suggestions in the Trades panel."
    )


def run_stress_test(df: pd.DataFrame, scenario: str) -> tuple[str, float]:
    """Very simple what-if shocks in % terms."""
    if df.empty:
        return "No holdings to stress test.", 0.0

    df = df.copy()
    df["Weight_wave"] = pd.to_numeric(df["Weight_wave"], errors="coerce").fillna(0.0)
    impact = 0.0
    description = ""

    if scenario == "NVDA -10% shock":
        if "NVDA" in df["Ticker"].values:
            w = float(df.loc[df["Ticker"] == "NVDA", "Weight_wave"].iloc[0])
            impact = w * -0.10
            description = "Shock: **NVDA down 10%** with other names unchanged."
        else:
            description = "NVDA not in this Wave; direct impact ~0%."
    elif scenario == "Market -5% shock":
        impact = float(df["Weight_wave"].sum()) * -0.05
        description = "Shock: **broad market down 5%** applied to full equity sleeve."
    else:  # Top holding shock
        top = df.sort_values("Weight_wave", ascending=False).head(1).iloc[0]
        w = float(top["Weight_wave"])
        impact = w * -0.08
        description = (
            f"Shock: **top holding {top['Ticker']} down 8%** with others unchanged."
        )

    return description, impact * 100  # percent move in Wave

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
universe_df = load_universe()
weights_df = load_wave_weights()

with st.expander("Debug: CSV status"):
    st.write("Universe file rows:", len(universe_df))
    st.write("Wave weights file rows:", len(weights_df))
    st.write("Universe columns:", list(universe_df.columns))
    st.write("Wave weights columns:", list(weights_df.columns))
    st.write("Universe preview:", universe_df.head())
    st.write("Weights preview:", weights_df.head())

if universe_df.empty or weights_df.empty:
    st.error("One or both CSV files have no usable data rows.")
    st.stop()

# ------------------------------------------------------------
# SIDEBAR – WAVE & MODE
# ------------------------------------------------------------
waves = sorted(weights_df["Wave"].unique())
selected_wave = st.sidebar.selectbox("Wave", waves, index=0)

risk_mode = st.sidebar.selectbox(
    "Risk Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic"],
)

cash_buffer_pct = st.sidebar.slider(
    "Cash Buffer (%)",
    min_value=0,
    max_value=50,
    value=5,
    step=1,
    help="Simulated cash held in SmartSafe / money market.",
)

st.sidebar.markdown(
    """
    <div style="margin-top:1.5rem;padding-top:0.75rem;
                border-top:1px solid #262654;
                color:#8084FF;font-size:0.7rem;
                text-transform:uppercase;letter-spacing:0.15em;">
        WAVES INTELLIGENCE&trade;
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# BUILD CURRENT WAVE VIEW
# ------------------------------------------------------------
wave_slice = weights_df[weights_df["Wave"] == selected_wave].copy()
merged = wave_slice.merge(universe_df, on="Ticker", how="left")

# Basic cleanup
merged["Sector"] = merged["Sector"].astype(str)
merged["Sector"] = merged["Sector"].replace(["", "nan", "NaN", "None"], "")

# Fill missing sectors via Yahoo Finance
missing_mask = merged["Sector"].str.strip() == ""
missing_tickers = (
    merged.loc[missing_mask, "Ticker"]
    .dropna()
    .astype(str)
    .str.upper()
    .unique()
    .tolist()
)
sector_map = {t: get_sector_from_yf(t) for t in missing_tickers} if missing_tickers else {}
merged.loc[missing_mask, "Sector"] = (
    merged.loc[missing_mask, "Ticker"].map(sector_map).fillna("Unknown")
)

merged["Sector"] = merged["Sector"].replace(["", "nan", "NaN", "None"], "Unknown")
merged["Price"] = pd.to_numeric(merged["Price"], errors="coerce")

# Live prices
live_price_map = fetch_live_prices(merged["Ticker"].unique().tolist())
merged["LivePrice"] = merged["Ticker"].map(live_price_map)
merged["LivePrice"] = merged["LivePrice"].fillna(merged["Price"])

PORTFOLIO_NOTIONAL = 1_000_000.0

valid_price_mask = merged["Price"] > 0
merged["Shares"] = 0.0
merged.loc[valid_price_mask, "Shares"] = (
    merged.loc[valid_price_mask, "Weight_wave"] * PORTFOLIO_NOTIONAL
) / merged.loc[valid_price_mask, "Price"]

merged["LiveValue"] = merged["Shares"] * merged["LivePrice"]

total_live_value = merged["LiveValue"].sum()
if total_live_value > 0:
    merged["LiveWeight"] = merged["LiveValue"] / total_live_value
else:
    merged["LiveWeight"] = merged["Weight_wave"]

merged["Drift"] = merged["LiveWeight"] - merged["Weight_wave"]

THRESH = 0.005  # 0.5% drift
merged["TradeAction"] = ""
merged.loc[merged["Drift"] > THRESH, "TradeAction"] = "Sell"
merged.loc[merged["Drift"] < -THRESH, "TradeAction"] = "Buy"

merged["TradeSize_$"] = merged["Drift"] * PORTFOLIO_NOTIONAL
merged["TradeShares"] = merged["TradeSize_$"] / merged["LivePrice"].replace(0, np.nan)

total_wave_weight = float(merged["Weight_wave"].sum()) if not merged.empty else 0.0
cash_exposure = cash_buffer_pct / 100.0
equity_exposure = max(0.0, 1.0 - cash_exposure) * total_wave_weight

top5_weight = merged["Weight_wave"].nlargest(5).sum() if not merged.empty else 0.0
top10_weight = merged["Weight_wave"].nlargest(10).sum() if not merged.empty else 0.0
top_holding = (
    merged.sort_values("Weight_wave", ascending=False).iloc[0]
    if not merged.empty
    else None
)

# ------------------------------------------------------------
# TOP SUMMARY STRIP
# ------------------------------------------------------------
st.subheader(f"Wave: {selected_wave}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Holdings", len(merged))
m2.metric("Total Wave Weight", f"{total_wave_weight:.4f}")
m3.metric("Equity Exposure", f"{equity_exposure * 100:,.1f}%")
m4.metric("Cash Buffer", f"{cash_buffer_pct:.1f}%")

if top_holding is not None:
    m5, m6, m7 = st.columns(3)
    m5.metric("Top Holding", f"{top_holding['Ticker']} – {top_holding['Company']}")
    m6.metric("Top 5 Concentration", f"{top5_weight * 100:,.1f}%")
    m7.metric("Top 10 Concentration", f"{top10_weight * 100:,.1f}%")

display_df = merged.copy()
display_df["TargetWeight"] = display_df["Weight_wave"]
display_df["CurrentWeight"] = display_df["LiveWeight"]

# ------------------------------------------------------------
# TABS – OVERVIEW / HOLDINGS / TRADES / ANALYTICS + STRESS
# ------------------------------------------------------------
tab_overview, tab_holdings, tab_trades, tab_analytics = st.tabs(
    ["📊 Overview", "📋 Holdings", "📝 Trades", "📈 Analytics & Stress"]
)

# ---------- OVERVIEW ----------
with tab_overview:
    st.markdown(
        "<h4 style='margin-top:0.5rem;margin-bottom:0.4rem;color:#E3E5FF;'>Wave Snapshot</h4>",
        unsafe_allow_html=True,
    )

    o1, o2 = st.columns(2)

    with o1:
        st.markdown(
            "<div style='font-size:0.8rem;color:#B6B9FF;margin-bottom:0.3rem;'>Top 10 by Target Weight</div>",
            unsafe_allow_html=True,
        )
        if not display_df.empty:
            top10 = (
                display_df.sort_values("TargetWeight", ascending=False)
                .head(10)[["Ticker", "Company", "Sector", "TargetWeight"]]
            )
            st.dataframe(top10, use_container_width=True, height=280)

    with o2:
        st.markdown(
            "<div style='font-size:0.8rem;color:#B6B9FF;margin-bottom:0.3rem;'>Top 10 by Current Weight</div>",
            unsafe_allow_html=True,
        )
        if not display_df.empty:
            top10_live = (
                display_df.sort_values("CurrentWeight", ascending=False)
                .head(10)[["Ticker", "Company", "Sector", "CurrentWeight", "Drift"]]
            )
            st.dataframe(top10_live, use_container_width=True, height=280)

    st.markdown(
        "<h4 style='margin-top:1.2rem;margin-bottom:0.4rem;color:#E3E5FF;'>AI Commentary</h4>",
        unsafe_allow_html=True,
    )
    st.write(generate_commentary(selected_wave, merged, risk_mode))

# ---------- HOLDINGS ----------
with tab_holdings:
    st.markdown(
        "<h4 style='margin-top:0.5rem;margin-bottom:0.4rem;color:#E3E5FF;'>Holdings (Live)</h4>",
        unsafe_allow_html=True,
    )

    holdings_cols = [
        "Ticker",
        "Company",
        "Sector",
        "TargetWeight",
        "CurrentWeight",
        "Drift",
        "Price",
        "LivePrice",
        "Shares",
        "LiveValue",
    ]
    holdings_cols = [c for c in holdings_cols if c in display_df.columns]

    st.dataframe(
        display_df[holdings_cols].sort_values("CurrentWeight", ascending=False),
        use_container_width=True,
    )

# ---------- TRADES / TURNOVER LOG ----------
with tab_trades:
    st.markdown(
        "<h4 style='margin-top:0.5rem;margin-bottom:0.4rem;color:#E3E5FF;'>Trade Suggestions (Demo Engine Output)</h4>",
        unsafe_allow_html=True,
    )

    trades = merged[merged["TradeAction"] != ""].copy()
    if trades.empty:
        st.write("No trades required – Wave is within drift thresholds.")
    else:
        trade_cols = [
            "Ticker",
            "Company",
            "TradeAction",
            "Drift",
            "TradeShares",
            "TradeSize_$",
            "LivePrice",
        ]
        trade_cols = [c for c in trade_cols if c in trades.columns]

        st.dataframe(
            trades[trade_cols].sort_values(
                "TradeSize_$", key=lambda s: s.abs(), ascending=False
            ),
            use_container_width=True,
        )

        st.markdown(
            "<div style='font-size:0.8rem;color:#B6B9FF;margin-top:0.3rem;'>"
            "This panel effectively functions as a daily turnover log – highest notional "
            "changes surface to the top for execution review.</div>",
            unsafe_allow_html=True,
        )

# ---------- ANALYTICS & STRESS ----------
with tab_analytics:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            "<h4 style='margin-top:0.5rem;margin-bottom:0.4rem;color:#E3E5FF;'>Simulated NAV (60 business days)</h4>",
            unsafe_allow_html=True,
        )
        nav_series = simulate_nav_series(selected_wave, days=60)
        st.line_chart(nav_series)

    with a2:
        st.markdown(
            "<h4 style='margin-top:0.5rem;margin-bottom:0.4rem;color:#E3E5FF;'>Sector Allocation (Current Weights)</h4>",
            unsafe_allow_html=True,
        )
        if "Sector" in display_df.columns and not display_df.empty:
            sector_weights = (
                display_df.groupby("Sector")["CurrentWeight"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(sector_weights)

    st.markdown(
        "<h4 style='margin-top:1.0rem;margin-bottom:0.4rem;color:#E3E5FF;'>Stress Testing (What-if Scenarios)</h4>",
        unsafe_allow_html=True,
    )

    scenario = st.selectbox(
        "Choose stress scenario",
        ["NVDA -10% shock", "Market -5% shock", "Top holding -8% shock"],
    )

    desc, impact_pct = run_stress_test(merged, scenario)
    st.markdown(desc)
    st.metric("Estimated Wave move under scenario", f"{impact_pct:,.2f}%")