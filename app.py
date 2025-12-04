import streamlit as st
import pandas as pd
from pathlib import Path

# =========================================================
# Helper functions
# =========================================================

def find_col(df: pd.DataFrame, *candidates):
    """Return existing column name matching any candidate (case-insensitive)."""
    cols_lower = {}
    for c in df.columns:
        try:
            key = str(c).lower().strip()
            cols_lower[key] = c
        except Exception:
            continue

    for cand in candidates:
        key = cand.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


@st.cache_data
def load_universe():
    """
    Load the master equity universe from one of several candidate CSVs.
    Try in order:
      1) Master_Stock_Sheet17.csv
      2) Master_Stock_Sheet5.csv
      3) Master_Stock_Sheet.csv
    Returns (df, filename) or (None, None) if nothing found.
    """
    candidates = [
        "Master_Stock_Sheet17.csv",
        "Master_Stock_Sheet5.csv",
        "Master_Stock_Sheet.csv",
    ]

    chosen = None
    for name in candidates:
        p = Path(name)
        if p.exists():
            chosen = name
            break

    if chosen is None:
        return None, None

    df = pd.read_csv(chosen)

    # Normalize key columns
    col_ticker = find_col(df, "Ticker", "Symbol")
    col_name   = find_col(df, "Company", "Name", "Security")
    col_sector = find_col(df, "Sector")
    col_price  = find_col(df, "Price")
    col_mktval = find_col(df, "Market Value", "MarketValue", "MktValue")
    col_weight = find_col(df, "Weight")  # index / benchmark weight

    rename_map = {}
    if col_ticker: rename_map[col_ticker] = "Ticker"
    if col_name:   rename_map[col_name]   = "Company"
    if col_sector: rename_map[col_sector] = "Sector"
    if col_price:  rename_map[col_price]  = "Price"
    if col_mktval: rename_map[col_mktval] = "MarketValue"
    if col_weight: rename_map[col_weight] = "BenchmarkWeight"

    df = df.rename(columns=rename_map)

    keep_cols = [c for c in ["Ticker", "Company", "Sector",
                             "Price", "MarketValue", "BenchmarkWeight"]
                 if c in df.columns]

    df = df[keep_cols].dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"])

    return df, chosen


@st.cache_data
def load_wave_weights():
    """
    Load wave definitions (Wave / Ticker / Weight) from wave_weights.csv.
    Returns df or None.
    """
    p = Path("wave_weights.csv")
    if not p.exists():
        return None

    try:
        df = pd.read_csv(p)
    except Exception:
        return None

    return df


def pct(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return ""


# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="WAVES Intelligence – Equity Wave Console",
    layout="wide",
)

st.markdown(
    "<h2 style='margin-bottom:0'>WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE</h2>",
    unsafe_allow_html=True,
)
st.caption("Equity Waves only – benchmark-aware, AI-directed, multi-mode demo.")
st.write("")

# =========================================================
# Load data
# =========================================================

universe_df, universe_file = load_universe()
waves_df = load_wave_weights()

if universe_df is None or universe_file is None:
    st.error(
        "Universe file not found.\n\n"
        "Place one of these CSVs in the app folder (repo root):\n"
        "- Master_Stock_Sheet17.csv\n"
        "- Master_Stock_Sheet5.csv\n"
        "- Master_Stock_Sheet.csv\n\n"
        "Export your master Sheet (Sheet5) from Google Sheets as CSV and name it "
        "`Master_Stock_Sheet17.csv` for best results."
    )
    st.stop()

st.sidebar.success(f"Universe loaded from **{universe_file}**")

if waves_df is None or waves_df.empty:
    st.error(
        "Wave definitions file **wave_weights.csv** not found or empty.\n\n"
        "Create wave_weights.csv in the repo root with at least these columns:\n"
        "- Wave\n- Ticker\n- Weight (0–1, not %)\n\n"
        "Example:\n"
        "Wave,Ticker,Weight\n"
        "S&P 500 Wave,NVDA,0.061\n"
        "S&P 500 Wave,AAPL,0.033\n"
        "S&P 500 Wave,MSFT,0.052\n"
    )
    st.stop()

# =========================================================
# Clean wave_weights
# =========================================================

col_wave    = find_col(waves_df, "Wave")
col_ticker  = find_col(waves_df, "Ticker", "Symbol")
col_weight  = find_col(waves_df, "Weight", "WaveWeight", "Wgt")
col_alpha   = find_col(waves_df, "Alpha", "AlphaCapture", "Alpha_Capture")
col_is_cash = find_col(waves_df, "IsCash", "CashFlag", "Is_Cash")

if not col_wave or not col_ticker or not col_weight:
    st.error("wave_weights.csv must contain at least Wave, Ticker, and Weight columns.")
    st.stop()

waves_df = waves_df.rename(columns={
    col_wave:   "Wave",
    col_ticker: "Ticker",
    col_weight: "WaveWeight",
})
if col_alpha:
    waves_df = waves_df.rename(columns={col_alpha: "Alpha"})
if col_is_cash:
    waves_df = waves_df.rename(columns={col_is_cash: "IsCash"})

waves_df["WaveWeight"] = pd.to_numeric(waves_df["WaveWeight"], errors="coerce")
waves_df = waves_df.dropna(subset=["Ticker", "WaveWeight"])

if waves_df.empty:
    st.error("After cleaning, no valid Wave rows remain. Check wave_weights.csv.")
    st.stop()

# =========================================================
# Join with universe
# =========================================================

joined_df = waves_df.merge(
    universe_df,
    on="Ticker",
    how="left",
    suffixes=("", "_universe"),
)

missing = joined_df["Company"].isna().sum()
if missing > 0:
    st.sidebar.warning(
        f"{missing} holdings in wave_weights.csv have no match in {universe_file}."
    )

# =========================================================
# Wave & mode selectors
# =========================================================

waves = sorted(joined_df["Wave"].dropna().unique().tolist())
if not waves:
    st.error("No Waves found in wave_weights.csv.")
    st.stop()

selected_wave = st.sidebar.selectbox("Select Wave", waves)

mode = st.sidebar.radio(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
)

mode_scale = {
    "Standard": 1.00,
    "Alpha-Minus-Beta": 0.80,
    "Private Logic™": 1.20,
}.get(mode, 1.00)

wave_df = joined_df[joined_df["Wave"] == selected_wave].copy()
if wave_df.empty:
    st.error("No holdings found for the selected Wave.")
    st.stop()

wave_df["EffectiveWeight"] = wave_df["WaveWeight"] * mode_scale

has_alpha = "Alpha" in wave_df.columns
has_cash  = "IsCash" in wave_df.columns

# =========================================================
# Header
# =========================================================

st.markdown(
    f"<h1 style='color:#4CAFEB; margin-top:0'>{selected_wave} (LIVE Demo)</h1>",
    unsafe_allow_html=True,
)
st.caption(
    f"Mode: **{mode}** – equities only. In production this flag would drive overlays and rebalancing."
)
st.write("")

# =========================================================
# Top holdings & analytics
# =========================================================

left, right = st.columns([1.3, 1.2])

with left:
    st.subheader("Top 10 holdings")

    wave_sorted = wave_df.sort_values("EffectiveWeight", ascending=False)
    top10 = wave_sorted.head(10).copy()

    table = {}
    if "Company" in top10.columns:
        table["Name"] = top10["Company"]
    table["Ticker"] = top10["Ticker"]
    table["Base weight"] = top10["WaveWeight"].apply(pct)
    table["Mode weight"] = top10["EffectiveWeight"].apply(pct)
    if "Sector" in top10.columns:
        table["Sector"] = top10["Sector"]

    st.caption("Ranked by Wave weight (mode-adjusted).")
    st.dataframe(pd.DataFrame(table), use_container_width=True)

with right:
    st.subheader("Top-10 by Wave weight")

    chart_data = top10[["Ticker", "EffectiveWeight"]].set_index("Ticker")
    st.bar_chart(chart_data)

    total_holdings = len(wave_df)
    largest_weight = wave_df["EffectiveWeight"].max()

    equity_weight = wave_df["WaveWeight"].sum()
    cash_weight = 0.0
    if has_cash:
        cash_mask = wave_df["IsCash"] == True
        cash_weight = wave_df.loc[cash_mask, "WaveWeight"].sum()
        equity_weight = wave_df["WaveWeight"].sum() - cash_weight

    alpha_capture = None
    if has_alpha:
        alpha_capture = wave_df["Alpha"].mean()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("TOTAL HOLDINGS", f"{total_holdings}")
    with c2:
        if equity_weight or cash_weight:
            st.metric(
                "EQUITY vs CASH (base)",
                f"{equity_weight * 100:.0f}% / {cash_weight * 100:.0f}%",
            )
        else:
            st.metric("EQUITY vs CASH", "n/a")
    with c3:
        st.metric("LARGEST POSITION (mode)", pct(largest_weight))

    c4, c5 = st.columns(2)
    with c4:
        if alpha_capture is not None:
            st.metric("ALPHA CAPTURE (est.)", f"{alpha_capture:.2f}")
        else:
            st.metric("ALPHA CAPTURE (est.)", "n/a")

# =========================================================
# Sector allocation & weight-decay
# =========================================================

st.markdown("---")
sec_col, decay_col = st.columns(2)

with sec_col:
    st.subheader("Sector allocation")
    if "Sector" in wave_df.columns:
        sector_data = (
            wave_df
            .dropna(subset=["Sector"])
            .groupby("Sector")["EffectiveWeight"]
            .sum()
            .sort_values(ascending=False)
        )
        if not sector_data.empty:
            st.bar_chart(sector_data)
        else:
            st.info("No sector data available for this Wave.")
    else:
        st.info("No Sector column in universe; cannot show sector allocation.")

with decay_col:
    st.subheader("Weight decay curve")
    decay_data = (
        wave_sorted[["EffectiveWeight"]]
        .reset_index(drop=True)
        .rename(columns={"EffectiveWeight": "Weight"})
    )
    decay_data["Rank"] = decay_data.index + 1
    decay_data = decay_data.set_index("Rank")
    st.line_chart(decay_data)

# =========================================================
# Mode overview / console status
# =========================================================

st.markdown("---")
st.subheader("Mode overview")
st.write(
    """
**Standard** – Wave aligned tightly to its benchmark with controlled tracking error.  

**Alpha-Minus-Beta** – same selection logic, but effective equity exposure dialed down
(e.g., 80%) to make room for SmartSafe™ / hedging overlays.  

**Private Logic™** – proprietary overlays for leadership, momentum, and SmartSafe™,
allowing equity exposure to expand or contract more aggressively.
"""
)

st.subheader("Console status")
st.write(
    """
- **Read-only demo** – no real orders are routed.  
- All analytics are calculated from **Master_Stock_SheetXX.csv** (your universe) 
  plus **wave_weights.csv** (your Waves).  
- Equities only in this version; crypto & income Waves can be layered later.
"""
)