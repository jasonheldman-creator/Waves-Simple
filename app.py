import os
import glob

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE",
    layout="wide",
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates):
    """
    Given a DataFrame and a list of candidate column names,
    return the first match (case-insensitive), or None.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        col = lower_map.get(cand.lower())
        if col is not None:
            return col
    return None


@st.cache_data
def load_universe(universe_path: str = "Master_Stock_Sheet5.csv"):
    """
    Load the master equity universe (S&P 500 + others) from CSV.

    Expected columns (case-insensitive):
      - Ticker / Symbol
      - Company / Name
      - Sector
      - Weight (optional)
    """
    if not os.path.exists(universe_path):
        st.error(
            f"Universe file **{universe_path}** not found in the repo root.\n\n"
            f"Make sure `Master_Stock_Sheet5.csv` is uploaded to the **Waves-Simple** "
            f"repository beside `app.py`."
        )
        return None

    df = pd.read_csv(universe_path)

    # Normalise column names
    ticker_col = _find_column(df, ["Ticker", "Symbol"])
    name_col = _find_column(df, ["Company", "Name"])
    sector_col = _find_column(df, ["Sector"])
    weight_col = _find_column(df, ["Weight", "IndexWeight", "Index Weight"])

    if ticker_col is None:
        st.error("Could not find a **Ticker** column in Master_Stock_Sheet5.csv.")
        return None

    # Build a cleaned view
    clean = pd.DataFrame()
    clean["Ticker"] = df[ticker_col].astype(str).str.strip()

    if name_col is not None:
        clean["Company"] = df[name_col].astype(str).str.strip()
    else:
        clean["Company"] = ""

    if sector_col is not None:
        clean["Sector"] = df[sector_col].astype(str).str.strip()
    else:
        clean["Sector"] = "Other"

    if weight_col is not None:
        # Universe / index weight (not the Wave weight)
        clean["UniverseWeight"] = pd.to_numeric(
            df[weight_col], errors="coerce"
        ).fillna(0.0)
    else:
        clean["UniverseWeight"] = 0.0

    # Add an equal-weight fallback if universe weights are missing
    if clean["UniverseWeight"].sum() <= 0:
        n = len(clean)
        if n > 0:
            clean["UniverseWeight"] = 1.0 / n

    return clean


@st.cache_data
def load_wave_weights():
    """
    Load the Wave → Ticker → Weight mapping from any CSV
    whose filename starts with 'WaveWeight' and ends with '.csv'.

    This is designed to catch Google Sheets exports like:
      - WaveWeight-Sheet1.csv
      - WaveWeight-Sheet1.csv - Sheet1.csv
    """
    candidates = sorted(glob.glob("WaveWeight*.csv"))
    if not candidates:
        st.error(
            "No **WaveWeight** CSV found in the repo root.\n\n"
            "Expected a file whose name starts with `WaveWeight` and ends with `.csv`, "
            "for example: `WaveWeight-Sheet1.csv` or `WaveWeight-Sheet1.csv - Sheet1.csv`."
        )
        return None, None

    path = candidates[0]
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read WaveWeight file `{path}`: {e}")
        return None, None

    # Normalise expected columns
    wave_col = _find_column(df, ["Wave", "Wave Name"])
    ticker_col = _find_column(df, ["Ticker", "Symbol"])
    weight_col = _find_column(df, ["Weight", "WaveWeight"])

    if wave_col is None or ticker_col is None or weight_col is None:
        st.error(
            f"Wave weight file `{path}` must contain at least these columns: "
            f"`Wave`, `Ticker`, `Weight` (case-insensitive)."
        )
        return None, None

    wf = pd.DataFrame()
    wf["Wave"] = df[wave_col].astype(str).str.strip()
    wf["Ticker"] = df[ticker_col].astype(str).str.strip()
    wf["Weight"] = pd.to_numeric(df[weight_col], errors="coerce")

    # Drop rows with missing ticker or wave
    wf = wf.dropna(subset=["Wave", "Ticker"]).copy()

    # Replace any invalid weights with equal-weight per Wave
    for wave_name, group in wf.groupby("Wave"):
        mask = wf["Wave"] == wave_name
        # Any non-positive or NaN weights?
        bad = (wf.loc[mask, "Weight"].isna()) | (wf.loc[mask, "Weight"] <= 0)
        if bad.any():
            count = mask.sum()
            wf.loc[mask, "Weight"] = 1.0 / count
        else:
            # Normalise weights per wave
            total = wf.loc[mask, "Weight"].sum()
            if total > 0:
                wf.loc[mask, "Weight"] = wf.loc[mask, "Weight"] / total

    wave_names = sorted(wf["Wave"].unique().tolist())
    return wf, wave_names


def apply_mode_weights(base_weights: pd.Series, mode: str) -> pd.Series:
    """
    Apply a simple, deterministic transformation to base Wave weights
    depending on the selected mode.
    """
    w = base_weights.clip(lower=0.0).astype(float)

    if w.sum() == 0:
        return w

    if mode == "Standard":
        adj = w.values
    elif mode == "Alpha-Minus-Beta":
        # Slightly flatten extremes to reduce concentration risk
        adj = np.sqrt(w.values)
    elif mode == "Private Logic™":
        # Slightly accentuate conviction names
        adj = np.power(w.values, 1.2)
    else:
        adj = w.values

    adj = adj / np.sum(adj)
    return pd.Series(adj, index=w.index)


def build_wave_holdings(universe_df: pd.DataFrame,
                        weights_df: pd.DataFrame,
                        wave_name: str,
                        mode: str) -> pd.DataFrame:
    """
    Merge Wave-specific weights with the master universe to get
    a clean holdings table for that Wave in the selected mode.
    """
    wave_weights = weights_df[weights_df["Wave"] == wave_name].copy()
    if wave_weights.empty:
        return pd.DataFrame()

    merged = pd.merge(
        wave_weights,
        universe_df,
        on="Ticker",
        how="left",
        suffixes=("", "_universe"),
    )

    # Base wave weights
    merged["BaseWeight"] = pd.to_numeric(
        merged["Weight"], errors="coerce"
    ).fillna(0.0)

    # Apply the mode transform
    merged["WaveWeight"] = apply_mode_weights(merged["BaseWeight"], mode)

    # Clean columns for display
    cols = ["Ticker", "Company", "Sector", "WaveWeight"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = "" if c in ["Company", "Sector"] else 0.0

    merged = merged[cols].copy()
    merged = merged.sort_values("WaveWeight", ascending=False)

    return merged


def format_pct(x):
    try:
        return f"{100 * float(x):.2f}%"
    except Exception:
        return ""


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------

universe = load_universe()
weights, wave_names = load_wave_weights()

if universe is None or weights is None or not wave_names:
    st.stop()

# Sidebar
st.sidebar.header("Data source & mode")

st.sidebar.success(
    "Universe: **Master_Stock_Sheet5.csv**\n\n"
    f"Wave weights file detected: **{sorted(glob.glob('WaveWeight*.csv'))[0]}**"
)

selected_wave = st.sidebar.selectbox(
    "Select Wave",
    options=wave_names,
    index=0,
)

mode = st.sidebar.radio(
    "Mode",
    options=["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "All Waves share the **same 5,000-stock equity universe**. "
    "Each Wave selects a different subset with different weights."
)

# Main header
st.markdown("### WAVES INTELLIGENCE™ – PORTFOLIO WAVE CONSOLE")
st.markdown(
    f"#### {selected_wave} (LIVE Demo) &nbsp;&nbsp;·&nbsp;&nbsp; Mode: **{mode}**"
)

holdings = build_wave_holdings(universe, weights, selected_wave, mode)

if holdings.empty:
    st.warning(
        f"No holdings found for Wave **{selected_wave}**. "
        f"Check your WaveWeight CSV for that Wave name."
    )
    st.stop()

# --- Key stats
total_holdings = len(holdings)
largest_position = holdings["WaveWeight"].max()
top10 = holdings.head(10).copy()

alpha_capture_est = 0.0  # placeholder until we plug in performance engine

# Layout: 2 columns for Top-10 table & chart
col1, col2 = st.columns([1.1, 1.2])

with col1:
    st.subheader("Top 10 holdings")
    display_top10 = top10.copy()
    display_top10["Weight"] = display_top10["WaveWeight"].apply(format_pct)
    display_top10 = display_top10.drop(columns=["WaveWeight"])
    st.dataframe(
        display_top10,
        use_container_width=True,
        hide_index=True,
    )

with col2:
    st.subheader("Top-10 by Wave weight")
    fig_top = px.bar(
        top10,
        x="Ticker",
        y="WaveWeight",
        hover_data=["Company", "Sector"],
        labels={"WaveWeight": "Weight"},
    )
    fig_top.update_layout(
        xaxis_title="Ticker",
        yaxis_title="Wave weight",
        margin=dict(l=10, r=10, t=30, b=30),
    )
    st.plotly_chart(fig_top, use_container_width=True)

# --- Metrics row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total holdings", f"{total_holdings:,}")
with m2:
    st.metric("Largest position (est.)", format_pct(largest_position))
with m3:
    st.metric("Equity vs cash", "100% / 0%")  # demo: fully invested
with m4:
    st.metric("Alpha capture (est.)", "n/a")

st.markdown("---")

# --- Sector allocation and weight decay
c1, c2 = st.columns(2)

with c1:
    st.subheader("Sector allocation")
    if "Sector" in holdings.columns:
        sector_alloc = (
            holdings.groupby("Sector")["WaveWeight"]
            .sum()
            .reset_index()
            .sort_values("WaveWeight", ascending=False)
        )
        fig_sector = px.bar(
            sector_alloc,
            x="Sector",
            y="WaveWeight",
            labels={"WaveWeight": "Weight"},
        )
        fig_sector.update_layout(
            xaxis_title="Sector",
            yaxis_title="Wave weight",
            margin=dict(l=10, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    else:
        st.info("No **Sector** column available in the universe file.")

with c2:
    st.subheader("Weight decay curve")
    decay = holdings.copy()
    decay["Rank"] = np.arange(1, len(decay) + 1)
    fig_decay = px.line(
        decay,
        x="Rank",
        y="WaveWeight",
        labels={"Rank": "Holding rank (largest to smallest)",
                "WaveWeight": "Weight"},
    )
    fig_decay.update_layout(
        margin=dict(l=10, r=10, t=30, b=30),
    )
    st.plotly_chart(fig_decay, use_container_width=True)

st.markdown("---")

# --- Console status / narrative
st.subheader("Mode overview & console status")

mode_notes = {
    "Standard": (
        "Mode **Standard** – equities only. In production, this mode would keep "
        "the Wave tightly aligned to its benchmark with controlled tracking error."
    ),
    "Alpha-Minus-Beta": (
        "Mode **Alpha-Minus-Beta** – de-emphasizes single-name concentration, "
        "targeting smoother risk with a focus on benchmark-aware alpha."
    ),
    "Private Logic™": (
        "Mode **Private Logic™** – proprietary leadership, regime-switching, and "
        "SmartSafe™ overlays tuned for institutional-grade alpha at higher risk."
    ),
}

st.write(mode_notes.get(mode, ""))

st.info(
    "This is a **demo console**. No real orders are routed from this screen. "
    "All analytics are calculated from the uploaded universe and WaveWeight CSVs. "
    "Every Wave and mode can be exported to a full institutional console."
)