# app.py  — WAVES Intelligence™ Institutional Console
# Uses the new function-style engine: build_engine() from waves_engine

import traceback
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

from waves_engine import build_engine  # <- NEW import


# -----------------------------
# CACHED ENGINE & METRICS
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_engine():
    """
    Create and cache the core Waves engine.
    Assumes waves_engine.build_engine() returns an object with
    .compute_all_metrics(risk_mode='standard') method.
    """
    return build_engine()


@st.cache_data(show_spinner=True, ttl=900)
def load_metrics(risk_mode: str) -> pd.DataFrame:
    """
    Ask the engine for metrics across all Waves and normalize
    to a pandas DataFrame.
    """
    eng = load_engine()
    metrics_dict: Dict[str, Any] = eng.compute_all_metrics(risk_mode=risk_mode)

    rows: List[Dict[str, Any]] = []
    for wave_name, m in metrics_dict.items():
        if is_dataclass(m):
            row = asdict(m)
        elif isinstance(m, dict):
            row = dict(m)
        else:
            # Fallback: just stuff everything into a generic row
            row = {"wave": wave_name, "value": m}
        # Ensure the wave name is always present
        if "wave" not in row:
            row["wave"] = wave_name
        rows.append(row)

    df = pd.DataFrame(rows)

    # Nice human-friendly column names (only if present)
    rename_map = {
        "wave": "Wave",
        "benchmark": "Benchmark",
        "beta_target": "Beta (≈60M)",
        "alpha_30d": "Alpha 30D (%)",
        "alpha_60d": "Alpha 60D (%)",
        "alpha_1y": "Alpha 1Y (%)",
        "wave_ret_30d": "Wave 30D (%)",
        "wave_ret_60d": "Wave 60D (%)",
        "wave_ret_1y": "Wave 1Y (%)",
        "bench_ret_30d": "Bench 30D (%)",
        "bench_ret_60d": "Bench 60D (%)",
        "bench_ret_1y": "Bench 1Y (%)",
        "vix_regime": "VIX Regime",
        "mode": "Mode",
        "as_of": "As Of",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df


@st.cache_data(show_spinner=False)
def load_wave_weights() -> pd.DataFrame:
    """
    Load wave_weights.csv to power the Top 10 holdings / Google links
    for each Wave. This does NOT depend on the engine.
    """
    try:
        weights = pd.read_csv("wave_weights.csv")
    except Exception as e:
        st.warning(f"Could not load wave_weights.csv ({e})")
        return pd.DataFrame(columns=["wave", "ticker", "weight"])

    # Normalize column names
    weights.columns = [c.strip().lower() for c in weights.columns]
    # Expect columns: wave, ticker, weight
    return weights


# -----------------------------
# UI HELPERS
# -----------------------------
def safe_mean(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("nan")
    return float(series.mean())


def format_pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.2f}%"
    except Exception:
        return "—"


def google_quote_link(ticker: str) -> str:
    # Generic Google search for robustness across tickers & exchanges
    return f"https://www.google.com/search?q={ticker}+stock+price"


def top_holdings_table(weights: pd.DataFrame, wave_name: str, top_n: int = 10) -> pd.DataFrame:
    """
    Return a small DataFrame with top holdings + markdown link for Streamlit.
    """
    if weights.empty:
        return pd.DataFrame(columns=["Ticker", "Weight", "Quote"])

    mask = weights["wave"].astype(str).str.strip() == wave_name
    subset = weights.loc[mask].copy()

    if subset.empty:
        return pd.DataFrame(columns=["Ticker", "Weight", "Quote"])

    # Coerce weight to float; if it fails, just treat as equal weight
    try:
        subset["weight"] = subset["weight"].astype(float)
    except Exception:
        subset["weight"] = 1.0 / len(subset)

    subset = subset.sort_values("weight", ascending=False).head(top_n)

    subset["Ticker"] = subset["ticker"].astype(str)
    subset["Weight"] = subset["weight"].round(4)
    subset["Quote"] = subset["Ticker"].apply(lambda t: f"[Google Quote]({google_quote_link(t)})")

    return subset[["Ticker", "Weight", "Quote"]]


# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.markdown(
    """
    # WAVES Intelligence™ Institutional Console  
    **Live Wave Engine • Alpha Capture • Benchmark-Relative Performance**
    """.strip()
)

# -----------------------------
# SIDEBAR — MODE & BASIC CONTROLS
# -----------------------------
with st.sidebar:
    st.header("Mode & Settings")

    risk_mode = st.selectbox(
        "Wave Mode",
        options=["standard", "alpha-minus-beta", "private-logic"],
        index=0,
        format_func=lambda x: {
            "standard": "Standard",
            "alpha-minus-beta": "Alpha-Minus-Beta",
            "private-logic": "Private Logic™",
        }.get(x, x),
        help="This is passed into the engine as the risk/trading mode.",
    )

    st.caption(
        "Engine uses full WAVES strategy stack including VIX-aware overlays, "
        "benchmark-blend alpha capture, and 30D / 60D / 1Y windows."
    )

# -----------------------------
# LOAD DATA (ENGINE + WEIGHTS)
# -----------------------------
weights_df = load_wave_weights()

try:
    metrics_df = load_metrics(risk_mode)
except Exception as e:
    st.error("Engine failed while computing metrics. See diagnostics below.")
    with st.expander("Full error traceback"):
        st.code("".join(traceback.format_exc()))
    st.stop()

if metrics_df.empty:
    st.warning("No metrics returned by engine yet. Check engine configuration and data sources.")
    st.stop()

# Convenience lookups
wave_col = "Wave" if "Wave" in metrics_df.columns else metrics_df.columns[0]
wave_names = sorted(metrics_df[wave_col].astype(str).unique())

# -----------------------------
# TABS
# -----------------------------
tab_dashboard, tab_explorer, tab_about = st.tabs(
    ["Dashboard", "Wave Explorer", "About / Diagnostics"]
)

# =============================
# DASHBOARD TAB
# =============================
with tab_dashboard:
    st.subheader(f"Dashboard — Mode: {risk_mode}")

    # Summary stats section
    col1, col2, col3 = st.columns(3)

    alpha30_col = "Alpha 30D (%)" if "Alpha 30D (%)" in metrics_df.columns else None
    alpha60_col = "Alpha 60D (%)" if "Alpha 60D (%)" in metrics_df.columns else None
    alpha1y_col = "Alpha 1Y (%)" if "Alpha 1Y (%)" in metrics_df.columns else None

    if alpha30_col:
        col1.metric("Avg 30-Day Alpha", format_pct(safe_mean(metrics_df[alpha30_col])))
    else:
        col1.metric("Avg 30-Day Alpha", "—")

    if alpha60_col:
        col2.metric("Avg 60-Day Alpha", format_pct(safe_mean(metrics_df[alpha60_col])))
    else:
        col2.metric("Avg 60-Day Alpha", "—")

    if alpha1y_col:
        col3.metric("Avg 1-Year Alpha", format_pct(safe_mean(metrics_df[alpha1y_col])))
    else:
        col3.metric("Avg 1-Year Alpha", "—")

    # All Waves Snapshot
    st.markdown("### All Waves Snapshot")

    # Choose sort key
    sort_options = []
    for candidate in ["Alpha 30D (%)", "Alpha 60D (%)", "Alpha 1Y (%)", "Wave 1Y (%)"]:
        if candidate in metrics_df.columns:
            sort_options.append(candidate)
    sort_label = st.selectbox(
        "Sort Waves by",
        options=sort_options or [wave_col],
        index=0,
    )

    snap_df = metrics_df.copy()
    if sort_label in snap_df.columns:
        snap_df = snap_df.sort_values(sort_label, ascending=False)

    # Rearrange columns so the main ones appear first if present
    preferred_order = [
        wave_col,
        "Benchmark",
        "Beta (≈60M)",
        "Alpha 30D (%)",
        "Alpha 60D (%)",
        "Alpha 1Y (%)",
        "Wave 1Y (%)",
        "Bench 1Y (%)",
        "VIX Regime",
        "Mode",
        "As Of",
    ]
    ordered_cols = [c for c in preferred_order if c in snap_df.columns] + [
        c for c in snap_df.columns if c not in preferred_order
    ]
    snap_df = snap_df[ordered_cols]

    st.dataframe(
        snap_df,
        use_container_width=True,
        height=480,
    )

# =============================
# WAVE EXPLORER TAB
# =============================
with tab_explorer:
    st.subheader("Wave Explorer")

    left, right = st.columns([1, 2])

    with left:
        selected_wave = st.selectbox(
            "Select Wave",
            options=wave_names,
            index=0,
        )

        current_row = metrics_df.loc[metrics_df[wave_col] == selected_wave].iloc[0]

        st.markdown(f"### {selected_wave}")

        # Key metrics
        m1, m2, m3 = st.columns(3)
        if alpha30_col:
            m1.metric("Alpha 30D", format_pct(current_row.get(alpha30_col)))
        if alpha60_col:
            m2.metric("Alpha 60D", format_pct(current_row.get(alpha60_col)))
        if alpha1y_col:
            m3.metric("Alpha 1Y", format_pct(current_row.get(alpha1y_col)))

        # Wave vs Benchmark 1Y returns (if available)
        wave1y_col = "Wave 1Y (%)"
        bench1y_col = "Bench 1Y (%)"
        if wave1y_col in metrics_df.columns or bench1y_col in metrics_df.columns:
            st.markdown("#### 1-Year Performance vs Benchmark")
            w = format_pct(current_row.get(wave1y_col)) if wave1y_col in current_row else "—"
            b = format_pct(current_row.get(bench1y_col)) if bench1y_col in current_row else "—"
            st.write(f"**Wave 1Y Return:** {w}")
            st.write(f"**Benchmark 1Y Return:** {b}")

        # Benchmark info
        if "Benchmark" in current_row:
            st.markdown("#### Benchmark Blend")
            st.write(str(current_row["Benchmark"]))

        if "Beta (≈60M)" in current_row:
            st.markdown("#### Target Beta")
            st.write(f"{current_row['Beta (≈60M)']}")

    with right:
        st.markdown("### Top Holdings (with Google Quote Links)")

        holdings = top_holdings_table(weights_df, selected_wave, top_n=10)
        if holdings.empty:
            st.info("No holdings found for this Wave in wave_weights.csv.")
        else:
            st.dataframe(
                holdings,
                use_container_width=True,
                height=400,
            )

        st.caption(
            "Quote links open Google in a new tab. "
            "Weights are taken directly from wave_weights.csv."
        )

# =============================
# ABOUT / DIAGNOSTICS TAB
# =============================
with tab_about:
    st.subheader("About WAVES Intelligence™ Console")

    st.markdown(
        """
        This console is a **presentation layer** on top of the WAVES Intelligence™ engine:

        - Loads **wave_weights.csv** to define each Wave’s holdings  
        - Engine pulls historical price data (via `yfinance` or equivalent)  
        - Applies **benchmark blends**, **VIX-aware risk overlays**, and **multi-window alpha capture**  
        - Exposes risk-mode aware metrics (Standard, Alpha-Minus-Beta, Private Logic™)  

        ### Diagnostics

        Use this section during debugging or live demos if something looks off.
        """
    )

    with st.expander("Raw Metrics DataFrame"):
        st.dataframe(metrics_df, use_container_width=True, height=500)

    with st.expander("wave_weights.csv preview"):
        st.dataframe(weights_df.head(120), use_container_width=True, height=500)

    st.markdown(
        """
        **Notes**

        - If you see messages like _"No price data found for any tickers"_  
          check ticker symbols, internet connectivity, and intraday/holiday timing.  
        - If a Wave appears in `wave_weights.csv` but not in the snapshot,  
          confirm it is discovered and mapped correctly inside `waves_engine.py`.
        """
    )