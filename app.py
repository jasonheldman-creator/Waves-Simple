import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ============================================================
# PAGE CONFIG & GLOBALS
# ============================================================

st.set_page_config(
    page_title="WAVES Institutional Console",
    page_icon="🌊",
    layout="wide",
)

ROOT_DIR = Path(__file__).parent.resolve()
LOGS_DIR = ROOT_DIR / "logs"
PERF_DIR = LOGS_DIR / "performance"
POS_DIR = LOGS_DIR / "positions"

for d in [LOGS_DIR, PERF_DIR, POS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Map internal wave IDs -> nice display names
WAVE_NAME_MAP = {
    "Growth_Wave": "Growth Wave",
    "SP500_Wave": "S&P 500 Wave",
    "Income_Wave": "Income Wave",
    "Future_Power_Wave": "Future Power & Energy Wave",
    "Crypto_Income_Wave": "Crypto Income Wave",
    "Quantum_Wave": "Quantum Computing Wave",
    "Clean_Transit_Wave": "Clean Transit & Infrastructure Wave",
    "SMID_Growth_Wave": "Small–Mid Cap Growth Wave",
    "AL_Wave": "AI Leaders Wave",
}

def pretty_wave_name(internal: str) -> str:
    return WAVE_NAME_MAP.get(internal, internal.replace("_", " "))


# ============================================================
# STYLING
# ============================================================

CUSTOM_CSS = """
<style>
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

[data-testid="stMetricValue"] {
    font-weight: 600;
    font-size: 1.4rem;
}

.wave-pill {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    background: rgba(0, 255, 135, 0.08);
    border: 1px solid rgba(0, 255, 135, 0.35);
    color: #00ff87;
}

.mode-pill {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    background: rgba(0, 168, 255, 0.08);
    border: 1px solid rgba(0, 168, 255, 0.35);
    color: #00a8ff;
}

/* compact top-10 table */
.top10-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.top10-table th {
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 0.25rem 0.4rem;
}
.top10-table td {
    padding: 0.18rem 0.4rem;
}
.top10-ticker {
    font-weight: 600;
    text-decoration: none;
}
.top10-weight {
    text-align: right;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================

def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_universe_file(root: Path) -> Optional[Path]:
    candidates = [
        root / "wave_weights.csv",
        root / "Wave_Weights.csv",
        root / "weights.csv",
        root / "universe.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    for c in root.glob("*.csv"):
        return c
    return None


def load_universe_df() -> Optional[pd.DataFrame]:
    universe_path = find_universe_file(ROOT_DIR)
    if universe_path is None:
        return None

    df = safe_read_csv(universe_path)
    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    col_map: Dict[str, str] = {}

    for candidate in ["wave", "wave_name", "portfolio", "strategy"]:
        if candidate in df.columns:
            col_map["wave"] = candidate
            break

    for candidate in ["ticker", "symbol"]:
        if candidate in df.columns:
            col_map["ticker"] = candidate
            break

    for candidate in ["weight", "target_weight", "wgt"]:
        if candidate in df.columns:
            col_map["weight"] = candidate
            break

    for candidate in ["name", "company", "security_name"]:
        if candidate in df.columns:
            col_map["name"] = candidate
            break

    for candidate in ["sector", "industry"]:
        if candidate in df.columns:
            col_map["sector"] = candidate
            break

    out = pd.DataFrame()
    if "wave" in col_map:
        out["wave"] = df[col_map["wave"]].astype(str)
    if "ticker" in col_map:
        out["ticker"] = df[col_map["ticker"]].astype(str)
    if "weight" in col_map:
        out["weight"] = pd.to_numeric(df[col_map["weight"]], errors="coerce")
    if "name" in col_map:
        out["name"] = df[col_map["name"]].astype(str)
    if "sector" in col_map:
        out["sector"] = df[col_map["sector"]].astype(str)

    if out.empty or "wave" not in out.columns or "ticker" not in out.columns:
        return None

    out["weight"] = out.get("weight", 0).fillna(0.0)
    return out


def get_available_waves(universe_df: Optional[pd.DataFrame]) -> List[str]:
    if universe_df is not None and "wave" in universe_df.columns:
        waves = sorted(universe_df["wave"].dropna().unique().tolist())
        if waves:
            return waves
    # fallback list
    return [
        "AL_Wave",
        "Growth_Wave",
        "SP500_Wave",
        "Income_Wave",
        "Future_Power_Wave",
        "Crypto_Income_Wave",
        "Quantum_Wave",
        "Clean_Transit_Wave",
        "SMID_Growth_Wave",
    ]


def find_latest_perf_file_for_wave(wave_name: str) -> Optional[Path]:
    if not PERF_DIR.exists():
        return None
    pattern_dated = list(PERF_DIR.glob(f"{wave_name}_performance_*_daily.csv"))
    pattern_simple = list(PERF_DIR.glob(f"{wave_name}_performance_daily.csv"))
    candidates = pattern_dated + pattern_simple
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def load_perf_df_for_wave(wave_name: str) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    perf_path = find_latest_perf_file_for_wave(wave_name)
    if perf_path is None:
        return None, None
    df = safe_read_csv(perf_path)
    if df is None or df.empty:
        return None, perf_path

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    for candidate in ["timestamp", "date", "asof"]:
        if candidate in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df[candidate])
            except Exception:
                df["timestamp"] = df[candidate].astype(str)
            break

    return df, perf_path


def compute_perf_metrics(perf_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    metrics = {
        "total_return": None,
        "today_return": None,
        "max_drawdown": None,
        "alpha": None,
    }

    if perf_df is None or perf_df.empty:
        return metrics

    cols = perf_df.columns

    for candidate in ["cumulative_return", "cum_return", "total_return", "portfolio_return"]:
        if candidate in cols:
            metrics["total_return"] = perf_df[candidate].iloc[-1]
            break

    for candidate in ["daily_return", "return", "day_return"]:
        if candidate in cols:
            metrics["today_return"] = perf_df[candidate].iloc[-1]
            break

    for candidate in ["max_drawdown", "drawdown"]:
        if candidate in cols:
            series = perf_df[candidate]
            metrics["max_drawdown"] = series.min() if series.min() < 0 else series.iloc[-1]
            break

    for candidate in ["alpha", "excess_return", "alpha_vs_benchmark"]:
        if candidate in cols:
            metrics["alpha"] = perf_df[candidate].iloc[-1]
            break

    return metrics


def load_positions_for_wave(wave_name: str) -> Optional[pd.DataFrame]:
    if not POS_DIR.exists():
        return None
    candidates = list(POS_DIR.glob(f"{wave_name}_positions_*.csv"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    df = safe_read_csv(latest)
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:,.2f}%"


def fmt_bp(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 10000:,.0f} bp"


def choose_perf_series(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Choose a good column (or few) to plot: prefer cum return / NAV over exposure."""
    if df is None or df.empty:
        return None

    num = df.select_dtypes(include=["number"]).copy()
    if num.empty:
        return None

    # score columns
    scores: Dict[str, int] = {}
    for c in num.columns:
        cl = c.lower()
        score = 0
        if "cum" in cl and "return" in cl:
            score += 100
        if "nav" in cl or "portfolio_value" in cl or "equity_curve" in cl:
            score += 90
        if "return" in cl:
            score += 60
        if "equity_exposure" in cl or "beta" in cl or "exposure" in cl:
            score -= 50
        if "weight" in cl:
            score -= 80
        scores[c] = score

    best_cols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    best_cols = [c for c in best_cols if scores[c] > -10]  # drop obviously bad ones

    if not best_cols:
        best_cols = num.columns.tolist()

    plot_df = num[best_cols[:2]]  # at most 2 lines
    return plot_df


def build_top10_html(df: pd.DataFrame, positions: Optional[pd.DataFrame]) -> str:
    """Build HTML table for top 10 with Google Finance links & red/green coloring."""
    df = df.copy()

    # join in daily change if we have positions
    change_map: Dict[str, float] = {}
    if positions is not None and not positions.empty:
        pos = positions.copy()
        pos.columns = [c.strip().lower() for c in pos.columns]
        change_col = None
        for cand in ["day_return", "daily_return", "return", "change_pct", "pct_change"]:
            if cand in pos.columns:
                change_col = cand
                break
        if change_col and "ticker" in pos.columns:
            for _, row in pos.iterrows():
                t = str(row["ticker"])
                val = row[change_col]
                try:
                    val = float(val)
                    change_map[t] = val
                except Exception:
                    continue

    rows_html = []
    for _, row in df.head(10).iterrows():
        ticker = str(row.get("ticker", ""))
        weight = row.get("weight_pct", np.nan)
        w_str = f"{weight * 100:,.2f}%" if pd.notnull(weight) else "—"

        change = change_map.get(ticker)
        if change is None or np.isnan(change):
            color = "#cccccc"
        elif change > 0:
            color = "#00ff87"  # green
        elif change < 0:
            color = "#ff4b4b"  # red
        else:
            color = "#cccccc"

        url = f"https://www.google.com/finance/quote/{ticker}"
        cell = (
            f"<a class='top10-ticker' href='{url}' target='_blank' "
            f"style='color:{color};'>{ticker}</a>"
        )
        row_html = (
            f"<tr><td>{cell}</td>"
            f"<td class='top10-weight'>{w_str}</td></tr>"
        )
        rows_html.append(row_html)

    table_html = """
    <table class="top10-table">
        <thead>
            <tr><th>Ticker</th><th class="top10-weight">Weight %</th></tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """.format(
        rows="\n".join(rows_html)
    )
    return table_html


# ============================================================
# SIDEBAR
# ============================================================

universe_df = load_universe_df()
available_waves = get_available_waves(universe_df)

with st.sidebar:
    st.markdown("## 🌊 WAVES Intelligence™")
    st.markdown(
        "<span class='wave-pill'>Desktop Engine + Cloud Snapshot</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "This console shows **live weights, holdings, and risk**. "
        "Full performance curves and alpha metrics are driven by the "
        "**WAVES desktop engine** writing CSVs into the `logs/` folder."
    )

    st.markdown("---")

    # selectbox returns internal ID, but shows pretty name
    selected_wave = st.selectbox(
        "Select Wave",
        available_waves,
        index=0,
        format_func=pretty_wave_name,
    )

    mode = st.radio(
        "Risk Mode (label in this view)",
        ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
        index=0,
    )

    st.markdown(
        f"<span class='mode-pill'>{mode}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        "Run `python3 waves_engine.py` in another terminal to keep "
        "performance logs updating while this console is open."
    )

# ============================================================
# HEADER
# ============================================================

col_title, col_badge = st.columns([0.8, 0.2])

with col_title:
    st.title("WAVES Institutional Console")
    st.caption(
        f"Live / demo console for **WAVES Intelligence™** — {pretty_wave_name(selected_wave)} "
        "shown below. Desktop engine handles trading & analytics; this view gives you "
        "**instant visibility** into each Wave."
    )

with col_badge:
    st.markdown(
        "<div style='text-align:right;margin-top:0.6rem;'>"
        "<span class='wave-pill'>Live Engine View</span>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ============================================================
# LOAD DATA FOR CURRENT WAVE
# ============================================================

perf_df, perf_path = load_perf_df_for_wave(selected_wave)
perf_metrics = compute_perf_metrics(perf_df)
positions_df = load_positions_for_wave(selected_wave)

if universe_df is not None:
    wave_universe = universe_df[universe_df["wave"] == selected_wave].copy()
else:
    wave_universe = None

# ============================================================
# SUMMARY METRICS ROW
# ============================================================

summary_cols = st.columns(4)
total_ret_str = fmt_pct(perf_metrics["total_return"])
today_ret_str = fmt_pct(perf_metrics["today_return"])
max_dd_str = fmt_pct(perf_metrics["max_drawdown"])
alpha_str = fmt_bp(perf_metrics["alpha"])

with summary_cols[0]:
    st.metric("Total Return (logs)", total_ret_str)

with summary_cols[1]:
    st.metric("Today", today_ret_str)

with summary_cols[2]:
    st.metric("Max Drawdown", max_dd_str)

with summary_cols[3]:
    st.metric("Alpha vs Benchmark", alpha_str)

st.caption(
    "If metrics show **0.00%** or **—**, there may not be enough performance history yet. "
    "As you run the **desktop engine**, these will fill in automatically."
)

st.markdown("---")

# ============================================================
# LAYOUT: LEFT = PERFORMANCE, RIGHT = HOLDINGS/RISK
# ============================================================

left, right = st.columns([0.55, 0.45])

# -------------------- LEFT: PERFORMANCE --------------------
with left:
    st.subheader("Performance Curve")

    if perf_df is None or perf_df.empty:
        if perf_path is None:
            st.info(
                "No performance logs were found yet for this Wave in `logs/performance/`.  "
                "Once you run the engine (`python3 waves_engine.py`), "
                "this panel will plot the **Wave equity curve / cumulative return**."
            )
        else:
            st.info(
                f"A performance file was found (`{perf_path.name}`), "
                "but it is empty or unreadable."
            )
    else:
        df = perf_df.copy()
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
            df = df.set_index("timestamp")

        plot_df = choose_perf_series(df)

        if plot_df is not None and not plot_df.empty:
            st.line_chart(plot_df)
            # optional: warn if series is flat
            if plot_df.nunique().max() <= 1:
                st.caption(
                    "Performance series appears flat. If this should not be the case, "
                    "check which columns are being written into the performance CSV."
                )
            else:
                st.caption(
                    f"Chart based on latest performance log for **{pretty_wave_name(selected_wave)}** "
                    "found in `logs/performance/`."
                )
        else:
            st.info(
                "Performance data is present, but no suitable numeric series "
                "was found to plot (avoiding beta/exposure columns)."
            )

# -------------------- RIGHT: HOLDINGS & RISK --------------------
with right:
    st.subheader("Holdings, Weights & Risk")

    if wave_universe is None or wave_universe.empty:
        st.warning(
            "No universe / weights file could be loaded for this Wave. "
            "Make sure `wave_weights.csv` (or similar) exists and includes this Wave."
        )
    else:
        df = wave_universe.copy()

        if "weight" in df.columns and df["weight"].sum() > 0:
            df["weight_pct"] = df["weight"] / df["weight"].sum()
        else:
            df["weight_pct"] = np.nan

        df = df.sort_values("weight_pct", ascending=False)

        st.caption("Top 10 Positions — Google Finance Links (Bloomberg-style)")
        top10_html = build_top10_html(df, positions_df)
        st.markdown(top10_html, unsafe_allow_html=True)

        with st.expander("Full Wave universe table"):
            display_df = df.copy()
            display_df.rename(
                columns={
                    "ticker": "Ticker",
                    "name": "Name",
                    "sector": "Sector",
                    "weight_pct": "Weight",
                },
                inplace=True,
            )
            if "Weight" in display_df.columns:
                display_df["Weight"] = display_df["Weight"].apply(
                    lambda x: f"{x * 100:,.2f}%" if pd.notnull(x) else "—"
                )
            st.dataframe(display_df, use_container_width=True)

# ============================================================
# ENGINE / LOGS EXPLAINER
# ============================================================

st.markdown("---")
st.subheader("Engine & Logs Snapshot")

status_cols = st.columns(2)

with status_cols[0]:
    st.markdown("**Local Engine Status**")
    st.caption(
        "The WAVES engine writes CSV logs into `logs/positions/` and `logs/performance/` "
        "whenever you run `python3 waves_engine.py`."
    )

    if PERF_DIR.exists():
        perf_files = sorted(PERF_DIR.glob(f"{selected_wave}_performance*.csv"))
    else:
        perf_files = []

    if perf_files:
        latest = max(perf_files, key=lambda p: p.stat().st_mtime)
        ts = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        st.success(
            f"Latest performance log for **{pretty_wave_name(selected_wave)}**: `{latest.name}`  \n"
            f"_Last modified: **{ts}**_"
        )
    else:
        st.info(
            f"No performance logs found for **{pretty_wave_name(selected_wave)}** yet. "
            "Run the engine script to start generating them."
        )

with status_cols[1]:
    st.markdown("**How to Use this Console**")
    st.markdown(
        """
        1. In **Terminal window A**:  
           `cd ~/Downloads/Wave-Simple-main`  
           `python3 -m streamlit run app.py`  

        2. In **Terminal window B**:  
           `cd ~/Downloads/Wave-Simple-main`  
           `python3 waves_engine.py`  

        3. Use this dashboard to:  
           - Switch between Waves (names fully spelled out)  
           - See **top holdings & weights** with **Google links** and red/green coloring  
           - Watch **performance curves** and alpha metrics as logs accumulate  
        """
    )

st.caption(
    "WAVES Intelligence™ — AI-managed Waves with a live desktop engine "
    "and an institutional-grade console."
)
