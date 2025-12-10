import numpy as np
import pandas as pd
import streamlit as st

from waves_engine import WavesEngine

# =========================================================
# Caching & Engine bootstrap
# =========================================================

@st.cache_resource
def load_engine():
    """
    Lazily create a single WavesEngine instance for this app.
    Adjust constructor args here if your engine needs any.
    """
    return WavesEngine()

engine = load_engine()

# =========================================================
# Helper formatting
# =========================================================

def fmt_pct(x):
    """Format a decimal as percentage, or em-dash if missing."""
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "—"
    except Exception:
        pass
    try:
        return f"{x * 100:0.2f}%"
    except Exception:
        return "—"


def fmt_beta(x):
    """Format beta with 2 decimals, or em-dash if missing."""
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "—"
        return f"{x:0.2f}"
    except Exception:
        return "—"


def fmt_exposure(m):
    """
    Format exposure from the metrics dict as a percentage string,
    or em-dash if missing.
    """
    val = m.get("exposure")
    if val is None:
        return "—"
    try:
        if isinstance(val, float) and np.isnan(val):
            return "—"
        return f"{val * 100:0.2f}%"
    except Exception:
        return "—"


def fmt_yield(y):
    """Format SmartSafe yield estimate (already a decimal)."""
    if y is None:
        return "—"
    try:
        if isinstance(y, float) and np.isnan(y):
            return "—"
        return f"{y * 100:0.2f}%"
    except Exception:
        return "—"


# =========================================================
# SmartSafe 3.0 configuration (UI-side only)
# =========================================================

SMARTSAFE_WAVE_NAME = "SmartSafe Wave"
SMARTSAFE_BASE_YIELD = 0.0425  # 4.25% estimated yield

# =========================================================
# Page layout & top-level config
# =========================================================

try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass

st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)

st.markdown(
    """
    <h1 style="font-size:2.4rem; margin-bottom:0;">
        WAVES Intelligence™ Institutional Console
    </h1>
    <p style="color:#AAAAAA; margin-top:0.25rem;">
        Live Wave Engine • Dynamic Weights • SmartSafe™ 3.0 Sweep • Benchmark-Relative Performance
    </p>
    <hr style="opacity:0.2;" />
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Sidebar controls
# =========================================================

st.sidebar.header("Wave Engine Controls")

mode = st.sidebar.selectbox(
    "Mode",
    ["Standard", "Alpha-Minus-Beta", "Private Logic™"],
    index=0,
)

sort_metric = st.sidebar.selectbox(
    "Sort Waves by",
    ["Alpha 30D", "Alpha 60D", "Alpha 1Y"],
    index=0,
)

# SmartSafe 3.0 controls
st.sidebar.markdown("---")
st.sidebar.subheader("SmartSafe™ 3.0 — Sweep Engine")

smartsafe_enabled = st.sidebar.checkbox(
    "Enable SmartSafe Sweep",
    value=True,
    help="When enabled, idle / defensive allocation is swept into SmartSafe Wave at the target level."
)

smartsafe_target_pct = st.sidebar.slider(
    "Target SmartSafe Allocation",
    min_value=0,
    max_value=40,
    value=10,
    step=1,
    help="Conceptual target share of portfolio that the sweep engine aims to keep in SmartSafe."
)

st.sidebar.markdown(
    f"**SmartSafe Estimated Yield:** {fmt_yield(SMARTSAFE_BASE_YIELD)}"
)

st.sidebar.caption(
    "Note: Sweep logic is implemented in the engine; these controls govern how the "
    "SmartSafe 3.0 overlay is interpreted and displayed here."
)

# =========================================================
# Data fetch from engine
# =========================================================

@st.cache_data(show_spinner=False)
def get_wave_metrics(selected_mode: str):
    """
    Adapter around WavesEngine to get a convenient metrics dict.

    Expected structure:
        {
          'S&P Wave': {
              'alpha_30d_blended': 0.12,
              'alpha_60d_blended': 0.08,
              'alpha_1y_blended': 0.15,
              'return_30d_wave_blended': 0.10,
              'return_60d_wave_blended': 0.12,
              'return_1y_wave_blended': 0.18,
              'exposure': 0.95,
              'beta': 0.90,
              ...
          },
          ...
        }
    """
    # IMPORTANT: if your engine uses a different method name,
    # update this call only.
    metrics = engine.get_wave_metrics(mode=selected_mode)
    return metrics

metrics_by_wave = get_wave_metrics(mode)

# Inject SmartSafe yield info if that wave exists
if SMARTSAFE_WAVE_NAME in metrics_by_wave:
    metrics_by_wave[SMARTSAFE_WAVE_NAME]["yield_estimate"] = SMARTSAFE_BASE_YIELD
    metrics_by_wave[SMARTSAFE_WAVE_NAME]["sweep_enabled"] = smartsafe_enabled
    metrics_by_wave[SMARTSAFE_WAVE_NAME]["sweep_target"] = smartsafe_target_pct / 100.0

# =========================================================
# Build DataFrames for views
# =========================================================

def build_all_waves_df(metrics_dict):
    rows = []
    for wave_name, m in metrics_dict.items():
        rows.append(
            {
                "Wave": wave_name,
                "Alpha 30D": m.get("alpha_30d_blended"),
                "Alpha 60D": m.get("alpha_60d_blended"),
                "Alpha 1Y": m.get("alpha_1y_blended"),
                "Return 30D": m.get("return_30d_wave_blended"),
                "Return 60D": m.get("return_60d_wave_blended"),
                "Return 1Y": m.get("return_1y_wave_blended"),
                "Exposure": m.get("exposure"),
                "Beta": m.get("beta"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort by selected alpha metric
    sort_col_map = {
        "Alpha 30D": "Alpha 30D",
        "Alpha 60D": "Alpha 60D",
        "Alpha 1Y": "Alpha 1Y",
    }
    sort_col = sort_col_map.get(sort_metric, "Alpha 30D")
    df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    return df


all_waves_df = build_all_waves_df(metrics_by_wave)

# High-level alpha snapshot
overall_alpha_1y = None
if not all_waves_df.empty:
    try:
        overall_alpha_1y = float(
            all_waves_df["Alpha 1Y"].replace({None: np.nan}).astype(float).mean()
        )
    except Exception:
        overall_alpha_1y = None

# =========================================================
# Tabs
# =========================================================

tab_dashboard, tab_alpha_matrix, tab_smartsafe, tab_explorer = st.tabs(
    ["Dashboard", "Alpha Matrix", "SmartSafe™ 3.0", "Wave Explorer"]
)

# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------

with tab_dashboard:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("#### 1-Year Alpha (All Waves)")

        st.markdown(
            f"""
            <div style="font-size:3.0rem; font-weight:600; margin:0 0 0.5rem 0;">
                {fmt_pct(overall_alpha_1y)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Average 1-year alpha across all active Waves in the selected mode "
            "(benchmark-relative, risk-adjusted)."
        )

        if smartsafe_enabled:
            st.markdown(
                f"**SmartSafe 3.0:** Sweep enabled at **{smartsafe_target_pct}%** "
                f"target allocation. Estimated yield {fmt_yield(SMARTSAFE_BASE_YIELD)}."
            )
        else:
            st.markdown("**SmartSafe 3.0:** Sweep currently **disabled** in this view.")

    with col_right:
        st.markdown("#### All Waves Snapshot")

        if all_waves_df.empty:
            st.info("No wave metrics available. Check engine logs.")
        else:
            display_df = all_waves_df.copy()

            # Apply formatting
            for col in ["Alpha 30D", "Alpha 60D", "Alpha 1Y",
                        "Return 30D", "Return 60D", "Return 1Y"]:
                display_df[col] = display_df[col].apply(fmt_pct)

            display_df["Exposure"] = [
                fmt_exposure(metrics_by_wave[row["Wave"]])
                for _, row in display_df.iterrows()
            ]
            display_df["Beta"] = display_df["Beta"].apply(fmt_beta)

            st.dataframe(
                display_df.set_index("Wave"),
                use_container_width=True,
                height=420,
            )

# ---------------------------------------------------------
# Alpha Matrix
# ---------------------------------------------------------

with tab_alpha_matrix:
    st.markdown("### Alpha Matrix (All Waves)")

    if all_waves_df.empty:
        st.info("No data available for Alpha Matrix.")
    else:
        alpha_df = all_waves_df[["Wave", "Alpha 30D", "Alpha 60D", "Alpha 1Y"]].copy()
        alpha_df["Alpha 30D"] = alpha_df["Alpha 30D"].apply(fmt_pct)
        alpha_df["Alpha 60D"] = alpha_df["Alpha 60D"].apply(fmt_pct)
        alpha_df["Alpha 1Y"] = alpha_df["Alpha 1Y"].apply(fmt_pct)

        alpha_df = alpha_df.set_index("Wave")
        st.dataframe(alpha_df, use_container_width=True, height=420)

# ---------------------------------------------------------
# SmartSafe 3.0 tab
# ---------------------------------------------------------

with tab_smartsafe:
    st.markdown("### SmartSafe™ 3.0 — Sweep & Yield")

    if SMARTSAFE_WAVE_NAME not in metrics_by_wave:
        st.warning(
            f"`{SMARTSAFE_WAVE_NAME}` not found in metrics. "
            "Confirm the wave name in wave_weights.csv and engine configuration."
        )
    else:
        m = metrics_by_wave[SMARTSAFE_WAVE_NAME]

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("SmartSafe Estimated Yield", fmt_yield(m.get("yield_estimate", SMARTSAFE_BASE_YIELD)))
        with colB:
            st.metric(
                "SmartSafe Alpha 1Y",
                fmt_pct(m.get("alpha_1y_blended")),
            )
        with colC:
            st.metric(
                "SmartSafe Exposure",
                fmt_exposure(m),
            )

        st.markdown("---")

        st.markdown(
            """
            **How SmartSafe 3.0 Works (Conceptual View)**

            * SmartSafe is a conservative Wave that behaves like a modern, AI-aware cash
              and short-term fixed-income sleeve.
            * The **sweep engine** monitors overall portfolio risk and idle allocation and
              shifts a configurable share into SmartSafe.
            * In this console, the controls you set in the sidebar (target % and on/off)
              are passed to the engine and also visualized here so you can see the impact
              on alpha and exposure.
            """
        )

        st.markdown("#### Yield & Sweep Settings")

        settings_df = pd.DataFrame(
            [
                {
                    "Wave": SMARTSAFE_WAVE_NAME,
                    "Sweep Enabled": "Yes" if smartsafe_enabled else "No",
                    "Target Allocation": f"{smartsafe_target_pct:.0f}%",
                    "Estimated Yield": fmt_yield(m.get("yield_estimate", SMARTSAFE_BASE_YIELD)),
                }
            ]
        ).set_index("Wave")

        st.table(settings_df)

# ---------------------------------------------------------
# Wave Explorer
# ---------------------------------------------------------

with tab_explorer:
    st.markdown("### Wave Explorer")

    if not metrics_by_wave:
        st.info("No wave data available.")
    else:
        wave_names = sorted(metrics_by_wave.keys())
        selected_wave = st.selectbox("Select a Wave", wave_names)

        m = metrics_by_wave[selected_wave]

        col1, col2, col3 = st.columns(3)
        col1.metric("Alpha 30D", fmt_pct(m.get("alpha_30d_blended")))
        col2.metric("Alpha 60D", fmt_pct(m.get("alpha_60d_blended")))
        col3.metric("Alpha 1Y", fmt_pct(m.get("alpha_1y_blended")))

        col4, col5, col6 = st.columns(3)
        col4.metric("Return 30D", fmt_pct(m.get("return_30d_wave_blended")))
        col5.metric("Return 60D", fmt_pct(m.get("return_60d_wave_blended")))
        col6.metric("Return 1Y", fmt_pct(m.get("return_1y_wave_blended")))

        col7, col8 = st.columns(2)
        col7.metric("Exposure", fmt_exposure(m))
        col8.metric("Beta", fmt_beta(m.get("beta")))

        if selected_wave == SMARTSAFE_WAVE_NAME:
            st.markdown(
                f"**SmartSafe Estimated Yield:** {fmt_yield(m.get('yield_estimate', SMARTSAFE_BASE_YIELD))}"
            )

        st.markdown("---")
        st.caption(
            "All metrics are generated by the live WavesEngine using wave_weights.csv "
            "and benchmark mappings. Values are benchmark-relative where applicable."
        )