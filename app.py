import streamlit as st
import pandas as pd

from waves_engine import build_engine


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Institutional Console",
    layout="wide",
)


# ---------------------------
# Engine loader (cached)
# ---------------------------
@st.cache_resource
def load_engine():
    """Build and cache the Waves engine."""
    engine = build_engine()
    return engine


# ---------------------------
# Metrics computation (cached)
# ---------------------------
@st.cache_data(show_spinner=False)
def compute_metrics_df() -> pd.DataFrame:
    """
    Ask the engine for all metrics and normalize them into a DataFrame.
    This is intentionally generic so it works whether the engine
    returns a dict of dataclass objects, a dict of dicts, or a DataFrame.
    """
    engine = load_engine()
    metrics = engine.compute_all_metrics()

    # If the engine already returns a DataFrame, just normalize column names.
    if isinstance(metrics, pd.DataFrame):
        df = metrics.copy()
        if "Wave" not in df.columns:
            # Try common alternatives
            for candidate in ["wave", "name", "wave_name"]:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "Wave"})
                    break
        return df

    # If nothing or empty
    if not metrics:
        return pd.DataFrame()

    # Assume a dict keyed by wave name
    rows = []
    for wave_name, m in metrics.items():
        row = {"Wave": wave_name}

        # dataclass / object with __dict__
        if hasattr(m, "__dict__"):
            for k, v in m.__dict__.items():
                if k.startswith("_"):
                    continue
                if k in ("wave", "wave_name", "name"):
                    continue
                row[k] = v

        # plain dict
        elif isinstance(m, dict):
            for k, v in m.items():
                if k in ("wave", "wave_name", "name"):
                    continue
                row[k] = v

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Put Wave column first if present
    cols = df.columns.tolist()
    if "Wave" in cols:
        cols = ["Wave"] + [c for c in cols if c != "Wave"]
        df = df[cols]

    return df


# ---------------------------
# UI
# ---------------------------
def main():
    st.title("WAVES Intelligence™ Institutional Console")
    st.caption("Live Wave Engine • Alpha Capture • Benchmarks • Diagnostics")

    metrics_df = compute_metrics_df()

    tab_dashboard, tab_explorer, tab_diag = st.tabs(
        ["Dashboard", "Wave Explorer", "Diagnostics"]
    )

    # ---- Dashboard tab ----
    with tab_dashboard:
        st.subheader("All Waves Snapshot")

        if metrics_df.empty:
            st.error(
                "No Wave metrics available. "
                "If this persists, check the Diagnostics tab and engine logs."
            )
        else:
            st.dataframe(
                metrics_df,
                use_container_width=True,
            )

    # ---- Wave Explorer tab ----
    with tab_explorer:
        st.subheader("Wave Explorer")

        if metrics_df.empty:
            st.info("No Wave metrics to explore yet.")
        else:
            wave_names = metrics_df["Wave"].tolist() if "Wave" in metrics_df.columns else []
            if not wave_names:
                st.warning("Metrics are loaded, but no 'Wave' column was found.")
            else:
                selected_wave = st.selectbox("Select a Wave", wave_names)
                wave_row = metrics_df[metrics_df["Wave"] == selected_wave]

                st.write(f"### {selected_wave}")
                # Show the metrics for this wave as a vertical table
                st.dataframe(
                    wave_row.set_index("Wave").T,
                    use_container_width=True,
                )

    # ---- Diagnostics tab ----
    with tab_diag:
        st.subheader("Diagnostics")

        st.markdown(
            """
            **If the dashboard is empty or shows an error:**

            1. Confirm that `wave_weights.csv` exists and is in the expected folder.
            2. Check that all tickers in `wave_weights.csv` are valid and download correctly with yfinance.
               - If a ticker repeatedly fails (rate limit or delisting), replace it with a similar liquid name.
            3. If you recently edited `waves_engine.py`, make sure:
               - `build_engine()` still exists and returns the engine instance.
               - The engine still has a `compute_all_metrics()` method.

            This app is now using a very generic metrics loader, so as long as the
            engine returns either:
            - a **DataFrame**, or
            - a **dict** of metric objects / dicts keyed by wave name,

            the dashboard should populate automatically.
            """
        )

        if not metrics_df.empty:
            st.markdown("#### Metrics preview")
            st.dataframe(metrics_df.head(), use_container_width=True)


if __name__ == "__main__":
    main()