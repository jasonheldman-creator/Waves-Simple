import streamlit as st
import pandas as pd

# ------------------------------------------------
# Try to import the engine safely
# ------------------------------------------------
try:
    import waves_engine
    ENGINE_IMPORT_ERROR = None
except Exception as e:
    waves_engine = None
    ENGINE_IMPORT_ERROR = e


# ------------------------------------------------
# Helpers to get engine & metrics
# ------------------------------------------------
@st.cache_resource
def get_engine():
    """
    Build and cache the Waves engine.
    Any errors are raised so the UI can display them nicely.
    """
    if ENGINE_IMPORT_ERROR is not None:
        raise RuntimeError("Could not import waves_engine") from ENGINE_IMPORT_ERROR

    if not hasattr(waves_engine, "build_engine"):
        raise RuntimeError(
            "waves_engine.build_engine() is missing. "
            "Make sure your waves_engine.py defines build_engine()."
        )

    engine = waves_engine.build_engine()
    return engine


@st.cache_data(show_spinner=False)
def get_metrics_df() -> pd.DataFrame:
    """
    Call engine.compute_all_metrics() and normalize the result into a DataFrame.
    This works if the engine returns:
      - a pandas DataFrame, OR
      - a dict keyed by wave name with metric objects or dicts.
    """
    engine = get_engine()
    metrics = engine.compute_all_metrics()

    # Case 1: already a DataFrame
    if isinstance(metrics, pd.DataFrame):
        df = metrics.copy()
        if "Wave" not in df.columns:
            for alt in ["wave", "wave_name", "name"]:
                if alt in df.columns:
                    df = df.rename(columns={alt: "Wave"})
                    break
        return df

    # Case 2: dict of metrics
    if not metrics:
        return pd.DataFrame()

    rows = []
    for wave_name, m in metrics.items():
        row = {"Wave": wave_name}

        if hasattr(m, "__dict__"):
            for k, v in m.__dict__.items():
                if k.startswith("_"):
                    continue
                if k in ("wave", "wave_name", "name"):
                    continue
                row[k] = v
        elif isinstance(m, dict):
            for k, v in m.items():
                if k in ("wave", "wave_name", "name"):
                    continue
                row[k] = v

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    cols = df.columns.tolist()
    if "Wave" in cols:
        cols = ["Wave"] + [c for c in cols if c != "Wave"]
        df = df[cols]

    return df


# ------------------------------------------------
# UI
# ------------------------------------------------
def main():
    st.set_page_config(
        page_title="WAVES Intelligence™ Institutional Console",
        layout="wide",
    )

    st.title("WAVES Intelligence™ Institutional Console")
    st.caption("Live Wave Engine • Alpha Capture • Benchmarks • Diagnostics")

    tab_dash, tab_wave, tab_diag = st.tabs(
        ["Dashboard", "Wave Explorer", "Diagnostics"]
    )

    # -------- Dashboard --------
    with tab_dash:
        st.subheader("All Waves Snapshot")

        try:
            metrics_df = get_metrics_df()
        except Exception as e:
            st.error(
                "Engine failed while computing metrics. "
                "See Diagnostics tab for details."
            )
            st.stop()

        if metrics_df.empty:
            st.warning(
                "No Wave metrics available yet. "
                "Check that wave_weights.csv and your engine are configured."
            )
        else:
            st.dataframe(metrics_df, use_container_width=True)

    # -------- Wave Explorer --------
    with tab_wave:
        st.subheader("Wave Explorer")

        try:
            metrics_df = get_metrics_df()
        except Exception:
            st.info("Engine is currently failing; see Diagnostics tab.")
        else:
            if metrics_df.empty:
                st.info("No Wave metrics to explore yet.")
            elif "Wave" not in metrics_df.columns:
                st.warning("Metrics loaded, but there is no 'Wave' column.")
                st.dataframe(metrics_df.head(), use_container_width=True)
            else:
                waves = metrics_df["Wave"].tolist()
                selected = st.selectbox("Select a Wave", waves)
                row = metrics_df[metrics_df["Wave"] == selected]
                st.write(f"### {selected}")
                st.dataframe(row.set_index("Wave").T, use_container_width=True)

    # -------- Diagnostics --------
    with tab_diag:
        st.subheader("Diagnostics")

        if ENGINE_IMPORT_ERROR is not None:
            st.error("Import error while loading waves_engine.py")
            st.exception(ENGINE_IMPORT_ERROR)
        else:
            st.success("waves_engine imported successfully.")

        st.markdown(
            """
            **If metrics are still failing:**

            1. Confirm `waves_engine.build_engine()` exists and returns an engine
               with `compute_all_metrics()`.
            2. Confirm `wave_weights.csv` is present and all tickers are valid.
            3. If a ticker keeps failing (e.g., BRK.B), replace it with another
               highly liquid name.
            """
        )

        # Try to show a small preview if possible
        try:
            df_preview = get_metrics_df()
        except Exception as e:
            st.error("Engine raised an exception while computing metrics:")
            st.exception(e)
        else:
            if not df_preview.empty:
                st.markdown("#### Metrics preview")
                st.dataframe(df_preview.head(), use_container_width=True)


if __name__ == "__main__":
    main()