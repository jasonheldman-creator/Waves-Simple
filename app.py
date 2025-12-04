import streamlit as st
import pandas as pd

from waves_equity_universe_v2 import run_equity_waves, WAVES_CONFIG

st.set_page_config(
    page_title="WAVES Intelligence – Equity Waves Console",
    layout="wide",
)

st.title("🌊 WAVES Intelligence™ – Equity Waves Console")
st.caption("Live view of all 10 Equity Waves (prototype)")

st.sidebar.header("Settings")

# Show which waves are configured
configured = [w for w in WAVES_CONFIG if w.holdings_csv_url]
missing = [w for w in WAVES_CONFIG if not w.holdings_csv_url]

st.sidebar.subheader("Configured Waves")
if configured:
    for w in configured:
        st.sidebar.write(f"✅ **{w.code}** – {w.name}  \nBench: `{w.benchmark}`")
else:
    st.sidebar.write("⚠️ No waves have holdings URLs configured yet.")

if missing:
    st.sidebar.subheader("Missing holdings URLs")
    for w in missing:
        st.sidebar.write(f"⚠️ {w.code} – {w.name}")

if st.button("🔁 Run Equity Waves"):
    with st.spinner("Running WAVES engine…"):
        df = run_equity_waves()

    if df.empty:
        st.error("No waves processed. Add holdings_csv_url values in WAVES_CONFIG.")
    else:
        # Pretty formatting
        fmt_df = df.copy()
        for col in ["wave_return", "benchmark_return", "alpha"]:
            fmt_df[col] = (fmt_df[col] * 100).round(2)

        fmt_df.rename(
            columns={
                "code": "Wave",
                "name": "Name",
                "benchmark": "Benchmark",
                "nav": "NAV ($)",
                "wave_return": "Wave Return (%)",
                "benchmark_return": "Benchmark Return (%)",
                "alpha": "Alpha (%)",
            },
            inplace=True,
        )

        st.success("Run complete.")
        st.dataframe(fmt_df, use_container_width=True)

        # Optional: highlight alpha
        st.subheader("Alpha by Wave")
        alpha_chart = df.set_index("code")["alpha"]
        st.bar_chart(alpha_chart)
else:
    st.info("Click **Run Equity Waves** to fetch data and compute returns.")