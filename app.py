# --------------------------------------------------------------------
# ROW 1 – S&P 500 + VIX (SAFE MODE)
# --------------------------------------------------------------------
sp_col, vix_col = st.columns([1.6, 1.0])

# ----- S&P 500 -----
with sp_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
        <div style="display:flex; justify-content:space-between; margin-bottom:0.15rem;">
            <div class="section-title">S&P 500 Index</div>
            <div class="section-caption">^GSPC · last 6 months</div>
        </div>
    """, unsafe_allow_html=True)

    sp_df = load_index_series("^GSPC", period="6mo")

    if sp_df is None or len(sp_df) == 0:
        st.warning("Live S&P data unavailable (no internet / blocked / yfinance missing).", icon="⚠️")
    else:
        fig_sp = px.area(sp_df, x="Date", y="Close")
        fig_sp.update_traces(mode="lines", line_shape="spline")
        fig_sp.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            margin=dict(l=10, r=10, t=5, b=10),
            height=200,
        )
        st.plotly_chart(fig_sp, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----- VIX -----
with vix_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("""
        <div style="display:flex; justify-content:space-between; margin-bottom:0.15rem;">
            <div class="section-title">VIX Volatility Index</div>
            <div class="section-caption">^VIX · last 6 months</div>
        </div>
    """, unsafe_allow_html=True)

    vix_df = load_index_series("^VIX", period="6mo")

    if vix_df is None or len(vix_df) == 0:
        st.warning("Live VIX data unavailable (no internet / blocked / yfinance missing).", icon="⚠️")
    else:
        fig_vix = px.line(vix_df, x="Date", y="Close")
        fig_vix.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            margin=dict(l=10, r=10, t=5, b=10),
            height=200,
        )
        st.plotly_chart(fig_vix, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)