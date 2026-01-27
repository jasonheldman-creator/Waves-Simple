def render_alpha_quality_and_confidence(
    snapshot_df,
    source_df,
    selected_wave,
    return_cols,
    benchmark_cols,
):
    st.subheader("Alpha Quality & Confidence")

    if selected_wave is None:
        st.info("No wave selected.")
        return

    # Load alpha history
    alpha_history_path = Path("data") / "alpha_history.parquet"
    if not alpha_history_path.exists():
        st.metric("Alpha Quality Score", "—")
        st.metric("Confidence Level", "—")
        return

    df = pd.read_parquet(alpha_history_path)

    # Filter to selected wave + DAILY only
    df = df[
        (df["wave_id"] == selected_wave)
        & (df["window"] == "DAILY")
    ].sort_values("timestamp")

    MIN_OBS = 20

    if len(df) < MIN_OBS:
        st.metric("Alpha Quality Score", "—")
        st.metric("Confidence Level", "—")
        st.caption(f"Requires ≥ {MIN_OBS} daily observations")
        return

    # ---------------------------
    # Alpha Quality (mean / vol)
    # ---------------------------
    alpha_mean = df["alpha"].mean()
    alpha_vol = df["alpha"].std()

    if pd.isna(alpha_vol) or alpha_vol == 0:
        quality_score = 0.0
    else:
        quality_score = alpha_mean / alpha_vol

    # Normalize for display (bounded, IC-safe)
    quality_score = max(min(quality_score * 10, 100), -100)

    # ---------------------------
    # Confidence Level
    # ---------------------------
    confidence_level = min(len(df) / 60, 1.0) * 100

    # ---------------------------
    # Render
    # ---------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Alpha Quality Score",
            f"{quality_score:.1f}"
        )

    with col2:
        st.metric(
            "Confidence Level",
            f"{confidence_level:.0f}%"
        )