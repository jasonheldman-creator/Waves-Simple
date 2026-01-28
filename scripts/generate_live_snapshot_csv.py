def generate_live_snapshot(
    output_path: str = "live_snapshot.csv",
    session_state: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate a comprehensive snapshot of all waves with returns, alpha,
    and attribution diagnostics.

    This function is an AGGREGATION LAYER ONLY.
    It does NOT compute attribution — it only merges it if available.

    Attribution source (optional):
        data/alpha_attribution_snapshot.csv
    """

    from waves_engine import compute_history_nav

    # ------------------------------------------------------------------
    # SAFE MODE / DEMO MODE GUARDS
    # ------------------------------------------------------------------
    if session_state is not None and session_state.get("safe_mode_no_fetch", True):
        print("⚠️ Safe Mode active - generate_live_snapshot suppressed")
        return pd.DataFrame()

    if session_state is not None and session_state.get("safe_demo_mode", False):
        print("⚠️ SAFE DEMO MODE active - generate_live_snapshot suppressed")
        return pd.DataFrame()

    print("=" * 70)
    print("Generating Live Snapshot (with Attribution Merge)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # LOAD ATTRIBUTION SNAPSHOT (OPTIONAL)
    # ------------------------------------------------------------------
    attribution_path = "data/alpha_attribution_snapshot.csv"
    attribution_df = None

    if os.path.exists(attribution_path):
        try:
            attribution_df = pd.read_csv(attribution_path)

            if "wave_id" not in attribution_df.columns:
                print("⚠️ attribution snapshot missing wave_id — ignoring attribution")
                attribution_df = None
            else:
                attribution_df = attribution_df.set_index("wave_id")

        except Exception as e:
            print(f"⚠️ Failed to load attribution snapshot: {e}")
            attribution_df = None
    else:
        print("ℹ️ No attribution snapshot found — continuing without attribution")

    # ------------------------------------------------------------------
    # SNAPSHOT BUILD
    # ------------------------------------------------------------------
    snapshot_rows = []
    timeframes = [1, 30, 60, 365]

    for wave_id in get_all_wave_ids():
        wave_name = get_display_name_from_wave_id(wave_id)

        try:
            readiness = compute_data_ready_status(wave_id)

            row = {
                "wave_id": wave_id,
                "wave_name": wave_name,
                "readiness_status": readiness.get("readiness_status", "unavailable"),
                "coverage_pct": readiness.get("coverage_pct", 0.0),
                "data_regime": readiness.get("readiness_status", "unavailable"),
            }

            # ----------------------------------------------------------
            # RETURNS + TOTAL ALPHA
            # ----------------------------------------------------------
            for days in timeframes:
                try:
                    nav_df = compute_history_nav(
                        wave_name,
                        mode="Standard",
                        days=days,
                        include_diagnostics=False,
                        session_state=session_state
                    )

                    if not nav_df.empty and len(nav_df) >= 2:
                        wave_ret = (
                            nav_df["wave_nav"].iloc[-1] / nav_df["wave_nav"].iloc[0] - 1
                            if "wave_nav" in nav_df.columns else np.nan
                        )
                        bm_ret = (
                            nav_df["bm_nav"].iloc[-1] / nav_df["bm_nav"].iloc[0] - 1
                            if "bm_nav" in nav_df.columns else np.nan
                        )

                        row[f"wave_return_{days}d"] = wave_ret
                        row[f"bm_return_{days}d"] = bm_ret
                        row[f"alpha_{days}d"] = (
                            wave_ret - bm_ret
                            if not np.isnan(wave_ret) and not np.isnan(bm_ret)
                            else np.nan
                        )
                    else:
                        row[f"wave_return_{days}d"] = np.nan
                        row[f"bm_return_{days}d"] = np.nan
                        row[f"alpha_{days}d"] = np.nan

                except Exception:
                    row[f"wave_return_{days}d"] = np.nan
                    row[f"bm_return_{days}d"] = np.nan
                    row[f"alpha_{days}d"] = np.nan

            # ----------------------------------------------------------
            # ATTRIBUTION MERGE (NO COMPUTATION HERE)
            # ----------------------------------------------------------
            for days in timeframes:
                for component in [
                    "momentum",
                    "volatility",
                    "regime",
                    "selection",
                    "exposure",
                    "residual",
                ]:
                    col = f"alpha_{component}_{days}d"
                    row[col] = np.nan

            if attribution_df is not None and wave_id in attribution_df.index:
                attr_row = attribution_df.loc[wave_id]

                for days in timeframes:
                    for component in [
                        "momentum",
                        "volatility",
                        "regime",
                        "selection",
                        "exposure",
                        "residual",
                    ]:
                        col = f"alpha_{component}_{days}d"
                        if col in attr_row and pd.notna(attr_row[col]):
                            row[col] = float(attr_row[col])

            # ----------------------------------------------------------
            # EXPOSURE / CASH (PLACEHOLDER — NON-BLOCKING)
            # ----------------------------------------------------------
            row["exposure"] = np.nan
            row["cash_pct"] = np.nan

            snapshot_rows.append(row)
            print(f"  ✓ {wave_name}")

        except Exception as e:
            print(f"  ✗ Error processing {wave_name}: {e}")

            snapshot_rows.append({
                "wave_id": wave_id,
                "wave_name": wave_name,
                "readiness_status": "error",
                "coverage_pct": 0.0,
                "data_regime": "error",
                **{f"wave_return_{d}d": np.nan for d in timeframes},
                **{f"bm_return_{d}d": np.nan for d in timeframes},
                **{f"alpha_{d}d": np.nan for d in timeframes},
                **{
                    f"alpha_{c}_{d}d": np.nan
                    for d in timeframes
                    for c in [
                        "momentum",
                        "volatility",
                        "regime",
                        "selection",
                        "exposure",
                        "residual",
                    ]
                },
                "exposure": np.nan,
                "cash_pct": np.nan,
            })

    # ------------------------------------------------------------------
    # FINALIZE + VALIDATE
    # ------------------------------------------------------------------
    snapshot_df = pd.DataFrame(snapshot_rows)

    expected = len(get_all_wave_ids())
    actual = len(snapshot_df)

    if actual != expected:
        raise AssertionError(
            f"Snapshot validation failed: expected {expected} waves, got {actual}"
        )

    snapshot_df.to_csv(output_path, index=False)

    print(f"\n✓ Live snapshot saved to: {output_path}")
    print(f"  Total waves: {actual}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return snapshot_df