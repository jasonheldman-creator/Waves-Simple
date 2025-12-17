# app.py — WAVES Intelligence™ Institutional Console (Vector OS Edition)
# MODULAR BUILD — vNEXT Stabilized
#
# Why this avoids “script killer” errors:
#   • app.py stays small (UI wiring only)
#   • all heavy logic lives in modules
#   • each panel is wrapped; app still boots if a panel fails
#
# Core features included:
#   1) Executive IC One-Pager
#   2) Cohesion Lock (Truth Table)
#   3) Benchmark Fidelity Inspector (+ composition diff + difficulty proxy)
#   4) Deterministic AI Explanation Layer (rules-based)
#   5) Wave-to-Wave Comparator
#   6) Diagnostics tab (engine/import/data status)
#   7) BIG wave header fix (always obvious what wave you’re viewing)
#   8) NEW: Beta Reliability / Benchmark Beta Fidelity (adds real institutional value)

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import ui_components as ui
import canonical as canon
import metrics as mx
import benchmark as bm
import scorecard as sc
import explain as ex
import comparator as comp
import diagnostics as diag

# -----------------------------
# Streamlit config + CSS
# -----------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_global_css()

# -----------------------------
# Feature flags (demo-safe)
# -----------------------------
ENABLE_SCORECARD = True
ENABLE_FIDELITY_INSPECTOR = True
ENABLE_AI_EXPLAIN = True
ENABLE_COMPARATOR = True
ENABLE_BETA_FIDELITY = True

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.markdown("## Controls")

all_waves = canon.get_all_waves_safe()
if not all_waves:
    st.sidebar.warning("No waves found. Ensure engine is available OR CSVs exist (wave_config.csv / wave_weights.csv / list.csv).")

mode = st.sidebar.selectbox("Mode", ["Standard", "Alpha-Minus-Beta", "Private Logic"], index=0)
selected_wave = st.sidebar.selectbox("Selected Wave", options=all_waves if all_waves else ["(none)"], index=0)
scan_mode = st.sidebar.toggle("Scan Mode (fast, fewer visuals)", value=True)
days = st.sidebar.selectbox("History Window", [365, 730, 1095, 2520], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Canonical rule: every metric shown is computed from ONE canonical dataset (hist_sel) for the selected Wave+Mode.")

# -----------------------------
# Canonical Source-of-Truth
# -----------------------------
hist_sel = canon.get_canonical_history(selected_wave, mode, days) if selected_wave and selected_wave != "(none)" else pd.DataFrame()
cov = canon.coverage_report(hist_sel)

# Benchmark governance inputs
bm_mix = bm.get_benchmark_mix()
bm_rows_now = bm.bm_rows_for_wave(bm_mix, selected_wave) if selected_wave and selected_wave != "(none)" else pd.DataFrame(columns=["Ticker", "Weight"])
bm_id = bm.benchmark_snapshot_id(selected_wave, bm_mix) if selected_wave and selected_wave != "(none)" else "BM-NA"
bm_drift = bm.benchmark_drift_status(selected_wave, mode, bm_id) if selected_wave and selected_wave != "(none)" else "stable"
bm_diff = bm.benchmark_diff_table(selected_wave, mode, bm_rows_now) if ENABLE_FIDELITY_INSPECTOR else pd.DataFrame()
difficulty = bm.benchmark_difficulty_proxy(bm_rows_now)

# Canonical metrics
m = mx.compute_metrics_from_hist(hist_sel)
conf_level, conf_reason = canon.confidence_from_integrity(cov, bm_drift)

# Scorecard + Risk Reaction
sel_score = sc.compute_analytics_score_for_selected(hist_sel, cov, bm_drift) if ENABLE_SCORECARD else {"AnalyticsScore": np.nan, "Grade": "N/A", "Flags": ""}
rr_score = mx.risk_reaction_score(m.get("te"), m.get("mdd"), m.get("cvar95"))
te_band = mx.te_risk_band(m.get("te"))

# Beta fidelity (optional)
beta_report = {}
if ENABLE_BETA_FIDELITY:
    beta_report = bm.beta_fidelity_report(hist_sel, bm_rows_now)

# -----------------------------
# BIG HEADER (fixes “what wave is this?”)
# -----------------------------
ui.big_wave_header(selected_wave, mode, days)

# -----------------------------
# Sticky scan bar (chips)
# -----------------------------
ui.sticky_open()
ui.chip(f"Confidence: {conf_level}")
if ENABLE_SCORECARD:
    ui.chip(f"Wave Analytics: {sel_score.get('Grade','N/A')} ({ui.fmt_num(sel_score.get('AnalyticsScore'), 1)})")
ui.chip(f"BM: {bm_id} · {str(bm_drift).capitalize()}")
ui.chip(f"Coverage: {ui.fmt_num(cov.get('completeness_score'),1)} · Age: {ui.fmt_int(cov.get('age_days'))}")
ui.chip(f"30D α {ui.fmt_pct(m.get('a30'))} · r {ui.fmt_pct(m.get('r30'))}")
ui.chip(f"60D α {ui.fmt_pct(m.get('a60'))} · r {ui.fmt_pct(m.get('r60'))}")
ui.chip(f"Risk: TE {ui.fmt_pct(m.get('te'))} ({te_band}) · MaxDD {ui.fmt_pct(m.get('mdd'))}")
ui.chip(f"Risk Reaction: {ui.fmt_num(rr_score,1)}/100")

if ENABLE_BETA_FIDELITY and beta_report:
    delta = beta_report.get("delta_beta")
    ui.chip(f"β Fidelity: Δβ {ui.fmt_num(delta,2)} · {beta_report.get('band','N/A')}")

ui.sticky_close()

# -----------------------------
# Tabs (consolidated)
# -----------------------------
tabs = st.tabs(
    ["IC Summary", "Overview", "Risk + Advanced", "Benchmark Governance", "Comparator", "Diagnostics"]
)

# ============================================================
# IC SUMMARY — Executive One-Pager
# ============================================================
with tabs[0]:
    def panel():
        ui.h3("Executive IC One-Pager")

        colA, colB = st.columns([1.2, 1.0], gap="large")

        # Left: Narrative + actions
        with colA:
            ui.card_open()
            st.markdown("#### What is this wave?")
            st.write(
                "A governance-native portfolio wave with a **benchmark-anchored analytics stack**. "
                "Designed to eliminate crisscross metrics and compress institutional analysis time."
            )
            st.markdown("**Trust + Governance**")
            st.write(f"**Confidence:** {conf_level} — {conf_reason}")

            st.markdown("**Performance vs Benchmark**")
            st.write(f"30D Return {ui.fmt_pct(m.get('r30'))} | 30D Alpha {ui.fmt_pct(m.get('a30'))}")
            st.write(f"60D Return {ui.fmt_pct(m.get('r60'))} | 60D Alpha {ui.fmt_pct(m.get('a60'))}")
            st.write(f"365D Return {ui.fmt_pct(m.get('r365'))} | 365D Alpha {ui.fmt_pct(m.get('a365'))}")
            ui.card_close()

            ui.card_open()
            st.markdown("#### Key Wins / Key Risks / Next Actions")

            wins, risks, actions = [], [], []

            if conf_level == "High":
                wins.append("Fresh + complete coverage supports institutional trust.")
            if bm_drift == "stable":
                wins.append("Benchmark snapshot is stable (governance green).")
            if ui.is_pos(m.get("a30")):
                wins.append("Positive 30D alpha versus benchmark mix.")

            if conf_level != "High":
                risks.append("Data trust flags present (coverage/age/rows).")
            if bm_drift != "stable":
                risks.append("Benchmark drift detected (composition changed in-session).")
            if ui.is_low(m.get("mdd"), thresh=-0.25):
                risks.append("Deep drawdown regime risk is elevated.")

            if bm_drift != "stable":
                actions.append("Freeze benchmark mix for demo/governance, then re-run.")
            if ui.is_high(m.get("te"), thresh=0.20):
                actions.append("Confirm exposure caps / SmartSafe posture (active risk is high).")
            if conf_level != "High":
                actions.append("Inspect history pipeline for missing/stale writes.")
            if ENABLE_BETA_FIDELITY and beta_report and beta_report.get("band") in ("Medium", "High"):
                actions.append("Check benchmark beta alignment (Δβ). Misalignment can distort perceived alpha.")

            if not actions:
                actions.append("Proceed: governance is stable; use comparator to position vs other waves.")

            st.markdown("**Key Wins**")
            for x in wins[:3] if wins else ["(none)"]:
                st.write("• " + x)

            st.markdown("**Key Risks**")
            for x in risks[:3] if risks else ["(none)"]:
                st.write("• " + x)

            st.markdown("**Next Actions**")
            for x in actions[:3] if actions else ["(none)"]:
                st.write("• " + x)

            ui.card_close()

        # Right: Tiles + Definitions
        with colB:
            st.markdown("#### IC Tiles")
            c1, c2 = st.columns(2, gap="medium")

            with c1:
                ui.tile("Confidence", conf_level, conf_reason)
                ui.tile("Benchmark", "Stable" if bm_drift == "stable" else "Drift", bm_id)
                ui.tile("30D Alpha", ui.fmt_pct(m.get("a30")), f"30D Return {ui.fmt_pct(m.get('r30'))}")

            with c2:
                ui.tile("Analytics Grade", f"{sel_score.get('Grade','N/A')}", f"{ui.fmt_num(sel_score.get('AnalyticsScore'),1)}/100")
                ui.tile("Active Risk (TE)", ui.fmt_pct(m.get("te")), f"Band: {te_band}")
                ui.tile("Risk Reaction", f"{ui.fmt_num(rr_score,1)}/100", f"MaxDD {ui.fmt_pct(m.get('mdd'))} | CVaR {ui.fmt_pct(m.get('cvar95'))}")

            if ENABLE_BETA_FIDELITY and beta_report:
                st.markdown("---")
                ui.card_open()
                st.markdown("#### Beta Reliability (Benchmark Fidelity)")
                st.write(f"**Wave β vs SPY:** {ui.fmt_num(beta_report.get('beta_wave'), 2)}")
                st.write(f"**Benchmark β vs SPY:** {ui.fmt_num(beta_report.get('beta_bm'), 2)}")
                st.write(f"**Δβ (Wave − BM):** {ui.fmt_num(beta_report.get('delta_beta'), 2)}")
                st.write(f"**Band:** {beta_report.get('band','N/A')} — {beta_report.get('note','')}")
                st.caption("This adds institutional value: beta-aligned benchmarks reduce false alpha attribution.")
                ui.card_close()

            st.markdown("---")
            ui.render_definitions(
                [
                    "Canonical (Source of Truth)",
                    "Return",
                    "Alpha",
                    "Tracking Error (TE)",
                    "Max Drawdown (MaxDD)",
                    "VaR 95% (daily)",
                    "CVaR 95% (daily)",
                    "Risk Reaction Score",
                    "Analytics Scorecard",
                    "Benchmark Snapshot / Drift",
                    "Beta Fidelity",
                ],
                title="Definitions (IC)",
            )

    ui.safe_panel("IC Summary", panel)

# ============================================================
# OVERVIEW — Cohesion Lock + Performance
# ============================================================
with tabs[1]:
    def panel():
        ui.h3("Overview (Canonical Metrics)")
        st.caption("Everything below is computed from the same canonical hist_sel object (no duplicate math).")

        st.markdown("#### Cohesion Lock (Truth Table)")
        truth = canon.build_truth_table(hist_sel, cov, bm_id, bm_drift, m, rr_score)
        st.dataframe(pd.DataFrame([truth]), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Performance vs Benchmark")
        perf_show = mx.performance_table(m)
        st.dataframe(perf_show, use_container_width=True, hide_index=True)

        if not scan_mode and hist_sel is not None and not hist_sel.empty:
            st.markdown("---")
            st.markdown("#### NAV Preview (Wave vs Benchmark)")
            nav_view = hist_sel[["wave_nav", "bm_nav"]].tail(120).rename(columns={"wave_nav": "Wave NAV", "bm_nav": "Benchmark NAV"})
            st.line_chart(nav_view, height=240, use_container_width=True)

    ui.safe_panel("Overview", panel)

# ============================================================
# RISK + ADVANCED — Definitions + AI explanation
# ============================================================
with tabs[2]:
    def panel():
        ui.h3("Risk + Advanced Analytics (Canonical)")

        mx.render_risk_blocks(hist_sel, m, rr_score)

        ui.render_definitions(
            ["Sharpe", "Sortino", "Max Drawdown (MaxDD)", "Tracking Error (TE)", "VaR 95% (daily)", "CVaR 95% (daily)", "Risk Reaction Score"],
            title="Definitions (Risk & Advanced)",
        )

        if ENABLE_AI_EXPLAIN:
            st.markdown("---")
            st.markdown("#### AI Explanation Layer (Rules-Based, Deterministic)")
            narrative = ex.ai_explain_narrative(
                wave=selected_wave,
                mode=mode,
                cov=cov,
                bm_drift=bm_drift,
                metrics=m,
                rr_score=rr_score,
                te_band=te_band,
                beta_report=beta_report if ENABLE_BETA_FIDELITY else None,
            )
            ex.render_narrative(narrative)

    ui.safe_panel("Risk + Advanced", panel)

# ============================================================
# BENCHMARK GOVERNANCE — Fidelity Inspector
# ============================================================
with tabs[3]:
    def panel():
        ui.h3("Benchmark Governance (Fidelity Inspector)")

        left, right = st.columns([1.0, 1.1], gap="large")

        with left:
            ui.card_open()
            st.markdown("#### Inspector Summary")
            st.write(f"**Snapshot:** {bm_id}")
            st.write(f"**Drift Status:** {bm_drift}")
            st.write(f"**Active Risk Band (TE):** {te_band}  (TE {ui.fmt_pct(m.get('te'))})")
            st.write(f"**Difficulty vs SPY (proxy):** {ui.fmt_num(difficulty.get('difficulty_vs_spy'), 1)}  (range ~ -25 to +25)")
            st.caption("Difficulty proxy is a concentration/diversification heuristic (not a promise).")
            ui.card_close()

            if bm_rows_now is not None and not bm_rows_now.empty:
                st.markdown("#### Current Benchmark Composition")
                st.dataframe(bm.format_bm_table(bm_rows_now), use_container_width=True, hide_index=True)
            else:
                st.info("No benchmark mix table available for this wave.")

        with right:
            st.markdown("#### Composition Change vs Last Session")
            if bm_diff is None or bm_diff.empty:
                st.info("No prior snapshot stored yet (or no meaningful changes). Drift will populate diffs once it occurs.")
            else:
                st.dataframe(bm_diff, use_container_width=True, hide_index=True)

            ui.render_definitions(
                ["Benchmark Snapshot / Drift", "Difficulty vs SPY", "Tracking Error (TE)", "Beta Fidelity"],
                title="Definitions (Benchmark Governance)",
            )

    ui.safe_panel("Benchmark Governance", panel)

# ============================================================
# COMPARATOR — Wave A vs Wave B
# ============================================================
with tabs[4]:
    def panel():
        ui.h3("Wave-to-Wave Comparator")
        if not ENABLE_COMPARATOR:
            st.info("Comparator disabled.")
            return

        wave_b = st.selectbox(
            "Compare against (Wave B)",
            options=[w for w in all_waves if w != selected_wave] if all_waves else ["(none)"],
            index=0,
        )

        if not wave_b or wave_b == "(none)" or selected_wave == "(none)":
            st.info("Select a valid Wave B.")
            return

        comp.render_comparator(
            wave_a=selected_wave,
            wave_b=wave_b,
            mode=mode,
            days=days,
            enable_scorecard=ENABLE_SCORECARD,
            enable_beta=ENABLE_BETA_FIDELITY,
        )

    ui.safe_panel("Comparator", panel)

# ============================================================
# DIAGNOSTICS — Always available
# ============================================================
with tabs[5]:
    def panel():
        ui.h3("Diagnostics")
        diag.render_diagnostics(
            selected_wave=selected_wave,
            mode=mode,
            days=days,
            hist_sel=hist_sel,
            cov=cov,
            bm_id=bm_id,
            bm_drift=bm_drift,
        )

    ui.safe_panel("Diagnostics", panel)
    