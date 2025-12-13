# app.py
# WAVES Intelligence™ — Institutional Console
# Upgrade: Recommendation system + dropdown style + session apply + persistent apply + clear persistent override
# Goal: Build on your current restored console without shrinking it.

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import waves_engine as we


# -----------------------------
# UI helpers
# -----------------------------
def fmt_pct(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x*100:.2f}%"


def df_to_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def status_for_summary(summary: dict) -> str:
    if summary.get("365D_return") is None:
        return "NO HISTORY / INSUFFICIENT DATA"
    return "OK"


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="WAVES Intelligence™ Console",
    layout="wide",
)

st.title("WAVES Intelligence™ Institutional Console")


# -----------------------------
# Auto-load weights on page load (no button required)
# -----------------------------
try:
    we.refresh_weights()
except Exception:
    pass


# -----------------------------
# Sidebar controls
# -----------------------------
all_modes = ["Standard", "Alpha-Minus-Beta", "Private Logic"]

with st.sidebar:
    st.header("Controls")
    mode = st.selectbox("Mode", all_modes, index=0)

    reco_style = st.selectbox(
        "Recommendation Style",
        ["Conservative", "Balanced", "Aggressive"],
        index=1,
        help="Controls how large suggested deltas can be (still capped by guardrails)."
    )

    st.markdown("---")
    st.caption("Wave discovery: wave_weights.csv + logs + fallback. No waves should disappear.")

    if st.button("Refresh Wave List / Weights"):
        we.refresh_weights()
        st.success("Refreshed weights.")


# -----------------------------
# Ensure ALL waves show up
# -----------------------------
all_waves = we.get_all_waves()
if not all_waves:
    st.error("No Waves found. Make sure wave_weights.csv exists and has wave/ticker/weight columns.")
    st.stop()

wave_selected = st.sidebar.selectbox("Select Wave", all_waves, index=0)


# ============================================================
# Overview: Multi-window Alpha Capture (ALL WAVES, NEVER HIDE)
# ============================================================
st.header(f"Multi-Window Alpha Capture (All Waves · Mode = {mode})")

rows = []
for w in all_waves:
    try:
        s = we.compute_multi_window_summary(w, mode)
        rows.append({
            "Wave": w,
            "Status": status_for_summary(s),
            "1D Return": fmt_pct(s.get("1D_return")),
            "1D Alpha": fmt_pct(s.get("1D_alpha")),
            "30D Return": fmt_pct(s.get("30D_return")),
            "30D Alpha": fmt_pct(s.get("30D_alpha")),
            "60D Return": fmt_pct(s.get("60D_return")),
            "60D Alpha": fmt_pct(s.get("60D_alpha")),
            "365D Return": fmt_pct(s.get("365D_return")),
            "365D Alpha": fmt_pct(s.get("365D_alpha")),
        })
    except Exception as e:
        rows.append({
            "Wave": w,
            "Status": f"ERROR: {type(e).__name__}",
            "1D Return": "—", "1D Alpha": "—",
            "30D Return": "—", "30D Alpha": "—",
            "60D Return": "—", "60D Alpha": "—",
            "365D Return": "—", "365D Alpha": "—",
        })

overview = pd.DataFrame(rows)
st.dataframe(overview, use_container_width=True, height=420)

st.download_button(
    "Download overview CSV",
    data=df_to_download(overview),
    file_name=f"waves_overview_{mode.replace(' ','_')}_{datetime.utcnow().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

st.markdown("---")


# ============================================================
# Wave detail + safe overrides
# ============================================================
st.header(f"Wave Detail — {wave_selected} (Mode: {mode})")

if "session_overrides" not in st.session_state:
    st.session_state["session_overrides"] = {}  # (wave,mode) -> {exposure, smartsafe}
if "last_apply" not in st.session_state:
    st.session_state["last_apply"] = None

key = (wave_selected, mode)
override = st.session_state["session_overrides"].get(key, {})

base_exposure = we.MODE_BASE_EXPOSURE.get(mode, 1.0)
base_smartsafe = we.MODE_SMARTSAFE_BASE.get(mode, 0.0)

cur_exposure = float(override.get("exposure", base_exposure))
cur_smartsafe = float(override.get("smartsafe", base_smartsafe))

with st.expander("Session-only Controls (preview / revert)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        cur_exposure = st.slider("Exposure (session-only)", 0.0, 1.25, cur_exposure, 0.01)
    with col2:
        cur_smartsafe = st.slider("SmartSafe (session-only)", 0.0, 0.90, cur_smartsafe, 0.01)

    if st.button("Revert session-only overrides"):
        st.session_state["session_overrides"].pop(key, None)
        st.session_state["last_apply"] = None
        st.success("Reverted overrides for this Wave/Mode.")
        st.experimental_rerun()

st.session_state["session_overrides"][key] = {"exposure": cur_exposure, "smartsafe": cur_smartsafe}

bundle = we.wave_detail_bundle(
    wave_selected,
    mode,
    exposure_override=cur_exposure,
    smartsafe_override=cur_smartsafe,
    reco_style=reco_style,
)

df365 = bundle.get("df365")


# -----------------------------
# NAV chart + headline stats
# -----------------------------
if df365 is None or (isinstance(df365, pd.DataFrame) and df365.empty):
    st.warning("Not enough market history to compute 365D NAV for this Wave (yet). The Wave still appears in the list.")
else:
    nav = df365.copy()
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.set_index("date")

    st.line_chart(nav[["wave_nav", "bm_nav"]], height=320)

    ret_365 = float(nav["wave_nav"].iloc[-1] / nav["wave_nav"].iloc[0] - 1.0)
    bm_365 = float(nav["bm_nav"].iloc[-1] / nav["bm_nav"].iloc[0] - 1.0)
    alpha_365 = float((1.0 + nav["alpha"]).prod() - 1.0)

    colA, colB, colC = st.columns(3)
    colA.metric("365D Return", f"{ret_365*100:.2f}%")
    colB.metric("365D Alpha", f"{alpha_365*100:.2f}%")
    colC.metric("Benchmark Return", f"{bm_365*100:.2f}%")

st.markdown("---")


# ============================================================
# Volatility Regime Attribution
# ============================================================
st.subheader("Volatility Regime Attribution (365D)")
vol_attr = bundle.get("vol_attr")
if vol_attr is None or (isinstance(vol_attr, pd.DataFrame) and vol_attr.empty):
    st.info("Not enough data for volatility regime attribution yet.")
else:
    show = vol_attr.copy()
    for c in ["wave_ret", "bm_ret", "alpha"]:
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{x*100:.2f}%")
    st.dataframe(show, use_container_width=True, height=220)

st.markdown("---")


# ============================================================
# Conditional Attribution Grid + persistent logging
# ============================================================
st.subheader("Conditional Attribution Grid (Regime × Trend) — 365D")
cond_grid = bundle.get("cond_grid")
log_path = bundle.get("cond_log_path", "")

if cond_grid is None or (isinstance(cond_grid, pd.DataFrame) and cond_grid.empty):
    st.info("Not enough data for conditional attribution grid yet.")
else:
    grid_show = cond_grid.copy()
    if "mean_daily_alpha" in grid_show.columns:
        grid_show["mean_daily_alpha_bp"] = grid_show["mean_daily_alpha"] * 10000.0
        grid_show["mean_daily_alpha_bp"] = grid_show["mean_daily_alpha_bp"].map(lambda x: f"{x:.2f}")
    if "cum_alpha" in grid_show.columns:
        grid_show["cum_alpha"] = grid_show["cum_alpha"].map(lambda x: f"{x*100:.2f}%")

    cols = [c for c in ["regime", "trend", "days", "mean_daily_alpha_bp", "cum_alpha"] if c in grid_show.columns]
    st.dataframe(grid_show[cols], use_container_width=True, height=260)

    if log_path:
        st.caption(f"Logged to: {log_path}")

    st.download_button(
        "Download conditional grid CSV",
        data=df_to_download(cond_grid),
        file_name=f"{wave_selected}_conditional_{mode.replace(' ','_')}_{datetime.utcnow().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

st.markdown("---")


# ============================================================
# Diagnostics
# ============================================================
st.subheader("Diagnostics")
diags = bundle.get("diagnostics") or []
levels = [d.get("level", "INFO") for d in diags]
if any(l == "WARN" for l in levels):
    for d in diags:
        if d.get("level") == "WARN":
            st.warning(f"WARN — {d.get('msg','')}")
else:
    d0 = diags[0] if diags else {"level": "PASS", "msg": "No issues detected."}
    st.success(f"{d0.get('level','PASS')} — {d0.get('msg','No issues detected.')}")

st.markdown("---")


# ============================================================
# Auto Recommendations (Preview-first + Session Apply + Persistent Apply)
# ============================================================
st.subheader("Auto Recommendations (Preview-First + Safe Apply)")
recos = bundle.get("recommendations", [])

if not recos:
    st.info("No high-confidence recommendations detected (or not enough history).")
else:
    st.caption("These are suggestions based on conditional attribution + diagnostics. Preview-first, capped deltas, reversible.")

    for i, r in enumerate(recos):
        title = r.get("title", "Recommendation")
        conf = r.get("confidence", "Low")
        why = r.get("why", "")
        deltas = r.get("deltas", {}) or {}
        action_only = bool(r.get("action_only", False))
        rid = r.get("id", title)

        with st.expander(f"{title} • Confidence: {conf}", expanded=False):
            st.write(why)

            if action_only or (not deltas):
                st.info("This recommendation is an action/weights suggestion (no parameter auto-apply).")
                continue

            st.code(str(deltas))

            # confidence gate for apply buttons
            if we.CONF_RANK.get(conf, 0) < we.CONF_RANK.get(we.SAFE_APPLY_LIMITS["min_confidence_to_apply"], 2):
                st.warning("Blocked by guardrails: confidence below minimum to apply.")
                continue

            # preview
            new_exp, new_ss, applied = we.apply_recommendation_preview(
                wave_selected,
                mode,
                current_exposure=cur_exposure,
                current_smartsafe=cur_smartsafe,
                deltas=deltas
            )

            st.markdown("**Preview if applied:**")
            st.write(f"• Exposure Δ: {applied['exposure_delta']:+.2f} → {new_exp:.2f}")
            st.write(f"• SmartSafe Δ: {applied['smartsafe_delta']:+.2f} → {new_ss:.2f}")

            colX, colY = st.columns(2)

            # SESSION APPLY
            with colX:
                if st.button(f"Apply (session-only) — {title}", key=f"apply_{i}"):
                    st.session_state["last_apply"] = {
                        "key": key,
                        "prev": {"exposure": cur_exposure, "smartsafe": cur_smartsafe},
                        "new": {"exposure": new_exp, "smartsafe": new_ss},
                        "title": title,
                        "ts": we._now_iso(),
                    }
                    st.session_state["session_overrides"][key] = {"exposure": new_exp, "smartsafe": new_ss}

                    we._log_recommendation_event({
                        "ts": we._now_iso(),
                        "type": "session_apply",
                        "wave": wave_selected,
                        "mode": mode,
                        "reco_id": rid,
                        "confidence": conf,
                        "why": why,
                        "applied": applied,
                        "new_params": {"exposure": new_exp, "smartsafe": new_ss},
                        "style": reco_style,
                    })

                    st.success("Applied (session-only). Scroll up to see updated outputs.")
                    st.experimental_rerun()

            # UNDO
            with colY:
                if st.button("Undo last apply (revert)", key=f"undo_{i}"):
                    last = st.session_state.get("last_apply")
                    if last and last.get("key") == key:
                        st.session_state["session_overrides"][key] = last["prev"]
                        we._log_recommendation_event({
                            "ts": we._now_iso(),
                            "type": "session_undo",
                            "wave": wave_selected,
                            "mode": mode,
                            "reco_id": rid,
                            "reverted_to": last["prev"],
                        })
                        st.success("Reverted last apply.")
                        st.experimental_rerun()
                    else:
                        st.info("No apply to undo for this Wave/Mode.")

            st.markdown("---")

            # PERSISTENT APPLY (opt-in)
            st.markdown("### Persistent Apply (optional)")
            st.caption("Persists beyond this session (writes to disk). Requires confirmation.")

            confirm_key = f"confirm_persist_{i}"
            if confirm_key not in st.session_state:
                st.session_state[confirm_key] = False

            st.session_state[confirm_key] = st.checkbox(
                "I understand this persists beyond this session",
                value=st.session_state[confirm_key],
                key=confirm_key
            )

            colP1, colP2 = st.columns(2)

            with colP1:
                if st.button(f"Persist Apply — {title}", key=f"persist_{i}", disabled=not st.session_state[confirm_key]):
                    # persistent write
                    we.persist_apply(
                        wave_selected,
                        mode,
                        new_exposure=float(new_exp),
                        new_smartsafe=float(new_ss),
                        reason=str(why),
                        confidence=str(conf),
                        reco_id=str(rid),
                    )
                    st.success("Persisted. This Wave/Mode will load these parameters automatically next run.")
                    st.experimental_rerun()

            with colP2:
                if st.button("Clear Persistent Override (this Wave/Mode)", key=f"clear_persist_{i}"):
                    we.persist_clear(wave_selected, mode)
                    st.success("Cleared persistent override.")
                    st.experimental_rerun()

st.markdown("---")


# ============================================================
# Top-10 holdings
# ============================================================
st.subheader("Top-10 Holdings")
holds = bundle.get("holdings")
if holds is None or (isinstance(holds, pd.DataFrame) and holds.empty):
    st.info("No holdings found for this Wave (check wave_weights.csv).")
else:
    show = holds.copy()
    show["weight"] = (show["weight"] * 100.0).map(lambda x: f"{x:.2f}%")
    show.columns = ["Ticker", "Weight (%)"]
    st.dataframe(show, use_container_width=True, height=260)