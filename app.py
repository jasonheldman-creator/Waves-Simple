# app.py
# WAVES Intelligence™ — Institutional Console
# Full upgrade (SAFE): ALL waves visible + wave-count guardrail + Conditional Attribution Grid
# + persistent logging + safe recommendations w/ session + persistent apply controls

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
    if x is None:
        return "—"
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return "—"
        return f"{x*100:.2f}%"
    except Exception:
        return "—"


def df_to_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def status_for_summary(summary: dict) -> str:
    if summary.get("365D_return") is None:
        return "NO HISTORY / INSUFFICIENT DATA"
    return "OK"


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="WAVES Intelligence™ Console", layout="wide")
st.title("WAVES Intelligence™ Institutional Console")


# -----------------------------
# Sidebar controls
# -----------------------------
modes = we.get_modes()  # engine source of truth
mode = st.sidebar.selectbox("Mode", modes, index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Wave discovery is sourced from your WEIGHTS file (auto-discovered) + optional orphan logs.")

min_expected = st.sidebar.number_input(
    "Wave Count Guardrail (min allowed)",
    min_value=1,
    max_value=100,
    value=15,
    step=1,
    help="If the app discovers fewer waves than this, it will STOP and warn you (prevents regressions).",
)

if st.sidebar.button("Refresh Wave List / Weights"):
    we.refresh_weights()
    st.sidebar.success("Refreshed weights + wave list. Scroll to confirm count updated.")


# -----------------------------
# Ensure ALL waves show up (with guardrail)
# -----------------------------
all_waves = we.get_all_waves()

st.sidebar.markdown(f"**Waves discovered:** `{len(all_waves)}`")
weights_path = we.get_weights_path()
if weights_path:
    st.sidebar.caption(f"Weights file: `{weights_path}`")
else:
    st.sidebar.caption("Weights file: (not found yet)")

if not all_waves:
    st.error("No Waves found. The engine could not read your weights file. Check file name/location and columns.")
    st.stop()

# Guardrail: stop if wave list regresses (prevents accidental deploy)
if len(all_waves) < int(min_expected):
    st.error(
        f"🚨 WAVE DISCOVERY REGRESSION BLOCKER 🚨\n\n"
        f"Discovered only **{len(all_waves)}** waves, but guardrail minimum is **{min_expected}**.\n\n"
        f"Do NOT deploy this build. Fix weights path/format first.\n"
        f"(Your business-safe stopgap to prevent the 10-wave collapse.)"
    )
    st.stop()

wave_selected = st.sidebar.selectbox("Select Wave", all_waves, index=0)


# -----------------------------
# Overview: Multi-window Alpha Capture (ALL WAVES, NEVER HIDE)
# -----------------------------
st.header(f"Portfolio-Level Overview (All Waves) — Mode = {mode}")

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
st.dataframe(overview, use_container_width=True, height=460)

st.download_button(
    "Download overview CSV",
    data=df_to_download(overview),
    file_name=f"waves_overview_{mode.replace(' ','_')}_{datetime.utcnow().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

st.markdown("---")


# -----------------------------
# Wave detail
# -----------------------------
st.header(f"Wave Detail — {wave_selected} (Mode: {mode})")

# Session-only overrides (safe preview applies)
if "session_overrides" not in st.session_state:
    st.session_state["session_overrides"] = {}  # key -> {exposure, smartsafe}
if "last_apply" not in st.session_state:
    st.session_state["last_apply"] = None

key = (wave_selected, mode)
override = st.session_state["session_overrides"].get(key, {})

# Base defaults
base_exposure = we.MODE_BASE_EXPOSURE.get(mode, 1.0)
base_smartsafe = we.MODE_SMARTSAFE_BASE.get(mode, 0.0)

# Persistent defaults (engine supports this)
povr = we.get_persistent_override(wave_selected, mode)
p_expo = float(povr.get("exposure", base_exposure)) if povr else base_exposure
p_ss = float(povr.get("smartsafe", base_smartsafe)) if povr else base_smartsafe

cur_exposure = float(override.get("exposure", p_expo))
cur_smartsafe = float(override.get("smartsafe", p_ss))

with st.expander("Controls (Preview-first)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        cur_exposure = st.slider("Exposure (session override)", 0.0, 1.25, float(cur_exposure), 0.01)
    with col2:
        cur_smartsafe = st.slider("SmartSafe (session override)", 0.0, 0.90, float(cur_smartsafe), 0.01)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Revert session override"):
            st.session_state["session_overrides"].pop(key, None)
            st.session_state["last_apply"] = None
            st.success("Session override reverted.")
            st.rerun()

    with colB:
        if st.button("Clear persistent override (if any)"):
            we.persist_clear(wave_selected, mode)
            st.success("Persistent override cleared.")
            st.rerun()

# Save override state
st.session_state["session_overrides"][key] = {"exposure": cur_exposure, "smartsafe": cur_smartsafe}

bundle = we.wave_detail_bundle(
    wave_selected,
    mode,
    exposure_override=cur_exposure,
    smartsafe_override=cur_smartsafe,
    include_persistent_defaults=True,
)

df365 = bundle.get("df365")

# NAV chart + headline stats
if df365 is None or df365.empty:
    st.warning("Not enough market history to compute 365D NAV for this Wave yet.")
else:
    nav = df365.copy()
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.set_index("date")

    st.line_chart(nav[["wave_nav", "bm_nav"]], height=320)

    ret_365 = (nav["wave_nav"].iloc[-1] / nav["wave_nav"].iloc[0] - 1.0)
    bm_365 = (nav["bm_nav"].iloc[-1] / nav["bm_nav"].iloc[0] - 1.0)
    alpha_365 = (1.0 + nav["alpha"]).prod() - 1.0

    colA, colB, colC = st.columns(3)
    colA.metric("365D Return", f"{ret_365*100:.2f}%")
    colB.metric("365D Alpha", f"{alpha_365*100:.2f}%")
    colC.metric("Benchmark Return", f"{bm_365*100:.2f}%")

st.markdown("---")


# -----------------------------
# Volatility Regime Attribution
# -----------------------------
st.subheader("Volatility Regime Attribution (365D)")
vol_attr = bundle.get("vol_attr")
if vol_attr is None or (isinstance(vol_attr, pd.DataFrame) and vol_attr.empty):
    st.info("Not enough data for regime attribution yet.")
else:
    show = vol_attr.copy()
    show["wave_ret"] = show["wave_ret"].map(lambda x: f"{x*100:.2f}%")
    show["bm_ret"] = show["bm_ret"].map(lambda x: f"{x*100:.2f}%")
    show["alpha"] = show["alpha"].map(lambda x: f"{x*100:.2f}%")
    st.dataframe(show, use_container_width=True, height=220)

st.markdown("---")


# -----------------------------
# Conditional Attribution Grid + persistent logging
# -----------------------------
st.subheader("Conditional Attribution Grid (Regime × Trend) — 365D")
cond_grid = bundle.get("cond_grid")
log_path = bundle.get("cond_log_path", "")

if cond_grid is None or (isinstance(cond_grid, pd.DataFrame) and cond_grid.empty):
    st.info("Not enough data for conditional attribution grid yet.")
else:
    grid_show = cond_grid.copy()
    grid_show["mean_daily_alpha_bp"] = grid_show["mean_daily_alpha"] * 10000.0
    grid_show["cum_alpha"] = grid_show["cum_alpha"].map(lambda x: f"{x*100:.2f}%")
    grid_show["mean_daily_alpha_bp"] = grid_show["mean_daily_alpha_bp"].map(lambda x: f"{x:.2f}")
    grid_show = grid_show[["regime", "trend", "days", "mean_daily_alpha_bp", "cum_alpha"]]
    st.dataframe(grid_show, use_container_width=True, height=260)
    if log_path:
        st.caption(f"Logged to: {log_path}")

    st.download_button(
        "Download conditional grid CSV",
        data=df_to_download(cond_grid),
        file_name=f"{wave_selected}_conditional_{mode.replace(' ','_')}_{datetime.utcnow().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

st.markdown("---")


# -----------------------------
# Diagnostics
# -----------------------------
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


# -----------------------------
# Auto Recommendations (Preview-first + safe apply)
# -----------------------------
st.subheader("Auto Recommendations (Preview-first)")

recos = bundle.get("recommendations") or []
if not recos:
    st.info("No high-confidence recommendations detected (or not enough history).")
else:
    st.caption("Default apply is session-only. You can optionally enable persistent apply with hard confirmation.")
    enable_persist = st.checkbox("Enable persistent apply (writes persistent_overrides.json)", value=False)
    confirm_text = ""
    if enable_persist:
        confirm_text = st.text_input("Type APPLY to enable persistent writes", value="")

    for i, r in enumerate(recos):
        title = r.get("title", "Recommendation")
        conf = r.get("confidence", "Low")
        why = r.get("why", "")
        deltas = r.get("deltas", {}) or {}
        reco_id = r.get("id", f"reco_{i}")

        with st.expander(f"{title} • Confidence: {conf}", expanded=False):
            st.write(why)
            st.code(str(deltas))

            if we.CONF_RANK.get(conf, 0) < we.CONF_RANK.get(we.SAFE_APPLY_LIMITS["min_confidence_to_apply"], 2):
                st.warning("Blocked by guardrails: confidence below minimum for apply.")
                continue

            if not deltas or ("exposure_delta" not in deltas and "smartsafe_delta" not in deltas):
                st.info("Informational recommendation (no auto-apply parameters).")
                continue

            new_exp, new_ss, applied = we.apply_recommendation_preview(
                wave_selected,
                mode,
                current_exposure=float(cur_exposure),
                current_smartsafe=float(cur_smartsafe),
                deltas=deltas,
            )

            st.markdown("**Preview result if applied:**")
            st.write(f"• SmartSafe Δ: {applied['smartsafe_delta']:+.2f} → {new_ss:.2f}")
            st.write(f"• Exposure Δ: {applied['exposure_delta']:+.2f} → {new_exp:.2f}")

            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button(f"Apply (session-only) — {title}", key=f"apply_session_{i}"):
                    st.session_state["last_apply"] = {
                        "key": key,
                        "prev": {"exposure": float(cur_exposure), "smartsafe": float(cur_smartsafe)},
                        "new": {"exposure": float(new_exp), "smartsafe": float(new_ss)},
                        "title": title,
                        "ts": we._now_iso(),
                        "kind": "session",
                    }
                    st.session_state["session_overrides"][key] = {"exposure": float(new_exp), "smartsafe": float(new_ss)}

                    we.log_event({
                        "ts": we._now_iso(),
                        "type": "session_apply",
                        "wave": wave_selected,
                        "mode": mode,
                        "reco_id": reco_id,
                        "confidence": conf,
                        "reason": why,
                        "new_exposure": float(new_exp),
                        "new_smartsafe": float(new_ss),
                    })
                    st.success("Applied (session-only).")
                    st.rerun()

            with c2:
                if st.button("Undo last apply", key=f"undo_{i}"):
                    last = st.session_state.get("last_apply")
                    if last and last.get("key") == key:
                        st.session_state["session_overrides"][key] = last["prev"]
                        we.log_event({
                            "ts": we._now_iso(),
                            "type": "session_undo",
                            "wave": wave_selected,
                            "mode": mode,
                            "reverted_to": last["prev"],
                        })
                        st.success("Reverted session apply.")
                        st.rerun()
                    else:
                        st.info("No apply to undo for this Wave/Mode.")

            with c3:
                persist_ok = enable_persist and (confirm_text.strip().upper() == "APPLY")
                if st.button("Persist apply (writes persistent_overrides.json)", key=f"persist_{i}", disabled=(not persist_ok)):
                    we.persist_apply(
                        wave_selected,
                        mode,
                        new_exposure=float(new_exp),
                        new_smartsafe=float(new_ss),
                        reason=why,
                        reco_id=reco_id,
                        confidence=conf,
                    )
                    st.success("Persisted. This will apply on future runs.")
                    st.rerun()

st.markdown("---")


# -----------------------------
# Top-10 holdings
# -----------------------------
st.subheader("Top-10 Holdings")
holds = bundle.get("holdings")
if holds is None or holds.empty:
    st.info("No holdings found for this Wave (check weights file).")
else:
    show = holds.copy().head(10)
    show["weight"] = show["weight"].map(lambda x: f"{float(x):.2f}%")
    show = show.rename(columns={"ticker": "Ticker", "weight": "Weight"})
    st.dataframe(show, use_container_width=True, height=260)

st.caption("Logs: logs/recommendations/reco_events.csv • Persistent overrides: logs/overrides/persistent_overrides.json")