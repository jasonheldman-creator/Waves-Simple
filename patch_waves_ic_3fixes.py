"""
patch_waves_ic_3fixes.py
WAVES IC: 3 micro-fixes applied to existing app.py WITHOUT rewriting the whole file.

Fix #1: Wave Purpose "yellow warning" -> neutral fallback line
Fix #2: Risk-On/Risk-Off share math -> absolute contribution shares (0..1, never negative, never >1)
Fix #3: build_final_verdict() -> remove globals()/implicit hist_sel; pass hist_sel explicitly

SAFE BEHAVIOR:
- Creates a timestamped backup copy of app.py before changes
- Refuses to patch if anchors aren't found (so it won't corrupt your file)
"""

from __future__ import annotations

import re
import sys
import shutil
from datetime import datetime
from pathlib import Path

APP_PATH = Path("app.py")  # change if needed

def die(msg: str, code: int = 1):
    print(f"\n❌ {msg}\n")
    sys.exit(code)

def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    shutil.copy2(path, bak)
    print(f"✅ Backup created: {bak}")
    return bak

def apply_regex_replace(text: str, pattern: str, repl: str, desc: str, flags=0) -> str:
    m = re.search(pattern, text, flags)
    if not m:
        die(f"Anchor not found for: {desc}\nPattern:\n{pattern}")
    new_text, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        die(f"Unexpected replace count ({n}) for: {desc}")
    print(f"✅ Patched: {desc}")
    return new_text

def main():
    if not APP_PATH.exists():
        die(f"Could not find {APP_PATH.resolve()}. Run this script in the same folder as app.py.")

    src = APP_PATH.read_text(encoding="utf-8")
    backup_file(APP_PATH)

    # ------------------------------------------------------------
    # FIX #1 — Wave Purpose: replace st.warning() missing-purpose UX with neutral st.write()
    #
    # This targets the common pattern:
    #   if (not purpose) or ("Purpose statement not set yet" in str(purpose)):
    #       st.markdown("**Wave Purpose Statement (Vector™):**")
    #       st.warning(...)
    #       ...
    #
    # We replace ONLY the st.warning line with st.write neutral text.
    # ------------------------------------------------------------
    fix1_pattern = r"""
(?P<prefix>
\s*if\s*\(not\s+purpose\)\s*or\s*\(\s*["']Purpose\ statement\ not\ set\ yet["']\s*in\s*str\(purpose\)\s*\)\s*:\s*\n
(?:.*\n){0,8}?
\s*st\.markdown\(\s*["']\*\*Wave\ Purpose\ Statement.*?\n
)
(?P<warning_line>
\s*st\.warning\([^\n]*\)\s*\n
)
"""
    fix1_repl = r"""\g<prefix>    st.write(str(purpose) if purpose else "Purpose statement not set yet for this Wave. (Non-blocking)")\n"""

    src = apply_regex_replace(
        src,
        fix1_pattern,
        fix1_repl,
        desc="Fix #1 (Wave Purpose warning -> neutral fallback line)",
        flags=re.MULTILINE | re.VERBOSE,
    )

    # ------------------------------------------------------------
    # FIX #2 — Risk-On/Risk-Off share math:
    # Replace the share computation block inside _risk_on_off_attrib()
    #
    # We look for the block that sets tot += ro/rf and then does ro/tot, rf/tot.
    # We replace it with absolute contribution shares:
    #   denom = abs(ro)+abs(rf)
    #   share = abs(x)/denom
    # ------------------------------------------------------------
    fix2_pattern = r"""
(?P<block>
\s*tot\s*=\s*0\.0\s*\n
\s*if\s+math\.isfinite\(ro\)\s*:\s*\n
\s*tot\s*\+=\s*ro\s*\n
\s*if\s+math\.isfinite\(rf\)\s*:\s*\n
\s*tot\s*\+=\s*rf\s*\n
\s*if\s+tot\s*!=\s*0\.0\s*:\s*\n
\s*out\["risk_on_share"\]\s*=\s*ro\s*/\s*tot.*\n
\s*out\["risk_off_share"\]\s*=\s*rf\s*/\s*tot.*\n
)
"""
    fix2_repl = r"""
            # ✅ FIX #2: ABS contribution shares (stable 0..1, never negative / >1)
            aro = abs(ro) if math.isfinite(ro) else 0.0
            arf = abs(rf) if math.isfinite(rf) else 0.0
            denom = aro + arf
            if denom > 0:
                out["risk_on_share"] = aro / denom
                out["risk_off_share"] = arf / denom
"""

    src = apply_regex_replace(
        src,
        fix2_pattern,
        fix2_repl,
        desc="Fix #2 (Risk-On/Risk-Off shares -> absolute contribution shares)",
        flags=re.MULTILINE | re.VERBOSE,
    )

    # ------------------------------------------------------------
    # FIX #3 — build_final_verdict(): remove globals()/implicit hist_sel use
    #
    # A) Update function signature to accept hist_sel explicitly:
    #    def build_final_verdict(..., hist_sel: pd.DataFrame, ...) -> Dict[str, Any]:
    #
    # We patch the def line by inserting hist_sel parameter after mode: str,
    # if it is not already present.
    #
    # B) Replace the internal line that tries:
    #    exp = _try_get_exposure_series(... pd.DatetimeIndex(hist_sel.index)) if 'hist_sel' in globals() else None
    # with:
    #    exp = _try_get_exposure_series(... pd.DatetimeIndex(hist_sel.index)) if (hist_sel is not None and not hist_sel.empty) else None
    #
    # C) Update the call site where final_verdict = build_final_verdict(...)
    # to pass hist_sel=hist_sel.
    # ------------------------------------------------------------

    # 3A: patch def build_final_verdict signature
    fix3a_pattern = r"""
def\s+build_final_verdict\(\s*\n
(?P<body>(?:.*\n){0,40}?)
\)\s*->\s*Dict\[str,\s*Any\]\s*:\s*\n
"""
    m = re.search(fix3a_pattern, src, flags=re.MULTILINE | re.VERBOSE)
    if not m:
        die("Anchor not found for Fix #3A (build_final_verdict def block).")

    def_block = m.group(0)
    if "hist_sel" not in def_block:
        # Insert hist_sel right after mode line if present, otherwise after selected_wave.
        def_block_new = def_block
        if re.search(r"\s*mode:\s*str\s*,\s*\n", def_block_new):
            def_block_new = re.sub(
                r"(\s*mode:\s*str\s*,\s*\n)",
                r"\1    hist_sel: pd.DataFrame,\n",
                def_block_new,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            def_block_new = re.sub(
                r"(\s*selected_wave:\s*str\s*,\s*\n)",
                r"\1    mode: str,\n    hist_sel: pd.DataFrame,\n",
                def_block_new,
                count=1,
                flags=re.MULTILINE,
            )

        src = src.replace(def_block, def_block_new, 1)
        print("✅ Patched: Fix #3A (build_final_verdict signature now takes hist_sel)")
    else:
        print("ℹ️ Fix #3A skipped: build_final_verdict already has hist_sel")

    # 3B: patch the bad globals() usage line (or similar)
    fix3b_pattern = r"""
exp\s*=\s*_try_get_exposure_series\(
\s*selected_wave,\s*mode,\s*pd\.DatetimeIndex\(hist_sel\.index\)
\)\s*if\s*['"]hist_sel['"]\s*in\s*globals\(\)\s*else\s*None
"""
    fix3b_repl = r"""exp = _try_get_exposure_series(selected_wave, mode, pd.DatetimeIndex(hist_sel.index)) if (hist_sel is not None and not hist_sel.empty) else None"""
    src = apply_regex_replace(
        src,
        fix3b_pattern,
        fix3b_repl,
        desc="Fix #3B (remove globals() hist_sel pattern in build_final_verdict)",
        flags=re.MULTILINE | re.VERBOSE,
    )

    # 3C: patch the call site to pass hist_sel
    # Matches: final_verdict = build_final_verdict(selected_wave, mode, metrics, cov, bm_drift, beta_score, rr_score)
    fix3c_pattern = r"""
final_verdict\s*=\s*build_final_verdict\(
\s*selected_wave\s*,\s*
mode\s*,\s*
metrics\s*,\s*
cov\s*,\s*
bm_drift\s*,\s*
beta_score\s*,\s*
rr_score\s*
\)
"""
    fix3c_repl = r"""final_verdict = build_final_verdict(selected_wave, mode, hist_sel, metrics, cov, bm_drift, beta_score, rr_score)"""
    src = apply_regex_replace(
        src,
        fix3c_pattern,
        fix3c_repl,
        desc="Fix #3C (pass hist_sel into build_final_verdict call site)",
        flags=re.MULTILINE | re.VERBOSE,
    )

    APP_PATH.write_text(src, encoding="utf-8")
    print("\n✅ DONE. app.py patched successfully.\n")
    print("Next: re-run Streamlit. If anything looks off, restore from the .bak_ file created above.")

if __name__ == "__main__":
    main()
