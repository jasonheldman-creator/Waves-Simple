"""
patch_waves_ic_3fixes.py
WAVES IC: micro-fixes applied to existing app.py WITHOUT rewriting the whole file.

Fix #0: Repair broken multiline condition:
        if len(ac) >=
            2:
        -> if len(ac) >= 2:

Fix #1: Wave Purpose "yellow warning" -> neutral fallback line (non-blocking UX)
Fix #2: Risk-On/Risk-Off share math -> absolute contribution shares (0..1, never negative, never >1)
Fix #3: build_final_verdict() -> remove globals()/implicit hist_sel; pass hist_sel explicitly

SAFE BEHAVIOR:
- Creates a timestamped backup copy of app.py before changes
- Refuses to patch when a REQUIRED anchor isn't found (so it won't corrupt your file)
- Some fixes are OPTIONAL (if already fixed / not present) and will skip safely
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


def apply_regex_replace_required(text: str, pattern: str, repl: str, desc: str, flags=0) -> str:
    m = re.search(pattern, text, flags)
    if not m:
        die(f"Anchor not found for: {desc}\nPattern:\n{pattern}")
    new_text, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        die(f"Unexpected replace count ({n}) for: {desc}")
    print(f"✅ Patched: {desc}")
    return new_text


def apply_regex_replace_optional(text: str, pattern: str, repl: str, desc: str, flags=0) -> str:
    m = re.search(pattern, text, flags)
    if not m:
        print(f"ℹ️ Skipped (not found / already fixed): {desc}")
        return text
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
    # FIX #0 — Repair broken multiline condition:
    #   if len(ac) >=
    #       2:
    # -> if len(ac) >= 2:
    #
    # OPTIONAL: skip if not present.
    # ------------------------------------------------------------
    fix0_pattern = r"""
(?P<indent>^[ \t]*)if[ \t]+len\([ \t]*ac[ \t]*\)[ \t]*>=[ \t]*\r?\n
(?P<indent2>[ \t]*)2[ \t]*:[ \t]*$
"""
    fix0_repl = r"""\g<indent>if len(ac) >= 2:"""
    src = apply_regex_replace_optional(
        src,
        fix0_pattern,
        fix0_repl,
        desc="Fix #0 (repair broken 'if len(ac) >=' newline '2:' condition)",
        flags=re.MULTILINE | re.VERBOSE,
    )

    # ------------------------------------------------------------
    # FIX #1 — Wave Purpose: replace st.warning() missing-purpose UX with neutral st.write()
    #
    # Targets:
    #   if (not purpose) or ("Purpose statement not set yet" in str(purpose)):
    #       st.markdown("**Wave Purpose Statement (Vector™):**")
    #       st.warning(...)
    #
    # We replace ONLY the st.warning line with st.write, preserving indentation.
    # REQUIRED: if this exact section exists, we patch it; if not, we skip safely.
    # ------------------------------------------------------------
    fix1_pattern = r"""
(?P<prefix>
^[ \t]*if[ \t]*\([ \t]*not[ \t]+purpose[ \t]*\)[ \t]*or[ \t]*\([ \t]*["']Purpose\ statement\ not\ set\ yet["'][ \t]*in[ \t]*str\(purpose\)[ \t]*\)[ \t]*:[ \t]*\r?\n
(?:.*\r?\n){0,10}?
^[ \t]*st\.markdown\([ \t]*["']\*\*Wave\ Purpose\ Statement.*?\r?\n
)
(?P<warn_indent>^[ \t]*)st\.warning\([^\r\n]*\)[ \t]*\r?\n
"""
    fix1_repl = (
        r"""\g<prefix>"""
        r"""\g<warn_indent>st.write(str(purpose) if purpose else "Purpose statement not set yet for this Wave. (Non-blocking)")"""
        r"""\n"""
    )
    src = apply_regex_replace_optional(
        src,
        fix1_pattern,
        fix1_repl,
        desc="Fix #1 (Wave Purpose warning -> neutral fallback line)",
        flags=re.MULTILINE | re.VERBOSE,
    )

    # ------------------------------------------------------------
    # FIX #2 — Risk-On/Risk-Off share math:
    # Replace the share computation block inside the risk-on/off attribution function.
    #
    # OLD:
    #   tot += ro/rf then ro/tot rf/tot (can go negative or >1)
    #
    # NEW:
    #   denom = abs(ro)+abs(rf); shares = abs(x)/denom
    #
    # OPTIONAL: skip if block not found (already fixed or code differs).
    # ------------------------------------------------------------
    fix2_pattern = r"""
(?P<block>
^[ \t]*tot[ \t]*=[ \t]*0\.0[ \t]*\r?\n
^[ \t]*if[ \t]+math\.isfinite\(ro\)[ \t]*:[ \t]*\r?\n
^[ \t]*tot[ \t]*\+=[ \t]*ro[ \t]*\r?\n
^[ \t]*if[ \t]+math\.isfinite\(rf\)[ \t]*:[ \t]*\r?\n
^[ \t]*tot[ \t]*\+=[ \t]*rf[ \t]*\r?\n
^[ \t]*if[ \t]+tot[ \t]*!=[ \t]*0\.0[ \t]*:[ \t]*\r?\n
^[ \t]*out\["risk_on_share"\][ \t]*=[ \t]*ro[ \t]*/[ \t]*tot.*\r?\n
^[ \t]*out\["risk_off_share"\][ \t]*=[ \t]*rf[ \t]*/[ \t]*tot.*\r?\n
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
    src = apply_regex_replace_optional(
        src,
        fix2_pattern,
        fix2_repl,
        desc="Fix #2 (Risk-On/Risk-Off shares -> absolute contribution shares)",
        flags=re.MULTILINE | re.VERBOSE,
    )

    # ------------------------------------------------------------
    # FIX #3 — build_final_verdict(): remove globals()/implicit hist_sel use
    #
    # 3A) Ensure signature accepts hist_sel explicitly:
    #     def build_final_verdict(selected_wave: str, mode: str, hist_sel: pd.DataFrame, ...)
    #
    # 3B) Replace any exp=... line that uses ('hist_sel' in globals()) with a proper check
    #
    # 3C) Ensure call site passes hist_sel
    # ------------------------------------------------------------

    # 3A: patch def build_final_verdict signature (best-effort)
    fix3a_def_block_pattern = r"""
def[ \t]+build_final_verdict\([ \t]*\r?\n
(?P<body>(?:.*\r?\n){0,60}?)
\)[ \t]*->[ \t]*Dict\[str,[ \t]*Any\][ \t]*:[ \t]*\r?\n
"""
    m = re.search(fix3a_def_block_pattern, src, flags=re.MULTILINE | re.VERBOSE)
    if not m:
        print("ℹ️ Skipped Fix #3A: build_final_verdict def block not found (pattern mismatch).")
    else:
        def_block = m.group(0)
        if "hist_sel" in def_block:
            print("ℹ️ Fix #3A skipped: build_final_verdict already has hist_sel")
        else:
            # Insert hist_sel right after the 'mode: str,' line if present; else after selected_wave: str,
            def_block_new = def_block
            if re.search(r"^[ \t]*mode:[ \t]*str[ \t]*,[ \t]*\r?\n", def_block_new, flags=re.MULTILINE):
                def_block_new = re.sub(
                    r"(^[ \t]*mode:[ \t]*str[ \t]*,[ \t]*\r?\n)",
                    r"\1    hist_sel: pd.DataFrame,\n",
                    def_block_new,
                    count=1,
                    flags=re.MULTILINE,
                )
            elif re.search(r"^[ \t]*selected_wave:[ \t]*str[ \t]*,[ \t]*\r?\n", def_block_new, flags=re.MULTILINE):
                def_block_new = re.sub(
                    r"(^[ \t]*selected_wave:[ \t]*str[ \t]*,[ \t]*\r?\n)",
                    r"\1    mode: str,\n    hist_sel: pd.DataFrame,\n",
                    def_block_new,
                    count=1,
                    flags=re.MULTILINE,
                )
            else:
                die("Fix #3A: could not locate where to insert hist_sel in build_final_verdict signature.")

            src = src.replace(def_block, def_block_new, 1)
            print("✅ Patched: Fix #3A (build_final_verdict signature now takes hist_sel)")

    # 3B: patch the bad globals() usage line (optional)
    # This is intentionally flexible: it looks for a line containing globals() and hist_sel together on exp assignment.
    fix3b_pattern = r"""
^[ \t]*exp[ \t]*=[^\r\n]*globals\(\)[^\r\n]*hist_sel[^\r\n]*\r?\n
"""
    fix3b_repl = r"""    exp = _try_get_exposure_series(selected_wave, mode, pd.DatetimeIndex(hist_sel.index)) if (hist_sel is not None and not hist_sel.empty) else None
"""
    src = apply_regex_replace_optional(
        src,
        fix3b_pattern,
        fix3b_repl,
        desc="Fix #3B (replace exp=... globals() hist_sel pattern)",
        flags=re.MULTILINE | re.VERBOSE,
    )

    # 3C: patch call site to pass hist_sel (best-effort)
    # If your call is positional, this fixes the common pattern:
    #   final_verdict = build_final_verdict(selected_wave, mode, metrics, cov, bm_drift, beta_score, rr_score)
    # -> final_verdict = build_final_verdict(selected_wave, mode, hist_sel, metrics, cov, bm_drift, beta_score, rr_score)
    fix3c_pattern = r"""
final_verdict[ \t]*=[ \t]*build_final_verdict\(
[ \t]*selected_wave[ \t]*,[ \t]*
mode[ \t]*,[ \t]*
metrics[ \t]*,[ \t]*
cov[ \t]*,[ \t]*
bm_drift[ \t]*,[ \t]*
beta_score[ \t]*,[ \t]*
rr_score[ \t]*
\)
"""
    fix3c_repl = r"""final_verdict = build_final_verdict(selected_wave, mode, hist_sel, metrics, cov, bm_drift, beta_score, rr_score)"""
    src = apply_regex_replace_optional(
        src,
        fix3c_pattern,
        fix3c_repl,
        desc="Fix #3C (pass hist_sel into build_final_verdict call site)",
        flags=re.MULTILINE | re.VERBOSE,
    )

    APP_PATH.write_text(src, encoding="utf-8")
    print("\n✅ DONE. app.py patched successfully.\n")
    print("Next: switch main file back to app.py and redeploy. If anything looks off, restore from the .bak_ file created above.")


if __name__ == "__main__":
    main()