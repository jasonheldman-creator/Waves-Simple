"""
patch_waves_ic_3fixes.py

WAVES IC — Safe micro-patch for app.py (NO rewrite)

Fix #1: Replace Wave Purpose st.warning() with neutral fallback text
Fix #2: Risk-On / Risk-Off shares → absolute contribution shares (0..1)
Fix #3: build_final_verdict() → explicit hist_sel (no globals)

GUARDS:
• Timestamped backup before changes
• Hard anchor checks (won’t corrupt app.py)
• One-time, idempotent execution
"""

from __future__ import annotations

import re
import sys
import shutil
from pathlib import Path
from datetime import datetime

APP = Path("app.py")


def abort(msg: str):
    print(f"\n❌ PATCH ABORTED: {msg}\n")
    sys.exit(1)


def backup(path: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    shutil.copy2(path, bak)
    print(f"✅ Backup created: {bak}")


def replace_once(src: str, pattern: str, repl: str, label: str, flags=0) -> str:
    if not re.search(pattern, src, flags):
        abort(f"Anchor not found for {label}")
    out, n = re.subn(pattern, repl, src, count=1, flags=flags)
    if n != 1:
        abort(f"Unexpected replacement count for {label}: {n}")
    print(f"✅ Applied {label}")
    return out


def main():
    if not APP.exists():
        abort("app.py not found — run patch from project root")

    src = APP.read_text(encoding="utf-8")
    backup(APP)

    # ------------------------------------------------------------
    # FIX #1 — Wave Purpose warning → neutral text
    # ------------------------------------------------------------
    fix1_pattern = r"""
(\s*st\.markdown\(\s*["']\*\*Wave\ Purpose\ Statement.*?\n)
\s*st\.warning\([^\n]*\)\s*\n
"""
    fix1_repl = r"""\1    st.write("Purpose statement not set yet for this Wave. (Non-blocking)")\n"""

    src = replace_once(
        src,
        fix1_pattern,
        fix1_repl,
        "Fix #1 — Wave Purpose neutral fallback",
        flags=re.MULTILINE | re.VERBOSE,
    )

    # ------------------------------------------------------------
    # FIX #2 — Risk-On / Risk-Off absolute shares
    # ------------------------------------------------------------
    fix2_pattern = r"""
\s*tot\s*=\s*0\.0\s*\n
\s*if\s+math\.isfinite\(ro\)\s*:\s*\n
\s*tot\s*\+=\s*ro\s*\n
\s*if\s+math\.isfinite\(rf\)\s*:\s*\n
\s*tot\s*\+=\s*rf\s*\n
\s*if\s+tot\s*!=\s*0\.0\s*:\s*\n
\s*out\["risk_on_share"\]\s*=\s*ro\s*/\s*tot.*\n
\s*out\["risk_off_share"\]\s*=\s*rf\s*/\s*tot.*\n
"""
    fix2_repl = """
            # FIX: absolute contribution shares (stable 0..1)
            aro = abs(ro) if math.isfinite(ro) else 0.0
            arf = abs(rf) if math.isfinite(rf) else 0.0
            denom = aro + arf
            if denom > 0:
                out["risk_on_share"] = aro / denom
                out["risk_off_share"] = arf / denom
"""

    src = replace_once(
        src,
        fix2_pattern,
        fix2_repl,
        "Fix #2 — Risk-On/Risk-Off absolute shares",
        flags=re.MULTILINE | re.VERBOSE,
    )

    # ------------------------------------------------------------
    # FIX #3 — build_final_verdict(hist_sel)
    # ------------------------------------------------------------

    # 3A — signature
    sig_pattern = r"def\s+build_final_verdict\(([^)]*)\):"
    sig_match = re.search(sig_pattern, src)
    if not sig_match:
        abort("build_final_verdict() definition not found")

    sig_body = sig_match.group(1)
    if "hist_sel" not in sig_body:
        new_sig = sig_body.replace(
            "mode: str,",
            "mode: str,\n    hist_sel,"
        )
        src = src.replace(sig_body, new_sig, 1)
        print("✅ Fix #3A — build_final_verdict signature updated")

    # 3B — globals() removal
    src = replace_once(
        src,
        r"if\s*['\"]hist_sel['\"]\s*in\s*globals\(\)\s*else\s*None",
        "if (hist_sel is not None and not hist_sel.empty) else None",
        "Fix #3B — remove globals() dependency",
    )

    # 3C — call site
    src = replace_once(
        src,
        r"build_final_verdict\(\s*selected_wave\s*,\s*mode\s*,",
        "build_final_verdict(selected_wave, mode, hist_sel,",
        "Fix #3C — pass hist_sel explicitly",
    )

    APP.write_text(src, encoding="utf-8")
    print("\n✅ PATCH COMPLETE — app.py updated safely\n")
    print("Restart Streamlit. Backup available if rollback is needed.")


if __name__ == "__main__":
    main()