"""
patch_waves_ic_scores_only.py
WAVES IC: Convert letter-grade tiles (A–F) into 0–100 score tiles WITHOUT rewriting app.py.

What it changes (display-only):
- "Benchmark Fit" big letter -> big numeric score (e.g., 53.7/100)
- "Beta Reliability" big letter -> big numeric score (e.g., 53.7/100)
- "Analytics Grade" big letter -> big numeric score (e.g., 90.2/100) and label -> "Analytics Score"

SAFE BEHAVIOR:
- Creates a timestamped backup copy of app.py before changes
- Tries multiple anchor patterns; applies what matches
- Refuses to write if nothing matched (so it won't silently do nothing / or corrupt)
"""

from __future__ import annotations

import re
import sys
import shutil
from datetime import datetime
from pathlib import Path

APP_PATH = Path("app.py")  # run from repo root (same folder as app.py)

def die(msg: str, code: int = 1):
    print(f"\n❌ {msg}\n")
    sys.exit(code)

def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    shutil.copy2(path, bak)
    print(f"✅ Backup created: {bak}")
    return bak

def try_regex_replace(text: str, pattern: str, repl: str, desc: str, flags=0):
    m = re.search(pattern, text, flags)
    if not m:
        return text, False
    new_text, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n != 1:
        die(f"Unexpected replace count ({n}) for: {desc}")
    print(f"✅ Patched: {desc}")
    return new_text, True

def main():
    if not APP_PATH.exists():
        die(f"Could not find {APP_PATH.resolve()}. Run this script in the same folder as app.py.")

    src = APP_PATH.read_text(encoding="utf-8")
    backup_file(APP_PATH)

    applied_any = False

    # ------------------------------------------------------------
    # FIX A — Benchmark Fit tile: replace letter-grade value with numeric score
    #
    # Targets a common pattern like:
    #   tile("Benchmark Fit", beta_fit_grade, f"BetaRel {beta_score:.1f}/100 ...")
    #
    # We capture the score variable used in the BetaRel string and display that as the big value.
    # ------------------------------------------------------------
    fixA_pattern = r"""
tile\(\s*["']Benchmark\s+Fit["']\s*,\s*
(?P<old_val>[^,]+?)\s*,\s*
(?P<sub>
    f?["'][^"']*BetaRel\s*\{(?P<score_var>[A-Za-z_]\w*)[^}]*\}\s*/\s*100[^"']*["'][^)]*
)
\)
"""
    fixA_repl = r"""tile("Benchmark Fit", f"{\g<score_var>:.1f}/100", \g<sub>)"""
    src, ok = try_regex_replace(
        src, fixA_pattern, fixA_repl,
        desc='Scores-only: "Benchmark Fit" tile letter -> 0-100',
        flags=re.MULTILINE | re.VERBOSE | re.DOTALL,
    )
    applied_any = applied_any or ok

    # ------------------------------------------------------------
    # FIX B — Beta Reliability tile: replace letter-grade value with numeric score
    #
    # Targets:
    #   tile("Beta Reliability", beta_grade, f"{beta_score:.1f}/100 · β ...")
    # or variants containing "/100" in the subtitle.
    # ------------------------------------------------------------
    fixB_pattern = r"""
tile\(\s*["']Beta\s+Reliability["']\s*,\s*
(?P<old_val>[^,]+?)\s*,\s*
(?P<sub>
    f?["'][^"']*\{(?P<score_var>[A-Za-z_]\w*)[^}]*\}\s*/\s*100[^"']*["'][^)]*
)
\)
"""
    fixB_repl = r"""tile("Beta Reliability", f"{\g<score_var>:.1f}/100", \g<sub>)"""
    src, ok = try_regex_replace(
        src, fixB_pattern, fixB_repl,
        desc='Scores-only: "Beta Reliability" tile letter -> 0-100',
        flags=re.MULTILINE | re.VERBOSE | re.DOTALL,
    )
    applied_any = applied_any or ok

    # ------------------------------------------------------------
    # FIX C — Analytics Grade tile: label -> Analytics Score, value -> numeric score
    #
    # Targets:
    #   tile("Analytics Grade", grade_letter, f"{risk_score:.1f}/100 RISK")
    # We display big numeric (risk_score/100).
    # ------------------------------------------------------------
    fixC_pattern = r"""
tile\(\s*["']Analytics\s+Grade["']\s*,\s*
(?P<old_val>[^,]+?)\s*,\s*
(?P<sub>
    f?["'][^"']*\{(?P<score_var>[A-Za-z_]\w*)[^}]*\}\s*/\s*100[^"']*["'][^)]*
)
\)
"""
    fixC_repl = r"""tile("Analytics Score", f"{\g<score_var>:.1f}/100", \g<sub>)"""
    src, ok = try_regex_replace(
        src, fixC_pattern, fixC_repl,
        desc='Scores-only: "Analytics Grade" tile -> "Analytics Score" 0-100',
        flags=re.MULTILINE | re.VERBOSE | re.DOTALL,
    )
    applied_any = applied_any or ok

    # ------------------------------------------------------------
    # FIX D — Optional: any st.metric("... Grade", "A", ".../100") -> numeric
    # (Some builds use st.metric instead of tile)
    # ------------------------------------------------------------
    fixD_pattern = r"""
st\.metric\(\s*["']Analytics\s+Grade["']\s*,\s*
(?P<old_val>[^,]+?)\s*,\s*
(?P<sub>
    f?["'][^"']*\{(?P<score_var>[A-Za-z_]\w*)[^}]*\}\s*/\s*100[^"']*["']
)
\s*\)
"""
    fixD_repl = r"""st.metric("Analytics Score", f"{\g<score_var>:.1f}/100", \g<sub>)"""
    src, ok = try_regex_replace(
        src, fixD_pattern, fixD_repl,
        desc='Scores-only: st.metric("Analytics Grade") -> numeric score',
        flags=re.MULTILINE | re.VERBOSE | re.DOTALL,
    )
    applied_any = applied_any or ok

    if not applied_any:
        die(
            "No score/grade anchors matched in your current app.py.\n"
            "This usually means the tile/metric calls are named differently.\n"
            "Send a screenshot of the exact code block that renders the tiles, "
            "or search in app.py for: Benchmark Fit / Beta Reliability / Analytics Grade."
        )

    APP_PATH.write_text(src, encoding="utf-8")
    print("\n✅ DONE. app.py patched to show 0–100 scores (no A–F letters) for the main tiles.\n")
    print("Next: redeploy / rerun Streamlit. If anything looks off, restore from the .bak_ file created above.")

if __name__ == "__main__":
    main()