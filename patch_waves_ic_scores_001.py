"""
patch_waves_ic_scores_001.py

WAVES IC — Score Normalization Patch
Removes letter grading (A–F) and standardizes on numeric 0–100 scores only.

Targets:
• Benchmark Fit (Beta Reliability) — IC Summary
• Beta Reliability — IC Tiles

SAFE:
• Creates timestamped backup of app.py
• Refuses to patch if anchors are not found
"""

from __future__ import annotations

import re
import sys
import shutil
from datetime import datetime
from pathlib import Path

APP_PATH = Path("app.py")

def die(msg: str):
    print(f"\n❌ {msg}\n")
    sys.exit(1)

def backup(path: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_scores_{ts}")
    shutil.copy2(path, bak)
    print(f"✅ Backup created: {bak}")

def replace_once(src: str, pattern: str, repl: str, desc: str) -> str:
    new, n = re.subn(pattern, repl, src, count=1, flags=re.MULTILINE | re.VERBOSE)
    if n != 1:
        die(f"Anchor not found or multiple matches for: {desc}")
    print(f"✅ Patched: {desc}")
    return new

def main():
    if not APP_PATH.exists():
        die("app.py not found. Run this in the same directory.")

    src = APP_PATH.read_text(encoding="utf-8")
    backup(APP_PATH)

    # ------------------------------------------------------------------
    # FIX 1 — IC SUMMARY: Remove letter grade from "Benchmark Fit"
    #
    # BEFORE:
    #   st.markdown("### Benchmark Fit")
    #   st.markdown("**F**")
    #   st.caption("BetaRel 53.7/100")
    #
    # AFTER:
    #   st.markdown("### Benchmark Fit")
    #   st.metric("Beta Reliability", f"{beta_rel:.1f}/100")
    # ------------------------------------------------------------------
    fix1_pattern = r"""
st\.markdown\(\s*["']###\s*Benchmark\s*Fit["']\s*\)\s*\n
\s*st\.markdown\(\s*["']\*\*[A-F]\*\*["']\s*\)\s*\n
\s*st\.caption\(\s*f?["']BetaRel.*?["']\s*\)
"""

    fix1_repl = """
st.markdown("### Benchmark Fit")
st.metric("Beta Reliability", f"{beta_rel:.1f}/100")
"""

    src = replace_once(
        src,
        fix1_pattern,
        fix1_repl,
        "IC Summary — remove A–F grade, show numeric Beta Reliability",
    )

    # ------------------------------------------------------------------
    # FIX 2 — IC TILES: Remove letter grade from Beta Reliability tile
    #
    # BEFORE:
    #   tile("Beta Reliability", "F", "53.7/100 • β 0.75 tgt 1.00")
    #
    # AFTER:
    #   tile("Beta Reliability", "53.7/100", "β 0.75 vs tgt 1.00")
    # ------------------------------------------------------------------
    fix2_pattern = r"""
tile\(\s*
["']Beta\s*Reliability["']\s*,\s*
["'][A-F]["']\s*,\s*
f?["']([^"']*?/100).*?["']\s*
\)
"""

    fix2_repl = r"""
tile(
    "Beta Reliability",
    f"{beta_rel:.1f}/100",
    f"β {beta:.2f} vs tgt {beta_tgt:.2f}"
)
"""

    src = replace_once(
        src,
        fix2_pattern,
        fix2_repl,
        "IC Tiles — remove A–F grade, numeric Beta Reliability only",
    )

    APP_PATH.write_text(src, encoding="utf-8")
    print("\n✅ DONE — Numeric scoring enforced. Letter grades fully removed.\n")
    print("Re-run Streamlit. Restore anytime from the .bak_scores_ file.")

if __name__ == "__main__":
    main()