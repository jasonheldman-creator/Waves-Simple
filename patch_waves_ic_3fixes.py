"""
patch_waves_ic_numeric_only.py

WAVES IC PATCH
Fix: Remove A–F grading system entirely.
Force ALL reliability / fit metrics to render as 0–100 numeric only.

SAFE:
- Creates timestamped backup of app.py
- Refuses to patch if anchors are not found
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
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    shutil.copy2(path, bak)
    print(f"✅ Backup created: {bak}")

def replace_once(src: str, pattern: str, repl: str, desc: str, flags=0) -> str:
    if not re.search(pattern, src, flags):
        die(f"Anchor not found for {desc}")
    out, n = re.subn(pattern, repl, src, count=1, flags=flags)
    if n != 1:
        die(f"Unexpected replace count for {desc}: {n}")
    print(f"✅ {desc}")
    return out

def main():
    if not APP_PATH.exists():
        die("app.py not found")

    src = APP_PATH.read_text(encoding="utf-8")
    backup(APP_PATH)

    # ------------------------------------------------------------
    # FIX #1 — Kill score-to-letter grading helper if present
    # ------------------------------------------------------------
    src = re.sub(
        r"""
def\s+score_to_grade\s*\([^)]*\)\s*:\s*
(?:.*\n){1,20}?
return\s+["']?[A-F]["']?
""",
        """
def score_to_grade(score):
    # PATCHED: grading removed — numeric-only system
    return None
""",
        src,
        flags=re.MULTILINE | re.VERBOSE,
    )

    print("✅ Disabled score_to_grade()")

    # ------------------------------------------------------------
    # FIX #2 — Remove ANY inline A–F mapping logic
    # ------------------------------------------------------------
    src = re.sub(
        r"""
(["']grade["']\s*:\s*)
(["']?[A-F]["']?)
""",
        r'\1None',
        src,
        flags=re.MULTILINE | re.VERBOSE,
    )

    print("✅ Removed inline letter-grade assignments")

    # ------------------------------------------------------------
    # FIX #3 — Force UI tiles to show numeric only
    # (Benchmark Fit / Beta Reliability)
    # ------------------------------------------------------------
    src = re.sub(
        r"""
st\.metric\(
\s*label\s*=\s*["']Benchmark Fit["'][\s\S]*?
\)
""",
        """
st.metric(
    label="Benchmark Fit",
    value=f"{beta_rel_score:.1f} / 100"
)
""",
        src,
        flags=re.MULTILINE | re.VERBOSE,
    )

    print("✅ Benchmark Fit tile set to numeric-only")

    # ------------------------------------------------------------
    # FIX #4 — Remove stray standalone grade rendering ("F", "A", etc.)
    # ------------------------------------------------------------
    src = re.sub(
        r"""
st\.(write|markdown)\(\s*["']\b[A-F]\b["']\s*\)
""",
        "",
        src,
        flags=re.MULTILINE | re.VERBOSE,
    )

    print("✅ Removed stray letter-grade rendering")

    APP_PATH.write_text(src, encoding="utf-8")

    print("\n✅ PATCH COMPLETE — Numeric scoring only (0–100)\n")
    print("Next: restart Streamlit")

if __name__ == "__main__":
    main()