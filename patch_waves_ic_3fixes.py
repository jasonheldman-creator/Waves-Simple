"""
patch_waves_ic_grade_numeric_v2.py
WAVES IC: Remove A–F letter display and show 0–100 score everywhere we can, safely.

Targets (UI only):
- tile("Benchmark Fit", "F", "BetaRel 53.7/100 ...")  -> tile(..., "53.7/100", ...)
- tile("Beta Reliability", "F", "53.7/100 ...")      -> tile(..., "53.7/100", ...)
- tile("Analytics Grade", "A", "90.2/100 RISK")      -> tile(..., "90.2/100", ...)

Also targets common text patterns like:
- "Beta Reliability: F (53.7/100)" -> "Beta Reliability: 53.7/100"
- "Benchmark Fit: F (53.7/100)"    -> "Benchmark Fit: 53.7/100"

SAFE BEHAVIOR:
- Creates timestamped backup of app.py
- Refuses to write if it can’t find anything to change (so we don’t “half patch”)
"""

from __future__ import annotations

import re
import sys
import shutil
from datetime import datetime
from pathlib import Path

APP_PATH = Path("app.py")

def die(msg: str, code: int = 1):
    print(f"\n❌ {msg}\n")
    sys.exit(code)

def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    shutil.copy2(path, bak)
    print(f"✅ Backup created: {bak}")
    return bak

def _first_score_in_text(expr: str) -> str | None:
    """
    Try to extract a "53.7/100" or f"{beta_rel:.1f}/100" style snippet from text.
    Returns an expression to use as the big value.
    """
    # f"{var:.1f}/100" inside an f-string / format
    m = re.search(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*\.?\d*f\}", expr)
    if m:
        var = m.group(1)
        return f'f"{{{var}:.1f}}/100"'

    # explicit var mention like beta_rel or risk_score (even if not in braces)
    for var in ["beta_rel", "beta_score", "risk_score", "risk_pts", "analytics_score", "fit_score"]:
        if re.search(rf"\b{var}\b", expr):
            return f'f"{{{var}:.1f}}/100"'

    # literal "53.7/100"
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*/\s*100", expr)
    if m2:
        val = m2.group(1)
        return f'"{val}/100"'

    return None

def patch_all_tile_calls(src: str, title: str) -> tuple[str, int]:
    """
    Patch ALL tile("title", BIG, SUB) occurrences by replacing BIG with a 0–100 string
    derived from SUB whenever possible.
    """
    # Capture tile("Title", big_expr, sub_expr) across newlines
    pattern = re.compile(
        rf"""
        tile\(\s*
            (?P<q>["']){re.escape(title)}(?P=q)\s*,\s*
            (?P<big>.*?)\s*,\s*
            (?P<sub>.*?)
        \)\s*
        """,
        re.VERBOSE | re.DOTALL,
    )

    out = []
    last = 0
    changed = 0

    for m in pattern.finditer(src):
        out.append(src[last:m.start()])
        old_call = m.group(0)
        big_expr = m.group("big")
        sub_expr = m.group("sub")

        new_big = _first_score_in_text(sub_expr)
        if not new_big:
            # If we can’t infer a numeric score from subtitle, leave unchanged (safe)
            out.append(old_call)
            last = m.end()
            continue

        new_call = old_call.replace(big_expr, new_big, 1)
        if new_call != old_call:
            changed += 1
            out.append(new_call)
        else:
            out.append(old_call)

        last = m.end()

    out.append(src[last:])
    return ("".join(out), changed)

def patch_grade_text_patterns(src: str) -> tuple[str, int]:
    """
    Replace common "X: F (53.7/100)" patterns.
    """
    patterns = [
        # Beta Reliability: F (53.7/100) -> Beta Reliability: 53.7/100
        (re.compile(r"(Beta\s*Reliability\s*:\s*)([A-F][+-]?)\s*\(\s*(\d+(?:\.\d+)?)\s*/\s*100\s*\)", re.IGNORECASE),
         r"\1\3/100"),
        # Benchmark Fit: F (53.7/100) -> Benchmark Fit: 53.7/100
        (re.compile(r"(Benchmark\s*Fit\s*:\s*)([A-F][+-]?)\s*\(\s*(\d+(?:\.\d+)?)\s*/\s*100\s*\)", re.IGNORECASE),
         r"\1\3/100"),
        # Analytics Grade: A (90.2/100) -> Analytics Score: 90.2/100  (optional rename)
        (re.compile(r"(Analytics\s*Grade\s*:\s*)([A-F][+-]?)\s*\(\s*(\d+(?:\.\d+)?)\s*/\s*100\s*\)", re.IGNORECASE),
         r"Analytics Score: \3/100"),
    ]

    total = 0
    for rx, repl in patterns:
        src, n = rx.subn(repl, src)
        total += n
    return src, total

def main():
    if not APP_PATH.exists():
        die(f"Could not find {APP_PATH.resolve()}. Run this in the same folder as app.py.")

    src = APP_PATH.read_text(encoding="utf-8")
    backup_file(APP_PATH)

    total_changes = 0

    # Patch ALL occurrences (Executive IC One-Pager + IC Tiles + anywhere else)
    for t in ["Benchmark Fit", "Beta Reliability", "Analytics Grade"]:
        src, n = patch_all_tile_calls(src, t)
        print(f'✅ tile("{t}") patched occurrences: {n}')
        total_changes += n

    # Patch common text patterns that still show A–F
    src, ntext = patch_grade_text_patterns(src)
    if ntext:
        print(f"✅ Text A–F patterns patched: {ntext}")
    total_changes += ntext

    if total_changes == 0:
        die(
            "No changes were applied.\n"
            "That means your UI is not using tile(\"Benchmark Fit\"/\"Beta Reliability\"/\"Analytics Grade\") "
            "or the strings differ.\n"
            "Next step: search app.py for 'Benchmark Fit' / 'Beta Reliability' / 'Analytics Grade' and we’ll target the exact pattern."
        )

    APP_PATH.write_text(src, encoding="utf-8")
    print(f"\n✅ DONE. Total changes applied: {total_changes}\n")
    print("Next: restart Streamlit / redeploy so the new app.py is actually loaded.")

if __name__ == "__main__":
    main()