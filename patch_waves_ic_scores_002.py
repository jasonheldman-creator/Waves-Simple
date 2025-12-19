"""
patch_waves_ic_scores_002.py
WAVES IC: Remove letter grades (A–F) from tiles and show 0–100 score instead.

Why this works:
- Your letters (A/F) are already baked into the displayed tile value.
- The subtitle already contains the numeric score (e.g., 53.7/100, 90.2/100).
- We patch the *tile renderer* once so ANY tile with a letter grade + "/100" subtitle
  displays numeric score and drops the letter grade automatically.

SAFE BEHAVIOR:
- Creates a timestamped backup of app.py
- Refuses to patch if it can't find the tile() function anchor
- Adds the logic only once (idempotent)
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


def main():
    if not APP_PATH.exists():
        die(f"Could not find {APP_PATH.resolve()}. Run this script in the same folder as app.py.")

    src = APP_PATH.read_text(encoding="utf-8")
    backup_file(APP_PATH)

    # ------------------------------------------------------------
    # Anchor: def tile(...)
    # We'll inject a "grade → score" coercion block near the top of tile().
    # Works regardless of where the grades are produced (engine/UI).
    # ------------------------------------------------------------

    # If already patched, exit cleanly.
    if "WAVES_PATCH_SCORE_TILES_V1" in src:
        print("ℹ️ Score tiles patch already applied (WAVES_PATCH_SCORE_TILES_V1). Nothing to do.")
        return

    # Find tile() function definition
    m = re.search(r"^([ \t]*)def\s+tile\s*\(", src, flags=re.MULTILINE)
    if not m:
        die("Anchor not found: def tile(...). Can't safely patch.")

    indent = m.group(1)
    # Find the colon line ending the def signature (handles multi-line signature)
    # We'll insert right after the first line that ends with "):"
    # More robust: find the first occurrence of "def tile(" then find the next line that ends with "):"
    start = m.start()

    # locate the end of the tile def line block
    lines = src[start:].splitlines(True)
    end_idx = None
    running_len = 0
    for i, ln in enumerate(lines):
        running_len += len(ln)
        # detect end of function signature
        if re.search(r"\)\s*:\s*(#.*)?$", ln):
            end_idx = start + running_len
            break
    if end_idx is None:
        die("Could not locate end of tile() signature (no '):' found).")

    inject = f"""
{indent}    # --- WAVES_PATCH_SCORE_TILES_V1 ---
{indent}    # Auto-convert letter grades (A–F) into numeric 0–100 score when subtitle includes '/100'.
{indent}    # Example:
{indent}    #   tile("Beta Reliability", "F", "53.7/100 · β 0.75 tgt 1.00")
{indent}    # becomes:
{indent}    #   value="53.7" subtitle="53.7/100 · β 0.75 tgt 1.00"
{indent}    try:
{indent}        import re as _re
{indent}        _v = str(value).strip() if value is not None else ""
{indent}        _s = str(subtitle).strip() if subtitle is not None else ""
{indent}        # Detect simple letter grade tokens commonly used in UI tiles.
{indent}        _is_letter_grade = _v in {{"A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"}}
{indent}        if _is_letter_grade:
{indent}            _m = _re.search(r"(\\d+(?:\\.\\d+)?)\\s*/\\s*100", _s)
{indent}            if _m:
{indent}                _num = float(_m.group(1))
{indent}                # Keep the remainder of the subtitle after the '/100' segment.
{indent}                _post = _s[_m.end():].strip()
{indent}                # Normalize separators
{indent}                if _post.startswith(("·", "•", "|")):
{indent}                    _post = _post[1:].strip()
{indent}                value = f"{{_num:.1f}}"
{indent}                subtitle = f"{{_num:.1f}}/100" + (f" · {{_post}}" if _post else "")
{indent}    except Exception:
{indent}        pass
"""

    # Insert inject block immediately after tile() signature
    new_src = src[:end_idx] + inject + src[end_idx:]

    APP_PATH.write_text(new_src, encoding="utf-8")
    print("✅ DONE. Letter-grade tiles will now display numeric 0–100 scores.\n")
    print("Next: reload Streamlit. If anything looks off, restore from the .bak_ file created above.")


if __name__ == "__main__":
    main()