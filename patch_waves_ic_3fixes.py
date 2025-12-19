"""
patch_waves_ic_grade_numeric.py
WAVES IC: Convert A–F grade DISPLAY to 0–100 score DISPLAY (no engine/math changes).

What this fixes (UI only):
- "Benchmark Fit" big letter (ex: F)  -> "53.7/100"
- "Beta Reliability" big letter (ex: F) -> "53.7/100"
- "Analytics Grade" big letter (ex: A) -> "90.2/100"

SAFE BEHAVIOR:
- Creates a timestamped backup copy of app.py before changes
- Refuses to patch if anchors aren't found (so it won't corrupt your file)
- Only touches the SECOND argument of tile("...", <BIG_VALUE>, <SUBTITLE>)
"""

from __future__ import annotations

import re
import sys
import shutil
from datetime import datetime
from pathlib import Path

APP_PATH = Path("app.py")  # run in same folder as app.py

def die(msg: str, code: int = 1):
    print(f"\n❌ {msg}\n")
    sys.exit(code)

def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    shutil.copy2(path, bak)
    print(f"✅ Backup created: {bak}")
    return bak

def _choose_score_var(subtitle_src: str, candidates: list[str]) -> str | None:
    """
    Pick the first candidate variable name that appears in the subtitle expression.
    """
    for v in candidates:
        # match whole identifier (avoid partial hits)
        if re.search(rf"\b{re.escape(v)}\b", subtitle_src):
            return v
    return None

def patch_tile_big_value_to_numeric(src: str, title: str, score_candidates: list[str]) -> str:
    """
    Find tile("TITLE", BIG, SUBTITLE) and replace BIG with f"{score:.1f}/100"
    using a numeric score variable referenced in SUBTITLE.
    """
    # Match tile("Title", <big_expr>, <subtitle_expr>)
    # Works across newlines; non-greedy.
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

    m = pattern.search(src)
    if not m:
        die(f'Anchor not found: tile("{title}", ...)  (Fix: {title})')

    big_expr = m.group("big")
    sub_expr = m.group("sub")

    score_var = _choose_score_var(sub_expr, score_candidates)
    if not score_var:
        die(
            f'Could not find any known numeric score variable in tile("{title}") subtitle.\n'
            f"Tried: {score_candidates}\n"
            f"Subtitle expression snippet:\n{sub_expr[:400]}"
        )

    # Replace ONLY the BIG expr inside this matched tile call
    new_big = f'f"{{{score_var}:.1f}}/100"'

    # Rebuild the matched tile(...) text with big replaced
    old_call = m.group(0)
    new_call = old_call.replace(big_expr, new_big, 1)

    if old_call == new_call:
        die(f'No change applied for tile("{title}") — unexpected pattern.')

    src2 = src[: m.start()] + new_call + src[m.end() :]
    print(f'✅ Patched: tile("{title}") big value -> {score_var}/100')
    return src2

def main():
    if not APP_PATH.exists():
        die(f"Could not find {APP_PATH.resolve()}. Run this script in the same folder as app.py.")

    src = APP_PATH.read_text(encoding="utf-8")
    backup_file(APP_PATH)

    # --- Fix A–F display to 0–100 display (UI only) ---

    # "Benchmark Fit" big letter -> numeric score (usually beta_rel or beta_score)
    src = patch_tile_big_value_to_numeric(
        src,
        title="Benchmark Fit",
        score_candidates=[
            "beta_rel", "beta_score", "beta_reliability", "betarel",
            "bench_fit", "benchmark_fit", "fit_score"
        ],
    )

    # "Beta Reliability" big letter -> numeric score (usually beta_rel or beta_score)
    src = patch_tile_big_value_to_numeric(
        src,
        title="Beta Reliability",
        score_candidates=[
            "beta_rel", "beta_score", "beta_reliability", "betarel",
            "br_score", "rel_score"
        ],
    )

    # "Analytics Grade" big letter -> numeric score (usually risk_score or risk_pts)
    src = patch_tile_big_value_to_numeric(
        src,
        title="Analytics Grade",
        score_candidates=[
            "risk_score", "risk_pts", "risk_total", "risk_val", "risk",
            "analytics_score", "grade_score"
        ],
    )

    APP_PATH.write_text(src, encoding="utf-8")
    print("\n✅ DONE. app.py updated: grade letters replaced with 0–100 display.\n")
    print("Next: restart Streamlit. If anything looks off, restore the .bak_ file created above.")

if __name__ == "__main__":
    main()