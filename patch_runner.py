# patch_runner.py
# Streamlit-safe patch executor (iPhone friendly)

import streamlit as st
from pathlib import Path
import shutil
from datetime import datetime
import re
import sys

APP = Path("app.py")

st.title("🧩 WAVES Patch Runner")

if not APP.exists():
    st.error("app.py not found in repo root")
    st.stop()

def backup(path: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(f".bak_{ts}.py")
    shutil.copy2(path, bak)
    return bak

def apply_patch(src: str):
    """
    PATCH: Remove A-F grading for Beta Reliability
    Replace with numeric-only display
    """

    pattern = r'''
    tile\(
        \s*["']Beta\s+Reliability["']\s*,\s*
        ["'][A-F]["']\s*,\s*
        ([^)]*)
    \)
    '''

    repl = r'''
tile(
    "Beta Reliability",
    f"{beta_rel:.1f}/100",
    f"β {beta:.2f} vs tgt {beta_tgt:.2f}"
)
'''

    new_src, n = re.subn(
        pattern,
        repl,
        src,
        flags=re.VERBOSE | re.MULTILINE
    )

    return new_src, n

if st.button("🚀 RUN SCORE PATCH"):
    src = APP.read_text(encoding="utf-8")
    bak = backup(APP)

    new_src, n = apply_patch(src)

    if n == 0:
        st.error("❌ Patch anchor not found - app.py unchanged")
    else:
        APP.write_text(new_src, encoding="utf-8")
        st.success(f"✅ Patch applied ({n} replacement)")
        st.caption(f"Backup created: {bak}")

st.caption("Safe: runs once · creates backup · no globals · no app rewrite")