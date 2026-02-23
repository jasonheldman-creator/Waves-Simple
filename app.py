# Canonical Streamlit entrypoint — UTF-8 safe launcher

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).parent / "app_min.py"),
    run_name="__main__"
)