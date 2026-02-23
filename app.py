import runpy
from pathlib import Path

# Canonical Streamlit entrypoint launcher
runpy.run_path(
    str(Path(__file__).parent / "app_min.py"),
    run_name="__main__"
)