"""
WAVES Streamlit Entrypoint (SAFE MODE)

Loads app_min.py directly so syntax errors
are visible and do not get hidden by exec().
"""

import runpy
import pathlib

APP_FILE = pathlib.Path(__file__).parent / "app_min.py"

runpy.run_path(str(APP_FILE))