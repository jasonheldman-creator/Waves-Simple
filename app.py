# WAVES Intelligence Streamlit entrypoint
import os

_here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_here, "app_min.py"), encoding="utf-8") as f:
    code = f.read()

exec(compile(code, os.path.join(_here, "app_min.py"), "exec"))
from pathlib import Path

exec(open(Path(__file__).parent / "app_min.py", encoding="utf-8").read(), globals())
