# WAVES Intelligence Streamlit entrypoint

with open("app_min.py", encoding="utf-8") as f:
    code = f.read()

exec(code)