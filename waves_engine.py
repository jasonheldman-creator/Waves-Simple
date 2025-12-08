name: Run WAVES Engine

on:
  # Run once per market day at 21:00 UTC (3pm US Central) – tweak as needed
  schedule:
    - cron: "0 21 * * 1-5"
  # Allow manual trigger from GitHub UI
  workflow_dispatch:

jobs:
  run-engine:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          else
            pip install yfinance pandas numpy
          fi

      - name: Run WAVES engine
        run: |
          python waves_engine.py

      - name: Commit updated logs
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add logs/performance logs/positions || true
          git commit -m "Update WAVES logs" || echo "No changes to commit"
          git push || echo "Nothing to push"