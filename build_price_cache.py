name: Update Price Cache

on:
  schedule:
    # Runs daily at 2:00 AM UTC (after US market close)
    - cron: "0 2 * * *"
  workflow_dispatch:

jobs:
  build-cache:
    name: Build Price Cache
    runs-on: ubuntu-latest

    permissions:
      contents: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Build price cache (includes validation)
        run: |
          python build_price_cache.py

      - name: Commit updated cache files
        run: |
          git config user.name "github-actions"
          git config user.email "github-actions@github.com"

          git add data/cache/prices_cache.parquet data/cache/prices_cache_meta.json

          if git diff --cached --quiet; then
            echo "No cache changes to commit."
          else
            git commit -m "Update price cache"
            git push
          fi