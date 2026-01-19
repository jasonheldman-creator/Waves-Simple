# Updated Portfolio Snapshot Logic

## Changes Made
- Stopped aggregating over an empty wave iterable.
- Replaced aggregation with direct processing over PRICE_BOOK.prices.columns.
- Adjusted portfolio-level metrics to equal-weight or default symbol weighting across symbols.
- Ensured benchmark alphas are computed for PRICE_BOOK, returning numeric values if valid prices exist.

## Impact
These changes maintain the integrity of Portfolio Snapshot by improving performance and reliability without affecting TruthFrame, diagnostics, or Adaptive Intelligence.