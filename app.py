import pandas as pd
from datetime import datetime


def recompute_intraday_portfolio_snapshot(snapshot: dict, price_book: pd.DataFrame) -> dict:
    # Load baseline returns from the snapshot
    baseline_returns = snapshot.get('baseline_returns', {})

    # Calculate returns based on current prices in the price book
    current_prices = price_book['current_price'].to_dict()
    recomputed_returns = {
        'return_1d': (current_prices.get('ticker') - baseline_returns.get('ticker')) / baseline_returns.get('ticker'),
        'return_30d': (current_prices.get('ticker') - baseline_returns.get('ticker')) / baseline_returns.get('ticker'),
        'return_60d': (current_prices.get('ticker') - baseline_returns.get('ticker')) / baseline_returns.get('ticker'),
        'return_365d': (current_prices.get('ticker') - baseline_returns.get('ticker')) / baseline_returns.get('ticker'),
        'alpha_1d': None,  # Calculate alpha if needed
        'alpha_30d': None,  # Calculate alpha if needed
        'alpha_60d': None,  # Calculate alpha if needed
        'alpha_365d': None   # Calculate alpha if needed
    }

    # Overwrite the specified fields in the snapshot
    snapshot.update(recomputed_returns)

    # Add metadata fields
    snapshot['intraday_mode'] = 'HYBRID_SNAPSHOT_RECOMPUTE'
    snapshot['computed_at_utc'] = datetime.utcnow().isoformat()

    return snapshot
