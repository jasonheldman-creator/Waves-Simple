def compute_portfolio_snapshot(price_book):
    if not price_book:
        return "N/A"

    total_value = 0.0
    for item in price_book:
        if 'value' in item:
            total_value += item['value']

    return total_value if total_value > 0 else 0.0
