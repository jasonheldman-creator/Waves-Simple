# Updated Portfolio Snapshot

class Portfolio:
    def __init__(self, price_book):
        self.price_book = price_book

    def compute_returns(self):
        returns = {}
        for asset, prices in self.price_book.items():
            returns[asset] = (prices[-1] - prices[0]) / prices[0] if prices[0] else None
        return returns

    def snapshot(self):
        return self.compute_returns()


# Example usage
if __name__ == '__main__':
    price_book_example = {
        'Asset A': [100, 110, 105],
        'Asset B': [200, 210, 215],
        'Asset C': [300, 320, 315],
    }

    portfolio = Portfolio(price_book_example)
    print(portfolio.snapshot())  # Outputs returns for each asset in the price book