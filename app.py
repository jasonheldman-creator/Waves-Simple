#!/usr/bin/env python3
"""
Waves-Simple Console

Simple text-based console to explore the Master_Stock_Sheet.cvs.csv universe
and generate Google Finance links for any symbol.
"""

import sys
import os
import textwrap
import pandas as pd

CSV_FILE = "Master_Stock_Sheet.cvs.csv"

# Default exchange to use in Google Finance URL if we don't have one in the CSV
DEFAULT_EXCHANGE = "NASDAQ"  # change to "NYSE" or others if you prefer


def clear_screen():
    """Clear the terminal screen (works on Mac, Linux, Windows)."""
    os.system("cls" if os.name == "nt" else "clear")


def load_universe():
    """Load the stock universe CSV with basic error handling."""
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: Could not find '{CSV_FILE}' in the current folder.")
        print("Make sure the file is in the root of the repo next to app.py.")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_FILE)
        print(f"DEBUG: Loaded DataFrame columns: {list(df.columns)}")
    except Exception as e:
        print(f"ERROR: Failed to read {CSV_FILE}: {e}")
        sys.exit(1)

    # Try to detect the symbol and name columns
    symbol_col = None
    name_col = None

    possible_symbol_cols = ["Symbol", "Ticker", "symbol", "ticker", "SYMBOL"]
    possible_name_cols = ["Name", "Company", "Security Name", "name", "company"]

    for c in possible_symbol_cols:
        if c in df.columns:
            symbol_col = c
            break

    for c in possible_name_cols:
        if c in df.columns:
            name_col = c
            break

    if symbol_col is None:
        print("ERROR: Could not find a symbol column in the CSV.")
        print("Look for a column like 'Symbol' or 'Ticker' and update the code.")
        print("Columns found:", list(df.columns))
        sys.exit(1)

    if name_col is None:
        # Not critical, we can still run without a name column
        print("WARNING: Could not find a company name column.")
        print("I will show symbols only. Columns found:", list(df.columns))

    return df, symbol_col, name_col


def get_exchange_col(df):
    """Try to detect an exchange column (optional)."""
    possible_ex_cols = ["Exchange", "Primary Exchange", "exchange", "EXCHANGE"]
    for c in possible_ex_cols:
        if c in df.columns:
            return c
    return None


def google_finance_url(symbol, exchange=None):
    """
    Build a Google Finance URL.

    If we know the exchange, we use SYMBOL:EXCHANGE format.
    Otherwise we use the DEFAULT_EXCHANGE as a fallback.
    """
    symbol = str(symbol).strip().upper()
    if not symbol:
        return None

    if exchange:
        ex = str(exchange).strip().upper()
    else:
        ex = DEFAULT_EXCHANGE

    # You can tweak this if your exchanges are different
    return f"https://www.google.com/finance/quote/{symbol}:{ex}"


def print_header():
    print("=" * 70)
    print("              WAVES-SIMPLE CONSOLE (Stock Universe)           ")
    print("=" * 70)
    print()


def show_menu():
    print("Choose an option:")
    print("  1) View a random sample of symbols")
    print("  2) Look up a symbol (exact match)")
    print("  3) Search by company name (contains text)")
    print("  4) Quit")
    print()


def pause():
    input("\nPress Enter to continue...")


def view_sample(df, symbol_col, name_col, ex_col):
    clear_screen()
    print_header()
    print("DEBUG: Attempting to sample up to 25 rows.")
    try:
        sample = df.sample(n=min(25, len(df)), random_state=None)
    except ValueError:
        print("No data available in the CSV.")
        return

    print(f"Showing a random sample of up to 25 symbols out of {len(df):,} rows:\n")
    for _, row in sample.iterrows():
        symbol = row[symbol_col]
        name = row[name_col] if name_col else ""
        exchange = row[ex_col] if ex_col else None
        url = google_finance_url(symbol, exchange)

        line = f"{str(symbol):<10}"
        if name:
            line += f" | {name}"
        print(line)
        if url:
            print(f"      Google Finance: {url}")

    pause()


def lookup_symbol(df, symbol_col, name_col, ex_col):
    clear_screen()
    print_header()
    user_symbol = input("Enter a symbol (e.g. AAPL): ").strip()
    print(f"DEBUG: User entered symbol: {user_symbol}")
    if not user_symbol:
        print("No symbol entered.")
        pause()
        return

    mask = df[symbol_col].astype(str).str.upper() == user_symbol.upper()
    matches = df[mask]
    print(f"DEBUG: Found {len(matches)} matches for symbol '{user_symbol}'.")

    if matches.empty:
        print(f"No exact match found for symbol '{user_symbol}'.")
        pause()
        return

    print(f"\nFound {len(matches)} match(es) for '{user_symbol}':\n")

    for _, row in matches.iterrows():
        symbol = row[symbol_col]
        name = row[name_col] if name_col else ""
        exchange = row[ex_col] if ex_col else None
        url = google_finance_url(symbol, exchange)

        print("-" * 70)
        print(f"Symbol   : {symbol}")
        if name:
            print(f"Name     : {name}")
        if ex_col:
            print(f"Exchange : {row[ex_col]}")
        if url:
            print(f"Google   : {url}")

        # Show any other interesting columns
        for col in df.columns:
            if col in (symbol_col, name_col, ex_col):
                continue
            value = row[col]
            if pd.isna(value) or value == "":
                continue
            print(f"{col:<9}: {value}")

    pause()


def search_by_name(df, symbol_col, name_col, ex_col):
    clear_screen()
    print_header()

    if name_col is None:
        print("This CSV does not have a company name column I can detect.")
        print("Search by name is disabled.")
        pause()
        return

    text = input("Enter part of the company name (e.g. 'apple', 'energy'): ").strip()
    print(f"DEBUG: User entered search text: {text}")
    if not text:
        print("No text entered.")
        pause()
        return

    mask = df[name_col].astype(str).str.contains(text, case=False, na=False)
    results = df[mask]
    print(f"DEBUG: Search found {len(results)} matches containing '{text}'.")

    if results.empty:
        print(f"No matches found containing '{text}'.")
        pause()
        return

    print(f"\nFound {len(results)} match(es) containing '{text}':\n")

    # Show up to 30 matches
    for _, row in results.head(30).iterrows():
        symbol = row[symbol_col]
        name = row[name_col]
        exchange = row[ex_col] if ex_col else None
        url = google_finance_url(symbol, exchange)

        print("-" * 70)
        print(f"{symbol:<10} | {name}")
        if url:
            print(f"Google: {url}")

    if len(results) > 30:
        print(f"\n... and {len(results) - 30} more results.")

    pause()


def main():
    df, symbol_col, name_col = load_universe()
    ex_col = get_exchange_col(df)

    while True:
        clear_screen()
        print_header()
        print(f"Loaded rows : {len(df):,}")
        print(f"Symbol col  : {symbol_col}")
        if name_col:
            print(f"Name col    : {name_col}")
        if ex_col:
            print(f"Exchange col: {ex_col}")
        print()

        show_menu()
        choice = input("Enter choice (1-4): ").strip()
        print(f"DEBUG: User entered choice: {choice}")

        if choice == "1":
            view_sample(df, symbol_col, name_col, ex_col)
        elif choice == "2":
            lookup_symbol(df, symbol_col, name_col, ex_col)
        elif choice == "3":
            search_by_name(df, symbol_col, name_col, ex_col)
        elif choice == "4":
            clear_screen()
            print("Goodbye from Waves-Simple Console.")
            print()
            break
        else:
            if not choice:
                print("DEBUG: User entered nothing. Showing error message for empty input.")
            else:
                print(f"DEBUG: Invalid choice '{choice}'. Showing error message.")
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
            pause()


if __name__ == "__main__":
    main()