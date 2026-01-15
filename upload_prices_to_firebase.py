#!/usr/bin/env python3
"""
Upload catalog prices from a TSV file to Firebase Realtime Database.
Preserves SKUs as strings (important for leading zeros).
Replaces all existing data in catalog_prices with new data.

Usage:
    python upload_prices_to_firebase.py [prices.txt]

    If no file is specified, defaults to ~/documents/currentprice.txt

Designed to be called automatically from ~/R/currentprice.R

Requirements:
    pip install firebase-admin

Setup:
    1. Go to Firebase Console > Project Settings > Service Accounts
    2. Click "Generate new private key"
    3. Save the JSON file as 'firebase-credentials.json' in the same directory
       (or set FIREBASE_CREDENTIALS environment variable to the path)
"""

import csv
import json
import os
import sys
from pathlib import Path

try:
    import firebase_admin
    from firebase_admin import credentials, db
except ImportError:
    print("Error: firebase-admin not installed.")
    print("Run: pip install firebase-admin")
    sys.exit(1)


# Configuration
DATABASE_URL = "https://dcatalog-image-hosting-default-rtdb.firebaseio.com"
DATABASE_PATH = "/catalog_prices"  # Path in the database to store prices
CREDENTIALS_FILE = "firebase-credentials.json"
DEFAULT_INPUT_FILE = Path.home() / "documents" / "currentprice.txt"


def sanitize_firebase_key(key: str) -> str:
    """
    Sanitize a string to be a valid Firebase Realtime Database key.
    Firebase keys cannot contain: . $ # [ ] / or be empty.
    """
    if not key:
        return "_empty_"
    # Replace invalid characters with underscores
    for char in '.#$[]/':
        key = key.replace(char, '_')
    return key


def load_prices_from_tsv(filepath: str) -> dict:
    """
    Load prices from a TSV file into a dictionary.
    Keys are SKUs (as strings, sanitized for Firebase), values are prices (as floats).
    """
    prices = {}
    skipped = 0
    sanitized = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        # Skip header row
        header = next(reader)
        print(f"Header: {header}")

        for row in reader:
            if len(row) >= 2:
                sku_original = row[0].strip()
                if not sku_original:
                    skipped += 1
                    continue
                sku = sanitize_firebase_key(sku_original)
                if sku != sku_original:
                    sanitized.append((sku_original, sku))
                try:
                    price = float(row[1].strip())
                    prices[sku] = price
                except ValueError:
                    print(f"Warning: Could not parse price for SKU '{sku}': {row[1]}")

    if skipped:
        print(f"Skipped {skipped} rows with empty SKUs")

    if sanitized:
        print(f"Sanitized {len(sanitized)} SKUs:")
        for original, clean in sanitized:
            print(f"  '{original}' -> '{clean}'")

    return prices


def initialize_firebase():
    """Initialize Firebase Admin SDK with credentials."""
    # Check for credentials file
    creds_path = os.environ.get('FIREBASE_CREDENTIALS', CREDENTIALS_FILE)
    
    if not Path(creds_path).exists():
        print(f"Error: Firebase credentials file not found at '{creds_path}'")
        print("\nTo get credentials:")
        print("1. Go to Firebase Console > Project Settings > Service Accounts")
        print("2. Click 'Generate new private key'")
        print(f"3. Save as '{CREDENTIALS_FILE}' in this directory")
        sys.exit(1)
    
    # Initialize the app
    cred = credentials.Certificate(creds_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': DATABASE_URL
    })
    print("Firebase initialized successfully")


def upload_prices(prices: dict, batch_size: int = 2000):
    """Upload prices to Firebase Realtime Database, replacing existing data."""
    ref = db.reference(DATABASE_PATH)

    print(f"Uploading {len(prices)} prices to {DATABASE_PATH}...")

    # First, clear existing data by getting keys and deleting in batches
    print("Clearing existing data...")
    try:
        existing = ref.get(shallow=True)
        if existing:
            existing_keys = list(existing.keys())
            print(f"Found {len(existing_keys)} existing keys to delete...")
            # Delete in batches by setting keys to None
            for i in range(0, len(existing_keys), batch_size):
                batch_keys = existing_keys[i:i + batch_size]
                delete_batch = {key: None for key in batch_keys}
                ref.update(delete_batch)
                print(f"Deleted batch {(i // batch_size) + 1}...")
            print("Existing data cleared.")
    except Exception as e:
        print(f"Note: Could not clear existing data ({e}), proceeding with upload...")

    # Upload in batches to avoid size limits
    items = list(prices.items())
    total_batches = (len(items) + batch_size - 1) // batch_size

    for i in range(0, len(items), batch_size):
        batch = dict(items[i:i + batch_size])
        batch_num = (i // batch_size) + 1
        print(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} items)...")
        ref.update(batch)

    print("Upload complete!")


def preview_database(limit: int = 100):
    """Query and display the first N records from the database to verify upload."""
    ref = db.reference(DATABASE_PATH)

    print(f"\nPreviewing first {limit} records from database...")
    print("-" * 50)

    # Get records with limit
    data = ref.order_by_key().limit_to_first(limit).get()

    if not data:
        print("No data found in database!")
        return

    print(f"{'SKU':<20} {'Price':>10}")
    print("-" * 50)

    for sku, price in data.items():
        print(f"{sku:<20} ${price:>9.2f}")

    print("-" * 50)
    print(f"Showing {len(data)} of {limit} requested records")


def main():
    # Use command-line argument if provided, otherwise default file
    if len(sys.argv) >= 2:
        input_file = Path(sys.argv[1])
    else:
        input_file = DEFAULT_INPUT_FILE

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    # Load prices from TSV
    print(f"Loading prices from '{input_file}'...")
    prices = load_prices_from_tsv(str(input_file))
    print(f"Loaded {len(prices)} prices")

    # Save JSON locally for inspection
    json_output = str(input_file).rsplit('.', 1)[0] + '.json'
    with open(json_output, 'w') as f:
        json.dump(prices, f, indent=2)
    print(f"JSON saved to '{json_output}' for inspection")

    # Initialize Firebase and upload (replaces all existing data)
    initialize_firebase()
    upload_prices(prices)

    print(f"Done! Uploaded {len(prices)} prices to {DATABASE_PATH}")

    # Preview database to verify upload
    preview_database(100)


if __name__ == "__main__":
    main()
