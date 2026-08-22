"""
scripts/test_live_mouser.py

Strict verification script for Mouser Search API.
Directly tests live API authentication and data retrieval from Mouser's servers.
NO mock fallback allowed — fails loudly if API key is invalid, missing, or rate-limited.
"""

import json
import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded
load_dotenv()

MOUSER_API_KEY = os.getenv("MOUSER_API_KEY", "").strip()
MOUSER_API_URL = "https://api.mouser.com/api/v1/search/partnumber"
TEST_PART_NUMBER = "STM32F407VGT6"


def main():
    print(f"\n=== Live Mouser API Authentication Test ===")
    print(f"Target Part Number: {TEST_PART_NUMBER}")

    if not MOUSER_API_KEY:
        print("\n❌ CRITICAL ERROR: MOUSER_API_KEY is missing or empty in .env file.")
        sys.exit(1)

    print(f"Using MOUSER_API_KEY: {MOUSER_API_KEY[:6]}...{MOUSER_API_KEY[-4:]}")

    payload = {
        "SearchByPartRequest": {
            "mouserPartNumber": TEST_PART_NUMBER,
            "partSearchOptions": "Exact"
        }
    }
    params = {"apiKey": MOUSER_API_KEY}
    headers = {"Content-Type": "application/json"}

    print("\nSending POST request to Mouser API...")
    try:
        response = requests.post(
            MOUSER_API_URL,
            params=params,
            json=payload,
            headers=headers,
            timeout=15,
        )

        print(f"HTTP Status Code: {response.status_code}")

        # Check for HTTP errors (401, 403, 500, etc.)
        if response.status_code != 200:
            print(f"\n❌ MOUSER API AUTHENTICATION / REQUEST FAILED!")
            print(f"Raw Server Response:\n{response.text}")
            sys.exit(1)

        data = response.json() or {}
        errors = data.get("Errors", [])
        if errors:
            print(f"\n❌ MOUSER API RETURNED ERRORS:")
            print(json.dumps(errors, indent=2))
            sys.exit(1)

        search_results = data.get("SearchResults") or {}
        parts = search_results.get("Parts") or []

        if not parts:
            print(f"\n⚠️ API Call Succeeded, but no exact parts were returned for '{TEST_PART_NUMBER}'.")
            print(f"Raw Response Body:\n{json.dumps(data, indent=2)}")
            sys.exit(1)

        part = parts[0]
        mpn = part.get("ManufacturerPartNumber") or part.get("MouserPartNumber", "N/A")
        manufacturer = part.get("Manufacturer", "N/A")
        availability = part.get("Availability", "N/A")
        lifecycle = part.get("LifecycleStatus", "N/A")

        price_breaks = part.get("PriceBreaks", [])
        price_exts = part.get("PriceExts", [])
        unit_price = "N/A"
        if price_breaks:
            unit_price = f"{price_breaks[0].get('Price', 'N/A')} ({price_breaks[0].get('Currency', 'USD')})"
        elif price_exts:
            unit_price = price_exts[0].get("Price", "N/A")

        print("\n✅ LIVE MOUSER API CONNECTION SUCCESSFUL!")
        print("----------------------------------------")
        print(f"Manufacturer Part Number : {mpn}")
        print(f"Manufacturer             : {manufacturer}")
        print(f"Lifecycle Status         : {lifecycle}")
        print(f"Live Availability (Stock): {availability}")
        print(f"Unit Price               : {unit_price}")
        print("----------------------------------------\n")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ NETWORK / REQUEST EXCEPTION ENCOUNTERED:")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
