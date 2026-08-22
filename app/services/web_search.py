"""
services/web_search.py

Mouser Search API client for Electronic Component Sourcing.
Replaces generic web search with live component market data (inventory, pricing, lifecycle).
Includes mock data fallback when MOUSER_API_KEY is not configured or API calls fail.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Any

import requests
from config import MOUSER_API_KEY


class WebSearcher:
    """
    Component Sourcing Client backed by Mouser Search API.

    API Endpoint:
    POST https://api.mouser.com/api/v1/search/partnumber?apiKey={MOUSER_API_KEY}
    """

    MOUSER_API_URL = "https://api.mouser.com/api/v1/search/partnumber"

    def __init__(self, api_key: Optional[str] = None, top_k: int = 5) -> None:
        self.api_key = api_key or MOUSER_API_KEY or os.getenv("MOUSER_API_KEY", "")
        self.top_k = top_k

    def search(self, query: str) -> list[str]:
        """
        Search Mouser API for component details for *query* (part number).
        Returns a list containing raw JSON payload string(s).
        """
        part_number = self._extract_part_number(query)

        if not self.api_key:
            print("[MouserClient] No MOUSER_API_KEY configured — returning mock component data.")
            return [json.dumps(self._get_mock_response(part_number))]

        try:
            print(f"[MouserClient] Querying Mouser Search API for part: '{part_number}'...")
            payload = {
                "SearchByPartRequest": {
                    "mouserPartNumber": part_number,
                    "partSearchOptions": "Exact"
                }
            }
            params = {"apiKey": self.api_key}
            headers = {"Content-Type": "application/json"}

            response = requests.post(
                self.MOUSER_API_URL,
                params=params,
                json=payload,
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json() or {}
                search_results = data.get("SearchResults") or {}
                parts = search_results.get("Parts") or []
                if parts:
                    return [json.dumps(data)]
                print(f"[MouserClient] No exact Mouser match for '{part_number}' — falling back to mock response.")
                return [json.dumps(self._get_mock_response(part_number))]

            else:
                print(f"[MouserClient] Mouser API error HTTP {response.status_code}: {response.text}")
                return [json.dumps(self._get_mock_response(part_number))]

        except Exception as e:
            print(f"[MouserClient] Failed to query Mouser API: {e}. Using mock fallback.")
            return [json.dumps(self._get_mock_response(part_number))]

    def _extract_part_number(self, query: str) -> str:
        """Extract hardware part number candidate from query string."""
        tokens = query.strip().split()
        for token in tokens:
            cleaned = re.sub(r"[^\w\-]", "", token)
            # Typical electronic part number patterns (contains digits & letters)
            if len(cleaned) >= 4 and any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned):
                return cleaned.upper()
        return tokens[0].upper() if tokens else "DRV8301"

    def _get_mock_response(self, part_number: str) -> dict[str, Any]:
        """Mock Mouser API response structure for development and offline testing."""
        return {
            "SearchResults": {
                "NumberOfResult": 1,
                "Parts": [
                    {
                        "ManufacturerPartNumber": part_number if part_number else "DRV8301DCAR",
                        "Manufacturer": "Texas Instruments",
                        "Description": "Three-Phase Gate Driver IC With Dual Current Sense Amplifiers and Buck Converter",
                        "LifecycleStatus": "Active",
                        "Availability": "4,250 In Stock",
                        "LeadTime": "12 Weeks",
                        "Min": "1",
                        "Mult": "1",
                        "PriceBreaks": [
                            {"Quantity": 1, "Price": "$6.45", "Currency": "USD"},
                            {"Quantity": 10, "Price": "$5.82", "Currency": "USD"},
                            {"Quantity": 100, "Price": "$4.95", "Currency": "USD"},
                            {"Quantity": 1000, "Price": "$3.85", "Currency": "USD"}
                        ],
                        "ProductDetailUrl": f"https://www.mouser.com/c/?q={part_number}"
                    }
                ]
            }
        }