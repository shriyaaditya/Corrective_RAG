"""
services/knowledge_refiner.py

Payload Pruner for Electronic Component Sourcing.
Prunes bloated JSON responses from Mouser / Sourcing APIs down to essential market fields:
- Part Number
- Manufacturer
- Lifecycle Status
- Availability (Stock)
- Unit Price (PriceExts / PriceBreaks)
- Lead Time
"""

from __future__ import annotations

import json
from typing import Optional, Any
from services.retrieval_evaluator import RetrievalEvaluator
from services.text_utils import split_into_strips
from config import STRIP_SIZE_SENTENCES, STRIP_FILTER_THRESHOLD, TOP_K_STRIPS


class KnowledgeRefiner:
    """
    Refines retrieved documents (unstructured text) or API payloads (JSON)
    into compact, high-signal context for downstream LLM generation.
    """

    def __init__(
        self,
        evaluator: Optional[RetrievalEvaluator] = None,
        strip_size: int = STRIP_SIZE_SENTENCES,
        filter_threshold: float = STRIP_FILTER_THRESHOLD,
        top_k: int = TOP_K_STRIPS,
    ) -> None:
        self.evaluator = evaluator
        self.strip_size = strip_size
        self.filter_threshold = filter_threshold
        self.top_k = top_k

    def refine(self, query: str, documents: list[str]) -> str:
        """
        Refine a list of document strings or API JSON payload strings.
        Determines if input is JSON API payload or unstructured text.
        """
        if not documents:
            return ""

        # Check if the first document is a JSON API payload (e.g., from Mouser)
        for doc in documents:
            doc_str = doc.strip()
            if doc_str.startswith("{") and doc_str.endswith("}"):
                try:
                    data = json.loads(doc_str)
                    pruned_summary = self._prune_mouser_json(data)
                    if pruned_summary:
                        return pruned_summary
                except json.JSONDecodeError:
                    pass

        # Fallback to unstructured text strip decompose-filter-recompose
        return self._refine_unstructured(query, documents)

    def _prune_mouser_json(self, data: dict[str, Any]) -> str:
        """
        Sanitizes and prunes bloated Mouser API JSON payloads down to essential fields:
        Part Number, Manufacturer, Lifecycle Status, Stock, Unit Price, and Lead Time.
        """
        parts = data.get("SearchResults", {}).get("Parts", [])
        if not parts:
            return ""

        pruned_parts = []
        for p in parts[:self.top_k]:
            mpn = p.get("ManufacturerPartNumber") or p.get("MouserPartNumber", "N/A")
            manufacturer = p.get("Manufacturer", "N/A")
            lifecycle = p.get("LifecycleStatus", "Unknown")
            stock = p.get("Availability", "Unknown")
            lead_time = p.get("LeadTime", "N/A")

            # Extract unit price (1-qty price or first price break)
            price_breaks = p.get("PriceBreaks", [])
            price_exts = p.get("PriceExts", [])
            unit_price = "N/A"
            if price_breaks:
                unit_price = f"{price_breaks[0].get('Price', 'N/A')} ({price_breaks[0].get('Currency', 'USD')})"
            elif price_exts:
                unit_price = price_exts[0].get("Price", "N/A")

            pruned_part = {
                "Part Number": mpn,
                "Manufacturer": manufacturer,
                "Lifecycle Status": lifecycle,
                "Availability (Stock)": stock,
                "Unit Price": unit_price,
                "Lead Time": lead_time,
            }
            pruned_parts.append(pruned_part)

        return json.dumps(pruned_parts, indent=2)

    def _refine_unstructured(self, query: str, documents: list[str]) -> str:
        """Sentence-level decompose-filter-recompose algorithm for unstructured text."""
        all_strips: list[str] = []
        for doc in documents:
            strips = split_into_strips(doc, strip_size=self.strip_size)
            all_strips.extend(strips)

        if not all_strips:
            return " ".join(documents)

        if self.evaluator:
            relevant_strips = self.evaluator.filter_strips(
                query=query,
                strips=all_strips,
                threshold=self.filter_threshold,
                top_k=self.top_k,
            )
            return " ".join(relevant_strips)

        return " ".join(all_strips[:self.top_k])