"""
scripts/score_gatekeeper.py

Scores the CRAG Retrieval Evaluator (Gatekeeper) against the golden
dataset (data/gatekeeper_eval_dataset.json). Includes persistent
file-based caching in data/gatekeeper_score_cache.json.

Usage:
    PYTHONPATH=$(pwd):$(pwd)/app python3 scripts/score_gatekeeper.py
    PYTHONPATH=$(pwd):$(pwd)/app python3 scripts/score_gatekeeper.py --no-cache
    PYTHONPATH=$(pwd):$(pwd)/app python3 scripts/score_gatekeeper.py --clear-cache
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root and app directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

from app.services.groq_client import get_client, chat
from app.services.retrieval_evaluator import (
    RetrievalEvaluator,
    _parse_score,
    _LABEL_TO_SCORE,
    _SYSTEM_PROMPT,
    _USER_TEMPLATE,
    EVALUATOR_MODEL,
    PROMPT_VERSION,
    truncate,
)

LABELS = ["CORRECT", "AMBIGUOUS", "INCORRECT"]
DATASET_PATH = root_dir / "data" / "gatekeeper_eval_dataset.json"
CACHE_PATH = root_dir / "data" / "gatekeeper_score_cache.json"

_SCORE_TO_LABEL = {
    1.0: "CORRECT",
    0.0: "AMBIGUOUS",
    -1.0: "INCORRECT",
}


def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CACHE_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(temp_path, CACHE_PATH)



def make_cache_key(query: str, chunk: str, model: str, prompt_version: str) -> str:
    raw_str = f"{query}||{chunk}||{model}||{prompt_version}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data["data"]


def confusion_matrix(rows: list[dict]) -> dict:
    """rows: [{expected, predicted}, ...] -> matrix[expected][predicted] = count"""
    matrix = {e: {p: 0 for p in LABELS} for e in LABELS}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1
    return matrix


def per_class_metrics(matrix: dict) -> dict:
    metrics = {}
    for label in LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label][p] for p in LABELS if p != label)
        fp = sum(matrix[e][label] for e in LABELS if e != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": tp + fn,
        }
    return metrics


def print_report(matrix: dict, metrics: dict, mismatches: list[dict]):
    print("\n=== Confusion Matrix (rows=expected, cols=predicted) ===")
    header = " " * 14 + "".join(f"{l:>12}" for l in LABELS)
    print(header)
    for e in LABELS:
        row = "".join(f"{matrix[e][p]:>12}" for p in LABELS)
        print(f"{e:<14}{row}")

    print("\n=== Per-Class Metrics ===")
    print(f"{'Label':<12}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for label in LABELS:
        m = metrics[label]
        print(
            f"{label:<12}{m['precision']:>10}{m['recall']:>10}{m['f1']:>10}{m['support']:>10}"
        )

    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[l][l] for l in LABELS)
    overall_acc = correct / total if total else 0.0
    print(f"\nOverall accuracy: {overall_acc:.3f} ({correct}/{total})")

    from app.services.retrieval_evaluator import PARSING_FAILURES_COUNT
    print(f"Parsing failures encountered: {PARSING_FAILURES_COUNT}")

    if mismatches:
        print(f"\n=== Mismatches ({len(mismatches)}) ===")
        for m in mismatches:
            print(f"\n  Query: {m['query'][:100]}")
            print(f"  Expected: {m['expected']}  |  Predicted: {m['predicted']}")
            print(f"  Chunk: {m['chunk'][:150]}...")


def main():
    parser = argparse.ArgumentParser(description="Score Gatekeeper against Golden Dataset")
    parser.add_argument("--no-cache", action="store_true", help="Bypass score cache and re-call LLM")
    parser.add_argument("--clear-cache", action="store_true", help="Wipe cache file before running")
    args = parser.parse_args()

    if args.clear_cache:
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
            print(f"Cleared cache file at {CACHE_PATH}")
        else:
            print("No cache file found to clear.")

    cache = {} if args.no_cache else load_cache()

    dataset = load_dataset(DATASET_PATH)
    client = get_client()
    evaluator = RetrievalEvaluator(client=client)

    rows = []
    mismatches = []
    cached_count = 0
    fresh_count = 0

    print(f"Scoring {len(dataset)} items against RetrievalEvaluator (Model: {EVALUATOR_MODEL}, Prompt: {PROMPT_VERSION})...")
    print(f"Cache Status: {'Bypassed' if args.no_cache else f'Active ({len(cache)} entries loaded)'}\n")

    for idx, item in enumerate(dataset, 1):
        query = item["query"]
        chunk = item["chunk"]
        expected = item["expected_label"].upper()

        key = make_cache_key(query, chunk, EVALUATOR_MODEL, PROMPT_VERSION)

        if not args.no_cache and key in cache:
            predicted = cache[key]["predicted"].upper()
            raw_response = cache[key].get("raw_response", "")
            cached_count += 1
            source_tag = "CACHE"
        else:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(
                        question=query,
                        document=truncate(chunk, max_chars=2000),
                    ),
                },
            ]
            raw_response = chat(client, EVALUATOR_MODEL, messages, max_tokens=512, temperature=0.0)
            score = _parse_score(raw_response)
            predicted = _SCORE_TO_LABEL.get(score, "AMBIGUOUS").upper()
            fresh_count += 1
            source_tag = "LLM  "

            # Update cache incrementally
            cache[key] = {
                "query": query[:100],
                "predicted": predicted,
                "raw_response": raw_response,
                "model": EVALUATOR_MODEL,
                "prompt_version": PROMPT_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_cache(cache)

        rows.append({"expected": expected, "predicted": predicted})

        if expected != predicted:
            mismatches.append(
                {
                    "query": query,
                    "chunk": chunk,
                    "expected": expected,
                    "predicted": predicted,
                    "notes": item.get("notes", ""),
                    "raw_response": raw_response
                }
            )

        print(f"[{idx:2d}/{len(dataset)}] [{source_tag}] Expected: {expected:<10} | Predicted: {predicted:<10} | Match: {expected == predicted}")

    matrix = confusion_matrix(rows)
    metrics = per_class_metrics(matrix)
    print_report(matrix, metrics, mismatches)

    print(f"\n--- Execution Cache Summary ---")
    print(f"  Served from Cache : {cached_count}")
    print(f"  Freshly Scored LLM: {fresh_count}")
    print(f"  Total Cache Size  : {len(cache)} entries in {CACHE_PATH}")

    # Dump mismatches for review
    out_path = root_dir / "data" / "gatekeeper_mismatches.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mismatches, f, indent=2)
    print(f"\nMismatches written to {out_path}")


if __name__ == "__main__":
    main()
