import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
with open(root_dir / "data" / "gatekeeper_mismatches.json", "r", encoding="utf-8") as f:
    mismatches = json.load(f)

incorrect_to_ambiguous = [m for m in mismatches if m["expected"] == "INCORRECT" and m["predicted"] == "AMBIGUOUS"]

print(f"Total INCORRECT -> AMBIGUOUS mismatches: {len(incorrect_to_ambiguous)}\n")

for idx, m in enumerate(incorrect_to_ambiguous, 1):
    print(f"[{idx}] Query: {m['query']}")
    print(f"    Chunk Source: {m['chunk'][:60]}...")
    print(f"    Note: {m['notes']}\n")
