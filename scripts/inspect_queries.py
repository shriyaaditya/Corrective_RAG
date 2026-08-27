import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
raw_file = root_dir / "data" / "raw_retrieved_chunks.json"

with open(raw_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print("Queries retrieved:")
for idx, q in enumerate(raw_data.keys(), 1):
    print(f"{idx}. {q} (chunks: {len(raw_data[q])})")
