import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
raw_file = root_dir / "data" / "raw_retrieved_chunks.json"

with open(raw_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

for q, chunks in raw_data.items():
    print(f"\n==========================================")
    print(f"QUERY: {q}")
    for idx, c in enumerate(chunks, 1):
        print(f"--- Chunk {idx} ---")
        print(c[:300].replace("\n", " "))
