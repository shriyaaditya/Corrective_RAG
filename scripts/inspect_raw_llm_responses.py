import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

from app.services.groq_client import get_client, chat
from app.services.retrieval_evaluator import _SYSTEM_PROMPT, _USER_TEMPLATE, EVALUATOR_MODEL

# Load dataset
with open(root_dir / "data" / "gatekeeper_eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Find INCORRECT expected items predicted as CORRECT
client = get_client()

incorrect_to_correct = []
for item in dataset:
    if item["expected_label"] == "INCORRECT":
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    question=item["query"],
                    document=item["chunk"][:2000]
                )
            }
        ]
        raw_resp = chat(client, EVALUATOR_MODEL, messages, max_tokens=512, temperature=0.0)
        
        # Check if last extracted label was CORRECT
        upper = raw_resp.upper()
        last_corr = upper.rfind("CORRECT")
        last_inc = upper.rfind("INCORRECT")
        
        if last_corr > last_inc:
            incorrect_to_correct.append({
                "query": item["query"],
                "chunk": item["chunk"],
                "notes": item.get("notes", ""),
                "raw_response": raw_resp
            })

print(f"Captured raw LLM responses for {len(incorrect_to_correct)} INCORRECT->CORRECT mismatches.\n")

for idx, ex in enumerate(incorrect_to_correct[:6], 1):
    print(f"==================================================")
    print(f"SAMPLE {idx}:")
    print(f"QUERY: {ex['query']}")
    print(f"NOTES: {ex['notes']}")
    print(f"CHUNK SNIPPET: {ex['chunk'][:150]}...")
    print(f"\n--- RAW LLM RESPONSE ---")
    print(ex['raw_response'])
    print(f"==================================================\n")
