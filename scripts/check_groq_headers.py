import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

from app.services.groq_client import get_client
from app.services.retrieval_evaluator import EVALUATOR_MODEL

client = get_client()

print("Testing direct Groq API request and inspecting rate limit headers...\n")

try:
    # Make raw request using httpx client inside groq client to capture response headers
    response = client.chat.completions.with_raw_response.create(
        model=EVALUATOR_MODEL,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
        temperature=0.0
    )
    print("STATUS CODE:", response.status_code)
    print("\n--- RATE LIMIT HEADERS ---")
    for k, v in response.headers.items():
        if "ratelimit" in k.lower() or "retry" in k.lower():
            print(f"{k}: {v}")

except Exception as e:
    print("Caught Exception:", type(e).__name__)
    if hasattr(e, "response") and e.response is not None:
        print("HTTP STATUS:", e.response.status_code)
        print("\n--- ERROR RATE LIMIT HEADERS ---")
        for k, v in e.response.headers.items():
            if "ratelimit" in k.lower() or "retry" in k.lower():
                print(f"{k}: {v}")
        print("\n--- ERROR BODY ---")
        print(e.response.text)
    else:
        print("Error details:", str(e))
