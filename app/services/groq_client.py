"""
utils/groq_client.py
Thin wrapper around the Groq SDK so every module imports from one place.
"""
from groq import Groq
from config import GROQ_API_KEY


def get_client() -> Groq:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    return Groq(api_key=GROQ_API_KEY)


import time


def chat(
    client: Groq,
    model: str,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Send a chat request with 429 rate limit backoff and return assistant text."""
    time.sleep(2.0)
    max_retries = 4
    backoff_seconds = 5.0

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "rate" in err_str or "limit" in err_str) and attempt < max_retries:
                sleep_time = backoff_seconds * (2 ** attempt)
                print(f"\n[GroqClient] 429 Rate limit hit. Retrying in {sleep_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                raise e