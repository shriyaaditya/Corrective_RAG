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


def chat(
    client: Groq,
    model: str,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Send a chat request and return the assistant text."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()