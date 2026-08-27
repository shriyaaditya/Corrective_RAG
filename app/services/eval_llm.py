"""
app/services/eval_llm.py

Custom Groq LLM Wrapper for DeepEval framework.
Inherits from DeepEvalBaseLLM and uses langchain_groq.ChatGroq.
Implements throttling, exponential backoff, and 429 RateLimitError handling for Groq's free tier.
"""

import asyncio
import os
import re
import time
from typing import Optional, Any
from dotenv import load_dotenv

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
    from langchain_groq import ChatGroq
    _DEEPEVAL_AVAILABLE = True
except ImportError:
    _DEEPEVAL_AVAILABLE = False
    DeepEvalBaseLLM = object

from config import GROQ_API_KEY




import threading

_GLOBAL_LAST_CALL_TIME = 0.0
_GLOBAL_LOCK = threading.Lock()


def _clean_json_output(text: str) -> str:
    """Clean reasoning tags, markdown wrappers, and extract raw valid JSON."""
    if not text:
        return '{"truths": [], "claims": [], "verdicts": [], "reason": ""}'

    # Remove reasoning/thinking tags emitted by DeepSeek / Qwen / Reasoning models
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)

    # Strip markdown code block wrappers
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()

    # Find candidates containing required keys: "truths", "claims", "verdicts", "reason"
    all_json_blocks = re.findall(r"\{[\s\S]*\}", cleaned)
    for cand in all_json_blocks:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict) and len(parsed.keys()) > 0:
                return cand
        except Exception:
            pass

    # Try non-greedy match
    candidates = re.findall(r"\{[\s\S]*?\}", cleaned)
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict) and len(parsed.keys()) > 0:
                return cand
        except Exception:
            pass

    # Fallback to standard extraction
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = cleaned[start:end+1]
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass

    return '{"truths": [], "claims": [], "verdicts": [], "reason": ""}'






    if start != -1:
        candidate = cleaned[start:]
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")

        repaired = candidate
        if open_brackets > 0:
            repaired += "]" * open_brackets
        if open_braces > 0:
            repaired += "}" * open_braces

        try:
            json.loads(repaired)
            return repaired
        except Exception:
            pass

        return candidate

    return cleaned or '{"truths": [], "verdicts": []}'










def _enforce_global_rate_limit(min_interval: float = 5.0) -> None:
    """Thread-safe global rate limiter to guarantee >= min_interval seconds between Groq calls."""
    global _GLOBAL_LAST_CALL_TIME
    with _GLOBAL_LOCK:
        now = time.time()
        elapsed = now - _GLOBAL_LAST_CALL_TIME
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _GLOBAL_LAST_CALL_TIME = time.time()


class GroqEvalLLM(DeepEvalBaseLLM):
    """
    Custom DeepEval judge powered by Groq (qwen/qwen3.6-27b).
    Includes global rate-limit throttling and exponential backoff retry.
    """

    def __init__(
        self,
        model_name: str = "qwen/qwen3.6-27b",
        api_key: Optional[str] = None,
        *args,
        **kwargs
    ):
        load_dotenv()
        self.api_key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.model_name = model_name




    def load_model(self) -> ChatGroq:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set for GroqEvalLLM.")
        return ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.0,
            max_tokens=2048,
        )


    def generate(self, prompt: str) -> str:
        _enforce_global_rate_limit(6.0)
        chat_model = self.load_model()
        max_retries = 6
        backoff_seconds = 10.0

        # Append system directive to guarantee JSON output
        system_prompt = "CRITICAL INSTRUCTION: Respond EXCLUSIVELY in valid JSON format. Do NOT include <think> reasoning blocks, introduction, or explanations.\n\n" + prompt

        for attempt in range(max_retries + 1):
            try:
                res = chat_model.invoke(system_prompt)
                raw_text = str(res.content)
                cleaned = _clean_json_output(raw_text)
                return cleaned
            except Exception as e:
                err_str = str(e).lower()
                if ("429" in err_str or "rate" in err_str or "limit" in err_str) and attempt < max_retries:
                    sleep_time = backoff_seconds * (1.5 ** attempt)
                    print(f"\n[GroqEvalLLM] Rate limit hit (429). Retrying in {sleep_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    raise e

    async def a_generate(self, prompt: str) -> str:
        _enforce_global_rate_limit(6.0)
        chat_model = self.load_model()
        max_retries = 6
        backoff_seconds = 10.0

        system_prompt = "CRITICAL INSTRUCTION: Respond EXCLUSIVELY in valid JSON format. Do NOT include <think> reasoning blocks, introduction, or explanations.\n\n" + prompt

        for attempt in range(max_retries + 1):
            try:
                res = await chat_model.ainvoke(system_prompt)
                raw_text = str(res.content)
                cleaned = _clean_json_output(raw_text)
                return cleaned
            except Exception as e:
                err_str = str(e).lower()
                if ("429" in err_str or "rate" in err_str or "limit" in err_str) and attempt < max_retries:
                    sleep_time = backoff_seconds * (1.5 ** attempt)
                    print(f"\n[GroqEvalLLM] Async Rate limit hit (429). Retrying in {sleep_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(sleep_time)
                else:
                    raise e




    def get_model_name(self) -> str:
        return f"Groq ({self.model_name})"

