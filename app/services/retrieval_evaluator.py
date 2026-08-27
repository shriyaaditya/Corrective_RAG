"""
evaluator/retrieval_evaluator.py

Replaces the T5-large retrieval evaluator from the paper with a Groq-hosted
LLM.  The evaluator scores each (query, document) pair and returns a float
in [-1, 1].

Scoring rubric
--------------
  1.0  → document clearly answers the question          (CORRECT)
  0.0  → document is partially / tangentially relevant  (AMBIGUOUS)
 -1.0  → document is unrelated / misleading             (INCORRECT)
"""

from __future__ import annotations

from groq import Groq

from config import EVALUATOR_MODEL, UPPER_THRESHOLD, LOWER_THRESHOLD
from services.groq_client import chat
from services.text_utils import truncate


import logging
import re

logger = logging.getLogger(__name__)

# Track parsing failure count globally
PARSING_FAILURES_COUNT = 0

# Prompt version constant to invalidate score cache when prompt/examples change
PROMPT_VERSION = "v1.0-revert"


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a strict Hardware Engineering & Procurement Gatekeeper for a Design for Manufacturing (DFM) and Electronic Sourcing Copilot.

Your task is to evaluate whether a retrieved document (PDF/CSV text) adequately and reliably answers the user's question, while respecting domain volatility constraints.

RULES & RUBRIC:
1. CORRECT:
   - Output CORRECT if the document contains clear, precise, and sufficient engineering facts/parameters directly answering the static question (e.g., reflow temperatures, ramp rates, dimensions, layer stackups).

2. AMBIGUOUS:
   - Output AMBIGUOUS if the query explicitly asks for volatile live market data (global stock availability, unit pricing, lead times, distributor inventory).
   - Also output AMBIGUOUS if the document is only partially relevant or incomplete for the question.

3. INCORRECT:
   - Output INCORRECT if the document is completely unrelated, misleading, or does NOT contain the specific information asked by the user.

CRITICAL FORMATTING INSTRUCTION:
Do NOT include reasoning, explanations, or thinking blocks. Respond EXCLUSIVELY with the single token: CORRECT, AMBIGUOUS, or INCORRECT.

EXAMPLES:
Example 1:
QUESTION: What is the maximum body temperature for a BGA-256 package during SAC305 reflow?
DOCUMENT: Peak Package Temperature Limit of 245 °C. Maximum delta Across BGA Package Body: <= 5 °C.
Gatekeeper Verdict: CORRECT

Example 2:
QUESTION: What is the global stock and unit price for part STM32F407VGT6?
DOCUMENT: IPN-MCU-002 | STM32F407VGT6 | STMicroelectronics | 32-bit ARM MCU 168MHz LQFP-100
Gatekeeper Verdict: AMBIGUOUS

Example 3:
QUESTION: What is the lead time for Xilinx Artix-7 FPGA XC7A35T-1FTG256C?
DOCUMENT: Reflow Profile Parameters: Preheat Zone 150 °C to 200 °C, Soak Duration 60s to 120s.
Gatekeeper Verdict: INCORRECT"""


# Preserve v1.1 expanded 5-shot prompt for future reference
_SYSTEM_PROMPT_V1_1 = """You are a strict Hardware Engineering & Procurement Gatekeeper for a Design for Manufacturing (DFM) and Electronic Sourcing Copilot.

Your task is to evaluate whether a retrieved document (PDF/CSV text) adequately and reliably answers the user's question, while respecting domain volatility constraints.

RULES & RUBRIC:
1. CORRECT:
   - Output CORRECT if the document contains clear, precise, and sufficient engineering facts/parameters directly answering the static question (e.g., reflow temperatures, ramp rates, dimensions, layer stackups).

2. AMBIGUOUS:
   - Output AMBIGUOUS if the query explicitly asks for volatile live market data (global stock availability, unit pricing, lead times, distributor inventory).
   - Output AMBIGUOUS if the document mentions the target component/topic and provides partial context, options, or high-level overview, but is missing the specific register bit, exact value, layout rule, or parameter requested.

3. INCORRECT:
   - Output INCORRECT if the document discusses a completely different component or topic (e.g., MCU datasheet retrieved for SMT reflow query).
   - Output INCORRECT if the document is entirely unrelated or misleading.

CRITICAL FORMATTING INSTRUCTION:
Do NOT include reasoning, explanations, or thinking blocks. Respond EXCLUSIVELY with the single token: CORRECT, AMBIGUOUS, or INCORRECT.

EXAMPLES:
Example 1 (CORRECT - Fully answers question):
QUESTION: What is the maximum body temperature for a BGA-256 package during SAC305 reflow?
DOCUMENT: Peak Package Temperature Limit of 245 °C. Maximum delta Across BGA Package Body: <= 5 °C.
Gatekeeper Verdict: CORRECT

Example 2 (AMBIGUOUS - Volatile market query):
QUESTION: What is the global stock and unit price for part STM32F407VGT6?
DOCUMENT: IPN-MCU-002 | STM32F407VGT6 | STMicroelectronics | 32-bit ARM MCU 168MHz LQFP-100
Gatekeeper Verdict: AMBIGUOUS

Example 3 (AMBIGUOUS - Target component present, but specific parameter missing):
QUESTION: What is the recommended gain setting for the DRV8301 current shunt amplifier to achieve a 40 V/V output?
DOCUMENT: DRV8301 Features: Dual Integrated Current Shunt Amplifiers With Adjustable Gain and Offset. 6-V to 60-V Operating Supply Voltage Range.
Gatekeeper Verdict: AMBIGUOUS

Example 4 (AMBIGUOUS - Target component package listed, but pinout/thermal layout details missing):
QUESTION: What is the pinout configuration and thermal pad ground requirement for DRV8301?
DOCUMENT: Device Package Information: PART NUMBER: DRV8301, PACKAGE: HTSSOP (56), BODY SIZE: 14.00 mm x 8.10 mm.
Gatekeeper Verdict: AMBIGUOUS

Example 5 (INCORRECT - Wrong component / completely off-topic):
QUESTION: What is the operating voltage range for DRV8301 gate driver?
DOCUMENT: STM32F405/STM32F407 Datasheet: Operating Voltage Range: 1.8 V to 3.6 V (Supply & I/Os).
Gatekeeper Verdict: INCORRECT"""

_USER_TEMPLATE = """QUESTION: {question}

DOCUMENT: {document}

Gatekeeper Verdict (CORRECT / AMBIGUOUS / INCORRECT):"""



# ── Token → score mapping ─────────────────────────────────────────────────────

_LABEL_TO_SCORE: dict[str, float] = {
    "CORRECT": 1.0,
    "AMBIGUOUS": 0.0,
    "INCORRECT": -1.0,
}


def _parse_score(raw: str) -> float:
    """
    Extract a numeric score from the model output.
    Uses exact word boundary regex r'\\b(INCORRECT|AMBIGUOUS|CORRECT)\\b'
    to prevent 'CORRECT' from matching inside 'INCORRECT'.
    Logs warning and increments PARSING_FAILURES_COUNT on unexpected output.
    """
    global PARSING_FAILURES_COUNT
    if not raw:
        PARSING_FAILURES_COUNT += 1
        logger.warning("[RetrievalEvaluator] Received empty raw response from LLM.")
        raise ValueError("RetrievalEvaluator received empty response from LLM.")

    # Remove reasoning/thinking tags emitted by reasoning models
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.IGNORECASE).strip()
    
    # Search for word-bounded labels in cleaned string first, then raw string
    for text_to_check in (cleaned, raw):
        upper = text_to_check.upper()
        # Find all isolated word matches with their positions
        matches = list(re.finditer(r"\b(INCORRECT|AMBIGUOUS|CORRECT)\b", upper))
        if matches:
            # Pick the last matched isolated label token in the text
            last_match_label = matches[-1].group(1)
            return _LABEL_TO_SCORE[last_match_label]

    # Explicit failure handling: increment counter and raise exception or log warning
    PARSING_FAILURES_COUNT += 1
    err_msg = f"[RetrievalEvaluator] Unable to extract valid label (CORRECT/AMBIGUOUS/INCORRECT) from response: '{raw[:100]}...'"
    logger.warning(err_msg)
    print(f"\n⚠️ {err_msg}")
    raise ValueError(err_msg)



# ── Public API ────────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Lightweight retrieval evaluator backed by a Groq-hosted LLM.

    Usage
    -----
    evaluator = RetrievalEvaluator(groq_client)
    score = evaluator.score(query, document)          # single pair → float
    scores = evaluator.score_batch(query, documents)  # list of docs → list[float]
    confidence = evaluator.judge(query, documents)    # → "CORRECT" | "INCORRECT" | "AMBIGUOUS"
    """

    def __init__(self, client: Groq, model: str = EVALUATOR_MODEL) -> None:
        self.client = client
        self.model = model

    def score(self, query: str, document: str) -> float:
        """Return relevance score in [-1, 1] for a single (query, doc) pair."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    question=query,
                    document=truncate(document, max_chars=2000),
                ),
            },
        ]
        raw = chat(self.client, self.model, messages, max_tokens=512, temperature=0.0)
        return _parse_score(raw)



    def score_batch(self, query: str, documents: list[str]) -> list[float]:
        """Score multiple documents for the same query."""
        return [self.score(query, doc) for doc in documents]

    def judge(
        self,
        query: str,
        documents: list[str],
        upper: float = UPPER_THRESHOLD,
        lower: float = LOWER_THRESHOLD,
    ) -> str:
        """
        Aggregate document scores into a single action label.

        Returns
        -------
        "CORRECT"   if any document score ≥ upper
        "INCORRECT" if all document scores < lower
        "AMBIGUOUS" otherwise
        """
        scores = self.score_batch(query, documents)
        if any(s >= upper for s in scores):
            return "CORRECT"
        if all(s < lower for s in scores):
            return "INCORRECT"
        return "AMBIGUOUS"

    def filter_strips(
        self,
        query: str,
        strips: list[str],
        threshold: float,
        top_k: int,
    ) -> list[str]:
        """
        Score every knowledge strip and keep the top-k above *threshold*.
        Returns strips in their original order.
        """
        scored = [(strip, self.score(query, strip)) for strip in strips]
        passed = [(s, sc) for s, sc in scored if sc >= threshold]
        # Sort by score descending, then take top-k
        passed.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in passed[:top_k]]
        # Restore original order
        original_order = [s for s in strips if s in selected]
        return original_order if original_order else [strips[0]]  # always return ≥1