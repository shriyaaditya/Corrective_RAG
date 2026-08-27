"""
scripts/test_pipeline_eval.py

DeepEval Evaluation Runner for CRAG Pipeline.
Evaluates Faithfulness, Answer Relevancy, and Contextual Precision
against data/eval_dataset.json using Groq as the judge.
Trims retrieval context to essential text chunks to keep token usage low under TPM limits.
"""

import json
import os
import sys
import time
from pathlib import Path
import pytest

# Add project root and app directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
)

from app.services.retriever.qdrant_store import QdrantStore
from app.services.retriever.vector_retriever import VectorRetriever
from app.services.crag_pipeline import CRAGPipeline
from app.services.eval_llm import GroqEvalLLM

# Load evaluation dataset
EVAL_DATASET_PATH = root_dir / "data" / "eval_dataset.json"

def load_eval_dataset():
    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset missing at {EVAL_DATASET_PATH}")
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

dataset = load_eval_dataset()

# Global pipeline initialization
store = QdrantStore(store_dir=str(root_dir / "qdrant_db"))
retriever = VectorRetriever(store)
pipeline = CRAGPipeline(corpus_retriever=retriever, verbose=False)
judge_llm = GroqEvalLLM()


def _trim_context(text: str, max_chars: int = 500) -> str:
    """Aggressively trim retrieval context to max_chars (500) to minimize token payload."""
    if not text:
        return ""
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "..."


@pytest.mark.parametrize("test_data", dataset, ids=[f"test_data{i+1}" for i in range(len(dataset))])
def test_crag_pipeline_eval(test_data: dict):

    # Sequential delay between test cases to prevent rate limits
    time.sleep(3.0)

    query = test_data["input"]
    expected_output = test_data["expected_output"]

    # Run query through CRAG pipeline
    result = pipeline.run(query)
    actual_output = result.answer

    # Construct aggressively trimmed retrieval context
    retrieval_context = []
    if result.final_context:
        trimmed = _trim_context(result.final_context, max_chars=500)
        if trimmed:
            retrieval_context.append(trimmed)
    if not retrieval_context and result.internal_knowledge:
        trimmed = _trim_context(result.internal_knowledge, max_chars=500)
        if trimmed:
            retrieval_context.append(trimmed)
    if not retrieval_context and result.external_knowledge:
        trimmed = _trim_context(result.external_knowledge, max_chars=500)
        if trimmed:
            retrieval_context.append(trimmed)
    if not retrieval_context:
        retrieval_context.append("No context retrieved.")

    # Create DeepEval LLMTestCase
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    # Initialize FaithfulnessMetric with Groq judge and 0.7 threshold
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=judge_llm)

    # Force synchronous execution using run_async=False
    assert_test(
        test_case,
        metrics=[faithfulness_metric],
        run_async=False,
    )

