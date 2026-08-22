"""
CRAG Configuration
All tuneable parameters in one place.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq & External APIs ──────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MOUSER_API_KEY = os.getenv("MOUSER_API_KEY", "")

# Model used for the retrieval evaluator (lightweight + fast)
EVALUATOR_MODEL = "openai/gpt-oss-20b"

# Model used for the generator (stronger)
GENERATOR_MODEL = "openai/gpt-oss-20b"

# Model used for query rewriting
REWRITER_MODEL = "openai/gpt-oss-20b"

# ── Action Thresholds ─────────────────────────────────────────────────────────
# Score range returned by the evaluator is [-1, 1]
UPPER_THRESHOLD = 0.5   # above → CORRECT
LOWER_THRESHOLD = -0.5  # below → INCORRECT
                        # in between → AMBIGUOUS

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DOCUMENTS = 5          # how many docs to retrieve from local corpus
TOP_K_WEB_RESULTS = 5        # how many web results to fetch
TOP_K_STRIPS = 5             # how many knowledge strips to keep after filtering
STRIP_FILTER_THRESHOLD = -0.5  # min score for a strip to be kept

# ── Knowledge Refinement ──────────────────────────────────────────────────────
# Number of sentences per knowledge strip when splitting a long document
STRIP_SIZE_SENTENCES = 3

# ── Generator ─────────────────────────────────────────────────────────────────
MAX_TOKENS_GENERATOR = 512
TEMPERATURE_GENERATOR = 0.2