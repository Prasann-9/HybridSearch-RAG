"""core/config.py — model names and tunable constants in one place.

Edit these for your use case. Keeping model names here makes them a one-line bump when a
provider deprecates a model (e.g. Gemini's embedding model name changes over time).
"""

# Groq chat models (the free tier serves these). Pick a default; offer more in a selector.
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Gemini embeddings — gemini-embedding-001 is the current stable text model.
# (text-embedding-004 was retired and now 404s.)
EMBED_MODEL = "models/gemini-embedding-001"

# ── Retrieval knobs ───────────────────────────────────────────────────────────
# Reciprocal Rank Fusion constant. Larger k flattens the weight of top ranks; 60 is
# the value from the original Cormack et al. RRF paper and a sane default.
RRF_K = 60
# How many results to show per method in the comparator / feed to the LLM.
TOP_K = 5
# How many candidates each retriever contributes into the RRF pool before fusing.
CANDIDATE_K = 20

# Worker-thread timeout for long phases (seconds).
WORK_TIMEOUT = 900
