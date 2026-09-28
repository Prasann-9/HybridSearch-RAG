"""core/retrieval.py — the three retrievers and their fusion.

Pure, torch-free, and free of any `st.*` call, so every function here is safe to run inside
`ui.async_utils.run_in_thread` and to unit-test without API keys.

  • Sparse  → BM25 (rank_bm25, exact-token matching over a lexical index).
  • Dense   → a Gemini bi-encoder: cosine similarity between the query embedding and each doc
              embedding. Embeddings are produced elsewhere (core.providers.embed_gemini*) and
              passed in as a plain numpy matrix — this module never calls the API itself.
  • Hybrid  → Reciprocal Rank Fusion (RRF) of the sparse and dense rankings.

The public entry point is `compare_retrievers(...)`, which returns, for each method, a ranked
list of results where every result also carries its rank under the *other* methods — that
cross-annotation is what makes "hybrid rescued this doc" visible in the UI.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np

from core.config import CANDIDATE_K, RRF_K, TOP_K

_TOKEN = re.compile(r"\w+")


# ──────────────────────────────────────────────────────────────────────────────
# Tokenisation + sparse (BM25)
# ──────────────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. Keeps acronyms and IDs as tokens ('bm25', 'rrf', '001')."""
    return _TOKEN.findall(text.lower())


def build_bm25(texts: list[str]):
    """Build a BM25 index over the corpus. Returns a rank_bm25.BM25Okapi."""
    from rank_bm25 import BM25Okapi
    return BM25Okapi([tokenize(t) for t in texts])


def bm25_scores(bm25, query: str) -> np.ndarray:
    """Per-doc BM25 relevance score for the query (higher = more relevant)."""
    return np.asarray(bm25.get_scores(tokenize(query)), dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Dense (bi-encoder cosine)
# ──────────────────────────────────────────────────────────────────────────────
def cosine_scores(query_vec: list[float] | np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between the query embedding and every doc embedding."""
    q = np.asarray(query_vec, dtype=float)
    m = np.asarray(doc_matrix, dtype=float)
    qn = np.linalg.norm(q) or 1.0
    mn = np.linalg.norm(m, axis=1)
    mn[mn == 0] = 1.0
    return (m @ q) / (mn * qn)


# ──────────────────────────────────────────────────────────────────────────────
# Ranking helpers + Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────────────────
def order_desc(scores: np.ndarray) -> list[int]:
    """Doc indices ordered best-first by score."""
    return list(np.argsort(-np.asarray(scores, dtype=float)))


def rrf_fuse(rank_lists: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: fused_score(d) = Σ 1 / (k + rank_i(d)) across input lists.

    rank_lists are ordered index lists (best-first); rank is 1-based position within each list.
    RRF needs only ranks — no score normalisation across incomparable BM25 / cosine scales.
    Returns (idx, fused_score) ordered best-first.
    """
    fused: dict[int, float] = defaultdict(float)
    for lst in rank_lists:
        for rank, idx in enumerate(lst, start=1):
            fused[idx] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: -kv[1])


# ──────────────────────────────────────────────────────────────────────────────
# High-level comparison
# ──────────────────────────────────────────────────────────────────────────────
def _rankmap(order: list[int]) -> dict[int, int]:
    return {idx: r for r, idx in enumerate(order, start=1)}


def compare_retrievers(query: str, docs: list[dict], bm25, doc_matrix: np.ndarray,
                       query_vec: list[float] | np.ndarray, *, k: int = RRF_K,
                       top_k: int = TOP_K, candidate_k: int = CANDIDATE_K) -> dict:
    """Run all three retrievers for one query and cross-annotate the results.

    Returns:
      {
        "bm25":   [result, ...],   # top_k, each a dict (see below)
        "dense":  [result, ...],
        "hybrid": [result, ...],
        "rescued": [idx, ...],     # docs in hybrid's top_k that neither single method ranked there
      }
    where a result is:
      {"doc_idx", "title", "snippet", "score",
       "ranks": {"bm25": r|None, "dense": r|None, "hybrid": r|None}}
    """
    b = bm25_scores(bm25, query)
    d = cosine_scores(query_vec, doc_matrix)

    b_order, d_order = order_desc(b), order_desc(d)
    b_rank, d_rank = _rankmap(b_order), _rankmap(d_order)

    fused = rrf_fuse([b_order[:candidate_k], d_order[:candidate_k]], k=k)
    h_order = [idx for idx, _ in fused]
    h_rank = _rankmap(h_order)
    h_score = dict(fused)

    def result(doc_idx: int, score: float) -> dict:
        doc = docs[doc_idx]
        return {
            "doc_idx": doc_idx,
            "title": doc.get("title", f"doc {doc_idx}"),
            "snippet": doc.get("text", "")[:160],
            "score": float(score),
            "ranks": {
                "bm25": b_rank.get(doc_idx),
                "dense": d_rank.get(doc_idx),
                "hybrid": h_rank.get(doc_idx),
            },
        }

    bm25_top = [result(i, b[i]) for i in b_order[:top_k]]
    dense_top = [result(i, d[i]) for i in d_order[:top_k]]
    hybrid_top = [result(i, h_score[i]) for i in h_order[:top_k]]

    # A doc is "rescued" by hybrid when it makes hybrid's top_k but neither single method's top_k.
    single_top = {r["doc_idx"] for r in bm25_top} | {r["doc_idx"] for r in dense_top}
    rescued = [r["doc_idx"] for r in hybrid_top if r["doc_idx"] not in single_top]

    return {"bm25": bm25_top, "dense": dense_top, "hybrid": hybrid_top, "rescued": rescued}


def top_doc_indices(query: str, docs: list[dict], bm25, doc_matrix: np.ndarray,
                    query_vec, method: str, *, k: int = RRF_K, top_k: int = TOP_K,
                    candidate_k: int = CANDIDATE_K) -> list[int]:
    """Return just the top-k doc indices for one method ('dense' or 'hybrid').

    Used by the RAG tab to assemble the context passed to the LLM.
    """
    if method == "dense":
        return order_desc(cosine_scores(query_vec, doc_matrix))[:top_k]
    if method == "hybrid":
        b_order = order_desc(bm25_scores(bm25, query))[:candidate_k]
        d_order = order_desc(cosine_scores(query_vec, doc_matrix))[:candidate_k]
        return [idx for idx, _ in rrf_fuse([b_order, d_order], k=k)][:top_k]
    if method == "bm25":
        return order_desc(bm25_scores(bm25, query))[:top_k]
    raise ValueError(f"unknown method: {method}")


def corpus_hash(docs: list[dict]) -> str:
    """Stable hash of the corpus text, for keying @st.cache_resource on corpus content."""
    h = hashlib.sha256()
    for d in docs:
        h.update((d.get("title", "") + "\x00" + d.get("text", "") + "\x01").encode("utf-8"))
    return h.hexdigest()[:16]
