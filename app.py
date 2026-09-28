"""Hybrid Search in RAG — a BYOK Streamlit demo.

Learners paste their own Groq + Gemini keys in the sidebar, validate, and the demo unlocks:

  1. 📘 Overview            — what hybrid search is + the pipeline diagram + corpus picker.
  2. 🔍 Retriever Comparator — one query, three retrievers (BM25 / Gemini dense / RRF hybrid)
                               side by side, each result annotated with its rank in every method.
  3. 🤖 RAG Answer          — the same question answered by Groq from semantic-only context vs
                               hybrid context, to show hybrid produces better-grounded answers.

Phase 1 has no evaluation tab; that arrives in Phase 2. The BYOK plumbing (sidebar keys,
validate-then-unlock, run_in_thread worker calls, torch-free deps) is the reusable template.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import numpy as np
import streamlit as st

from core import keys as keymod
from core import retrieval as R
from core.config import (CANDIDATE_K, DEFAULT_GROQ_MODEL, GROQ_MODELS, RRF_K, TOP_K,
                         WORK_TIMEOUT)
from core.providers import chat_groq, embed_gemini, embed_gemini_query
from data import corpus as corpusmod
from ui.async_utils import run_in_thread
from ui.tabs import df_table, render_mermaid

APP_TITLE = "🔀 Hybrid Search in RAG"
APP_TAGLINE = "See why sparse + dense + RRF beats either retriever alone — with your own keys."

st.set_page_config(page_title=APP_TITLE, page_icon="🔀", layout="wide")


# ──────────────────────────────────────────────────────────────────────────────
# Session
# ──────────────────────────────────────────────────────────────────────────────
def init_state():
    ss = st.session_state
    ss.setdefault("valid", {k: False for k in keymod.PROVIDERS})
    ss.setdefault("gen_model", DEFAULT_GROQ_MODEL)
    ss.setdefault("corpus_source", "Bundled corpus")
    ss.setdefault("pasted_text", "")
    ss.setdefault("comp_q", "")
    ss.setdefault("rag_q", "")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — registry-driven keys + validation + status
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    ss = st.session_state
    st.sidebar.title("🔑 API keys")
    st.sidebar.caption("Bring your own keys — they stay in this browser session only. "
                       "Get free keys from the links on each field.")
    overrides = {}
    for name, p in keymod.PROVIDERS.items():
        overrides[name] = st.sidebar.text_input(
            p.label, type="password", value=keymod.get_secret(name) or "",
            help=p.help_url + ("" if p.required else "  (optional)"))
    keymod.apply_keys(overrides)

    if st.sidebar.button("Validate keys", type="primary", use_container_width=True):
        valid = {}
        with st.sidebar, st.spinner("Validating…"):
            for name, p in keymod.PROVIDERS.items():
                ok, msg = p.validate(os.environ.get(name))
                valid[name] = ok
                st.write(f"{'✅' if ok else '❌'} {p.label} — {msg}")
        ss["valid"] = valid

    st.sidebar.divider()
    st.sidebar.markdown("**Status**")
    for name, p in keymod.PROVIDERS.items():
        mark = "✅" if ss["valid"].get(name) else ("⬜" if p.required else "▫️")
        st.sidebar.write(f"{mark} {p.label}")

    st.sidebar.divider()
    ss["gen_model"] = st.sidebar.selectbox(
        "Answer model (Groq)", GROQ_MODELS,
        index=GROQ_MODELS.index(ss["gen_model"]) if ss["gen_model"] in GROQ_MODELS else 0)

    if not keymod.all_required_valid(ss["valid"]):
        st.sidebar.info("Add both keys and click **Validate** to unlock the demo.")


def unlocked() -> bool:
    return keymod.all_required_valid(st.session_state["valid"])


# ──────────────────────────────────────────────────────────────────────────────
# Corpus + cached indexes
# ──────────────────────────────────────────────────────────────────────────────
def current_docs() -> list[dict]:
    """The active corpus: bundled, or parsed from the pasted text (falling back to bundled)."""
    ss = st.session_state
    if ss["corpus_source"].startswith("Paste"):
        docs = corpusmod.parse_pasted_corpus(ss["pasted_text"])
        if docs:
            return docs
    return corpusmod.default_docs()


def doc_texts(docs: list[dict]) -> list[str]:
    return [f"{d['title']}. {d['text']}" for d in docs]


@st.cache_resource(show_spinner=False)
def get_bm25(corpus_key: str, _texts: tuple):
    """BM25 index, cached per corpus (no key needed). corpus_key drives cache invalidation."""
    return R.build_bm25(list(_texts))


@st.cache_resource(show_spinner=False)
def get_doc_matrix(corpus_key: str, _texts: tuple) -> np.ndarray:
    """Gemini embeddings for the corpus, cached per corpus. Built once via the worker thread."""
    vecs = run_in_thread(embed_gemini, list(_texts), timeout=WORK_TIMEOUT)
    return np.asarray(vecs, dtype=float)


@st.cache_data(show_spinner=False)
def get_query_vec(query: str) -> list[float]:
    """Embed one query, cached by text so re-running (e.g. moving the k slider) is free."""
    return run_in_thread(embed_gemini_query, query, timeout=WORK_TIMEOUT)


def build_indexes(docs: list[dict]):
    """Return (corpus_key, texts, bm25, doc_matrix), building/warming caches as needed."""
    texts = doc_texts(docs)
    corpus_key = R.corpus_hash(docs)
    bm25 = get_bm25(corpus_key, tuple(texts))
    doc_matrix = get_doc_matrix(corpus_key, tuple(texts))
    return corpus_key, texts, bm25, doc_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview
# ──────────────────────────────────────────────────────────────────────────────
PIPELINE = """
flowchart LR
  Q[User query] --> B[BM25 sparse retriever]
  Q --> D[Gemini dense bi-encoder]
  B --> F[RRF fusion]
  D --> F
  F --> C[Top passages]
  C --> L[Groq LLM answer]
"""


def tab_overview():
    st.subheader("📘 What this demo shows")
    st.markdown(
        "- **Sparse (BM25)** matches *exact words* — great for acronyms, IDs, and rare terms, "
        "blind to paraphrase.\n"
        "- **Dense (Gemini bi-encoder)** matches *meaning* — great for synonyms and paraphrase, "
        "weak on exact literal tokens.\n"
        "- **Hybrid (RRF)** fuses both rankings, so a document strong in *either* signal surfaces.\n\n"
        "Move to **🔍 Retriever Comparator** to see all three ranked side by side, then "
        "**🤖 RAG Answer** to see the effect on the final grounded answer."
    )
    render_mermaid(PIPELINE, height=260)

    st.divider()
    st.subheader("📚 Corpus")
    ss = st.session_state
    ss["corpus_source"] = st.radio(
        "Which documents should the retrievers search?",
        ["Bundled corpus", "Paste your own"], horizontal=True,
        index=0 if ss["corpus_source"].startswith("Bundled") else 1)
    if ss["corpus_source"].startswith("Paste"):
        ss["pasted_text"] = st.text_area(
            "One document per blank-line-separated block. The first line becomes the title.",
            value=ss["pasted_text"], height=200,
            placeholder="BM25 basics\nBM25 scores exact term overlap with saturation and length "
                        "normalisation.\n\nCosine similarity\nRanks by the angle between embeddings.")
    docs = current_docs()
    st.caption(f"Active corpus: **{len(docs)} documents**"
               + ("  (bundled AI/ML & RAG set)" if ss["corpus_source"].startswith("Bundled")
                  else "  (your pasted text)"))
    with st.expander("Preview the corpus"):
        df_table([{"id": d["id"], "title": d["title"], "text": d["text"]} for d in docs],
                 wide_cols=("text",))


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 — Retriever Comparator
# ──────────────────────────────────────────────────────────────────────────────
def _rank_rows(results: list[dict]) -> list[dict]:
    def fmt(v):
        return v if v is not None else "—"
    rows = []
    for i, r in enumerate(results, start=1):
        rows.append({
            "#": i,
            "Document": r["title"],
            "BM25": fmt(r["ranks"]["bm25"]),
            "Dense": fmt(r["ranks"]["dense"]),
            "Hybrid": fmt(r["ranks"]["hybrid"]),
        })
    return rows


def tab_comparator():
    st.subheader("🔍 Three retrievers, one query")
    if not unlocked():
        st.info("Validate your keys in the sidebar to unlock this step.")
        return
    st.caption("Each table shows one retriever's top results. The **BM25 / Dense / Hybrid** columns "
               "give that document's rank under *every* method — watch a doc that one retriever "
               "buries get rescued by the others.")

    docs = current_docs()
    try:
        with st.spinner("Preparing indexes (embedding the corpus once)…"):
            corpus_key, texts, bm25, doc_matrix = build_indexes(docs)
    except Exception as e:
        st.error(f"Could not build the embedding index — {type(e).__name__}: {e}")
        return

    st.markdown("**Try an example** (label = the effect to expect):")
    cols = st.columns(len(corpusmod.COMPARATOR_QUERIES))
    for col, ex in zip(cols, corpusmod.COMPARATOR_QUERIES):
        with col:
            if st.button(ex["tag"], key=f"ex_{ex['q']}", use_container_width=True, help=ex["q"]):
                st.session_state["comp_q"] = ex["q"]
                st.rerun()

    st.text_input("Query", key="comp_q", placeholder="Type a query or pick an example above")
    k = st.slider("RRF k constant", min_value=1, max_value=120, value=RRF_K,
                  help="Smaller k lets rank-1 dominate; larger k flattens the top ranks. Paper default: 60.")

    query = st.session_state["comp_q"].strip()
    if not query:
        return

    # Narrate the expected effect if this is a known example.
    match = next((e for e in corpusmod.COMPARATOR_QUERIES if e["q"] == query), None)
    if match:
        st.info(f"**{match['tag']}** — {match['why']}")

    try:
        query_vec = get_query_vec(query)
        res = R.compare_retrievers(query, docs, bm25, doc_matrix, query_vec,
                                   k=k, top_k=TOP_K, candidate_k=CANDIDATE_K)
    except Exception as e:
        st.error(f"Retrieval failed — {type(e).__name__}: {e}")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔤 Sparse · BM25")
        df_table(_rank_rows(res["bm25"]), wide_cols=("Document",))
    with c2:
        st.markdown("#### 🧠 Dense · Gemini")
        df_table(_rank_rows(res["dense"]), wide_cols=("Document",))
    with c3:
        st.markdown("#### 🔀 Hybrid · RRF")
        df_table(_rank_rows(res["hybrid"]), wide_cols=("Document",))

    if res["rescued"]:
        titles = ", ".join(docs[i]["title"] for i in res["rescued"])
        st.success(f"🎯 **Hybrid rescued:** {titles} — in hybrid's top {TOP_K}, but not the top "
                   f"{TOP_K} of *either* single retriever.")
    else:
        st.caption("No rescue on this query — the single retrievers already agreed. Try a "
                   "'Hybrid shines' example to see fusion change the ranking.")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 — RAG Answer
# ──────────────────────────────────────────────────────────────────────────────
RAG_SYSTEM = (
    "You are a retrieval-augmented assistant. Answer the question using ONLY the numbered context "
    "passages provided. If the passages do not contain the answer, say you don't have enough "
    "information — do not use outside knowledge. Be concise (2–4 sentences) and cite the passage "
    "titles you used."
)


def _format_context(docs: list[dict], idxs: list[int]) -> str:
    return "\n\n".join(f"[{n}] {docs[i]['title']}: {docs[i]['text']}"
                       for n, i in enumerate(idxs, start=1))


def tab_rag():
    st.subheader("🤖 Same question, two retrievers, two answers")
    if not unlocked():
        st.info("Validate your keys in the sidebar to unlock this step.")
        return
    st.caption("The LLM answers twice from the same corpus — once with **semantic-only** context, "
               "once with **hybrid** context — so you can see retrieval quality flow into the answer.")

    docs = current_docs()
    try:
        with st.spinner("Preparing indexes (embedding the corpus once)…"):
            corpus_key, texts, bm25, doc_matrix = build_indexes(docs)
    except Exception as e:
        st.error(f"Could not build the embedding index — {type(e).__name__}: {e}")
        return

    st.markdown("**Try an example** (⚠️ = a gotcha where semantic-only tends to slip):")
    cols = st.columns(len(corpusmod.RAG_QUERIES))
    for col, ex in zip(cols, corpusmod.RAG_QUERIES):
        with col:
            label = ("⚠️ " if ex["gotcha"] else "") + ex["q"][:40] + "…"
            if st.button(label, key=f"rag_{ex['q']}", use_container_width=True, help=ex["q"]):
                st.session_state["rag_q"] = ex["q"]
                st.rerun()

    st.text_input("Question", key="rag_q", placeholder="Ask a question about the corpus")
    query = st.session_state["rag_q"].strip()
    if not query:
        return

    match = next((e for e in corpusmod.RAG_QUERIES if e["q"] == query), None)
    if match and match["gotcha"]:
        st.warning(f"**Gotcha** — {match['why']}")

    if not st.button("Generate both answers", type="primary"):
        return

    model = st.session_state["gen_model"]
    try:
        query_vec = get_query_vec(query)
        dense_idxs = R.top_doc_indices(query, docs, bm25, doc_matrix, query_vec,
                                       "dense", top_k=TOP_K, candidate_k=CANDIDATE_K)
        hybrid_idxs = R.top_doc_indices(query, docs, bm25, doc_matrix, query_vec,
                                        "hybrid", k=RRF_K, top_k=TOP_K, candidate_k=CANDIDATE_K)
    except Exception as e:
        st.error(f"Retrieval failed — {type(e).__name__}: {e}")
        return

    dense_ctx = _format_context(docs, dense_idxs)
    hybrid_ctx = _format_context(docs, hybrid_idxs)
    try:
        with st.spinner("Generating both answers via Groq…"):
            dense_ans = run_in_thread(
                chat_groq, f"Question: {query}\n\nContext:\n{dense_ctx}",
                system=RAG_SYSTEM, model=model, timeout=WORK_TIMEOUT)
            hybrid_ans = run_in_thread(
                chat_groq, f"Question: {query}\n\nContext:\n{hybrid_ctx}",
                system=RAG_SYSTEM, model=model, timeout=WORK_TIMEOUT)
    except Exception as e:
        st.error(f"Generation failed — {type(e).__name__}: {e}")
        return

    left, right = st.columns(2)
    with left:
        st.markdown("#### 🧠 Semantic-only (dense)")
        st.markdown(dense_ans)
        with st.expander(f"Retrieved context ({len(dense_idxs)} passages)"):
            for n, i in enumerate(dense_idxs, start=1):
                st.markdown(f"**[{n}] {docs[i]['title']}** — {docs[i]['text']}")
    with right:
        st.markdown("#### 🔀 Hybrid (BM25 + dense + RRF)")
        st.markdown(hybrid_ans)
        with st.expander(f"Retrieved context ({len(hybrid_idxs)} passages)"):
            for n, i in enumerate(hybrid_idxs, start=1):
                st.markdown(f"**[{n}] {docs[i]['title']}** — {docs[i]['text']}")

    only_hybrid = [i for i in hybrid_idxs if i not in dense_idxs]
    if only_hybrid:
        titles = ", ".join(docs[i]["title"] for i in only_hybrid)
        st.success(f"🎯 Hybrid pulled in passages semantic-only missed: **{titles}** — "
                   "often the exact-fact document that grounds the answer.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    init_state()
    render_sidebar()

    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)

    overview, comparator, rag = st.tabs(
        ["📘 Overview", "🔍 Retriever Comparator", "🤖 RAG Answer"])
    with overview:
        tab_overview()
    with comparator:
        tab_comparator()
    with rag:
        tab_rag()


if __name__ == "__main__":
    main()
