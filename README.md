# 🔀 Hybrid Search in RAG — a bring-your-own-key (BYOK) Streamlit demo

A dashboard on hybrid search where user can paste their **own** Groq + Gemini
keys into the sidebar; once the keys validate, the demo unlocks. Deployable as one public
Streamlit Community Cloud link — no author keys shipped.

It shows, on a corpus of AI/ML & RAG docs, **why hybrid search wins**:

- **🔍 Retriever Comparator** — one query, three retrievers side by side (BM25 sparse, Gemini
  dense bi-encoder, RRF hybrid). Each result is annotated with its rank under *every* method, so
  you can watch a document one retriever buries get rescued by fusion.
- **🤖 RAG Answer** — the same question answered by Groq from **semantic-only** context vs
  **hybrid** context, showing retrieval quality flow into the final grounded answer.

> Phase 1 (this build) has **no evaluation tab**. Phase 2 adds RAGAS-style Context Precision &
> Recall (a lightweight custom LLM judge, not the `ragas` package).

## Keys

Both are free-tier and required:

| Key | Used for | Get it |
| --- | --- | --- |
| `GROQ_API_KEY`   | LLM answers (RAG tab)              | https://console.groq.com/keys |
| `GEMINI_API_KEY` | Embeddings (dense / bi-encoder)   | https://aistudio.google.com/apikey |

Add them in the sidebar at runtime, or put them in `.streamlit/secrets.toml` for local dev.

## Run locally

```bash
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # optional: fill real keys
./.venv/bin/streamlit run app.py
```

You can skip `secrets.toml` and just paste keys into the sidebar.

## Deploy to Streamlit Community Cloud

1. Push to GitHub.
2. Create an app on https://share.streamlit.io pointing at this repo/branch.
3. Set **Main file path** to `app.py`.
4. (Optional) Paste keys into **Settings → Secrets**; learners can instead use their own.

## Layout

```
app.py             sidebar (BYOK) + three tabs
core/keys.py       provider registry (Groq + Gemini), resolution, validation, gating
core/config.py     model names + retrieval knobs (RRF_K, TOP_K, CANDIDATE_K)
core/providers.py  provider call helpers (Groq chat, Gemini embed docs + query)
core/retrieval.py  BM25 + dense cosine + Reciprocal Rank Fusion + comparison
data/corpus.py     bundled AI/ML & RAG corpus + curated example / gotcha queries
ui/async_utils.py  worker-thread runner (event-loop-safe)
ui/tabs.py         table / mermaid / styling helpers
```

Torch-free by design (BM25 via `rank-bm25`, embeddings via the Gemini API) so it fits Cloud's
~1 GB limit.
