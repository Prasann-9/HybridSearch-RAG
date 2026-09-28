"""data/corpus.py — the bundled demo corpus + curated example queries.

The corpus is deliberately engineered for a webinar on hybrid search: it mixes docs that carry
**exact, rare tokens** (acronyms, metric names, model IDs like `gemini-embedding-001`) — where
BM25 wins and a dense bi-encoder often misses — with **paraphrase-only** docs described without
their canonical keyword — where the dense retriever wins and BM25 misses. Reciprocal Rank Fusion
(hybrid) is what survives both cases, which is the whole point of the demo.

Each doc is {"id", "title", "text"}. Keep texts short (1–3 sentences) so retrieval is fast and
the comparator table stays readable.
"""
from __future__ import annotations

DOCS: list[dict] = [
    # ── Sparse / lexical retrieval ──────────────────────────────────────────
    {"id": "d01", "title": "Sparse retrieval basics",
     "text": "Sparse retrieval represents text as high-dimensional vectors where most entries are zero, "
             "one dimension per vocabulary term. Matching relies on exact term overlap between query and document."},
    {"id": "d02", "title": "TF-IDF weighting",
     "text": "TF-IDF scores a term by its frequency in a document times its inverse document frequency across the corpus, "
             "so common words like 'the' get low weight and distinctive words get high weight."},
    {"id": "d03", "title": "The BM25 ranking function",
     "text": "BM25 improves on TF-IDF with term-frequency saturation and document-length normalisation. Its two knobs are "
             "k1, which controls saturation, and b, which controls how strongly length is normalised."},
    {"id": "d04", "title": "BM25 parameter defaults",
     "text": "Common BM25 defaults are k1 between 1.2 and 2.0 and b around 0.75. Tuning k1 and b on a labelled set can "
             "meaningfully change ranking quality for a given corpus."},
    {"id": "d05", "title": "Term-frequency saturation",
     "text": "Raw term frequency rewards repetition without limit. Saturation means the tenth occurrence of a word adds far "
             "less than the second, preventing keyword stuffing from dominating a score."},
    {"id": "d06", "title": "Inverted index",
     "text": "Keyword engines store an inverted index mapping each token to the postings list of documents that contain it, "
             "which makes exact-token lookup extremely fast even over millions of documents."},

    # ── Dense / semantic retrieval ──────────────────────────────────────────
    {"id": "d07", "title": "Semantic search",
     "text": "Instead of matching words, this approach captures the intent behind a phrase, so a question can retrieve a "
             "passage that answers it even when the two share almost none of the same words."},
    {"id": "d08", "title": "Text embeddings",
     "text": "An embedding maps a passage to a fixed-length list of numbers such that pieces of text with related meaning land "
             "close together in the space, letting nearest-neighbour lookup act as a proxy for relevance."},
    {"id": "d09", "title": "Measuring closeness of vector directions",
     "text": "To compare two embeddings you look at the angle between them: normalise each to unit length and take their dot "
             "product. A value near one means they point the same way; near zero means they are unrelated."},
    {"id": "d10", "title": "The vocabulary mismatch problem",
     "text": "Keyword search fails when the searcher and the author choose different words for the same idea — asking about a "
             "'car' will miss a page that only ever says 'automobile'. Meaning-based retrieval closes that gap."},
    {"id": "d11", "title": "Approximate nearest neighbour with HNSW",
     "text": "Vector databases use graph indexes such as HNSW to find the closest embeddings without scanning every vector, "
             "trading a little recall for a large speedup at query time."},
    {"id": "d12", "title": "Chunking documents for retrieval",
     "text": "Long documents are split into passages before embedding so that a retrieved unit is focused enough to answer a "
             "question. Overlapping windows avoid cutting an idea across a boundary."},

    # ── Bi-encoders vs cross-encoders ───────────────────────────────────────
    {"id": "d13", "title": "Bi-encoder architecture",
     "text": "A bi-encoder embeds the query and each document separately, so document vectors can be computed once and stored. "
             "Scoring is then a cheap similarity lookup, which is what makes first-stage retrieval scale."},
    {"id": "d14", "title": "A model that reads query and passage together",
     "text": "A cross-encoder feeds the query and a candidate passage through the network jointly and outputs a single relevance "
             "score. It is far more accurate than separate encoding but too slow to run over a whole corpus."},
    {"id": "d15", "title": "Two-stage retrieve-then-rerank",
     "text": "A fast first stage fetches a few dozen candidates, then a heavier reranker re-scores just those. This keeps latency "
             "low while getting most of the accuracy of the expensive model."},
    {"id": "d16", "title": "Why not cross-encode everything",
     "text": "Because a joint encoder must run once per query-document pair, applying it to every item in a large index would be "
             "prohibitively slow, so it is reserved for reranking a short candidate list."},

    # ── Hybrid search + fusion ──────────────────────────────────────────────
    {"id": "d17", "title": "Hybrid search",
     "text": "Hybrid search runs a lexical retriever and a semantic retriever in parallel and merges their result lists, so exact "
             "matches and meaning matches both contribute to the final ranking."},
    {"id": "d18", "title": "Reciprocal Rank Fusion",
     "text": "RRF combines several ranked lists by summing 1 / (k + rank) for each document across the lists. It needs only the "
             "positions, not the raw scores, so it fuses incomparable BM25 and cosine scores cleanly."},
    {"id": "d19", "title": "The RRF k constant",
     "text": "In Reciprocal Rank Fusion the smoothing constant k=60 comes from the original Cormack paper. A larger k flattens the "
             "advantage of the very top ranks; a smaller k makes rank-one dominate."},
    {"id": "d20", "title": "Score normalisation versus rank fusion",
     "text": "One way to merge retrievers is to rescale their scores onto a common range and add them, but that is sensitive to "
             "outliers. Fusing by position sidesteps the need to calibrate different scoring scales."},
    {"id": "d21", "title": "Why hybrid beats either alone",
     "text": "Lexical retrieval anchors on exact identifiers and rare terms; semantic retrieval handles paraphrase and synonymy. "
             "Combining them recovers documents that a single method ranks too low to surface."},

    # ── Embeddings model / stack specifics (exact-ID docs) ──────────────────
    {"id": "d22", "title": "Gemini embedding model",
     "text": "This stack embeds text with gemini-embedding-001, the current stable Gemini text-embedding model, called through "
             "the langchain-google-genai integration."},
    {"id": "d23", "title": "Retired embedding models",
     "text": "The older text-embedding-004 model has been retired and now returns a 404, so pipelines that hard-coded it must be "
             "updated to gemini-embedding-001."},
    {"id": "d24", "title": "Groq for generation",
     "text": "The answer-generation step calls a Llama model hosted on Groq, for example llama-3.3-70b-versatile, which returns "
             "grounded answers quickly given the retrieved context."},

    # ── Evaluation metrics ──────────────────────────────────────────────────
    {"id": "d25", "title": "Recall at k",
     "text": "Recall@k is the fraction of the known relevant documents that appear anywhere in the top k results. It measures "
             "whether the retriever put the right material within reach of the reader or reranker."},
    {"id": "d26", "title": "Mean Reciprocal Rank",
     "text": "MRR averages one divided by the rank of the first relevant result across queries, rewarding systems that place a "
             "correct answer high rather than merely somewhere in the list."},
    {"id": "d27", "title": "Normalised discounted cumulative gain",
     "text": "nDCG@10 rewards placing highly relevant documents near the top of the first ten results and discounts relevance "
             "logarithmically by position, then normalises against the ideal ordering."},
    {"id": "d28", "title": "Precision versus recall in retrieval",
     "text": "Precision asks how many of the returned items are relevant; recall asks how many of the relevant items were "
             "returned. First-stage retrieval usually optimises recall and lets a reranker raise precision."},

    # ── RAG pipeline + grounding ────────────────────────────────────────────
    {"id": "d29", "title": "Retrieval-augmented generation",
     "text": "RAG grounds a language model on retrieved passages so its answer cites real source text instead of relying only on "
             "parametric memory, which reduces hallucination on niche or fast-changing facts."},
    {"id": "d30", "title": "Why retrieval quality caps answer quality",
     "text": "A generator can only be as accurate as the context it is given: if the right passage never reaches the prompt, no "
             "amount of prompting will recover the fact, so retrieval is the ceiling on RAG accuracy."},
    {"id": "d31", "title": "Grounding and citations",
     "text": "Grounded generation constrains the model to answer from the supplied passages and to say when the context does not "
             "contain the answer, rather than inventing one."},
    {"id": "d32", "title": "Context window budgeting",
     "text": "Only a handful of passages fit in a prompt before cost and distraction grow, so ranking the very best few chunks "
             "first matters more than retrieving many mediocre ones."},

    # ── Distractors / adjacent topics (make ranking non-trivial) ────────────
    {"id": "d33", "title": "Query expansion",
     "text": "Adding synonyms or related terms to a query before searching can help a lexical engine, but over-expansion drifts "
             "the intent and hurts precision."},
    {"id": "d34", "title": "Stop words and stemming",
     "text": "Classic text pipelines drop very common words and reduce words to a root form so that 'running' and 'ran' match, "
             "which helps lexical recall at some cost to precision."},
    {"id": "d35", "title": "Cosine versus Euclidean distance",
     "text": "For normalised embeddings, ranking by cosine similarity and by Euclidean distance give the same order, so most "
             "vector stores default to cosine for text."},
    {"id": "d36", "title": "Embedding dimensionality",
     "text": "Higher-dimensional embeddings can capture finer distinctions but cost more memory and compute per comparison, so "
             "there is a practical trade-off in choosing a model's output size."},
    {"id": "d37", "title": "Negative sampling for training encoders",
     "text": "Retrieval encoders are trained to pull relevant pairs together and push hard negatives apart; the quality of the "
             "negatives strongly shapes how well the model separates close cases."},
    {"id": "d38", "title": "Latency budget of a search stack",
     "text": "Each stage — lexical lookup, vector search, reranking, generation — adds milliseconds, and a production system "
             "allocates a latency budget across them to stay responsive."},
    {"id": "d39", "title": "Freshness and re-indexing",
     "text": "When documents change, both the inverted index and the vector index must be updated, or the system will keep "
             "returning stale results that no longer reflect the source."},
    {"id": "d40", "title": "Multilingual retrieval",
     "text": "Multilingual embeddings place translations of the same sentence near each other, letting a query in one language "
             "retrieve documents written in another without translation."},
    {"id": "d41", "title": "Filtering with metadata",
     "text": "Retrieval is often combined with structured filters — date, author, product — so that semantic ranking happens only "
             "within the subset a user is allowed or wants to see."},
    {"id": "d42", "title": "Duplicate and near-duplicate results",
     "text": "Without dedup, a retriever can fill the top results with near-identical passages, wasting the context budget; "
             "diversity-aware ranking spreads coverage across distinct facts."},
]


# ── Curated example queries for the Comparator tab ──────────────────────────
# `tag` labels the expected dynamic so you can narrate it live. The retrieval is real — these are
# the queries that most reliably show each effect on this corpus, not hard-coded answers.
COMPARATOR_QUERIES: list[dict] = [
    {"q": "RRF k=60 smoothing constant",
     "tag": "BM25 shines",
     "why": "A rare literal token (k, 60) — exact lexical match pins the right doc; embeddings treat it as low-signal."},
    {"q": "nDCG@10 ranking metric",
     "tag": "BM25 shines",
     "why": "The metric name is a distinctive token BM25 locks onto, while several eval docs look semantically alike."},
    {"q": "which model replaced text-embedding-004",
     "tag": "BM25 shines",
     "why": "An exact model ID; lexical match beats fuzzy similarity across the many embedding-related docs."},
    {"q": "how to find text with the same meaning but different words",
     "tag": "Dense shines",
     "why": "Pure paraphrase with almost no keyword overlap — the vocabulary-mismatch doc has none of these words."},
    {"q": "ranking documents by how closely their vectors point the same way",
     "tag": "Dense shines",
     "why": "Describes cosine similarity without saying 'cosine' — semantic retrieval gets it, BM25 does not."},
    {"q": "a model that reads the question and the passage together to score them",
     "tag": "Dense shines",
     "why": "Paraphrase of a cross-encoder; the canonical keyword never appears in the query."},
    {"q": "combine keyword and semantic search results into one ranking",
     "tag": "Hybrid shines",
     "why": "Needs both the fusion concept and exact terms — RRF floats the best doc that each method ranks only mid-list."},
]


# ── Queries for the RAG Answer tab (some are 'gotchas' for semantic-only) ───
RAG_QUERIES: list[dict] = [
    {"q": "What embedding model does this stack use, and is text-embedding-004 still supported?",
     "gotcha": True,
     "why": "Needs two exact-ID facts (gemini-embedding-001; text-embedding-004 retired). Semantic-only can miss the "
            "retired-model doc and the LLM may guess; hybrid retrieves both, so the answer is grounded."},
    {"q": "What is the default value of the k constant in Reciprocal Rank Fusion and what does changing it do?",
     "gotcha": True,
     "why": "The exact value 60 lives in one doc that BM25 nails; dense may surface general fusion docs instead."},
    {"q": "Why is a cross-encoder not used to score every document in the index?",
     "gotcha": False,
     "why": "A conceptual question both retrievers handle — useful as a control where answers largely agree."},
    {"q": "Why can hybrid search retrieve documents that neither keyword nor semantic search ranks highly alone?",
     "gotcha": False,
     "why": "The headline message of the webinar, answered from the fusion docs."},
]


def default_docs() -> list[dict]:
    """The bundled corpus."""
    return list(DOCS)


def parse_pasted_corpus(text: str) -> list[dict]:
    """Turn pasted text into docs. Split on blank lines; each block is one doc.

    The first line of a block becomes the title; the rest (or the whole block) becomes the text.
    Empty input returns [] so the caller can fall back to the bundled corpus.
    """
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    docs: list[dict] = []
    for i, block in enumerate(blocks):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        title = lines[0][:80]
        body = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
        docs.append({"id": f"u{i:02d}", "title": title, "text": body})
    return docs
