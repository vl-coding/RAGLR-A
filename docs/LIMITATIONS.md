# Limitations — RAGLR-A

## Data

**OAI-PMH metadata quality**
arXiv OAI-PMH records use Dublin Core (`oai_dc`), which is a lowest-common-denominator metadata format. Abstracts are sometimes truncated, whitespace-collapsed, or encoded with LaTeX that is not rendered. Author lists and dates may be incomplete in older records.

**Coverage gaps**
Papers without a title or abstract are dropped during ingestion. Records with no parseable arXiv ID (e.g. old-format identifiers that do not match the extractor) are also skipped. A small fraction of valid papers may be lost as a result.

**Temporal freshness**
The corpus reflects the state of arXiv at the time of the most recent harvest. New preprints submitted after that date will not appear in results until the next `update_arxiv_data.py --incremental` run.

**Category assignment**
arXiv categories come from author self-reporting and are stored on each `Paper` record, but are not used for retrieval filtering. A paper may be cross-listed in multiple categories, or placed in a category that does not fully reflect its content.

---

## Retrieval

**Dense retrieval candidate pre-filtering**
ChromaDB's `query()` method does not support native set-based ID filtering. RAGLR-A compensates by querying a larger result pool (up to 5× `top_n` or the candidate set size) and post-filtering. For very large candidate sets this may not retrieve all relevant papers within the top-k from the dense search.

**Embedding model**
`all-MiniLM-L6-v2` is a lightweight 22M-parameter model optimized for speed. It may underperform larger embedding models (e.g. `all-mpnet-base-v2`, `text-embedding-3-large`) on highly technical or domain-specific queries, especially in fields like mathematics or physics where notation-heavy text is common.

**BM25 tokenization**
The tokenizer uses a simple regex (`\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b`) and lowercases all tokens. This keeps two-character alphanumeric tokens like `AI`, `ML`, `T5`, and `CV`, which were previously dropped entirely. Mathematical symbols and Greek letters (e.g. `α`, `β`) are still outside `[a-zA-Z0-9\-]` and are dropped, reducing BM25 recall for queries involving equations or symbol-heavy notation. A built index must be rebuilt (`scripts/build_indexes.py`) to pick up tokenizer changes.

**HyDE quality**
Claude's hypothetical abstract is conditioned solely on the user query. If the query is ambiguous or very short, the generated abstract may not accurately represent the desired research direction, which can skew dense retrieval.

**RRF score interpretability**
RRF scores are not probabilities and are not directly comparable across queries with different candidate set sizes. They should be treated as ordinal ranks, not cardinal relevance scores.

**Old but significant papers rank poorly**
Foundational papers (e.g. "Attention Is All You Need", BERT, GPT-3, CLIP) often fall outside both retrievers' top-200 candidates for queries phrased in current terminology — their decade-old vocabulary doesn't compete lexically (BM25) or semantically (dense) with thousands of newer papers describing the same ideas. This is an inherent property of similarity-based retrieval over a large, fast-moving corpus, not something `rrf_k` tuning or a larger candidate set fixes. See `docs/EVALUATION.md` for the full diagnosis.

---

## Models

**Qwen2.5-0.5B-Instruct**
The keyword extractor is a 0.5B-parameter model running locally. It may generate irrelevant, redundant, or overly broad keywords for complex or interdisciplinary queries. The pipeline falls back to tokenizing the raw query if the model output cannot be parsed as a JSON list.

**Claude justifications**
Claude's relevance and specificity scores (1–10) are not calibrated across queries. A score of 8 for one query may not be comparable to a score of 8 for another. Scores should be used for relative ranking within a single result set, not for cross-query comparisons. Justifications may occasionally be verbose, overly positive, or hallucinated if Claude lacks domain expertise in the subject area.

**API rate limits and cost**
Claude is called once for HyDE and once per top-k result for justification. With `top_k=10`, each query makes 11 Claude API calls. High-volume use will accumulate cost and may hit rate limits.

---

## Performance

**Index build time**
Building the dense index (ChromaDB + SBERT) over millions of papers is CPU/GPU-intensive and can take several hours on a standard machine without a GPU. BM25 and keyword index builds are faster but also memory-intensive for large corpora.

**Qwen local inference**
Qwen2.5-0.5B-Instruct runs in float16 on GPU or float32 on CPU. On CPU, keyword extraction can add several seconds of latency per query. Consider disabling (`--no-qwen`) for low-latency use cases.

**Memory**
The BM25 index is loaded into RAM at pipeline startup; for a corpus of several million papers this can use several GB. The keyword index is SQLite-backed and queried via point lookups, so it relies on the OS page cache rather than being loaded wholesale into the Python process.

---

## Scope

**Not a citation graph system**
RAGLR-A does not model citation relationships, co-authorship networks, or temporal trends. Results are ranked purely by lexical and semantic similarity to the query.

**English-only**
The embedding model and keyword tokenizer are optimized for English text. Non-English arXiv submissions (a small fraction of the corpus) may be retrieved with lower accuracy.

**No deduplication of near-duplicates**
Multiple versions of the same paper (v1, v2, v3...) are deduplicated by arXiv ID. However, substantively similar papers from different authors are not identified as near-duplicates.
