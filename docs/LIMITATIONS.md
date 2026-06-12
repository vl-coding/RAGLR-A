# Limitations — RAGLR-A

## Data

**OAI-PMH metadata quality**
arXiv OAI-PMH records use Dublin Core (`oai_dc`), which is a lowest-common-denominator metadata format. Author lists and dates may be incomplete in older records. Whitespace is collapsed during ingestion (`normalize_whitespace`), and a conservative set of unrendered LaTeX artifacts is cleaned up (`clean_latex_artifacts`): inline formatting commands like `\textbf{...}`/`\emph{...}` are unwrapped to their argument, escaped characters like `\%`/`\&`/`\_` are unescaped, `\\` line breaks become spaces, and bare `$...$` math delimiters are dropped. More complex LaTeX (nested commands, custom macros, tables/equations) is not reconstructed. Abstracts that are truncated at the source are not detected or repaired — the original (possibly incomplete) text is kept.

**Coverage gaps**
Papers without a title or abstract are dropped during ingestion, as are records marked `deleted` by arXiv or with no parseable arXiv ID. `scripts/update_arxiv_data.py` now tracks per-run drop counts (`kept`, `deleted`, `no_arxiv_id`, `missing_title_or_abstract`), printed at the end of each harvest and recorded in `artifacts/manifest.json` under `ingestion_drop_stats_this_run` so the drop rate is visible over time. Old-format arXiv identifiers (e.g. `math/0309136`) are already handled by `extract_arxiv_id_from_identifier`, so `no_arxiv_id` drops should be rare in practice; the main expected source of loss is `missing_title_or_abstract`.

**Temporal freshness**
The corpus reflects the state of arXiv at the time of the most recent harvest. `scripts/run_scheduler.py` runs `scripts/incremental_update.py` twice daily (00:00 and 12:00 UTC by default) to harvest new papers and hot-reload the indexes with no downtime — see [setup.md](setup.md#7-start-the-incremental-update-scheduler). If the scheduler isn't running, new preprints won't appear until `update_arxiv_data.py --incremental` is run manually.

**Category assignment**
arXiv categories come from author self-reporting and are stored on each `Paper` record, but are not used for retrieval filtering. A paper may be cross-listed in multiple categories, or placed in a category that does not fully reflect its content.

---

## Retrieval

**Dense retrieval candidate pre-filtering**
ChromaDB's `query()` method does not support native set-based ID filtering. RAGLR-A compensates by querying a larger result pool (up to 5× `top_n` or the candidate set size) and post-filtering. For very large candidate sets this may not retrieve all relevant papers within the top-k from the dense search.

**Embedding model**
`all-MiniLM-L6-v2` is a lightweight 22M-parameter model optimized for speed. It may underperform larger embedding models (e.g. `all-mpnet-base-v2`, `text-embedding-3-large`) on highly technical or domain-specific queries, especially in fields like mathematics or physics where notation-heavy text is common.

**BM25 tokenization**
The tokenizer uses a simple regex (`\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b`) and lowercases all tokens. This keeps two-character alphanumeric tokens like `AI`, `ML`, `T5`, and `CV`, which were previously dropped entirely, and hyphenated model names like `GPT-4` tokenize as a single token. Lowercase and uppercase Greek letters (e.g. `α`, `Σ`) are transliterated to their English names (`alpha`, `sigma`) before tokenization, so queries written as words can match abstracts using the glyph and vice versa. Other mathematical symbols (e.g. `∇`, `∫`, `≤`) have no natural word form and are still dropped, reducing BM25 recall for symbol-heavy equation notation. A built index must be rebuilt (`scripts/build_indexes.py`) to pick up tokenizer changes.

**HyDE quality**
Claude's hypothetical abstract is conditioned solely on the user query. If the query is ambiguous or very short, the generated abstract may not accurately represent the desired research direction, which can skew dense retrieval.

**RRF score interpretability**
RRF scores are not probabilities and are not directly comparable across queries with different candidate set sizes. They should be treated as ordinal ranks, not cardinal relevance scores.

**Old but significant papers rank poorly**
Foundational papers (e.g. "Attention Is All You Need", BERT, GPT-3, CLIP) often fall outside both retrievers' top-200 candidates for queries phrased in current terminology — their decade-old vocabulary doesn't compete lexically (BM25) or semantically (dense) with thousands of newer papers describing the same ideas. This is an inherent property of similarity-based retrieval over a large, fast-moving corpus, not something `rrf_k` tuning or a larger candidate set fixes. As a partial mitigation, `data/canonical_papers.yaml` curates a small registry of "signature papers" per CS/ML topic with topic-phrase tags; when a query's text or extracted keywords match a paper's topics, that paper is injected into RRF fusion as its own ranked list (`src/rag_lit/canonical_boost.py`), giving it a chance to surface even if neither retriever's candidate pool included it. This only covers the curated topics/papers in the registry — it does not generalize to arbitrary foundational papers outside it. See `docs/EVALUATION.md` for the full diagnosis.

---

## Models

**Qwen2.5-0.5B-Instruct**
The keyword extractor is a 0.5B-parameter model running locally. It may generate irrelevant, redundant, or overly broad keywords for complex or interdisciplinary queries. The pipeline falls back to tokenizing the raw query if the model output cannot be parsed as a JSON list.

**Claude justifications**
Claude's relevance and specificity scores (1–10) are not calibrated across queries. A score of 8 for one query may not be comparable to a score of 8 for another. Scores should be used for relative ranking within a single result set, not for cross-query comparisons. Justifications may occasionally be verbose, overly positive, or hallucinated if Claude lacks domain expertise in the subject area.

**API rate limits and cost**
Claude is called once for HyDE and once per top-k result for justification. With `top_k=10`, each query makes 11 Claude API calls. High-volume use will accumulate cost. The client retries rate-limited/transient errors with backoff (`models.claude_max_retries`, `models.claude_timeout_seconds` in config.yaml), and justification concurrency is capped (`models.claude_justifier_max_concurrency`, default 5) to avoid a thundering herd of simultaneous requests at high `top_k`. There is no request batching or response caching.

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
