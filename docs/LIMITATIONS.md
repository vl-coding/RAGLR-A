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
ChromaDB's `query()` method does not support native set-based ID filtering efficiently — filtering by `ids=`/`where $in` against a ~425K-id candidate set takes ~21s, vs. ~0.4s for an unfiltered query of the same size. RAGLR-A compensates with two heuristics in `DenseRetriever.search()`: (1) if the candidate set covers at least `retrieval.dense_skip_filter_threshold_percent` (default 40%) of the corpus, it isn't meaningfully narrowing the search, so filtering is skipped entirely and Chroma's unfiltered global top-k is used directly; (2) otherwise, RAGLR-A queries a pool of `top_n * 50` results and post-filters by the candidate set, doubling the pool size (up to a hard ceiling of `top_n * 400`, or the full corpus) if fewer than `top_n` candidate-set members are found. For candidate sets just under the 40% threshold with relevant papers ranked very low in the embedding space, the doubling may still bottom out at the ceiling before finding `top_n` matches.

**Embedding model**
`all-MiniLM-L6-v2` is a lightweight 22M-parameter model optimized for speed. Issue #12 benchmarked the hypothesis that a larger embedding model (`all-mpnet-base-v2`, 110M params) would do better on math/physics queries, using a fixed 49,950-paper subset indexed with both models (see `docs/EVALUATION.md` > "Embedding model benchmark"). Result: mpnet showed **no improvement** on the math/physics gold queries (NDCG@10 1.000 vs. 0.976) and was slightly worse on CS/ML queries, while being ~10x slower to build a full-corpus index (~11 days vs. ~1 day) and ~1.6x slower per query. Switching to `all-mpnet-base-v2` is not recommended. `text-embedding-3-large` was not benchmarked (API dependency/cost) and remains out of scope absent new evidence.

**BM25 tokenization**
The tokenizer uses a simple regex (`\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b`) and lowercases all tokens. This keeps two-character alphanumeric tokens like `AI`, `ML`, `T5`, and `CV`, which were previously dropped entirely, and hyphenated model names like `GPT-4` tokenize as a single token. Lowercase and uppercase Greek letters (e.g. `α`, `Σ`) are transliterated to their English names (`alpha`, `sigma`) before tokenization, so queries written as words can match abstracts using the glyph and vice versa. Other mathematical symbols (e.g. `∇`, `∫`, `≤`) have no natural word form and are still dropped, reducing BM25 recall for symbol-heavy equation notation. A built index must be rebuilt (`scripts/build_indexes.py`) to pick up tokenizer changes.

**HyDE quality**
Claude's hypothetical abstract is conditioned solely on the user query. If the query is ambiguous or very short, the generated abstract may not accurately represent the desired research direction, which can skew dense retrieval.

To mitigate this for short queries (issue #14), `pipeline.run()` treats any query with fewer than `SHORT_QUERY_WORD_THRESHOLD` (default 4) whitespace-separated words as "short/ambiguous" and automatically runs a second dense search against the raw query text (in addition to the usual HyDE-document dense search), folding both dense rankings into the RRF fusion that produces the default result set — reusing the dual raw-query/HyDE dense search + fusion machinery that `hyde_ablation=True` added for debugging (issue #2). This gives results that rank well on the literal query wording a chance to surface even when the generated hypothetical abstract drifts off-target. `response.trace.short_query_dual_dense` reports whether this path was taken. Normal-length queries (>= 4 words) keep the existing HyDE-only dense search as the default, to avoid the extra dense search's latency cost in the common case; `hyde_ablation=True` still independently runs and exposes the raw-query dense/fusion results for debugging regardless of query length. The 4-word threshold is a simple heuristic (not a learned ambiguity classifier) and may misclassify some short-but-precise queries (e.g. "BERT fine-tuning") or some longer-but-vague ones; it has not been tuned against retrieval-quality metrics.

**RRF score interpretability**
RRF scores are not probabilities and are not directly comparable across queries with different candidate set sizes. They should be treated as ordinal ranks, not cardinal relevance scores.

**Old but significant papers rank poorly**
Foundational papers (e.g. "Attention Is All You Need", BERT, GPT-3, CLIP) often fall outside both retrievers' top-200 candidates for queries phrased in current terminology — their decade-old vocabulary doesn't compete lexically (BM25) or semantically (dense) with thousands of newer papers describing the same ideas. This is an inherent property of similarity-based retrieval over a large, fast-moving corpus, not something `rrf_k` tuning or a larger candidate set fixes. As a partial mitigation, `data/canonical_papers.yaml` curates a small registry of "signature papers" per CS/ML topic with topic-phrase tags; when a query's text or extracted keywords match a paper's topics, that paper is injected into RRF fusion as its own ranked list (`src/rag_lit/canonical_boost.py`), giving it a chance to surface even if neither retriever's candidate pool included it. This only covers the curated topics/papers in the registry — it does not generalize to arbitrary foundational papers outside it. See `docs/EVALUATION.md` for the full diagnosis.

---

## Models

**Qwen2.5-0.5B-Instruct**
The keyword extractor is a 0.5B-parameter model running locally. `prompts/qwen_keywords_v1.txt` asks for multi-word technical phrases (named methods, architectures, application domains) over generic single words, with a worked example to steer the small model toward concrete phrasing. `qwen_prefilter._filter_keywords` then post-processes the extracted (or fallback-tokenized) list: it drops single-word keywords whose normalized form is in a small generic-term stoplist (e.g. "model", "method", "learning", "approach"), and dedupes case/plural near-duplicates (e.g. "Transformer" vs "transformers"). Multi-word phrases pass through even if one of their words is generic (e.g. "graph neural networks" is kept). This stoplist is heuristic and CS/ML-biased — it may help less for bio/math/physics queries, and a 0.5B model may still occasionally emit generic terms or overly broad phrases not caught by the filter. The pipeline falls back to tokenizing the raw query (then applying the same filter) if the model output cannot be parsed as a JSON list.

**Claude justifications**
`prompts/claude_justifier_v1.txt` defines a fixed rubric for `relevance_score` and `specificity_score`: each level (1, 2-3, 4-5, 6-7, 8-9, 10) has a query-independent description (e.g. relevance 10 = "central contribution is this exact topic / primary citation for the query", 6-7 = "related but main contribution targets a different problem", 1 = "unrelated"), so an 8 for one query is intended to mean the same thing as an 8 for another — see `docs/EVALUATION.md` > Justifier score calibration for before/after distributions. Scores still cluster in the upper half of the scale (mean ~9 for relevance, ~8 for specificity) because the justifier only sees already-retrieved, pre-filtered results; this is an inherent property of scoring a candidate set that's been narrowed by retrieval, not a calibration defect. Scores remain best used for relative ranking within a result set rather than precise cross-query arithmetic. Justifications may occasionally be verbose, overly positive, or hallucinated if Claude lacks domain expertise in the subject area.

**API rate limits and cost**
Claude is called once for HyDE and once per top-k result for justification. With `top_k=10`, each query makes 11 Claude API calls. High-volume use will accumulate cost. The client retries rate-limited/transient errors with backoff (`models.claude_max_retries`, `models.claude_timeout_seconds` in config.yaml), and justification concurrency is capped (`models.claude_justifier_max_concurrency`, default 5) to avoid a thundering herd of simultaneous requests at high `top_k`. There is no request batching or response caching.

---

## Performance

**Index build time**
The *initial* full-corpus dense index build (ChromaDB + SBERT, `scripts/build_dense_index_fast.py`) is CPU-intensive: benchmarked at ~34.5 papers/sec on a 12-core/no-GPU machine (`all-MiniLM-L6-v2`, torch backend, batch_size=128), which is ~24.7h for a 3.07M-paper corpus — consistent with the ~24-28h reported in practice.

This is a **one-time cost**, not a per-harvest cost: `scripts/incremental_update.py` (run twice daily by `scripts/run_scheduler.py`) embeds only newly-harvested papers via `DenseRetriever.build_index` (typically dozens-to-hundreds of papers, seconds-to-minutes) and upserts them into the existing ChromaDB collection — a full rebuild is never required after the initial build.

For the initial build (or disaster recovery), backend/batching tuning was investigated but gives only modest gains on CPU:
- The ONNX Runtime backend's *default* file selection (`onnx/model.onnx`, fp32) measured **slower** than the torch backend (~27.9 vs ~34.5 papers/sec) — the "ONNX is 2-3x faster" assumption did not hold on this hardware. The "optimized" fp32 variant (`model_O4.onnx`) was worse still (~17.1 papers/sec).
- The int8-quantized AVX-512 ONNX variant (`onnx/model_qint8_avx512.onnx`, pass via `--onnx-file`) measured ~38.7 papers/sec, only ~12% faster than torch — a ~2.7h saving on the full corpus, not the order-of-magnitude improvement the issue hoped for.
- Larger batch sizes (256, 512) did not help and were slightly slower than 128; torch already saturates ~10 of 12 CPU threads via MKL at batch_size=128.
- **GPU is the only lever with large headroom**: if `torch.cuda.is_available()` (or `onnxruntime-gpu` is installed for the ONNX backend), both `SentenceTransformer` backends use the GPU automatically with no code changes — for a 22M-parameter model like MiniLM, this typically gives an order-of-magnitude speedup over CPU, but wasn't available to benchmark on this dev machine.

`scripts/build_dense_index_fast.py` streams the JSONL in chunks (bounded ~1-2GB RAM) and upserts are idempotent by `arxiv_id`, so a crashed/restarted initial build resumes from whatever ChromaDB already has. BM25 and keyword index builds are faster but also memory-intensive for large corpora.

**Qwen local inference**
Qwen2.5-0.5B-Instruct runs in float16 on GPU or float32 on CPU. On CPU, keyword extraction can add several seconds of latency per query. `generate_keywords` uses a `StoppingCriteria` (`_JSONArrayStoppingCriteria`) that stops generation as soon as the first JSON keyword array's brackets balance, rather than always running to `max_new_tokens=160` — this typically cuts generation short well before the token budget, since the answer array is usually much shorter than 160 tokens and the model sometimes continues with hallucinated extra output after it. `max_new_tokens=160` remains as a hard ceiling if the model never emits a balanced array. Consider disabling (`--no-qwen`) for low-latency use cases.

**Memory**
`retrieval.bm25_mmap: true` (the default, issue #5) memory-maps the BM25 index's CSC arrays (`bm25s.BM25.load(..., mmap=True)`) instead of loading the full index into RAM at pipeline startup — for a corpus of several million papers, loading fully into RAM can use several GB. With mmap, each query pages in only the score-array slices for its own tokens, and the OS page cache keeps hot tokens resident across queries, similar to how the keyword index (SQLite-backed, point lookups) already behaves. This trades a small amount of per-query I/O latency (mostly on cold pages) for a much smaller resident set, and requires no index rebuild — only `BM25Retriever.load()`'s `mmap` argument changes. Set `retrieval.bm25_mmap: false` to revert to the fully-in-RAM behavior. The delta BM25 index (`bm25_delta`) is small and always loaded fully into RAM regardless of this setting.

---

## Scope

**Not a citation graph system**
RAGLR-A does not model citation relationships, co-authorship networks, or temporal trends. Results are ranked purely by lexical and semantic similarity to the query.

**English-only**
The embedding model and keyword tokenizer are optimized for English text. Non-English arXiv submissions (a small fraction of the corpus) may be retrieved with lower accuracy.

**Near-duplicate detection is result-set-scoped, not corpus-wide**
Multiple versions of the same paper (v1, v2, v3...) are deduplicated by arXiv ID. Separately, `src/rag_lit/dedup.py` (`find_near_duplicates`) flags substantively similar papers that ended up in the *same result set* under different arXiv IDs: after RRF fusion, the pipeline embeds each result's title+abstract with the same SBERT model used for dense retrieval (`all-MiniLM-L6-v2`) and computes pairwise cosine similarity across the (small, top-k) result list. Pairs scoring at or above `retrieval.near_duplicate_threshold` (default 0.92) have each other's arXiv ID recorded in `PaperResult.possible_duplicate_of`. This is O(N^2) on the result-set size (cheap for N<=25) but does **not** detect near-duplicates corpus-wide -- two near-identical papers that don't both make it into the same query's top-k will not be flagged. The 0.92 threshold favors precision over recall: it is intended to catch near-identical title+abstract text (e.g. the same work submitted under two different arXiv IDs), not merely topically-similar papers, which commonly score 0.6-0.85 with this embedding model.
