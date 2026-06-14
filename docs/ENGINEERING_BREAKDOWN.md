# Engineering Breakdown — RAGLR-A

A map of how the system is built — query pipeline, data pipeline, evaluation,
and engineering practices — with pointers to the doc that covers each piece
in depth. This doc intentionally stays high-level; it doesn't restate what's
already written elsewhere.

---

## 1. RAG query pipeline

`RagLiteraturePipeline.run()` (`src/rag_lit/pipeline.py`) runs six stages per
query:

1. **Keyword prefilter** (`qwen_prefilter.py`, `keyword_index.py`) — local
   Qwen2.5-0.5B extracts keywords, intersected with an inverted index to
   shrink the candidate set (with a full-corpus fallback).
2. **HyDE** (`hyde.py`) — Claude writes a hypothetical abstract used as the
   dense query vector; runs in parallel with step 1.
3. **Dual retrieval** — dense (`dense_retriever.py`, SBERT/ChromaDB) and
   sparse (`bm25_retriever.py`, bm25s) over the candidate set.
4. **RRF fusion** (`rrf.py`) — combines dense, BM25, delta-BM25, and the
   canonical-paper boost list (`canonical_boost.py`); short queries (<4
   words) also fuse a raw-query dense list.
5. **Justification** (`justifier.py`) — Claude scores each top-k result
   (contribution, relevance/specificity 1–10) concurrently.
6. **Near-duplicate flagging** (`dedup.py`) — pairwise cosine similarity over
   the top-k result embeddings.

Output is a `SearchResponse` (`schemas.py`): results + a `RetrievalTrace`
(latencies, candidate-set sizes, generated keywords/HyDE doc).

**Full stage-by-stage detail, config knobs, and the data model →
[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).** **Known trade-offs and tuning
rationale (skip-filter thresholds, tokenizer choices, HyDE failure modes,
etc.) → [LIMITATIONS.md](LIMITATIONS.md).**

---

## 2. Data & indexing pipeline

- **Harvest** (`scripts/update_arxiv_data.py`) — OAI-PMH, resumable, drops
  malformed/deleted records. Field provenance and schema →
  [DATA_CARD.md](DATA_CARD.md).
- **Initial build** (`scripts/orchestrate_indexing.py`) — builds the dense
  (ChromaDB), BM25, keyword, and metadata indexes from scratch (~24–28h CPU
  for 3M papers, one-time cost).
- **Steady state** (`scripts/incremental_update.py` via `run_scheduler.py`,
  twice daily) — harvests new papers, upserts into ChromaDB, rebuilds a small
  delta BM25 index, merges keyword tokens. The running pipeline hot-reloads
  these via `_maybe_reload()` (mtime checks each query) — no restart, no
  downtime.

**Build steps, timings, and artifact layout → [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)
("Index build process").** **Operational setup, process management, and the
monthly full-rebuild procedure → [setup.md](setup.md).**

---

## 3. Evaluation pipeline

`scripts/evaluate_retrieval.py` runs the 26-query gold set
(`tests/eval/gold_queries.yaml`) through the real pipeline once per query
(`debug=True`) and derives four analyses from that single response:

| Mode | Question |
|---|---|
| `prefilter` | Does the keyword filter ever drop a relevant paper? |
| `hyde` | Does HyDE-document dense search beat raw-query dense search? |
| `e2e` | Do final top-k results match the gold `relevant_ids`? |
| `calibration` | Do Claude's relevance/specificity scores discriminate top-k from random papers? |

Metric primitives (precision/recall/NDCG/MRR, bootstrap CI) live in
`src/rag_lit/eval_metrics.py`. Component ablations (`--no-qwen`,
`--no-justification` on `run_query.py`) isolate each stage's contribution
outside the eval harness.

**Full methodology, gold set design, headline numbers, issue-driven
improvement history, and known gaps → [EVALUATION.md](EVALUATION.md).**

---

## 4. Engineering practices

- **Config** (`config.py` + `configs/config.yaml`/`configs/arxiv_taxonomy.yaml`)
  — single source of runtime parameters; `ensure_project_dirs` creates
  missing artifact dirs.
- **Schemas** (`schemas.py`) — all pipeline I/O is Pydantic; field
  descriptions double as FastAPI docs and as guardrails (e.g. `rrf_score` is
  documented as ordinal, not cross-query comparable).
- **Resilience** — Claude calls (`hyde.py`, `justifier.py`) retry with
  backoff (`models.claude_max_retries`/`claude_timeout_seconds`); justifier
  concurrency is capped (`claude_justifier_max_concurrency`).
  `rate_limiter.py` is a thread-safe sliding-window limiter for demo
  rate-limiting (no external deps).
- **Testing** — unit tests for pure logic (RRF, eval metrics, filters, dedup,
  canonical boost, LaTeX cleanup, keyword index) need no I/O. Pipeline smoke
  tests (`tests/test_pipeline_smoke.py`) build a real `RagLiteraturePipeline`
  against a 5-paper corpus with every model/index mocked, covering response
  shape, ablations, debug output, category filtering, short-query fusion, and
  near-duplicate flagging — no models, indexes, or API keys required. Harvest
  and incremental-update tests cover OAI-PMH edge cases and delta rebuild
  atomicity.
- **Interfaces** — Streamlit (`app/streamlit_app.py`), FastAPI
  (`api/main.py`, `/search` + `/health`), CLI (`scripts/run_query.py`). All
  three load one `RagLiteraturePipeline`.

**Deployment, process management, logging → [setup.md](setup.md).**
**Performance tuning history (latency/memory optimizations, each tied to its
issue) → [EVALUATION.md](EVALUATION.md) and [LIMITATIONS.md](LIMITATIONS.md).**

---

## 5. Source map

```
src/rag_lit/
  pipeline.py         orchestrates all stages (run())
  qwen_prefilter.py   Qwen keyword extraction + stopping criteria
  keyword_index.py    inverted index build/query/merge
  hyde.py             Claude HyDE
  dense_retriever.py  SBERT + ChromaDB
  bm25_retriever.py   bm25s wrapper, tokenizer
  rrf.py              Reciprocal Rank Fusion
  canonical_boost.py  curated "signature paper" registry matching
  dedup.py            result-set near-duplicate detection
  justifier.py        Claude relevance/specificity scoring
  metadata_db.py      SQLite metadata + category index
  preprocessing.py    candidate filtering, text cleanup, reduction stats
  eval_metrics.py     P/R/NDCG/MRR/bootstrap CI primitives
  rate_limiter.py     sliding-window rate limiter
  config.py           config + taxonomy loader
  schemas.py          Pydantic models (Paper, SearchResponse, ...)

scripts/
  update_arxiv_data.py     OAI-PMH harvester
  orchestrate_indexing.py  full index build
  build_dense_index_fast.py / build_bm25_index.py / build_keyword_index.py / build_metadata_db.py
  incremental_update.py    twice-daily delta update
  run_scheduler.py         scheduler loop
  evaluate_retrieval.py    eval harness (prefilter/hyde/e2e/calibration)
  run_query.py             CLI runner
```
