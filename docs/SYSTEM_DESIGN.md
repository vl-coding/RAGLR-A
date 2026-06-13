# System Design — RAGLR-A

## Overview

RAGLR-A is a hybrid retrieval-augmented generation system built on top of arXiv metadata. It combines a keyword-based candidate prefilter, sparse and dense retrieval, LLM query expansion (HyDE), and LLM-generated relevance justifications into a single end-to-end pipeline.

---

## Pipeline

```
User query
    │
    ▼
┌─────────────────────────────┐
│  1. Qwen keyword prefilter   │  Qwen2.5-0.5B-Instruct extracts ≤18 keywords
│                             │  → intersected with inverted index
│                             │  → reduces candidate set
└─────────────────────────────┘
    │
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
┌─────────────────────┐              ┌─────────────────────────┐
│  2. Claude HyDE      │              │  (BM25 uses raw query)   │
│  Generates a         │              └─────────────────────────┘
│  hypothetical        │
│  abstract as         │
│  dense query         │
└─────────────────────┘
    │
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
┌─────────────────────┐              ┌─────────────────────────┐
│  3. Dense retrieval  │              │  4. BM25 retrieval       │
│  SBERT embeddings   │              │  bm25s over              │
│  + ChromaDB         │              │  tokenized abstracts     │
│  top-200 candidates │              │  top-200 candidates      │
└─────────────────────┘              └─────────────────────────┘
    │                                              │
    └──────────────────────┬───────────────────────┘
                           ▼
              ┌────────────────────────┐
              │  5. Reciprocal Rank    │
              │  Fusion (k=60)         │
              │  Single fused ranking  │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  6. Claude justifier   │
              │  Per-paper: contribu-  │
              │  tion, justification,  │
              │  relevance score,      │
              │  specificity score     │
              └────────────────────────┘
                           │
                           ▼
                    SearchResponse
```

---

## Stage details

### 1. Qwen keyword prefilter (`qwen_prefilter.py`, `keyword_index.py`)

`QwenKeywordExtractor` loads `Qwen/Qwen2.5-0.5B-Instruct` locally and prompts it (`prompts/qwen_keywords_v1.txt`) to return a JSON list of ≤18 academic keywords for the query, favoring multi-word technical phrases over generic single words. Generation uses a `_JSONArrayStoppingCriteria` that stops as soon as the first JSON array's brackets balance (rather than always running to `max_new_tokens=160`), reducing typical CPU latency. The extracted (or raw-query-fallback) list is post-filtered by `_filter_keywords`, which drops generic single-word keywords (stoplist of terms like "model", "method", "learning") and dedupes case/plural near-duplicates while preserving multi-word phrases. Each remaining keyword is tokenized and looked up in a prebuilt inverted index stored in SQLite (`keyword_index.sqlite3`). The union of matching paper IDs is intersected with the full corpus. If the result is smaller than `min_prefilter_candidates` (default 500), the keyword filter is skipped and the full corpus is used.

The inverted index maps lowercase tokens (≥3 characters, alphanumeric+hyphen) to sets of arXiv IDs, stored as a `token -> comma-separated arxiv_ids` table. It is built at index time from paper titles and abstracts (`build_keyword_inverted_index` / `save_keyword_index_db` in `keyword_index.py`).

### 2. Claude HyDE (`hyde.py`)

Hypothetical Document Embeddings: Claude Sonnet is prompted to write a 3–5 sentence hypothetical academic abstract that would be highly relevant to the query. This abstract is used as the dense query vector instead of the raw query string. HyDE improves recall by bridging the vocabulary gap between short user queries and longer paper abstracts.

### 3. Dense retrieval (`dense_retriever.py`)

`sentence-transformers/all-MiniLM-L6-v2` encodes both papers (at index time) and the HyDE query (at search time) with L2-normalized embeddings. Vectors are stored in a ChromaDB persistent collection. At query time, a nearest-neighbor search returns up to `dense_candidates` (default 200) results from the candidate set, then post-filters by candidate ID. If the candidate set covers at least `dense_skip_filter_threshold_percent` (default 40%) of the corpus, filtering is skipped entirely and Chroma's unfiltered global top-k is used, since native ID-based filtering at that scale is ~50x slower than an unfiltered query. Otherwise, the result pool starts at `dense_candidates * 50` and doubles (up to `dense_candidates * 400` or the full corpus) until `dense_candidates` candidate-set members are found.

### 4. BM25 retrieval (`bm25_retriever.py`)

A `bm25s.BM25` index is built over tokenized paper texts (title + abstract). At query time, BM25 scores are computed for all papers, then filtered to the candidate set and the top `bm25_candidates` (default 200) are returned.

### 5. Reciprocal Rank Fusion (`rrf.py`)

RRF combines the dense and BM25 ranked lists without requiring score calibration:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
              lists i
```

Default k=60. The fused list is sorted by descending RRF score, and per-document `dense_rank` and `bm25_rank` are tracked for diagnostics.

### 6. Claude justifier (`justifier.py`)

For each of the top-k results, Claude Sonnet receives the query, paper title, and abstract and returns structured JSON:

```json
{
  "contribution":             "One-sentence summary of the paper's main contribution",
  "relevance_justification":  "Why this paper is relevant to the query",
  "relevance_score":          8,
  "specificity_score":        7
}
```

Scores are 1–10. If Claude returns malformed JSON, the raw text is stored in `relevance_justification` and the scores are `null`.

---

## Data model

### `Paper`
Parsed arXiv record stored in `arxiv_papers.jsonl`.

| Field | Type | Description |
|---|---|---|
| `arxiv_id` | str | arXiv identifier |
| `title` | str | Paper title |
| `abstract` | str | Paper abstract |
| `authors` | List[str] | Author names |
| `categories` | List[str] | arXiv category codes |
| `year` | int | Publication year |
| `url` | str | `https://arxiv.org/abs/<id>` |

### `PaperResult`
One entry in the pipeline output.

| Field | Type | Description |
|---|---|---|
| `rank` | int | Final RRF rank |
| `arxiv_id` | str | arXiv identifier |
| `rrf_score` | float | Fused RRF score |
| `dense_rank` | int \| null | Rank from dense retrieval |
| `bm25_rank` | int \| null | Rank from BM25 |
| `relevance_score` | float \| null | Claude score 1–10 |
| `specificity_score` | float \| null | Claude score 1–10 |
| `relevance_justification` | str \| null | Claude explanation |
| `contribution` | str \| null | Claude summary |

### `RetrievalTrace`
Per-query diagnostics included in every `SearchResponse`.

| Field | Description |
|---|---|
| `total_corpus_size` | Papers before any filter |
| `keyword_filtered_size` | After keyword prefilter |
| `reduction_percent_after_keyword_filter` | Search-space reduction % |
| `generated_keywords` | Keywords from Qwen |
| `hyde_document` | Hypothetical abstract from Claude |
| `dense_latency_seconds` | Dense retrieval wall time |
| `bm25_latency_seconds` | BM25 retrieval wall time |
| `total_latency_seconds` | Full pipeline wall time |

---

## Configuration

`configs/config.yaml` controls all runtime parameters:

```yaml
retrieval:
  dense_candidates: 200       # papers retrieved by dense search
  bm25_candidates: 200        # papers retrieved by BM25
  rrf_k: 60                   # RRF smoothing constant
  min_prefilter_candidates: 500  # fallback threshold for keyword filter
  dense_skip_filter_threshold_percent: 40  # skip dense candidate filtering above this % of corpus

models:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  qwen_model: Qwen/Qwen2.5-0.5B-Instruct
  claude_model: claude-sonnet-4-6
```

---

## Interfaces

| Interface | Entry point | Notes |
|---|---|---|
| Streamlit UI | `app/streamlit_app.py` | Query input, progress tracking, results display |
| FastAPI | `api/main.py` | REST endpoints: `/search`, `/health` |
| CLI | `scripts/run_query.py` | `--query`, `--top-k`, `--no-qwen`, `--no-justification` |

---

## Index build process

**Initial build** (one-time, or disaster recovery):

```
update_arxiv_data.py  →  arxiv_papers.jsonl
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       dense_index/      bm25_index/    keyword_index.sqlite3
      (ChromaDB)         (bm25s)        (token → {arxiv_id})
```

`scripts/orchestrate_indexing.py` runs `build_dense_index_fast.py`, `build_bm25_index.py`, `build_keyword_index.py`, and `build_metadata_db.py` end-to-end and writes `artifacts/manifest.json`. The dense build is the long pole (~24-28h on CPU for a 3M-paper corpus — see `docs/LIMITATIONS.md` > Performance > Index build time).

**Steady state** (twice daily, via `run_scheduler.py`): `scripts/incremental_update.py` harvests new papers since the last run, appends them to `arxiv_delta.jsonl`, embeds and upserts just those papers into the existing ChromaDB collection, rebuilds the small delta BM25 index, and merges new tokens into the keyword index — all in seconds-to-minutes. The full `orchestrate_indexing.py` pipeline is **not** re-run per harvest.
