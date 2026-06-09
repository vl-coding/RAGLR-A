# System Design — RAGLR-A

## Overview

RAGLR-A is a hybrid retrieval-augmented generation system built on top of arXiv metadata. It combines field-based filtering, sparse and dense retrieval, LLM query expansion (HyDE), and LLM-generated relevance justifications into a single end-to-end pipeline.

---

## Pipeline

```
User query
    │
    ▼
┌─────────────────────────────┐
│  1. Field filter             │  Narrows corpus to allowed arXiv categories
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  2. Qwen keyword prefilter   │  Qwen2.5-3B-Instruct extracts ≤18 keywords
│                             │  → intersected with inverted index
│                             │  → further reduces candidate set
└─────────────────────────────┘
    │
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
┌─────────────────────┐              ┌─────────────────────────┐
│  3. Claude HyDE      │              │  (BM25 uses raw query)   │
│  Generates a         │              └─────────────────────────┘
│  hypothetical        │
│  abstract as         │
│  dense query         │
└─────────────────────┘
    │
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
┌─────────────────────┐              ┌─────────────────────────┐
│  4. Dense retrieval  │              │  5. BM25 retrieval       │
│  SBERT embeddings   │              │  rank-bm25 over          │
│  + ChromaDB         │              │  tokenized abstracts     │
│  top-200 candidates │              │  top-200 candidates      │
└─────────────────────┘              └─────────────────────────┘
    │                                              │
    └──────────────────────┬───────────────────────┘
                           ▼
              ┌────────────────────────┐
              │  6. Reciprocal Rank    │
              │  Fusion (k=60)         │
              │  Single fused ranking  │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  7. Claude justifier   │
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

### 1. Field filter (`preprocessing.py`)

The arXiv taxonomy (`configs/arxiv_taxonomy.yaml`) defines 14 top-level fields, each with a list of arXiv category codes (e.g. `cs.AI`, `stat.ML`). Selecting `"all"` disables filtering. Any paper whose `categories` list overlaps the allowed set passes through.

### 2. Qwen keyword prefilter (`qwen_prefilter.py`, `keyword_index.py`)

`QwenKeywordExtractor` loads `Qwen/Qwen2.5-3B-Instruct` locally and prompts it to return a JSON list of ≤18 academic keywords for the query. Each keyword is tokenized and looked up in a prebuilt inverted index (`keyword_inverted_index.pkl`). The union of matching paper IDs is intersected with the field-filtered set. If the result is smaller than `min_prefilter_candidates` (default 500), the keyword filter is skipped and the full field-filtered set is used.

The inverted index maps lowercase tokens (≥3 characters, alphanumeric+hyphen) to sets of arXiv IDs. It is built at index time from paper titles and abstracts.

### 3. Claude HyDE (`hyde.py`)

Hypothetical Document Embeddings: Claude Sonnet is prompted to write a 3–5 sentence hypothetical academic abstract that would be highly relevant to the query. This abstract is used as the dense query vector instead of the raw query string. HyDE improves recall by bridging the vocabulary gap between short user queries and longer paper abstracts.

### 4. Dense retrieval (`dense_retriever.py`)

`sentence-transformers/all-MiniLM-L6-v2` encodes both papers (at index time) and the HyDE query (at search time) with L2-normalized embeddings. Vectors are stored in a ChromaDB persistent collection. At query time, a nearest-neighbor search returns up to `dense_candidates` (default 200) results from the candidate set, then post-filters by candidate ID.

### 5. BM25 retrieval (`bm25_retriever.py`)

`BM25Okapi` from `rank-bm25` is built over tokenized paper texts (title + abstract). At query time, BM25 scores are computed for all papers, then filtered to the candidate set and the top `bm25_candidates` (default 200) are returned.

### 6. Reciprocal Rank Fusion (`rrf.py`)

RRF combines the dense and BM25 ranked lists without requiring score calibration:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
              lists i
```

Default k=60. The fused list is sorted by descending RRF score, and per-document `dense_rank` and `bm25_rank` are tracked for diagnostics.

### 7. Claude justifier (`justifier.py`)

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
| `field_filtered_size` | After field filter |
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

models:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  qwen_model: Qwen/Qwen2.5-3B-Instruct
  claude_model: claude-sonnet-4-6
```

`configs/arxiv_taxonomy.yaml` defines the full arXiv field hierarchy. Each field entry has a `label` (human-readable) and `categories` (list of arXiv codes, or `"*"` for all).

---

## Interfaces

| Interface | Entry point | Notes |
|---|---|---|
| Streamlit UI | `app/streamlit_app.py` | Interactive field/subcategory selection |
| FastAPI | `api/main.py` | REST endpoints: `/search`, `/fields`, `/health` |
| CLI | `scripts/run_query.py` | `--query`, `--fields`, `--top-k`, `--no-qwen`, `--no-justification` |

---

## Index build process

Run once after each data harvest:

```
update_arxiv_data.py  →  arxiv_papers.jsonl
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       dense_index/    bm25_index.pkl   keyword_inverted_index.pkl
      (ChromaDB)       (BM25Okapi)       (token → {arxiv_id})
```

`scripts/build_indexes.py` orchestrates all three and writes `artifacts/manifest.json`.
