# RAGLR-A — RAG Literature Review Assistant (Demo)

A domain-general arXiv retrieval system built to explore multi-stage RAG pipelines for academic literature search. RAGLR-A combines sparse and dense retrieval with LLM-powered query expansion and relevance justification to surface relevant papers from across the full arXiv taxonomy (3M+ papers).

---

## Demo

![RAG Literature Review Assistant — Streamlit UI](docs/images/streamlit_demo.png)

[Full results page (PDF)](docs/images/streamlit_demo_full.pdf) — shows all 5 ranked papers with relevance justifications, contributions, and scores for the same query.

This project is **not hosted as a live demo**. The corpus is 3M+ arXiv papers backed by ~40GB of dense (ChromaDB), BM25, and keyword indexes, plus a locally-run Qwen2.5-3B model for keyword extraction. On CPU, a single query takes roughly **2 minutes end to end** (keyword extraction + HyDE generation + dual retrieval + per-result relevance justification). That's not a great experience for a "click a link from a resume" demo, and keeping 40GB of indexes + a 3B-parameter model warm on a hosted instance isn't practical for an occasional-traffic portfolio project.

If you don't want to run and build the indexes yourself — building the full dense index alone can take **overnight** (~24-28 hours on CPU, see [Build indexes](#4-build-indexes)) — watch the video walkthrough instead:

[![Watch the demo video](https://img.youtube.com/vi/-YcnMkPnDR4/hqdefault.jpg)](https://youtu.be/-YcnMkPnDR4)

Click the thumbnail above for a ~2 minute walkthrough of a query running end-to-end.

Instead, this README documents the architecture and how to run it locally — see [Running it yourself](#running-it-yourself) below.

---

## Architecture

Each query flows through the following stages:

1. **Qwen keyword prefilter** — a local Qwen2.5-3B-Instruct model extracts up to 18 search keywords and intersects them against a prebuilt inverted index, reducing the candidate pool before expensive retrieval
2. **Claude HyDE** — Claude Sonnet generates a hypothetical paper abstract representing an ideal result; this is used as the dense query vector (runs in parallel with step 1)
3. **Dual retrieval** — SBERT (all-MiniLM-L6-v2) dense retrieval via ChromaDB and BM25 sparse retrieval run in parallel over the candidate set
4. **Reciprocal Rank Fusion** — results from both retrievers are fused using RRF (k=60) to produce a single ranked list
5. **Claude justifications** — for each top-k result, Claude Sonnet generates a structured relevance justification: contribution summary, relevance reasoning, and relevance/specificity scores (1–10)

Every response also includes a `RetrievalTrace`: search-space reduction stats, latency breakdowns, and the keywords generated along the way.

### Interfaces
- **Streamlit UI** — query input, progress tracking, and results display
- **FastAPI REST server** — `/search` endpoint with interactive docs
- **CLI runner** — `scripts/run_query.py` for quick ad-hoc queries

### Data pipeline
- OAI-PMH harvester (`scripts/update_arxiv_data.py`) with incremental update support and recovery
- JSONL preprocessing and multi-index build (dense, BM25, keyword, metadata)

---

## Project structure

```
RAGLR-A/
├── api/                    FastAPI REST server
├── app/                    Streamlit UI
├── artifacts/              Built indexes (gitignored — ~40GB)
├── configs/
│   └── config.yaml         Runtime configuration
├── data/
│   ├── raw/                Raw OAI-PMH snapshot (gitignored)
│   └── processed/          Cleaned paper JSONL (gitignored)
├── docs/
│   └── EVALUATION.md       Retrieval evaluation methodology and results
├── prompts/                Versioned prompt templates
├── scripts/
│   ├── update_arxiv_data.py     Harvest arXiv via OAI-PMH
│   ├── orchestrate_indexing.py  Build all indexes end-to-end
│   ├── build_bm25_index.py      Build BM25 index
│   ├── build_keyword_index.py   Build keyword inverted index
│   ├── build_dense_index_fast.py Build ChromaDB dense index
│   ├── build_metadata_db.py     Build SQLite metadata index
│   ├── incremental_update.py    Apply incremental data updates
│   ├── run_query.py             CLI query runner
│   └── run_scheduler.py         Scheduled incremental harvesting
├── src/rag_lit/            Core library
│   ├── config.py           Config loader
│   ├── schemas.py          Pydantic output schemas
│   ├── pipeline.py         End-to-end pipeline
│   ├── preprocessing.py    Candidate ID filtering
│   ├── keyword_index.py    Inverted index build/query
│   ├── metadata_db.py      SQLite metadata index
│   ├── qwen_prefilter.py   Qwen keyword extractor
│   ├── hyde.py             Claude HyDE
│   ├── dense_retriever.py  SBERT + ChromaDB retriever
│   ├── bm25_retriever.py   BM25 retriever
│   ├── rrf.py              Reciprocal Rank Fusion
│   ├── justifier.py        Claude relevance justifier
│   └── rate_limiter.py     Demo rate limiting
└── tests/                  Unit and smoke tests
    └── eval/               Gold query set for retrieval evaluation
```

---

## Running it yourself

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

### 3. Harvest arXiv data

Test run (1,000 papers — recommended for trying this out):
```bash
python scripts/update_arxiv_data.py --max-records 1000
```

Full harvest (millions of records — takes time; be polite to arXiv):
```bash
python scripts/update_arxiv_data.py
```

Incremental update from last harvest:
```bash
python scripts/update_arxiv_data.py --incremental
```

### 4. Build indexes

```bash
python scripts/orchestrate_indexing.py
```

This builds the artifacts in `artifacts/`:
- `dense_index/` — ChromaDB vector store (SBERT embeddings)
- `bm25_index/` — BM25 index
- `keyword_inverted_index.pkl` / `keyword_index.sqlite3` — token → paper ID inverted index
- `metadata.sqlite3` — paper metadata index (categories, byte offsets)

### 5. Run the app

```bash
streamlit run app/streamlit_app.py
```

Enter a research question and click **Find Papers**.

> **Performance note:** with the full 3M-paper corpus on CPU, expect ~2 minutes per query (local Qwen keyword extraction + Claude HyDE + dual retrieval + per-result justification). A smaller corpus (e.g. the 1,000-paper test harvest) returns in seconds, since most of the latency comes from running the local Qwen model and per-result Claude calls over a large candidate set.

### FastAPI server

```bash
uvicorn api.main:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

**POST /search**
```json
{
  "query": "attention mechanisms in transformers",
  "top_k": 10,
  "use_qwen_prefilter": true,
  "use_claude_justification": true
}
```

### CLI

```bash
python scripts/run_query.py \
  --query "graph neural networks for drug discovery" \
  --top-k 5
```

Add `--no-qwen` to skip keyword prefiltering, `--no-justification` to skip Claude scoring.

---

## Running tests

```bash
# Unit tests — no data or API keys required
pytest tests/test_schemas.py tests/test_rrf.py tests/test_filters.py -v

# Pipeline smoke tests — uses mocks, no data or API keys required
pytest tests/test_pipeline_smoke.py -v
```

---

## Models used

| Role | Model |
|---|---|
| Keyword extraction | Qwen/Qwen2.5-3B-Instruct (local) |
| Dense embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| HyDE query expansion | claude-sonnet-4-6 (API) |
| Relevance justification | claude-sonnet-4-6 (API) |

---

## Evaluation

RAGLR-A is evaluated against a 14-query gold set (8 CS/ML + 2 biology + 2 math + 2 physics) with hand-curated known-relevant `arxiv_id`s, run through the real pipeline at `top_k=10`. Highlights from the latest run:

| Metric | Result |
|---|---|
| Prefilter recall (keyword filter never drops a known-relevant paper) | 1.000 |
| End-to-end Precision@10 / Recall@10 / NDCG@10 / MRR | 0.086 / 0.214 / 0.197 / 0.345 |
| Justifier decoy-discrimination gap (top-k vs. random papers) | 8.07 / 10 |

See **[docs/EVALUATION.md](docs/EVALUATION.md)** for the full methodology, the gold query set, HyDE-vs-raw-query ablation results, and per-query breakdowns.

---

## Corpus coverage

The harvested corpus spans the full arXiv taxonomy: Computer Science · Mathematics · Physics · Astrophysics · Condensed Matter · High Energy Physics · Nuclear Physics · Nonlinear Sciences · Statistics · Quantitative Biology · Quantitative Finance · Economics · Electrical Engineering and Systems Science · Other Physics
