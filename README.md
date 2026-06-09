# RAGLR-A — RAG Literature Review Assistant (Demo)

A domain-general arXiv retrieval demo built to explore multi-stage RAG pipelines for academic literature search. RAGLR-A combines sparse and dense retrieval with LLM-powered query expansion and relevance justification to surface relevant papers from across the full arXiv taxonomy.

---

## What's been built

### Core pipeline (complete)
- **Field filter** — narrows the corpus to papers matching selected academic fields (e.g. Computer Science, Statistics, Physics)
- **Qwen keyword prefilter** — a local Qwen2.5-3B-Instruct model extracts up to 18 search keywords and intersects them against a prebuilt inverted index, reducing the candidate pool before expensive retrieval
- **Claude HyDE** — Claude Sonnet generates a hypothetical paper abstract representing an ideal result; this is used as the dense query vector
- **Dual retrieval** — SBERT (all-MiniLM-L6-v2) dense retrieval via ChromaDB and BM25 sparse retrieval run in parallel over the candidate set
- **Reciprocal Rank Fusion** — results from both retrievers are fused using RRF (k=60) to produce a single ranked list
- **Claude justifications** — for each top-k result, Claude Sonnet generates a structured relevance justification: contribution summary, relevance reasoning, and relevance/specificity scores (1–10)
- **RetrievalTrace** — every response includes search-space reduction stats, latency breakdowns, and generated keywords

### Interfaces (complete)
- **Streamlit UI** — interactive field/subcategory selection, query input, and results display
- **FastAPI REST server** — `/search` and `/fields` endpoints with interactive docs
- **CLI runner** — `scripts/run_query.py` for quick ad-hoc queries

### Data pipeline (complete)
- OAI-PMH harvester with incremental update support and recovery
- JSONL preprocessing and multi-index build (dense, BM25, keyword)

### Infrastructure improvements
- Parallelized query pipeline (HyDE + dual retrieval run concurrently)
- ChromaDB batch limit handling and indexing recovery pipeline
- UTF-16 LE encoding support for log detection
- Configurable `top_k` presets

---

## Project structure

```
RAGLR-A/
├── api/                    FastAPI REST server
├── app/                    Streamlit UI
├── artifacts/              Built indexes (gitignored)
├── configs/
│   ├── config.yaml         Runtime configuration
│   └── arxiv_taxonomy.yaml Full arXiv field/category taxonomy
├── data/
│   ├── raw/                Raw OAI-PMH snapshot (gitignored)
│   └── processed/          Cleaned paper JSONL (gitignored)
├── evaluation/             Eval queries and runner
├── prompts/                Versioned prompt templates
├── scripts/
│   ├── update_arxiv_data.py   Harvest arXiv via OAI-PMH
│   ├── build_indexes.py       Build dense, BM25, keyword indexes
│   ├── run_query.py           CLI query runner
│   └── inspect_outputs.py     Inspect saved results
├── src/rag_lit/            Core library
│   ├── config.py           Config loader
│   ├── schemas.py          Pydantic output schemas
│   ├── pipeline.py         End-to-end pipeline
│   ├── preprocessing.py    Field/candidate filtering
│   ├── data_ingestion.py   JSONL load/save
│   ├── keyword_index.py    Inverted index build/query
│   ├── qwen_prefilter.py   Qwen keyword extractor
│   ├── hyde.py             Claude HyDE
│   ├── dense_retriever.py  SBERT + ChromaDB retriever
│   ├── bm25_retriever.py   BM25 retriever
│   ├── rrf.py              Reciprocal Rank Fusion
│   ├── justifier.py        Claude relevance justifier
│   ├── taxonomy.py         Taxonomy utilities
│   ├── logger.py           Logging setup
│   └── utils.py            General utilities
└── tests/                  Unit and smoke tests
```

---

## Setup

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

Test run (1,000 papers):
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
python scripts/build_indexes.py
```

This builds three artifacts in `artifacts/`:
- `dense_index/` — ChromaDB vector store (SBERT embeddings)
- `bm25_index.pkl` — BM25 index
- `keyword_inverted_index.pkl` — token → paper ID inverted index

---

## Usage

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Select academic fields, optionally restrict to specific arXiv subcategories, enter a research question, and click **Find Papers**.

### FastAPI server

```bash
uvicorn api.main:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

**POST /search**
```json
{
  "query": "attention mechanisms in transformers",
  "selected_fields": ["computer_science"],
  "top_k": 10,
  "use_qwen_prefilter": true,
  "use_claude_justification": true
}
```

**GET /fields** — list all available academic fields.

### CLI

```bash
python scripts/run_query.py \
  --query "graph neural networks for drug discovery" \
  --fields computer_science quantitative_biology \
  --top-k 10
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

## Supported arXiv fields

Computer Science · Mathematics · Physics · Astrophysics · Condensed Matter · High Energy Physics · Nuclear Physics · Nonlinear Sciences · Statistics · Quantitative Biology · Quantitative Finance · Economics · Electrical Engineering and Systems Science · Other Physics
