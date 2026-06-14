# RAG Literature Review Assistant — Deployment Guide

## Overview

The system has two surfaces:

- **Streamlit app** (`app/streamlit_app.py`) — browser UI for interactive queries
- **FastAPI server** (`api/main.py`) — REST API for programmatic access

Both load the same `RagLiteraturePipeline`, which serves queries from pre-built indexes held in memory. A background scheduler (`scripts/run_scheduler.py`) updates the corpus and indexes twice a day with no downtime.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested on Python 3.14 |
| 16 GB RAM minimum | Pipeline metadata index (~3 M papers) uses ~4 GB; BM25 uses ~3 GB (set `retrieval.bm25_mmap: true` in `config.yaml` to memory-map the BM25 index instead of loading it fully — see `docs/LIMITATIONS.md` > Performance > Memory) |
| 50 GB free disk | Corpus JSONL ~3.8 GB, dense index ~8 GB, BM25 + keyword ~2 GB |
| GPU (optional) | CUDA GPU cuts initial index build from ~28 h to ~4 h; not needed at query time |
| Anthropic API key | Used by HyDE and justifier (Claude) |

---

## 1. Clone and install

```bash
git clone <repo-url> RAG-L-R-Assistant
cd RAG-L-R-Assistant
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

---

## 2. Environment variables

Create a `.env` file in the project root (already gitignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

The pipeline reads this via `python-dotenv` at startup. No other environment variables are required.

---

## 3. Initial corpus harvest

This fetches all arXiv metadata via OAI-PMH (~2.7 M papers, takes 6–10 hours depending on network):

```bash
python scripts/update_arxiv_data.py
```

The result is written to `data/processed/arxiv_papers.jsonl`. Progress is printed per OAI-PMH batch. If interrupted, re-running continues from the resumption token saved in `artifacts/update_state.json`.

To test with a small sample first:

```bash
python scripts/update_arxiv_data.py --max-records 5000
```

---

## 4. Build the pre-query indexes

Run the orchestrator, which chains all build steps automatically:

```bash
python scripts/orchestrate_indexing.py
```

What it runs in order:

| Step | Script | Time estimate |
|---|---|---|
| Dense index (ChromaDB, all papers) | `build_dense_index_fast.py` | 4–28 h (GPU/CPU) |
| BM25 index | `build_bm25_index.py` | 30–90 min |
| Keyword inverted index | `build_keyword_index.py` | 30–90 min |

The dense index builder is crash-safe — upserts are idempotent by `arxiv_id`, so a restart resumes without re-embedding already-indexed papers.

To build indexes manually (e.g., if you need to re-run one step):

```bash
python scripts/build_dense_index_fast.py --backend torch
python scripts/build_bm25_index.py
python scripts/build_keyword_index.py
```

---

## 5. Verify the build

After all indexes complete, check artifact sizes:

```bash
# Windows PowerShell
(Get-Item artifacts\dense_index\chroma.sqlite3).Length / 1GB
(Get-ChildItem artifacts\bm25_index).Count
Test-Path artifacts\keyword_index.sqlite3
```

Expected: ChromaDB sqlite3 ~1–2 GB, bm25_index directory with `index.*` and `arxiv_ids.npy`, keyword index SQLite DB present.

---

## 6. Start the application

### Option A — Streamlit (browser UI)

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. Pipeline loads on first page load (~30–60 s for metadata index + model loading).

### Option B — FastAPI (REST API)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Pipeline loads at startup. Health check: `GET /health`. Search endpoint: `POST /search`.

Example request:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformers for time series forecasting", "top_k": 10, "categories": ["cs.LG", "cs.AI"]}'
```

### Option C — CLI

```bash
python scripts/run_query.py \
  --query "diffusion models for image synthesis" \
  --top-k 10
```

Add `--no-qwen` to skip keyword prefiltering, `--no-justification` to skip Claude scoring. Results are printed to stdout and saved to `outputs/latest_results.json`.

---

## 7. Start the incremental update scheduler

Run this as a persistent background process alongside the app server:

```bash
python scripts/run_scheduler.py
```

This calls `scripts/incremental_update.py` at **00:00 and 12:00 UTC** each day. Each run:

1. Harvests new arXiv papers since the last run (OAI-PMH)
2. Filters to papers not already in the corpus
3. Appends new papers to `data/processed/arxiv_delta.jsonl`
4. Embeds and upserts new papers into ChromaDB
5. Rebuilds the delta BM25 index (new papers only, fast)
6. Merges new paper tokens into the keyword index (atomic file swap)

The running pipeline detects index file changes between queries and hot-reloads them — no restart required. Users never experience downtime or a stale corpus.

To run an update manually at any time:

```bash
python scripts/incremental_update.py
```

To preview what would change without writing anything:

```bash
python scripts/incremental_update.py --dry-run
```

To run updates more frequently (e.g., every 6 hours):

```bash
python scripts/run_scheduler.py --hours 0 6 12 18
```

Scheduler output is logged to `logs/scheduler.log`.

---

## 8. Process management (production)

For a long-running deployment, manage the app server and scheduler as system services or use a process manager.

### Windows — Task Scheduler

Create two scheduled tasks:

1. **App server** — runs `uvicorn api.main:app` at system startup, restarts on failure
2. **Scheduler** — runs `python scripts/run_scheduler.py` at system startup

Or use NSSM (Non-Sucking Service Manager) to wrap both as Windows services:

```
nssm install RAGLRA-API "C:\..\.venv\Scripts\python.exe" "C:\..RAG-L-R-Assistant\-m uvicorn api.main:app --host 0.0.0.0 --port 8000"
nssm install RAGLRA-Scheduler "C:\..\.venv\Scripts\python.exe" "C:\..RAG-L-R-Assistant\scripts\run_scheduler.py"
nssm start RAGLRA-API
nssm start RAGLRA-Scheduler
```

### Linux — systemd

Create `/etc/systemd/system/raglra-api.service`:

```ini
[Unit]
Description=RAG Literature Review API
After=network.target

[Service]
WorkingDirectory=/opt/RAG-L-R-Assistant
ExecStart=/opt/RAG-L-R-Assistant/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
EnvironmentFile=/opt/RAG-L-R-Assistant/.env
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/raglra-scheduler.service` similarly, replacing `ExecStart` with:

```
ExecStart=/opt/RAG-L-R-Assistant/.venv/bin/python scripts/run_scheduler.py
```

Then:

```bash
systemctl daemon-reload
systemctl enable --now raglra-api raglra-scheduler
```

---

## 9. Directory layout (post-build)

```
RAG-L-R-Assistant/
├── artifacts/
│   ├── dense_index/          # ChromaDB persistent store
│   ├── bm25_index/           # Main BM25 (full corpus)
│   ├── bm25_delta/           # Delta BM25 (new papers since last full build)
│   ├── keyword_index.sqlite3 # Token -> arxiv_ids inverted index
│   ├── metadata.sqlite3      # Paper metadata index (categories, byte offsets)
│   ├── known_ids.npy         # Set of all indexed arxiv_ids
│   ├── update_state.json     # Last OAI-PMH harvest date
│   └── manifest.json
├── data/
│   └── processed/
│       ├── arxiv_papers.jsonl    # Main corpus (~3.8 GB, stable)
│       └── arxiv_delta.jsonl     # Accumulates new papers between full rebuilds
├── logs/
│   ├── scheduler.log
│   └── ...
└── .env                      # API keys — never committed
```

---

## 10. Periodic full rebuild (recommended monthly)

The delta BM25 grows over time. Once it exceeds ~100 K papers, query quality may degrade slightly (BM25 scores are corpus-size dependent). Do a full rebuild monthly:

```bash
# 1. Stop the scheduler temporarily
# 2. Merge delta into main corpus
python scripts/merge_recovered_papers.py   # or write a merge_delta.py equivalent
# 3. Rebuild BM25 and keyword from the merged corpus
python scripts/build_bm25_index.py
python scripts/build_keyword_index.py
# 4. Clear the delta
del data\processed\arxiv_delta.jsonl      # Windows
# rm data/processed/arxiv_delta.jsonl    # Linux
# 5. Restart the scheduler
python scripts/run_scheduler.py
```

After step 3 completes, restart the pipeline server so it reloads the new main BM25 index and clears the delta state.

---

## 11. Logs

| Log file | Contains |
|---|---|
| `logs/scheduler.log` | Scheduler run times and update outcomes |
| `logs/build_dense.log` | Initial dense index build |
| `logs/build_bm25.log` | BM25 index build |
| `logs/build_keyword.log` | Keyword index build |
| `logs/orchestrate.log` | Orchestrator step sequencing |

Uvicorn and Streamlit write to stdout/stderr — redirect these to files or a logging service in production.
