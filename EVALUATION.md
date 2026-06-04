# Evaluation — RAGLR-A

## Goals

The evaluation framework for RAGLR-A targets two concerns:

1. **Retrieval quality** — do the top-k results contain papers genuinely relevant to the query?
2. **Pipeline diagnostics** — how much does each filtering stage reduce the search space, and at what latency?

Because RAGLR-A is a retrieval system without a fixed ground-truth test set, evaluation combines automated pipeline metrics with Claude-generated relevance scoring as a proxy for human judgment.

---

## Eval query set

The bundled query set (`evaluation/eval_queries.json`) contains representative queries drawn from common literature review topics:

| Query | Fields |
|---|---|
| Retrieval augmented generation for literature review | NLP, Machine Learning |
| Early exit transformers for efficient inference | Machine Learning, AI |
| Citation recommendation using language models | NLP, Information Retrieval |

Each entry specifies a `query` string and a `fields` list that constrains retrieval to relevant arXiv categories.

To extend the eval set, add entries to `evaluation/eval_queries.json` in the same format:

```json
{
  "query": "your research question here",
  "fields": ["computer_science", "statistics"]
}
```

Valid field keys are the top-level keys in `configs/arxiv_taxonomy.yaml` (e.g. `computer_science`, `mathematics`, `physics`).

---

## Running evaluation

```bash
python evaluation/run_evaluation.py
```

This runs the full pipeline for every query in the eval set and writes a timestamped results file to `outputs/`. Each run produces a JSON object per query containing the full `SearchResponse` (results + trace).

---

## Metrics tracked

### Search-space reduction

For each query the `RetrievalTrace` records:

| Metric | Description |
|---|---|
| `total_corpus_size` | Total papers in the index |
| `field_filtered_size` | Papers remaining after field filter |
| `keyword_filtered_size` | Papers remaining after Qwen prefilter |
| `reduction_percent_after_field_filter` | `(1 - field / total) × 100` |
| `reduction_percent_after_keyword_filter` | `(1 - keyword / total) × 100` |

Target reduction range after both filters: **86–99%**. If reduction is below 86%, the candidate set is too large and retrieval quality may degrade. If reduction exceeds 99%, the keyword filter is too aggressive and relevant papers may be excluded — the pipeline falls back to the field-filtered set automatically when candidate count drops below `min_prefilter_candidates` (default 500).

### Latency

| Metric | Description |
|---|---|
| `dense_latency_seconds` | SBERT encoding + ChromaDB ANN search |
| `bm25_latency_seconds` | BM25 scoring over candidate set |
| `total_latency_seconds` | Full pipeline including Claude calls |

Claude API calls (HyDE + justifications) typically dominate total latency.

### Relevance scores

Each `PaperResult` carries Claude-generated scores:

| Field | Scale | Description |
|---|---|---|
| `relevance_score` | 1–10 | How directly relevant the paper is to the query |
| `specificity_score` | 1–10 | How specific (vs. tangential) the match is |

These are used as a proxy for human relevance judgments. A well-tuned pipeline should consistently return mean `relevance_score ≥ 7` for the provided eval queries.

### Keyword quality

The `generated_keywords` field in `RetrievalTrace` lists the terms extracted by Qwen. Qualitatively, good keywords are:
- Specific academic phrases (e.g. `"attention mechanism"`, `"contrastive learning"`)
- Not redundant or overly generic (e.g. `"machine"`, `"data"`)

---

## Ablation comparisons

The eval runner supports ablation via command-line flags passed to `run_query.py`:

| Ablation | Flag | Effect |
|---|---|---|
| No keyword prefilter | `--no-qwen` | Skips Qwen extraction; uses full field-filtered set |
| No Claude justification | `--no-justification` | Returns results without relevance scores |

Run each ablation and compare `reduction_percent_after_keyword_filter`, `total_latency_seconds`, and result overlap to understand the contribution of each component.

---

## Known evaluation gaps

- **No gold-standard relevance labels.** Relevance scores rely on Claude as judge, which introduces model bias. Human annotation of a held-out query set would provide a stronger ground truth.
- **Corpus size dependency.** Reduction percentages and candidate counts depend heavily on how many papers were harvested. Numbers will differ significantly between a 1,000-paper test corpus and a multi-million-paper full harvest.
- **No recall measurement.** Without knowing which papers in the corpus are truly relevant to a query, precision can be approximated (via relevance scores) but recall cannot.
