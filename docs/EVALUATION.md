# Evaluation — RAGLR-A

## Goals

The evaluation framework for RAGLR-A targets three concerns:

1. **Prefilter safety** — does the Qwen keyword prefilter ever drop a paper that should have been retrievable?
2. **HyDE value** — does embedding a HyDE-generated hypothetical document retrieve better than embedding the raw query?
3. **End-to-end relevance** — do the final fused top-k results actually contain the papers a domain expert would expect for that query?

RAGLR-A has no external ground-truth test collection, so evaluation is built on a small hand-curated **gold query set** with known-relevant `arxiv_id`s (qrels), plus Claude-as-judge relevance/specificity scoring as a secondary signal.

---

## Gold query set

`tests/eval/gold_queries.yaml` contains **14 queries**, each with a `query` string and 4 `relevant_ids` (arxiv IDs of papers that are known to exist in `data/processed/arxiv_papers.jsonl` and are topically relevant):

| Domain | # queries | Example |
|---|---|---|
| Computer science / ML | 8 | "transformer architectures for sequence modeling" → Attention Is All You Need, Transformer-XL, T5, GPT-3 |
| Biological sciences | 2 | "deep learning for protein structure prediction" |
| Mathematics | 2 | "convergence analysis of stochastic gradient descent optimization" |
| Physics | 2 | "quantum error correction codes for fault-tolerant computing" |

For the CS/ML queries, `relevant_ids` are canonical/seminal papers for that topic. For the biology/math/physics queries (where there's no single "canonical" paper), `relevant_ids` are strong topical matches found via BM25 search over the live corpus.

To extend the set, add entries in the same format:

```yaml
- query: "your research question here"
  relevant_ids:
    - "2106.09685"
    - "1706.03762"
```

---

## Running evaluation

```powershell
$env:PYTHONPATH="."
python scripts/evaluate_retrieval.py --evals prefilter hyde e2e calibration --top-k 10 --output outputs/eval_report.json
```

`--evals` accepts any combination of `prefilter`, `hyde`, `e2e`, `calibration` (default: all four). Each gold query is run through the real pipeline **once** (`debug=True`, with `hyde_ablation` / `use_claude_justification` enabled as needed) and all requested analyses are derived from that single response — no redundant Qwen/HyDE/Claude calls.

Cost/latency notes:
- `prefilter` alone is cheapest (~5 min/query: Qwen keyword extraction + HyDE + one dense search + BM25 over up to ~3M candidates).
- Adding `hyde` doubles the dense search (HyDE-document query vs. raw-query). Adding `e2e`/`calibration` adds one Claude justification call per top-k result (10 per query at `--top-k 10`, plus `--decoys` per query for `calibration`).
- With `hyde` + `e2e` on the full 14-query set: ~12–15 min/query, ~3 hours total.

---

## Eval modes & metrics

All metric implementations live in `src/rag_lit/eval_metrics.py`.

### `prefilter` — keyword-filter recall

Checks whether the Qwen keyword prefilter (and the final candidate set after fallback logic) ever excludes a `relevant_id` before dense/BM25 retrieval sees it. Metric: `set_recall` — fraction of `relevant_ids` present in the candidate set (unordered, no `k`).

### `hyde` — HyDE vs. raw-query dense search

Runs dense (SBERT/ChromaDB) retrieval twice per query: once embedding the Claude-generated HyDE hypothetical document, once embedding the raw query string. Reports `recall_at_k`, `ndcg_at_k`, and `mrr` for both, a win/tie/loss count on NDCG@k, and a Wilcoxon signed-rank test comparing the two NDCG@k distributions.

### `e2e` — end-to-end relevance

Compares the final fused top-k (dense + BM25 via RRF, after Claude justification) against `relevant_ids` using `precision_at_k`, `recall_at_k`, `ndcg_at_k`, and `mrr`.

### `calibration` — justifier score calibration

Examines the distribution of Claude's `relevance_score` / `specificity_score` across all top-k results, and compares the mean top-k relevance score against the mean score for `--decoys` randomly sampled papers per query (the "gap" — larger is better discrimination).

---

## Latest results (14-query gold set, `--top-k 10`)

### Prefilter recall

`outputs/eval_report_prefilter_expanded.json` — **mean `recall_final_candidates = 1.000`** across all 14 queries (keyword-candidate sizes ranged from ~35k to ~3.0M out of 3,067,125 papers). The keyword prefilter never drops a known-relevant paper for this gold set.

### HyDE ablation

`outputs/eval_report_hyde_e2e_expanded.json`:

| Metric | HyDE | Raw query |
|---|---|---|
| mean NDCG@10 | 0.066 | 0.085 |
| NDCG@10 wins/ties/losses (HyDE vs. raw) | 3 / 8 / 3 | — |
| Wilcoxon signed-rank | statistic=8.5, **p=0.674** | (not significant) |

On this 14-query set, HyDE-document dense search does **not** show a statistically significant advantage over embedding the raw query — mean NDCG is actually slightly higher for raw-query embedding, and the Wilcoxon test is far from significant (p=0.674, n=14). This is consistent with the earlier 8-query CS/ML-only run (p=0.1875, also not significant, though that run favored HyDE on mean NDCG). HyDE's value likely depends heavily on the specific query phrasing rather than being a uniform win.

### End-to-end relevance

| Metric | Mean |
|---|---|
| Precision@10 | 0.121 |
| Recall@10 | 0.304 |
| NDCG@10 | 0.210 |
| MRR | 0.205 |

Results are bimodal by domain:

- **CS/ML queries (1–8)**: 0 hits for all 8 — none of the canonical seminal papers (e.g. "Attention Is All You Need", BERT, CLIP) appear in the final top-10 for their respective queries.
- **Biology/math/physics queries (9–14)**: 0.25–1.0 recall@10, with the lattice-QFT query hitting all 4 relevant_ids (recall@10 = 1.0).

This split is at least partly a property of how the gold IDs were chosen: the bio/math/physics `relevant_ids` were themselves sourced via BM25 search over this corpus, so they're biased toward papers the pipeline can lexically match. The CS/ML canonical papers are older (2017–2022), topically "obvious" to a human, but apparently not surfaced in the top 10 by the current dense+BM25+RRF fusion — worth investigating directly (see below).

---

## Ablation comparisons

`scripts/run_query.py` supports component ablations independent of the eval harness:

| Ablation | Flag | Effect |
|---|---|---|
| No keyword prefilter | `--no-qwen` | Skips Qwen extraction; searches the full corpus |
| No Claude justification | `--no-justification` | Returns results without relevance/specificity scores |

Run each ablation and compare result overlap and ranking against the default pipeline to isolate each component's contribution.

---

## Known evaluation gaps

- **Small qrels per query.** Each query has only 4 `relevant_ids`, which is a *lower bound* on relevance — there are almost certainly other relevant papers in a 3M-paper corpus that aren't in the gold set, so `recall@10` understates true recall and the metric mostly measures whether the curated IDs specifically surface.
- **Bio/math/physics qrels are BM25-sourced**, which somewhat favors lexical-match-friendly retrieval over the CS/ML qrels (canonical papers chosen independent of this corpus's retrieval behavior).
- **HyDE ablation is underpowered** (n=14) — the Wilcoxon test cannot reliably detect small or query-dependent effects at this sample size.
- **`calibration` has not yet been run on the expanded 14-query set** — the only calibration data available is from the original 8-query CS/ML set.
- **Claude-as-judge relevance/specificity scores** are a secondary signal and inherit whatever bias the judging model has.
