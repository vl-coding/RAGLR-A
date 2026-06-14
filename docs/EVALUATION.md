# Evaluation — RAGLR-A

## Summary

The evaluation framework targets three concerns:

1. **Prefilter safety** — does the Qwen keyword prefilter ever drop a paper that should have been retrievable?
2. **HyDE value** — does embedding a HyDE-generated hypothetical document retrieve better than embedding the raw query?
3. **End-to-end relevance** — do the final fused top-k results actually contain the papers a domain expert would expect?

RAGLR-A has no external ground-truth test collection, so evaluation runs against a hand-curated **gold query set** (26 queries with known-relevant `arxiv_id`s, see below) plus Claude-as-judge relevance/specificity scoring as a secondary signal.

**Headline numbers** (26-query gold set, `--top-k 10`, with canonical-paper boost enabled, `outputs/eval_report_canonical_extended.json`):

| Metric | Result |
|---|---|
| Prefilter recall (keyword filter rarely drops a known-relevant paper — see issue #4) | 0.990 |
| End-to-end Precision@10 / Recall@10 / NDCG@10 / MRR | 0.154 / 0.385 / 0.375 / 0.599 |
| Justifier decoy-discrimination gap (top-k vs. random papers) | 8.04 / 10 |

**HyDE vs. raw-query dense search** (post-RRF, mean over 26 queries — differences are directionally in HyDE's favor but not statistically significant at this sample size; see "Known evaluation gaps"):

| Metric | HyDE | Raw query | Mean diff (HyDE − raw) | 95% CI |
|---|---|---|---|---|
| P@10 | 0.154 | 0.131 | +0.023 | [-0.027, 0.065] |
| R@10 | 0.385 | 0.327 | +0.058 | [-0.067, 0.164] |
| NDCG@10 | 0.375 | 0.283 | +0.091 | [-0.017, 0.196] |
| MRR | 0.599 | 0.410 | +0.189 | [0.013, 0.368] |

**Issue-driven improvements** (full diagnostics recorded as comments on the linked issues):

| Area | Outcome | Issue |
|---|---|---|
| Gold query set size/bias | CS/ML queries expanded 8 → 20 to dilute dense-search bias in pooled metrics | [#1](https://github.com/vl-coding/RAGLR-A/issues/1) |
| HyDE-vs-raw-query eval methodology | 6-item overhaul: rank-delta, pooled judgments, stratification, bootstrap CIs | [#2](https://github.com/vl-coding/RAGLR-A/issues/2) |
| Canonical-paper boost for zero-hit CS/ML queries | 3/4 zero-hit queries recovered, CS/ML P@10/R@10/NDCG@10/MRR roughly doubled | [#3](https://github.com/vl-coding/RAGLR-A/issues/3) |
| Canonical-papers registry extended to issue #1's 12 new CS/ML queries | 10/12 newly-covered queries went from 0 hits to nonzero; full 26-query P@10/R@10/NDCG@10/MRR 0.088/0.221/0.207/0.340 → 0.154/0.385/0.375/0.599 | extension of [#3](https://github.com/vl-coding/RAGLR-A/issues/3) |
| Qwen keyword-extraction quality | Mean candidate-set size 45.8% → 27.4% of corpus, recall 1.000 → 0.990 | [#4](https://github.com/vl-coding/RAGLR-A/issues/4) |
| Dense retrieval over-fetch & skip-filter | Large-candidate-set queries 3-10x faster, no recall regression | [#11](https://github.com/vl-coding/RAGLR-A/issues/11) |
| Embedding model benchmark (MiniLM vs. mpnet) | mpnet showed no improvement at ~10x build cost — kept MiniLM | [#12](https://github.com/vl-coding/RAGLR-A/issues/12) |
| Justifier score calibration (rubric/anchors) | Decoy-discrimination gap unchanged, but scores now have a fixed per-level meaning | [#16](https://github.com/vl-coding/RAGLR-A/issues/16) |
| Qwen keyword-extraction latency | Mean per-query latency 9.5s → 3.8s (-60%) via early-stopping | [#19](https://github.com/vl-coding/RAGLR-A/issues/19) |
| BM25 index memory footprint + category filtering | mmap-backed BM25 index + indexed SQL category lookup instead of full-corpus Python scans | [#5](https://github.com/vl-coding/RAGLR-A/issues/5) |

---

## Gold query set

`tests/eval/gold_queries.yaml` contains **26 queries**, each with a `query` string, a `stratum` tag (`terminology_aligned` or `terminology_gap`, used by the `hyde` eval's stratified report — 4/26 are `terminology_gap`), and 4 `relevant_ids`:

| Domain | # queries | Example |
|---|---|---|
| Computer science / ML | 20 | "transformer architectures for sequence modeling" → Attention Is All You Need, Transformer-XL, T5, GPT-3 |
| Biological sciences | 2 | "deep learning for protein structure prediction" |
| Mathematics | 2 | "convergence analysis of stochastic gradient descent optimization" |
| Physics | 2 | "quantum error correction codes for fault-tolerant computing" |

For CS/ML queries, `relevant_ids` are canonical/seminal papers chosen independent of any retrieval method. For bio/math/physics queries, `relevant_ids` are strong topical matches found via dense (SBERT) search over the live corpus — see "Known evaluation gaps" for how this affects the `hyde` ablation.

All four eval modes (`prefilter`, `hyde`, `e2e`, `calibration`) have been run against the full 26-query set; results are summarized above (`outputs/eval_report_canonical_extended.json`).

To extend the set, add entries in the same format:

```yaml
- query: "your research question here"
  stratum: terminology_aligned  # or terminology_gap
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

**issue #2 flags** (only take effect when `hyde` is in `--evals`):

```powershell
python scripts/evaluate_retrieval.py --evals hyde e2e --top-k 10 `
  --rank-delta-top-n 5000 --pooled-judgments --pool-size 10 --output outputs/eval_report_hyde_v2.json
```

- `--rank-delta-top-n N` (default 0/off): for each `relevant_id`, run an extra dense search with `top_n=N` for both the HyDE-document and the raw query, and record `rank_raw - rank_hyde` (capped at `N+1` for ids absent from a list). Turns mostly-zero NDCG@10 values into per-paper paired rank observations.
- `--pooled-judgments` (default off): pools the top `--pool-size` ids from the HyDE-fused and raw-fused post-RRF results per query, judges any un-judged pool members with Claude's justifier, and computes NDCG@k for both rankings against the pooled graded relevance — a TREC-style comparison that doesn't depend on the 4-id qrels. Adds up to `2 * --pool-size` Claude calls per query.

**Cost/latency**: `hyde` doubles the dense search (HyDE-document vs. raw-query). `e2e`/`calibration` each add one Claude justification call per top-k result, plus `--decoys` per query for `calibration`. `--rank-delta-top-n 5000` adds two large ChromaDB queries per query but no extra Claude calls. With all four evals enabled, expect roughly 30-60s per query depending on candidate-set size and `top_k`.

---

## Eval modes & metrics

All metric implementations live in `src/rag_lit/eval_metrics.py`.

### `prefilter` — keyword-filter recall

Checks whether the Qwen keyword prefilter (and the final candidate set after fallback logic) ever excludes a `relevant_id` before dense/BM25 retrieval sees it. Metric: `set_recall` — fraction of `relevant_ids` present in the candidate set (unordered, no `k`).

### `hyde` — HyDE vs. raw-query dense search

Runs dense (SBERT/ChromaDB) retrieval twice per query: once embedding the Claude-generated HyDE hypothetical document, once embedding the raw query string. The report has several layers:

- **Dense-stage (pre-RRF)**: `recall_at_k`, `ndcg_at_k`, and `mrr` for both, a win/tie/loss count on NDCG@k, a bootstrap CI on the mean NDCG@k difference, and a Wilcoxon signed-rank test.
- **Post-RRF end-to-end**: the pipeline also fuses the raw-query dense results with the *same* BM25/delta/canonical ranked lists used for the HyDE-document fusion (`debug.fused_results_raw_query` in `src/rag_lit/schemas.py`), so `precision/recall/ndcg/mrr@k` can be compared end-to-end, holding everything except the dense-stage query identical.
- **Rank-delta @ `--rank-delta-top-n`** (opt-in): for each `relevant_id`, `rank_raw - rank_hyde` from a large-`top_n` dense search, with a bootstrap CI on the mean delta. Positive = HyDE ranks the paper higher (better).
- **Stratified by `stratum`**: separate dense-stage NDCG@k diff + bootstrap CI for `terminology_gap` vs. `terminology_aligned` queries.
- **TREC-style pooled-judgment NDCG@k** (opt-in via `--pooled-judgments`): NDCG@k for both rankings against Claude-judged graded relevance over the pooled top-`--pool-size` results, instead of the 4-id qrels.
- A closing note reiterates that Claude-as-judge scores aren't calibrated across queries (see `docs/LIMITATIONS.md`).

### `e2e` — end-to-end relevance

Compares the final fused top-k (dense + BM25 via RRF, after Claude justification) against `relevant_ids` using `precision_at_k`, `recall_at_k`, `ndcg_at_k`, and `mrr`.

### `calibration` — justifier score calibration

Examines the distribution of Claude's `relevance_score` / `specificity_score` across all top-k results, and compares the mean top-k relevance score against the mean score for `--decoys` randomly sampled papers per query (the "gap" — larger is better discrimination).

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

Tracked in [issue #1](https://github.com/vl-coding/RAGLR-A/issues/1) (qrels) and [issue #2](https://github.com/vl-coding/RAGLR-A/issues/2) (HyDE ablation methodology).

- **Small qrels per query.** Each query has only 4 `relevant_ids` — a *lower bound* on relevance, since a 3M-paper corpus almost certainly contains other relevant papers not in the gold set. So `recall@10` understates true recall and mostly measures whether the curated IDs specifically surface.
- **Bio/math/physics qrels are dense-search-sourced**, which favors raw-query dense retrieval for those 6 queries in the `hyde` ablation specifically. The `e2e` and `prefilter` metrics are less affected since they depend on the full fused pipeline. Any HyDE-vs-raw comparison should be read primarily from the CS/ML queries, whose qrels are retrieval-method-independent.
- **HyDE ablation is underpowered even at n=26** — the Wilcoxon test (post-RRF NDCG@10, p=0.1194) cannot reliably detect small or query-dependent effects at this sample size, and bootstrap CIs remain wide (e.g. post-RRF MRR diff 95% CI = [0.013, 0.368]).
- **Claude-as-judge relevance/specificity scores** are a secondary signal and inherit whatever bias the judging model has — including for the `--pooled-judgments` NDCG variant, which is judged by the same model.
- **`--rank-delta-top-n`/`--pooled-judgments` have not yet been re-run** against the full 26-query set — the latest run (`outputs/eval_report_canonical_extended.json`) used the default flags. A follow-up run with `--rank-delta-top-n 5000 --pooled-judgments` would add per-paper rank-delta and TREC-style pooled NDCG numbers.
