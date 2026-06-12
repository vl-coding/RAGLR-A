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

For the CS/ML queries, `relevant_ids` are canonical/seminal papers for that topic, chosen independent of any retrieval method. For the biology/math/physics queries (where there's no single "canonical" paper), `relevant_ids` are strong topical matches found via dense (SBERT) search over the live corpus, independent of BM25.

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
- Adding `hyde` doubles the dense search (HyDE-document query vs. raw-query). Adding `e2e`/`calibration` adds one Claude justification call per top-k result (10 per query at `--top-k 10`, plus `--decoys` per query for `calibration`).
- With all four evals (`prefilter hyde e2e calibration`) on the full 14-query set: **~45 sec/query, ~10.5 minutes total** (measured for the run behind the "Latest results" section below).

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

## Latest results (14-query gold set, `--top-k 10`, `outputs/eval_report_full_14q.json`)

These results were produced **after** fixing a BM25 index-build bug (`scripts/build_bm25_index.py` was selecting the "most recent 1M papers" via a lexicographic sort on `arxiv_id`, which put every pre-2007 `category/YYMMNNN`-style id ahead of all `YYMM.NNNNN` ids and silently dropped the entire 2007–2024 range — including every CS/ML canonical paper in the gold set — from the BM25 index). The index now covers the 2M most recent papers by corrected chronological order, and the bio/math/physics `relevant_ids` were re-derived via dense search (see above), so this run is not directly comparable to the pre-fix numbers below.

### Prefilter recall

**mean `recall_final_candidates = 1.000`** across all 14 queries (keyword-candidate sizes ranged from ~35k to ~3.0M out of 3,067,125 papers). The keyword prefilter never drops a known-relevant paper for this gold set.

### HyDE ablation

| Metric | HyDE | Raw query |
|---|---|---|
| mean NDCG@10 | 0.123 | 0.434 |
| NDCG@10 wins/ties/losses (HyDE vs. raw) | 4 / 4 / 6 | — |
| Wilcoxon signed-rank | statistic=10.0, **p=0.0735** | (not significant at α=0.05) |

Raw-query dense search now scores much higher than before (mean NDCG@10 0.434 vs. the prior run's 0.085), but this is largely an artifact of how the bio/math/physics qrels were re-derived: those 6 queries' `relevant_ids` were chosen by running the *exact same query text* through dense search and picking from the results, so raw-query dense search trivially recalls them (`raw_recall = 1.000` for queries 9–14). For the 8 CS/ML queries — whose `relevant_ids` were chosen independent of any retrieval method — HyDE and raw query are much closer (both near-zero NDCG for queries 1–3 and 8, with HyDE ahead on queries 7–8 and behind on 9–14). The p=0.0735 result is not significant and should not be read as "raw query beats HyDE"; see "Known evaluation gaps" below.

### End-to-end relevance

| Metric | Mean |
|---|---|
| Precision@10 | 0.086 |
| Recall@10 | 0.214 |
| NDCG@10 | 0.197 |
| MRR | 0.345 |

Per-query hits against `relevant_ids`:

- **CS/ML queries (1–8)**: 4 of 8 now get at least one hit in the top-10 — vision transformers (ViT, 2010.11929), contrastive learning (SimCLR, 2002.05709), denoising diffusion (DDPM + DDIM, 2/4), and graph neural networks (GAT, 1710.10903). The other 4 (transformer architectures, pretrained language models, parameter-efficient fine-tuning, CLIP) still get 0 hits. This is a major improvement from the pre-fix run, where **all 8** CS/ML queries scored 0 — the BM25 fix recovers some, but not all, of the canonical papers into the top-10.

  Diagnosing the remaining 4: across all 16 `relevant_ids` for these queries, **none appear in BM25's top-200 candidates** (`bm25_candidates: 200`) — the canonical papers' lexical content just doesn't compete with thousands of more recent, more specifically-worded papers on the same topics in a 3M-paper corpus. Dense retrieval fares slightly better (e.g. CLIP's own paper, 2103.00020, ranks #9 in HyDE-document dense search), but a paper that's strong in only *one* of the two retrievers is penalized by RRF: dense-rank-9-only gives `1/(60+9) = 0.0145`, below the top-10 RRF cutoff (~0.018–0.021), while papers ranking moderately in *both* lists (e.g. dense #30 + BM25 #30 → `1/90 + 1/90 = 0.0222`) win out. This is an inherent property of RRF + a large modern corpus rather than a bug: it rewards cross-retriever consensus, and old/terse foundational papers often lose that consensus to newer papers that are both more lexically and semantically aligned with how the topic is phrased today.
- **Biology/math/physics queries (9–14)**: all 6 get at least one hit (recall@10 0.25–0.75), e.g. protein structure prediction hits 3/4 relevant_ids (recall=0.75).

Overall precision/recall/NDCG dropped slightly versus the pre-fix run (0.121→0.086 P@10, 0.304→0.214 R@10) while MRR rose (0.205→0.345). This is expected: the bio/math/physics qrels are no longer hand-picked to match BM25's output, so they're harder to hit exactly, while several CS/ML canonical papers are now retrievable at all (raising MRR by giving more queries a non-zero top hit) even if not always landing in the top-10.

### Justifier score calibration

| Score | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| `relevance_score` | 140 | 9.071 | 0.957 | 6 | 10 |
| `specificity_score` | 140 | 8.121 | 0.714 | 6 | 10 |

Decoy discrimination (mean top-k `relevance_score` vs. mean score for 5 random decoy papers per query, `--decoys 5`): **mean gap = 8.071** (per-query gaps range 6.8–9.0, decoy mean = 1.0 for every query). Claude's relevance scoring clearly separates retrieved top-k results from random papers, but scores are tightly clustered near the top of the 1–10 scale (mean ~9, stdev <1) — the justifier is better at flagging "not relevant at all" than at finely ranking degrees of relevance among already-retrieved papers.

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
- **Bio/math/physics qrels are dense-search-sourced**, which favors raw-query dense retrieval for those 6 queries in the `hyde` ablation specifically (their `relevant_ids` were selected by running the same query through dense search). The `e2e` and `prefilter` metrics are less affected since they depend on the full fused pipeline, not raw dense search alone — but any HyDE-vs-raw comparison should be read primarily from the 8 CS/ML queries, whose qrels are retrieval-method-independent.
- **HyDE ablation is underpowered** (n=14) — the Wilcoxon test cannot reliably detect small or query-dependent effects at this sample size.
- **Claude-as-judge relevance/specificity scores** are a secondary signal and inherit whatever bias the judging model has.
