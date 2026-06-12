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

  **Diagnosing the remaining 4** (queries 1–3 and 8 — transformer architectures, pretrained LMs, parameter-efficient fine-tuning, CLIP-style pretraining; tracked in [issue #3](https://github.com/vl-coding/RAGLR-A/issues/3)). Ruled out an indexing bug: all 16 `relevant_ids` are present in both the dense index (3,067,125 docs) and the BM25 index (2,000,000 docs) — 16/16 hits in each. The gap is purely a *ranking* problem.

  Per-query retrieval debug (rrf_k=60, dense_candidates=bm25_candidates=200) shows **every one of the 16 `relevant_ids` scores below the rank-10 RRF cutoff**, and 11/16 don't appear in either retriever's top-200 at all:

  | arxiv_id | paper | dense (HyDE) rank | dense (raw) rank | bm25 rank | rrf_score | rank-10 cutoff |
  |---|---|---|---|---|---|---|
  | 1706.03762 | Attention Is All You Need | — | — | — | 0.0000 | 0.0203 |
  | 1901.02860 | Transformer-XL | — | 199 | — | 0.0000 | 0.0203 |
  | 1910.10683 | T5 | — | — | — | 0.0000 | 0.0203 |
  | 2005.14165 | GPT-3 | — | — | — | 0.0000 | 0.0203 |
  | 1810.04805 | BERT | 43 | — | — | 0.0097 | 0.0159 |
  | 1907.11692 | RoBERTa | — | — | — | 0.0000 | 0.0159 |
  | 1909.11942 | ALBERT | — | — | — | 0.0000 | 0.0159 |
  | 2003.10555 | ELECTRA | — | — | — | 0.0000 | 0.0159 |
  | 2106.09685 | LoRA | — | — | — | 0.0000 | 0.0190 |
  | 1902.00751 | Adapters | — | — | — | 0.0000 | 0.0190 |
  | 2101.00190 | Prefix-Tuning | — | — | — | 0.0000 | 0.0190 |
  | 2110.07602 | P-Tuning v2 | 129 | 182 | — | 0.0053 | 0.0190 |
  | 2103.00020 | CLIP | 7 | — | — | 0.0149 | 0.0206 |
  | 2102.05918 | ALIGN | 103 | — | — | 0.0061 | 0.0206 |
  | 2201.12086 | BLIP | — | — | — | 0.0000 | 0.0206 |
  | 2205.01917 | CoCa | — | — | — | 0.0000 | 0.0206 |

  Even CLIP (2103.00020, dense rank 7, RRF 0.0149) misses the cutoff (0.0206) because it registers in only *one* retriever — RRF rewards cross-retriever consensus, so dense-#30 + BM25-#30 (`1/90 + 1/90 = 0.0222`) outscores dense-#7-only.

  An unfiltered raw-query search with `top_n=5000` (vs. the pipeline's 200) shows how far the misses are: ranks range from 182 (best case, P-Tuning v2 dense) to >5000 (9/16 don't appear in the top 5000 of a 3M-doc corpus) — e.g. Attention Is All You Need ranks BM25 #4579 / dense #2028, and GPT-3 doesn't appear in either. **This is an inherent retrieval-difficulty property, not a bug**: short, decade-old foundational papers use far less of today's terminology than the thousands of newer papers describing the same ideas, so they lose on both lexical and semantic similarity to modern phrasing. Raising `dense_candidates`/`bm25_candidates` to 5000 would barely help — even rank-182 scores `1/(60+182) ≈ 0.0041`, an order of magnitude below the rank-10 cutoff (~0.02).

### RRF k sensitivity

The same cached dense/BM25 ranked lists were re-fused at `rrf_k ∈ {10, 20, 30, 40, 60, 80, 100, 150, 200}` — a pure re-fusion sweep that doesn't touch `configs/config.yaml` or re-run retrieval (see [issue #3](https://github.com/vl-coding/RAGLR-A/issues/3)):

| rrf_k | P@10 (all) | R@10 (all) | NDCG@10 (all) | MRR (all) | P@10 (CS/ML) | R@10 (CS/ML) | NDCG@10 (CS/ML) | MRR (CS/ML) | 0-hit recovered |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.086 | 0.214 | 0.199 | 0.332 | 0.050 | 0.125 | 0.127 | 0.250 | 0/4 |
| 20 | 0.093 | 0.232 | 0.204 | 0.332 | 0.050 | 0.125 | 0.129 | 0.250 | 0/4 |
| 30 | 0.079 | 0.196 | 0.185 | 0.309 | 0.050 | 0.125 | 0.119 | 0.212 | 0/4 |
| **60 (current)** | 0.079 | 0.196 | 0.186 | 0.307 | 0.050 | 0.125 | 0.118 | 0.208 | 0/4 |
| 100 | 0.079 | 0.196 | 0.179 | 0.295 | 0.050 | 0.125 | 0.118 | 0.208 | 0/4 |
| 200 | 0.071 | 0.179 | 0.170 | 0.286 | 0.050 | 0.125 | 0.118 | 0.208 | 0/4 |

`rrf_k=20` is marginally best on the all-queries mean, but the gain comes entirely from bio/math/physics queries (whose `relevant_ids` already rank #1 in raw-query dense search, so a smaller `k` lets that #1 dominate). CS/ML columns are flat across the sweep, and **0/4 zero-hit queries are recovered at any `rrf_k`** — the bottleneck is upstream of fusion, so no `rrf_k` value fixes it. `rrf_k=60` is not a meaningfully worse choice than alternatives tested; switching to 20 would be a marginal, bio/math/physics-driven gain at n=14 and isn't recommended without a larger query set.
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

Tracked in [issue #1](https://github.com/vl-coding/RAGLR-A/issues/1) (qrels) and [issue #2](https://github.com/vl-coding/RAGLR-A/issues/2) (HyDE ablation methodology).

- **Small qrels per query.** Each query has only 4 `relevant_ids`, which is a *lower bound* on relevance — there are almost certainly other relevant papers in a 3M-paper corpus that aren't in the gold set, so `recall@10` understates true recall and the metric mostly measures whether the curated IDs specifically surface.
- **Bio/math/physics qrels are dense-search-sourced**, which favors raw-query dense retrieval for those 6 queries in the `hyde` ablation specifically (their `relevant_ids` were selected by running the same query through dense search). The `e2e` and `prefilter` metrics are less affected since they depend on the full fused pipeline, not raw dense search alone — but any HyDE-vs-raw comparison should be read primarily from the 8 CS/ML queries, whose qrels are retrieval-method-independent.
- **HyDE ablation is underpowered** (n=14) — the Wilcoxon test cannot reliably detect small or query-dependent effects at this sample size.
- **Claude-as-judge relevance/specificity scores** are a secondary signal and inherit whatever bias the judging model has.

---

## Proposed HyDE evaluation improvements

Tracking checklist in [issue #2](https://github.com/vl-coding/RAGLR-A/issues/2).

The current `hyde` ablation compares HyDE-document vs. raw-query dense search using `NDCG@10` against the same 4-id `relevant_ids` qrels used for `e2e`. At n=8 (CS/ML-only) or n=14 (all), with a metric that floors at 0 once the relevant doc falls past rank 10, this produces a single borderline p-value (p=0.0735) that's hard to act on. Some options that would give a clearer signal, roughly ordered by effort:

1. **Compare post-RRF, not just pre-RRF.** Today's `hyde` ablation only compares the dense-stage ranked lists. Add an e2e variant: run the pipeline twice (HyDE vs. raw query feeding dense, BM25/RRF identical) and compare final fused `precision/recall/ndcg/mrr@10` — this answers whether HyDE changes what the user actually sees, since RRF can amplify or wash out a dense-stage difference.

2. **Switch from NDCG@10 to rank-delta on `relevant_ids`.** Record `rank_raw − rank_hyde` per id with a large `top_n` (e.g. 5000) so most ids get a finite rank. This turns 8 mostly-zero NDCG@10 values into up to 32 paired per-paper observations, giving Wilcoxon far more data and sensitivity to effects like "HyDE moved this paper from rank 800 to 150" that NDCG@10 can't see.

3. **TREC-style pooled relevance judgments.** Pool the HyDE-top-50 and raw-top-50 per query, have Claude score every pooled document, and use those graded scores as the relevance vector for NDCG@k on both rankings — standard practice for comparing rankers without exhaustive qrels, and removes the 4-ids-out-of-3M sparsity problem.

4. **Stratify queries by expected HyDE benefit.** Tag each gold query as "terminology-gap" (colloquial/short, differs from paper phrasing) vs. "terminology-aligned" (already reads like an abstract), and report HyDE-vs-raw per stratum — a clean win in one and no effect in the other is far more actionable than one pooled p=0.07.

5. **Expand the CS/ML-only set.** Growing the 8 retrieval-method-independent CS/ML queries to ~20–24 (still canonical/seminal papers) would meaningfully increase Wilcoxon power without touching the bio/math/physics qrels' dense-search provenance issue.

6. **Bootstrap CI instead of/alongside Wilcoxon.** Resample queries with replacement (e.g. 1000x) and report a 95% CI on the mean NDCG/rank-delta difference — more interpretable than a single p-value at n=8/14, and distinguishes "not significant" from "no detectable effect at this sample size."
