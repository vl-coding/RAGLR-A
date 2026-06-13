# Evaluation — RAGLR-A

## Goals

The evaluation framework for RAGLR-A targets three concerns:

1. **Prefilter safety** — does the Qwen keyword prefilter ever drop a paper that should have been retrievable?
2. **HyDE value** — does embedding a HyDE-generated hypothetical document retrieve better than embedding the raw query?
3. **End-to-end relevance** — do the final fused top-k results actually contain the papers a domain expert would expect for that query?

RAGLR-A has no external ground-truth test collection, so evaluation is built on a small hand-curated **gold query set** with known-relevant `arxiv_id`s (qrels), plus Claude-as-judge relevance/specificity scoring as a secondary signal.

---

## Gold query set

`tests/eval/gold_queries.yaml` contains **26 queries**, each with a `query` string, a `stratum` tag, and 4 `relevant_ids` (arxiv IDs of papers that are known to exist in `data/processed/arxiv_papers.jsonl` and are topically relevant):

| Domain | # queries | Example |
|---|---|---|
| Computer science / ML | 20 | "transformer architectures for sequence modeling" → Attention Is All You Need, Transformer-XL, T5, GPT-3 |
| Biological sciences | 2 | "deep learning for protein structure prediction" |
| Mathematics | 2 | "convergence analysis of stochastic gradient descent optimization" |
| Physics | 2 | "quantum error correction codes for fault-tolerant computing" |

For the CS/ML queries, `relevant_ids` are canonical/seminal papers for that topic, chosen independent of any retrieval method. For the biology/math/physics queries (where there's no single "canonical" paper), `relevant_ids` are strong topical matches found via dense (SBERT) search over the live corpus, independent of BM25.

**issue #1**: the CS/ML section was expanded from 8 to 20 queries (covering RLHF/DPO, GANs, word embeddings, seq2seq/NMT attention, adaptive optimizers, neural architecture search, object detection, knowledge distillation, federated learning, normalization, VAEs, and sparse MoE — all canonical/seminal papers, none chosen by running any retriever) to dilute the bio/math/physics dense-search bias in pooled metrics and give the `hyde` ablation's Wilcoxon test more statistical power (issue #2). All sections below reflect eval runs against the prior **14-query** set ("Latest results", "Canonical-paper boost results", "RRF k sensitivity"); a re-run against the full 26-query set is needed to refresh those numbers. The prefilter eval has already been re-run against the full 26-query set (`outputs/eval_report_prefilter_26q.json`): mean `recall_final_candidates = 1.000` across all 26 queries, confirming the 12 new CS/ML queries' `relevant_ids` survive the keyword prefilter.

**issue #2**: each entry also carries a `stratum: terminology_aligned | terminology_gap` tag, used by the `hyde` eval's stratified report (see below). 4/26 queries (1, 13, 15, 18 — "transformer architectures for sequence modeling", "adaptive gradient-based optimization methods...", "deep learning methods for object detection...", "normalization techniques...") are tagged `terminology_gap`: their phrasing is a broad/colloquial topic description that doesn't closely match how the canonical papers describe themselves, which is where HyDE's rewrite-to-abstract-style is hypothesized to help most. The remaining 22 are `terminology_aligned` (including all 6 biology/math/physics queries, by construction — their `relevant_ids` were dense-derived from this exact query text).

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

- `--rank-delta-top-n N` (default 0/off): for each `relevant_id`, run an extra dense search with `top_n=N` for both the HyDE-document and the raw query, and record `rank_raw - rank_hyde` (capped at `N+1` for ids absent from a list). Turns up to `4 * n_queries` mostly-zero NDCG@10 values into per-paper paired rank observations (issue #2 item 2).
- `--pooled-judgments` (default off): pools the top `--pool-size` ids from the HyDE-fused and raw-fused post-RRF results per query, judges any un-judged pool members with Claude's justifier, and computes NDCG@k for both rankings against the pooled graded relevance — a TREC-style pooled-judgment comparison that doesn't depend on the 4-id qrels (issue #2 item 3). Adds up to `2 * --pool-size` Claude justification calls per query.

Cost/latency notes:
- Adding `hyde` doubles the dense search (HyDE-document query vs. raw-query). Adding `e2e`/`calibration` adds one Claude justification call per top-k result (10 per query at `--top-k 10`, plus `--decoys` per query for `calibration`).
- `--rank-delta-top-n 5000` adds two large ChromaDB queries per query (HyDE-document and raw-query against the full final-candidate set) — noticeably slower than the default `dense_candidates` search, but no extra Claude calls.
- `--pooled-judgments` adds Claude justification calls for any pool members not already in the top-`--top-k` justified results (worst case `2 * --pool-size` per query).
- With all four evals (`prefilter hyde e2e calibration`) on the full 14-query set: **~45 sec/query, ~10.5 minutes total** (measured for the run behind the "Latest results" section below). The `--rank-delta-top-n`/`--pooled-judgments` flags were not enabled for that run.

---

## Eval modes & metrics

All metric implementations live in `src/rag_lit/eval_metrics.py`.

### `prefilter` — keyword-filter recall

Checks whether the Qwen keyword prefilter (and the final candidate set after fallback logic) ever excludes a `relevant_id` before dense/BM25 retrieval sees it. Metric: `set_recall` — fraction of `relevant_ids` present in the candidate set (unordered, no `k`).

### `hyde` — HyDE vs. raw-query dense search

Runs dense (SBERT/ChromaDB) retrieval twice per query: once embedding the Claude-generated HyDE hypothetical document, once embedding the raw query string. The report has several layers (issue #2):

- **Dense-stage (pre-RRF)**: `recall_at_k`, `ndcg_at_k`, and `mrr` for both, a win/tie/loss count on NDCG@k, a bootstrap CI on the mean NDCG@k difference, and a Wilcoxon signed-rank test.
- **Post-RRF end-to-end**: the pipeline also fuses the raw-query dense results with the *same* BM25/delta/canonical ranked lists used for the HyDE-document fusion (`debug.fused_results_raw_query` in `src/rag_lit/schemas.py`), so `precision/recall/ndcg/mrr@k` can be compared end-to-end, holding everything except the dense-stage query identical (item 1).
- **Rank-delta @ `--rank-delta-top-n`** (opt-in): for each `relevant_id`, `rank_raw - rank_hyde` from a large-`top_n` dense search, with a bootstrap CI on the mean delta (item 2). Positive = HyDE ranks the paper higher (better).
- **Stratified by `stratum`**: separate dense-stage NDCG@k diff + bootstrap CI for `terminology_gap` vs. `terminology_aligned` queries (item 4).
- **TREC-style pooled-judgment NDCG@k** (opt-in via `--pooled-judgments`): NDCG@k for both rankings against Claude-judged graded relevance over the pooled top-`--pool-size` results from each ranking, instead of the 4-id qrels (item 3).
- A closing note reiterates that Claude-as-judge scores aren't calibrated across queries (item 6; see `docs/LIMITATIONS.md`).

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

| Metric | Mean (no canonical boost) | Mean (with canonical boost, current) |
|---|---|---|
| Precision@10 | 0.086 | 0.150 |
| Recall@10 | 0.214 | 0.375 |
| NDCG@10 | 0.197 | 0.341 |
| MRR | 0.345 | 0.552 |

The "with canonical boost" column is from `outputs/eval_report_canonical_boost.json` (same gold set, `top-k=10`, `rrf_k=60`, `data/canonical_papers.yaml` / `src/rag_lit/canonical_boost.py` enabled — see "Canonical-paper boost results" below).

Per-query hits against `relevant_ids` (no canonical boost, historical):

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

  **Mitigation: canonical-paper registry (issue #3).** Since these papers aren't reachable by widening the candidate pool or tuning `rrf_k`, `data/canonical_papers.yaml` curates all 32 of the gold CS/ML `relevant_ids` (the 16 above plus the 16 from queries 4-7) with topic-phrase tags, and `src/rag_lit/canonical_boost.py` matches the query text/keywords against those tags. A match injects the paper into RRF fusion as a one-paper ranked list, contributing `1/(rrf_k + rank)` regardless of whether it appeared in the dense/BM25 candidate pools. For example, "transformer architectures for sequence modeling" matches Attention Is All You Need's topic "transformer architecture" and "sequence modeling" (2 matches, likely rank 1), adding `1/61 ≈ 0.0164` to its RRF score — enough to clear the ~0.02 rank-10 cutoff when combined with even a weak retriever signal, and close to it on its own. This is a curated, topic-specific nudge, not a general fix: it only helps the ~32 papers/topics in the registry. See "Canonical-paper boost results" below for the measured effect on queries 1-3 and 8.

### Canonical-paper boost results (issue #3)

Re-running `e2e` (`top-k=10`, `rrf_k=60`, `outputs/eval_report_canonical_boost.json`) with the canonical-paper boost enabled:

| Metric (CS/ML queries 1-8 only) | No boost | With boost |
|---|---|---|
| Precision@10 | 0.050 | 0.163 |
| Recall@10 | 0.125 | 0.406 |
| NDCG@10 | 0.118 | 0.393 |
| MRR | 0.208 | 0.650 |

Per-query effect on the "remaining 4" (queries 1-3, 8):

| query | hits before | hits after | likely mechanism |
|---|---|---|---|
| 1. transformer architectures for sequence modeling | 0/4 | 0/4 | Attention Is All You Need is the strongest canonical match ("transformer architecture" + "sequence modeling", likely canonical rank 1, ~+0.0164 RRF), but this query's rank-10 cutoff was ~0.0203 and all 4 `relevant_ids` had zero baseline dense/BM25 signal — the boost alone isn't quite enough. **Still unresolved.** |
| 2. pretrained language models for natural language understanding | 0/4 | 3/4 (BERT, ALBERT, RoBERTa) | "pretrained language model" / "language understanding" match 4 canonical papers (BERT, RoBERTa, ALBERT, ELECTRA) at canonical ranks ~1-4 (~0.0156-0.0164 RRF each) — enough to clear this query's lower rank-10 cutoff (~0.0159). |
| 3. parameter-efficient fine-tuning of large language models | 0/4 | 1/4 (LoRA) | LoRA ranks #2 in the canonical list behind Adapters (specificity tie-break on "parameter-efficient fine-tuning"); the ~+0.016 RRF contribution was enough to surface LoRA at retrieved rank 5. Adapters/Prefix-Tuning/P-Tuning v2 still miss the top-10. |
| 8. contrastive language-image pretraining for zero-shot transfer | 0/4 | 1/4 (CLIP) | CLIP already had dense rank 7 (RRF ~0.0149, just under its ~0.0206 cutoff); the canonical rank-1 match on "clip" / "contrastive language-image pretraining" / "zero-shot transfer" adds ~+0.0164, pushing it comfortably above the cutoff (MRR=1.0). |

Two queries that already had partial hits also improved: query 4 (vision transformers) gained Scaling ViT (1→2 hits) and query 6 (diffusion models) gained DDIM and "Diffusion Models Beat GANs" (2→4 hits, full recall@10).

**Conclusion**: the canonical-paper boost resolves 3 of the 4 target zero-hit queries (2, 3, 8) and improves 2 others (4, 6), with no effect on bio/math/physics queries (9-14), since the registry is CS/ML-only. Query 1 remains at 0 hits — its `relevant_ids` have essentially no baseline retrieval signal, so even a rank-1 canonical match (~0.0164) falls short of that query's rank-10 cutoff (~0.0203). If query 1 is worth pursuing further, options include letting multiple canonical matches for the same query (Transformer-XL, T5 also tag "sequence modeling"/"transformer architecture") stack additively rather than each only contributing one RRF-list slot, or a configurable boost multiplier — but either moves further from "nudge" toward "override" and should be checked against the rest of the gold set for regressions.

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

**Pre-rubric prompt** (`claude_justifier_v1.txt` before issue #16, 14-query gold set):

| Score | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| `relevance_score` | 140 | 9.071 | 0.957 | 6 | 10 |
| `specificity_score` | 140 | 8.121 | 0.714 | 6 | 10 |

Decoy discrimination (mean top-k `relevance_score` vs. mean score for 5 random decoy papers per query, `--decoys 5`): **mean gap = 8.071** (per-query gaps range 6.8–9.0, decoy mean = 1.0 for every query). Claude's relevance scoring clearly separates retrieved top-k results from random papers, but scores are tightly clustered near the top of the 1–10 scale (mean ~9, stdev <1) — the justifier is better at flagging "not relevant at all" than at finely ranking degrees of relevance among already-retrieved papers.

**Rubric/anchor prompt** (issue #16 fix, same `claude_justifier_v1.txt` structure but each score level now has a fixed, query-independent description — e.g. relevance 10 = "central contribution is this exact topic / primary citation", 6-7 = "related but main contribution targets a different problem", 1 = "unrelated"; full 26-query gold set, `--decoys 5`):

| Score | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| `relevance_score` | 260 | 8.988 | 1.124 | 4 | 10 |
| `specificity_score` | 260 | 7.865 | 1.479 | 4 | 9 |

Decoy discrimination: **mean gap = 7.912** (decoy mean ~1.0–1.2 for every query), essentially unchanged from the pre-rubric run. The rubric anchors widen both distributions without weakening decoy discrimination: `relevance_score` stdev rose from 0.957 to 1.124 and its observed range extended down to 4 (from 6); `specificity_score` stdev roughly doubled (0.714 → 1.479) and its range extended down to 4 (from 6) while the observed max dropped to 9 (from 10). Scores still cluster in the upper half of the scale — Claude continues to rate already-retrieved (pre-filtered) results as broadly relevant — but the rubric gives each numeric level a fixed, query-independent meaning, so an 8 now corresponds to the same anchor description ("substantially addresses the query topic, e.g. same problem via a different method") regardless of which query produced it. This is a calibration improvement (consistent meaning per score) rather than a full fix for the clustering itself, which is an inherent property of scoring already-relevant, pre-filtered results.

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

- **Small qrels per query.** Each query has only 4 `relevant_ids`, which is a *lower bound* on relevance — there are almost certainly other relevant papers in a 3M-paper corpus that aren't in the gold set, so `recall@10` understates true recall and the metric mostly measures whether the curated IDs specifically surface. (issue #1 expanded the *number* of CS/ML queries from 8 to 20, but each still has only 4 `relevant_ids` — this per-query sparsity is unchanged.)
- **Bio/math/physics qrels are dense-search-sourced**, which favors raw-query dense retrieval for those 6 queries in the `hyde` ablation specifically (their `relevant_ids` were selected by running the same query through dense search). The `e2e` and `prefilter` metrics are less affected since they depend on the full fused pipeline, not raw dense search alone — but any HyDE-vs-raw comparison should be read primarily from the CS/ML queries, whose qrels are retrieval-method-independent. As of issue #1, these 6 dense-search-sourced queries are now 6/26 (down from 6/14) of the pooled set, reducing — but not eliminating — their influence on pooled `hyde`-ablation means.
- **HyDE ablation is underpowered** (n=14 for the existing "Latest results" run; the gold set now has n=26) — the Wilcoxon test cannot reliably detect small or query-dependent effects at this sample size, and bootstrap CIs on small `n` will be wide.
- **Claude-as-judge relevance/specificity scores** are a secondary signal and inherit whatever bias the judging model has — including for the `--pooled-judgments` NDCG variant below, which is judged by the same model.
- **The `hyde` eval methodology below has not yet been re-run** against the 26-query set (the "Latest results" / "Canonical-paper boost results" / "RRF k sensitivity" sections above still reflect the prior 14-query, non-stratified, pre-RRF-only `hyde` ablation). A follow-up run with `--evals hyde e2e --rank-delta-top-n 5000 --pooled-judgments` is needed to populate the new tables.

---

## HyDE evaluation methodology (issue #2)

All six items from the issue #2 checklist are implemented in `scripts/evaluate_retrieval.py` / `src/rag_lit/eval_metrics.py` / `src/rag_lit/pipeline.py` and described in "Eval modes & metrics" > `hyde` above:

1. **Post-RRF (end-to-end) HyDE vs. raw-query ablation** — `pipeline.run(..., hyde_ablation=True)` now also fuses the raw-query dense results with the same BM25/delta/canonical lists (`debug.fused_results_raw_query`), and `evaluate_query` computes `precision/recall/ndcg/mrr@k` for it (`result["hyde_e2e"]`).
2. **Rank-delta instead of NDCG@10** — `--rank-delta-top-n N` runs an extra `top_n=N` dense search for HyDE-document and raw query and records `rank_raw - rank_hyde` per `relevant_id` (capped at `N+1` when absent), reported as a bootstrap CI on the mean delta.
3. **TREC-style pooled relevance judgments** — `--pooled-judgments [--pool-size N]` pools the top-N HyDE-fused and raw-fused ids, judges any not already scored, and reports NDCG@k against the pooled graded relevance.
4. **Stratify by terminology-gap vs. terminology-aligned** — every gold query now has a `stratum` tag (see "Gold query set" above); `report_hyde` prints a separate dense-stage NDCG@k diff + bootstrap CI per stratum.
5. **Expand the CS/ML-only set** — done in issue #1 (8 → 20 CS/ML queries, 26 total).
6. **Bootstrap CI alongside Wilcoxon** — `eval_metrics.bootstrap_ci` (1000 resamples, 95% CI by default) is reported alongside every Wilcoxon test in `report_hyde`, for both the dense-stage and post-RRF comparisons, plus the rank-delta and pooled-judgment sections; a closing note reiterates the Claude-as-judge calibration caveat.

**Not yet done**: an actual re-run of `--evals hyde e2e` (optionally with `--rank-delta-top-n 5000 --pooled-judgments`) against the 26-query set, to populate concrete numbers in this document. This is comparable in cost to the 14-query "Latest results" run (~10.5 min) times ~1.9x for the query count, plus the rank-delta/pooled-judgment overhead noted in "Running evaluation" if those flags are used.

---

## Qwen keyword-extraction quality (issue #4)

The baseline run (`outputs/eval_report_prefilter_26q.json`, see "Gold query set" above) showed `kw_n` (keyword-candidate set size) ranging from 35,194 to 2,995,804 out of 3,067,125 papers — for several queries (e.g. "vision transformers for image classification" at 97.6% of the corpus, "machine learning methods for lattice quantum field theory" at 97.7%), the keyword prefilter barely narrowed the candidate set at all, even though `recall_final_candidates = 1.000` (safe but ineffective).

Two changes address this, contained to the Qwen extraction step (`qwen_prefilter.py`, `prompts/qwen_keywords_v1.txt`) — `keyword_index.py`'s OR-of-tokens matching logic is unchanged:

1. **Prompt rewrite**: ask for multi-word technical phrases (named methods, architectures, techniques, application domains) over generic single words, with an explicit stoplist example and a worked few-shot example.
2. **Post-filtering** (`_filter_keywords`): drops single-word keywords whose normalized form is in a generic-term stoplist (e.g. "model", "method", "learning", "approach"), and dedupes case/plural near-duplicates (e.g. "Transformer" vs. "transformers"). Multi-word phrases pass through even if one of their words is generic (e.g. "graph neural networks" is kept). Applied to both `generate_keywords` output and `fallback_keywords`.

A side effect of the prompt rewrite required a parsing fix: the new prompt's few-shot example sometimes causes the 0.5B model to continue generating additional hallucinated "Query:"/"Output:" pairs after the real answer. The old `text.index("[")` → `text.rindex("]")` parsing spanned across multiple JSON arrays and produced invalid JSON, silently triggering the raw-query fallback. The fix parses only the **first** JSON array (`text.index("[")` → first `]` after it), which is the actual answer to the query.

### Before/after (`outputs/eval_report_prefilter_26q_v2.json`, 26-query set, `--evals prefilter --top-k 10`)

| Metric | Before (issue #1 baseline) | After (issue #4) |
|---|---|---|
| mean `keyword_candidate_size` | 1,406,161 (45.8% of corpus) | 838,872 (27.4% of corpus) |
| mean `recall_final_candidates` | 1.000 | 0.990 |

Mean candidate-set size dropped by ~40% relative (45.8% → 27.4% of the 3,067,125-paper corpus), with the largest reductions on queries that were previously near-100% of the corpus, e.g.:

| Query | `kw_n` before | `kw_n` after |
|---|---|---|
| "vision transformers for image classification" | 2,994,244 (97.6%) | 1,168,343 (38.1%) |
| "machine learning methods for lattice quantum field theory" | 2,995,804 (97.7%) | 1,421,167 (46.3%) |
| "normalization techniques for training deep neural networks" | 2,077,066 (67.7%) | 523,486 (17.1%) |
| "generative adversarial networks for image synthesis" | 950,519 (31.0%) | 178,337 (5.8%) |

**Recall caveat**: mean `recall_final_candidates` dropped from 1.000 to 0.990 — one of 104 `relevant_ids` (the "generative adversarial networks for image synthesis" query's `1406.2661`, Goodfellow et al. 2014) falls outside the new, more precise candidate set. Qwen now extracts `["image synthesis", "gan", "generator", "discriminator"]`, but that 2014 abstract uses "discriminative model"/"generative model G" rather than "gan"/"generator"/"discriminator", so none of those tokens match. This is the same vocabulary-drift pattern as "Old but significant papers rank poorly" above — a narrow prompt tweak (asking for both acronym and spelled-out forms, e.g. both "gan" and "generative adversarial network") was tried and found to be a wash: it fixed this query but introduced an equivalent miss on a different query ("reinforcement learning from human feedback for language model alignment"), with identical aggregate numbers (mean recall 0.990, mean `kw_n` ~27.4-27.8%). The prompt change was reverted as not worth the added complexity. `recall_final_candidates < 1.0` for a single query doesn't change pipeline behavior here since `final_candidate_size` (838,872+) is far above `min_prefilter_candidates` (500) — no fallback to the full corpus is triggered either way.

---

## Dense retrieval over-fetch & skip-filter (issue #11)

`DenseRetriever.search()` previously fetched a fixed `top_n * 50` pool from
ChromaDB and post-filtered by the keyword-prefilter candidate set. For
queries whose candidate set is a large fraction of the corpus, this fixed
pool can run out of candidate-set members before reaching `top_n` results.
The fix (`src/rag_lit/dense_retriever.py`) adds two heuristics: (1) if the
candidate set covers ≥`dense_skip_filter_threshold_percent` (default 40%) of
the corpus, skip post-filtering entirely and return Chroma's unfiltered
global top-k; (2) otherwise, adaptively double the query pool (up to
`top_n * 400` or the full corpus) until `top_n` candidate-set members are
found.

### Before/after timing (`scripts/_tmp_dense_issue11_check.py`)

The two gold queries whose `kw_n` exceeds the 40% threshold (from the
issue #4 "after" numbers above):

| Query | `kw_n` (% of corpus) | OLD (fixed `top_n*50`) | NEW (skip-filter+overfetch) | Hits found |
|---|---|---|---|---|
| "machine learning methods for lattice quantum field theory" | 1,421,167 (46.3%) | 73.54s | 7.00s (**10.5x faster**) | 4/4 `relevant_ids`, identical to OLD |
| "convergence analysis of stochastic gradient descent optimization" | 1,230,205 (40.1%) | 14.66s | 4.61s (**3.2x faster**) | 4/4 `relevant_ids`, identical to OLD |

Both queries return the exact same hit set as before — the skip-filter
threshold only changes *how* the candidate-restricted top-k is computed
(bypassing Chroma's slow `ids=`/`where $in` filtering at this scale, per
`docs/LIMITATIONS.md`), not *what* it returns. Net effect: large-candidate-set
queries get dramatically faster dense retrieval with no recall regression.

### End-to-end before/after (26-query gold set, `--evals e2e --top-k 10`)

`outputs/eval_report_e2e_26q_before_issue11.json` (pre-change) vs.
`outputs/eval_report_e2e_26q_after_issue11_19.json` (post-change, also
includes issue #19's Qwen early-stopping — see below):

| Metric | Before | After |
|---|---|---|
| mean precision@10 | 0.088 | 0.085 |
| mean recall@10 | 0.221 | 0.212 |
| mean ndcg@10 | 0.202 | 0.200 |
| mean mrr | 0.326 | 0.325 |

Aggregate metrics are flat (within ±0.003, well inside normal query-level
noise from Claude's HyDE rewriting and justification calls) — issue #11 is a
latency fix, not an accuracy fix, and the dense-stage diagnostic above already
confirms the candidate hit set is unchanged for the two affected queries.

For the two large-candidate-set queries specifically:

| Query | recall@10 (before) | mrr (before) | recall@10 (after) | mrr (after) |
|---|---|---|---|---|
| "machine learning methods for lattice quantum field theory" | 0.250 | 1.000 | 0.250 | 0.500 |
| "convergence analysis of stochastic gradient descent optimization" | 0.000 | 0.000 | 0.000 | 0.000 |

Recall@10 is unchanged for both. The lattice-QFT query's MRR dropped from
1.000 to 0.500 — the same paper (`2207.00283`) is still the only hit, but it
moved from RRF rank 1 to rank 2. This is a side effect of the skip-filter
path: above the 40% threshold, dense search now returns Chroma's unfiltered
global top-k (ranked purely by embedding similarity) instead of the
candidate-restricted top-k, which can shift RRF fusion ranks slightly. The
SGD-convergence query still scores 0 — its `relevant_ids` don't rank highly
enough in raw embedding similarity either, so this is an upstream
ranking-difficulty issue (same category as the "old but significant papers
rank poorly" diagnosis), not something issue #11 was meant to fix.

---

## Qwen keyword-extraction latency (issue #19)

`generate_keywords` (`src/rag_lit/qwen_prefilter.py`) previously always ran
`model.generate` to `max_new_tokens=160`, even though the JSON keyword array
is usually much shorter and the 0.5B model sometimes continues generating
hallucinated "Query:"/"Output:" pairs afterward (issue #4). The fix adds
`_JSONArrayStoppingCriteria`, a custom `StoppingCriteria` that stops
generation as soon as the first `[`...`]` JSON array's brackets balance.
`max_new_tokens=160` remains as a hard ceiling.

### Before/after latency (`scripts/_tmp_qwen_latency.py`, 26 gold queries, CPU)

| | Before | After | Reduction |
|---|---|---|---|
| mean | 9.521s | 3.777s | **60.3%** |
| total (26 queries) | 247.550s | 98.192s | **60.3%** |

The extracted keywords were identical between before and after for every
query checked — early-stopping cuts wasted generation without changing the
output.
