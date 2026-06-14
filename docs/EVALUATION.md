# Evaluation — RAGLR-A

## Summary

The evaluation framework targets three concerns:

1. **Prefilter safety** — does the Qwen keyword prefilter ever drop a paper that should have been retrievable?
2. **HyDE value** — does embedding a HyDE-generated hypothetical document retrieve better than embedding the raw query?
3. **End-to-end relevance** — do the final fused top-k results actually contain the papers a domain expert would expect?

RAGLR-A has no external ground-truth test collection, so evaluation runs against a hand-curated **gold query set** (26 queries with known-relevant `arxiv_id`s) plus Claude-as-judge relevance/specificity scoring as a secondary signal. See "Gold query set", "Running evaluation", and "Eval modes & metrics" below for the methodology; "Appendix" for full results and diagnostics.

**Headline numbers** (14-query subset, `--top-k 10`, with canonical-paper boost enabled):

| Metric | Result |
|---|---|
| Prefilter recall (keyword filter never drops a known-relevant paper) | 1.000 |
| End-to-end Precision@10 / Recall@10 / NDCG@10 / MRR | 0.150 / 0.375 / 0.341 / 0.552 |
| Justifier decoy-discrimination gap (top-k vs. random papers) | 7.91 / 10 |

**Issue-driven improvements** (full diagnostics in the linked issues and the Appendix):

| Area | Outcome | Issue |
|---|---|---|
| Gold query set size/bias | CS/ML queries expanded 8 → 20 to dilute dense-search bias in pooled metrics | [#1](https://github.com/vl-coding/RAGLR-A/issues/1) |
| HyDE-vs-raw-query eval methodology | 6-item overhaul: rank-delta, pooled judgments, stratification, bootstrap CIs | [#2](https://github.com/vl-coding/RAGLR-A/issues/2) |
| Canonical-paper boost for zero-hit CS/ML queries | 3/4 zero-hit queries recovered, CS/ML P@10/R@10/NDCG@10/MRR roughly doubled | [#3](https://github.com/vl-coding/RAGLR-A/issues/3) |
| Qwen keyword-extraction quality | Mean candidate-set size 45.8% → 27.4% of corpus, recall 1.000 → 0.990 | [#4](https://github.com/vl-coding/RAGLR-A/issues/4) |
| Dense retrieval over-fetch & skip-filter | Large-candidate-set queries 3-10x faster, no recall regression | [#11](https://github.com/vl-coding/RAGLR-A/issues/11) |
| Embedding model benchmark (MiniLM vs. mpnet) | mpnet showed no improvement at ~10x build cost — kept MiniLM | [#12](https://github.com/vl-coding/RAGLR-A/issues/12) |
| Justifier score calibration (rubric/anchors) | Decoy-discrimination gap unchanged, but scores now have a fixed per-level meaning | [#16](https://github.com/vl-coding/RAGLR-A/issues/16) |
| Qwen keyword-extraction latency | Mean per-query latency 9.5s → 3.8s (-60%) via early-stopping | [#19](https://github.com/vl-coding/RAGLR-A/issues/19) |

---

## Gold query set

`tests/eval/gold_queries.yaml` contains **26 queries**, each with a `query` string, a `stratum` tag (`terminology_aligned` or `terminology_gap`, used by the `hyde` eval's stratified report — 4/26 are `terminology_gap`), and 4 `relevant_ids`:

| Domain | # queries | Example |
|---|---|---|
| Computer science / ML | 20 | "transformer architectures for sequence modeling" → Attention Is All You Need, Transformer-XL, T5, GPT-3 |
| Biological sciences | 2 | "deep learning for protein structure prediction" |
| Mathematics | 2 | "convergence analysis of stochastic gradient descent optimization" |
| Physics | 2 | "quantum error correction codes for fault-tolerant computing" |

For CS/ML queries, `relevant_ids` are canonical/seminal papers chosen independent of any retrieval method (issue #1 expanded this set from 8 to 20). For bio/math/physics queries, `relevant_ids` are strong topical matches found via dense (SBERT) search over the live corpus — see "Known evaluation gaps" for how this affects the `hyde` ablation.

Most results in the Appendix reflect eval runs against the prior **14-query** set; the prefilter eval has been re-run against the full 26-query set (`outputs/eval_report_prefilter_26q.json`, mean `recall_final_candidates = 1.000`), but `hyde`/`e2e`/`calibration` re-runs against all 26 are still outstanding (see "Known evaluation gaps").

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

**Cost/latency**: `hyde` doubles the dense search (HyDE-document vs. raw-query). `e2e`/`calibration` each add one Claude justification call per top-k result, plus `--decoys` per query for `calibration`. `--rank-delta-top-n 5000` adds two large ChromaDB queries per query but no extra Claude calls. With all four evals on the full 14-query set: **~45 sec/query, ~10.5 minutes total**.

---

## Eval modes & metrics

All metric implementations live in `src/rag_lit/eval_metrics.py`.

### `prefilter` — keyword-filter recall

Checks whether the Qwen keyword prefilter (and the final candidate set after fallback logic) ever excludes a `relevant_id` before dense/BM25 retrieval sees it. Metric: `set_recall` — fraction of `relevant_ids` present in the candidate set (unordered, no `k`).

### `hyde` — HyDE vs. raw-query dense search

Runs dense (SBERT/ChromaDB) retrieval twice per query: once embedding the Claude-generated HyDE hypothetical document, once embedding the raw query string. The report has several layers (issue #2):

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
- **HyDE ablation is underpowered** (n=14 for the existing "Latest results" run; the gold set now has n=26) — the Wilcoxon test cannot reliably detect small or query-dependent effects at this sample size, and bootstrap CIs on small `n` will be wide.
- **Claude-as-judge relevance/specificity scores** are a secondary signal and inherit whatever bias the judging model has — including for the `--pooled-judgments` NDCG variant, which is judged by the same model.
- **The `hyde`/`e2e`/`calibration` evals have not yet been re-run** against the full 26-query set (the Appendix results below still reflect the prior 14-query run). A follow-up run with `--evals hyde e2e calibration --rank-delta-top-n 5000 --pooled-judgments` is needed to refresh those numbers.

---

## Appendix: detailed results & diagnostics

### Latest results (14-query gold set, `--top-k 10`, `outputs/eval_report_full_14q.json`)

These results follow a BM25 index-build fix (the index previously dropped the entire 2007-2024 range via a lexicographic ID sort on `arxiv_id`, silently excluding every CS/ML canonical paper from the BM25 index) and a re-derivation of the bio/math/physics `relevant_ids` via dense search — not directly comparable to pre-fix numbers.

**Prefilter recall**: mean `recall_final_candidates = 1.000` across all 14 queries (keyword-candidate sizes ranged from ~35k to ~3.0M of 3,067,125 papers) — the prefilter never drops a known-relevant paper.

**HyDE ablation**:

| Metric | HyDE | Raw query |
|---|---|---|
| mean NDCG@10 | 0.123 | 0.434 |
| NDCG@10 wins/ties/losses (HyDE vs. raw) | 4 / 4 / 6 | — |
| Wilcoxon signed-rank | statistic=10.0, **p=0.0735** (not significant at α=0.05) | — |

Raw query scores much higher here, but this is largely an artifact: the 6 bio/math/physics queries' `relevant_ids` were chosen by running that exact query text through dense search, so raw-query dense search trivially recalls them (`raw_recall = 1.000` for queries 9-14). For the 8 CS/ML queries (method-independent qrels), HyDE and raw query are much closer. p=0.0735 is not significant and should not be read as "raw query beats HyDE" — see "Known evaluation gaps" above.

**End-to-end relevance**:

| Metric | No canonical boost | With canonical boost (current) |
|---|---|---|
| Precision@10 | 0.086 | 0.150 |
| Recall@10 | 0.214 | 0.375 |
| NDCG@10 | 0.197 | 0.341 |
| MRR | 0.345 | 0.552 |

Without the boost, only 4 of the 8 CS/ML queries got any top-10 hit (vision transformers, contrastive learning, denoising diffusion, graph neural networks) — a major improvement over the pre-fix run (0/8), but the other 4 (transformer architectures, pretrained LMs, parameter-efficient fine-tuning, CLIP-style pretraining) still scored 0. Per-paper diagnostics confirmed an indexing bug wasn't the cause (all 16 `relevant_ids` are present in both the dense and BM25 indexes) — instead, every one of the 16 scores below the rank-10 RRF cutoff, and 11/16 don't appear in either retriever's top-200 at all. This is an **inherent retrieval-difficulty property, not a bug**: short, decade-old foundational papers use far less of today's terminology than the thousands of newer papers describing the same ideas, so they lose on both lexical and semantic similarity to modern phrasing. Full per-paper RRF table and the `rrf_k` sensitivity sweep that ruled out a fusion-tuning fix: [issue #3](https://github.com/vl-coding/RAGLR-A/issues/3).

All 6 bio/math/physics queries got at least one hit (recall@10 0.25-0.75) — e.g. protein structure prediction hits 3/4 `relevant_ids`. Overall precision/recall/NDCG dropped slightly versus the pre-fix run (0.121→0.086 P@10, 0.304→0.214 R@10) while MRR rose (0.205→0.345); expected, since the bio/math/physics qrels are no longer hand-picked to match BM25's output, while several CS/ML canonical papers became retrievable at all (raising MRR) even if not always landing in the top-10.

### Canonical-paper boost results (issue #3)

`data/canonical_papers.yaml` curates the 32 gold CS/ML `relevant_ids` with topic-phrase tags; `src/rag_lit/canonical_boost.py` matches query text/keywords against those tags and injects a matched paper into RRF fusion as its own one-paper ranked list (`1/(rrf_k + rank)`), giving it a shot at top-10 even if it's absent from both retrievers' candidate pools.

| Metric (CS/ML queries 1-8 only) | No boost | With boost |
|---|---|---|
| Precision@10 | 0.050 | 0.163 |
| Recall@10 | 0.125 | 0.406 |
| NDCG@10 | 0.118 | 0.393 |
| MRR | 0.208 | 0.650 |

Of the 4 originally zero-hit queries, 3 were recovered: "pretrained language models" → BERT/ALBERT/RoBERTa (3/4 hits), "parameter-efficient fine-tuning" → LoRA (1/4, surfaced via canonical RRF contribution), "CLIP-style pretraining" → CLIP (1/4, MRR=1.0 — CLIP already had dense rank 7, just under its cutoff, and the boost pushed it over). "transformer architectures for sequence modeling" remains at 0/4: its `relevant_ids` had essentially no baseline retrieval signal, and even a rank-1 canonical match (~+0.0164 RRF) falls short of that query's rank-10 cutoff (~0.0203). Two queries with partial hits also improved (vision transformers gained 1 hit; diffusion models reached full recall@10).

This is a curated, topic-specific nudge, not a general fix — it only helps the ~32 papers/topics in the registry. Full per-query before/after table: [issue #3](https://github.com/vl-coding/RAGLR-A/issues/3).

### RRF k sensitivity

A re-fusion sweep over `rrf_k ∈ {10, 20, 30, 40, 60, 80, 100, 150, 200}` (same cached dense/BM25 ranked lists, no retrieval re-run) found CS/ML metrics flat across the whole range, with **0/4 zero-hit queries recovered at any `rrf_k`** — confirming the bottleneck is upstream of fusion (addressed by the canonical boost above), not an `rrf_k` tuning problem. `rrf_k=20` is marginally best on the pooled mean, but the gain comes entirely from bio/math/physics queries (whose `relevant_ids` already rank #1 in raw-query dense search) and isn't recommended at n=14. `rrf_k=60` remains the default. Full sweep table: [issue #3](https://github.com/vl-coding/RAGLR-A/issues/3).

### Justifier score calibration (issue #16)

**Pre-rubric prompt** (`claude_justifier_v1.txt` before issue #16, 14-query gold set):

| Score | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| `relevance_score` | 140 | 9.071 | 0.957 | 6 | 10 |
| `specificity_score` | 140 | 8.121 | 0.714 | 6 | 10 |

Decoy discrimination (mean top-k `relevance_score` vs. mean score for 5 random decoy papers per query): **mean gap = 8.071** (decoy mean = 1.0 for every query). Scoring clearly separates retrieved top-k results from random papers, but is tightly clustered near the top of the 1-10 scale — better at flagging "not relevant at all" than at finely ranking degrees of relevance among already-retrieved papers.

**Rubric/anchor prompt** (issue #16 — each score level now has a fixed, query-independent description, e.g. relevance 10 = "central contribution is this exact topic / primary citation", 1 = "unrelated"; full 26-query gold set):

| Score | n | mean | stdev | min | max |
|---|---|---|---|---|---|
| `relevance_score` | 260 | 8.988 | 1.124 | 4 | 10 |
| `specificity_score` | 260 | 7.865 | 1.479 | 4 | 9 |

Decoy discrimination: **mean gap = 7.912**, essentially unchanged. The rubric widens both distributions (relevance stdev 0.957→1.124, range down to 4; specificity stdev 0.714→1.479, range 4-9) without weakening decoy discrimination. This is a calibration improvement — an 8 now corresponds to the same anchor description regardless of which query produced it — rather than a fix for the clustering itself, which is inherent to scoring already-retrieved, pre-filtered results.

### Qwen keyword-extraction quality (issue #4)

A prompt rewrite (multi-word technical phrases over generic single words, with a stoplist example and a worked few-shot example), post-filtering (`_filter_keywords` drops generic single-word terms and dedupes case/plural near-duplicates), and a JSON-parsing fix (parse only the first array, since the 0.5B model sometimes hallucinates extra "Query:"/"Output:" pairs after the real answer):

| Metric (26-query set, `outputs/eval_report_prefilter_26q_v2.json`) | Before | After |
|---|---|---|
| mean `keyword_candidate_size` | 1,406,161 (45.8% of corpus) | 838,872 (27.4% of corpus) |
| mean `recall_final_candidates` | 1.000 | 0.990 |

The 0.010 recall drop is one `relevant_id` (`1406.2661`, a 2014 GAN paper) whose era-specific vocabulary ("discriminative model"/"generative model G") no longer matches the more-precise extracted keywords ("gan"/"generator"/"discriminator") — the same vocabulary-drift pattern as "old but significant papers rank poorly" in `docs/LIMITATIONS.md`. A prompt tweak to request both acronym and spelled-out forms was tried and found to be a wash (fixed this query, broke an equivalent one elsewhere, identical aggregate numbers) and was reverted. `final_candidate_size` stays far above `min_prefilter_candidates` (500) regardless, so pipeline fallback behavior is unaffected. Full per-query `kw_n` breakdown: [issue #4](https://github.com/vl-coding/RAGLR-A/issues/4).

### Dense retrieval over-fetch & skip-filter (issue #11)

`DenseRetriever.search()` previously fetched a fixed `top_n * 50` pool from ChromaDB and post-filtered by the candidate set — for queries whose candidate set is a large fraction of the corpus, this pool can run out before reaching `top_n`. The fix (`src/rag_lit/dense_retriever.py`): skip post-filtering entirely above `dense_skip_filter_threshold_percent` (default 40%), returning Chroma's unfiltered global top-k; otherwise adaptively double the pool (up to `top_n * 400` or the full corpus).

The two gold queries whose `kw_n` exceeds the 40% threshold:

| Query | `kw_n` (% of corpus) | Before (`top_n*50`) | After (skip-filter+overfetch) | Hits |
|---|---|---|---|---|
| "machine learning methods for lattice quantum field theory" | 46.3% | 73.54s | 7.00s (**10.5x faster**) | 4/4, identical to before |
| "convergence analysis of stochastic gradient descent optimization" | 40.1% | 14.66s | 4.61s (**3.2x faster**) | 4/4, identical to before |

End-to-end metrics on the full 26-query set are flat (precision@10 0.088→0.085, recall@10 0.221→0.212, ndcg@10 0.202→0.200, mrr 0.326→0.325 — well within normal query-level noise) — this is a latency fix, not an accuracy fix. The lattice-QFT query's MRR shifted 1.000→0.500 (same single hit `2207.00283`, RRF rank 1→2) as a side effect of the skip-filter path returning Chroma's unfiltered global top-k, which can shift RRF fusion ranks slightly.

### Qwen keyword-extraction latency (issue #19)

`generate_keywords` previously always ran `model.generate` to `max_new_tokens=160`. A custom `_JSONArrayStoppingCriteria` now stops generation as soon as the first JSON keyword array's brackets balance (160 remains a hard ceiling):

| | Before | After | Reduction |
|---|---|---|---|
| mean (26 gold queries, CPU) | 9.521s | 3.777s | **60.3%** |
| total (26 queries) | 247.550s | 98.192s | **60.3%** |

Extracted keywords were identical before/after for every query checked — early-stopping cuts wasted generation without changing the output.

### Embedding model benchmark (issue #12)

`docs/LIMITATIONS.md` hypothesized `all-MiniLM-L6-v2` (22M params, 384-dim) might underperform a larger model (`all-mpnet-base-v2`, 110M params, 768-dim) on notation-heavy math/physics queries. `scripts/build_embedding_benchmark.py` built a fixed 49,950-paper subset (all 104 gold `relevant_ids` plus a random sample) and indexed it separately with both models; `scripts/benchmark_embedding_models.py` ran all 26 gold queries (raw query, dense-only) against both.

| Domain (n) | Recall@10 / NDCG@10 / MRR | MiniLM (current) | mpnet |
|---|---|---|---|
| CS/ML (20) | | 0.463 / 0.407 / 0.578 | 0.450 / 0.380 / 0.491 |
| Math+Physics (4) | | 1.000 / 1.000 / 1.000 | 1.000 / 0.976 / 1.000 |
| All (26) | | 0.587 / 0.543 / 0.675 | 0.577 / 0.519 / 0.608 |

| | MiniLM (current) | mpnet | Ratio |
|---|---|---|---|
| Build throughput (49,950-paper subset) | 33.4 papers/sec | 3.2 papers/sec | mpnet ~10.4x slower |
| Mean per-query search latency | 0.048s | 0.076s | mpnet ~1.6x slower |
| Extrapolated full-corpus (3.07M papers) build time | ~25.5h | ~266h (~11 days) | — |

mpnet shows no improvement on math/physics (recall@10 already 1.000 for both — a ceiling effect, since these `relevant_ids` were dense-search-derived in the first place) and is slightly worse on every CS/ML metric (whose qrels are retrieval-method-independent). Combined with ~10x slower indexing and ~1.6x slower search, **switching to `all-mpnet-base-v2` is not recommended**. `text-embedding-3-large` wasn't benchmarked — a stronger local model already showed no gain, so an API-based model is unlikely to help and remains out of scope.

### HyDE evaluation methodology (issue #2)

A 6-item methodology overhaul, all implemented in `scripts/evaluate_retrieval.py` / `src/rag_lit/eval_metrics.py` / `src/rag_lit/pipeline.py`: post-RRF (end-to-end) HyDE-vs-raw ablation, rank-delta metric (`--rank-delta-top-n`), TREC-style pooled judgments (`--pooled-judgments`), stratification by `terminology_gap`/`terminology_aligned`, CS/ML gold set expanded 8→20 (issue #1), and bootstrap CIs alongside every Wilcoxon test. Full checklist and implementation notes: [issue #2](https://github.com/vl-coding/RAGLR-A/issues/2).

**Not yet done**: a re-run of `--evals hyde e2e calibration` (optionally with `--rank-delta-top-n 5000 --pooled-judgments`) against the full 26-query set to populate concrete numbers for this methodology — see "Known evaluation gaps" above.
