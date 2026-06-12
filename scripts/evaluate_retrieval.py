"""Retrieval evaluation harness.

Runs the gold query set (tests/eval/gold_queries.yaml) through the real
pipeline and reports four things:

  prefilter   -- does the qwen keyword prefilter drop known-relevant papers
                 before dense/BM25 ever see them?
  hyde        -- does HyDE-document dense search beat embedding the raw
                 query directly, on Recall/NDCG/MRR @k?
  e2e         -- end-to-end Precision/Recall/NDCG/MRR @k of the final
                 fused top-k against the gold relevant_ids.
  calibration -- distribution of Claude's relevance_score/specificity_score,
                 and whether they discriminate top-k papers from random
                 "decoy" papers.

Each gold query is run through the pipeline ONCE (debug=True, optionally
hyde_ablation=True / use_claude_justification=True depending on which evals
were requested) and all four analyses are derived from that single response,
to avoid redundant Qwen/HyDE/Claude calls.

Cost note: with --evals all on the default 8-query gold set, this makes
roughly 8 HyDE calls + ~80 justification calls + (8 * --decoys) decoy
justification calls to the configured Claude model.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --evals prefilter hyde
    python scripts/evaluate_retrieval.py --top-k 10 --decoys 5 --output outputs/eval_report.json
"""
import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from src.rag_lit.config import load_config
from src.rag_lit.eval_metrics import (
    bootstrap_ci,
    load_gold_queries,
    mrr,
    ndcg_at_k,
    precision_at_k,
    rank_of,
    recall_at_k,
    set_recall,
)
from src.rag_lit.pipeline import RagLiteraturePipeline

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


EVAL_CHOICES = ["prefilter", "hyde", "e2e", "calibration"]


def _describe(values: List[float]) -> Dict[str, Optional[float]]:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _print_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def _fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(
    pipeline,
    entry,
    top_k,
    hyde_ablation,
    use_justification,
    rank_delta_top_n=0,
    pooled_judgments=False,
    pool_size=10,
):
    relevant_ids = set(entry["relevant_ids"])

    response = pipeline.run(
        query=entry["query"],
        top_k=top_k,
        use_qwen_prefilter=True,
        use_claude_justification=use_justification,
        debug=True,
        hyde_ablation=hyde_ablation,
    )

    debug = response.debug
    relevance = {doc_id: 1.0 for doc_id in relevant_ids}

    result = {
        "query": entry["query"],
        "relevant_ids": sorted(relevant_ids),
        "stratum": entry.get("stratum"),
    }

    # -- prefilter recall --
    kw_ids = set(debug.keyword_candidate_ids) if debug.keyword_candidate_ids is not None else None
    final_ids = set(debug.final_candidate_ids)
    result["prefilter"] = {
        "keyword_candidate_size": len(kw_ids) if kw_ids is not None else None,
        "final_candidate_size": len(final_ids),
        "recall_keyword_prefilter": set_recall(kw_ids, relevant_ids) if kw_ids is not None else None,
        "recall_final_candidates": set_recall(final_ids, relevant_ids),
    }

    # -- HyDE ablation: dense-stage (pre-RRF) --
    if hyde_ablation:
        hyde_ids = [r["doc_id"] for r in debug.dense_results]
        raw_ids = [r["doc_id"] for r in debug.dense_results_raw_query]
        result["hyde"] = {
            "hyde_recall@k": recall_at_k(hyde_ids, relevant_ids, top_k),
            "raw_recall@k": recall_at_k(raw_ids, relevant_ids, top_k),
            "hyde_ndcg@k": ndcg_at_k(hyde_ids, relevance, top_k),
            "raw_ndcg@k": ndcg_at_k(raw_ids, relevance, top_k),
            "hyde_mrr": mrr(hyde_ids, relevant_ids),
            "raw_mrr": mrr(raw_ids, relevant_ids),
        }

    # -- end-to-end relevance (HyDE-document fusion, the default pipeline) --
    retrieved_ids = [r.arxiv_id for r in response.results]
    result["e2e"] = {
        "precision@k": precision_at_k(retrieved_ids, relevant_ids, top_k),
        "recall@k": recall_at_k(retrieved_ids, relevant_ids, top_k),
        "ndcg@k": ndcg_at_k(retrieved_ids, relevance, top_k),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "hits": [aid for aid in retrieved_ids if aid in relevant_ids],
        "retrieved_ids": retrieved_ids,
    }

    # -- HyDE ablation: post-RRF end-to-end (issue #2 item 1) --
    if hyde_ablation:
        raw_fused_ids = [item["doc_id"] for item in debug.fused_results_raw_query][:top_k]
        result["hyde_e2e"] = {
            "raw_precision@k": precision_at_k(raw_fused_ids, relevant_ids, top_k),
            "raw_recall@k": recall_at_k(raw_fused_ids, relevant_ids, top_k),
            "raw_ndcg@k": ndcg_at_k(raw_fused_ids, relevance, top_k),
            "raw_mrr": mrr(raw_fused_ids, relevant_ids),
            "raw_hits": [aid for aid in raw_fused_ids if aid in relevant_ids],
            "raw_retrieved_ids": raw_fused_ids,
        }

    # -- HyDE ablation: rank-delta on relevant_ids at a large top_n (issue #2 item 2) --
    if hyde_ablation and rank_delta_top_n:
        hyde_large = pipeline.dense.search(
            query_text=response.trace.hyde_document,
            candidate_ids=final_ids,
            top_n=rank_delta_top_n,
        )
        raw_large = pipeline.dense.search(
            query_text=entry["query"],
            candidate_ids=final_ids,
            top_n=rank_delta_top_n,
        )
        hyde_large_ids = [r["doc_id"] for r in hyde_large]
        raw_large_ids = [r["doc_id"] for r in raw_large]

        per_id = []
        for rid in sorted(relevant_ids):
            r_hyde = rank_of(rid, hyde_large_ids)
            r_raw = rank_of(rid, raw_large_ids)
            capped_hyde = r_hyde if r_hyde is not None else rank_delta_top_n + 1
            capped_raw = r_raw if r_raw is not None else rank_delta_top_n + 1
            per_id.append({
                "arxiv_id": rid,
                "rank_hyde": r_hyde,
                "rank_raw": r_raw,
                "delta": capped_raw - capped_hyde,
            })
        result["rank_delta"] = {"top_n": rank_delta_top_n, "per_id": per_id}

    # -- justifier records (consumed by calibration and pooled judgments) --
    if use_justification:
        result["justifier_records"] = [
            {
                "arxiv_id": r.arxiv_id,
                "is_known_relevant": r.arxiv_id in relevant_ids,
                "relevance_score": r.relevance_score,
                "specificity_score": r.specificity_score,
            }
            for r in response.results
        ]

    # -- TREC-style pooled relevance judgments (issue #2 item 3) --
    if hyde_ablation and pooled_judgments:
        raw_fused_ids = result["hyde_e2e"]["raw_retrieved_ids"]
        hyde_pool = retrieved_ids[:pool_size]
        raw_pool = raw_fused_ids[:pool_size]
        pool_ids = sorted(set(hyde_pool) | set(raw_pool))

        known_scores = {
            rec["arxiv_id"]: rec["relevance_score"]
            for rec in result.get("justifier_records", [])
            if rec["relevance_score"] is not None
        }

        pooled_relevance = {}
        for doc_id in pool_ids:
            if doc_id in known_scores:
                pooled_relevance[doc_id] = known_scores[doc_id]
                continue
            paper = pipeline._load_paper(doc_id)
            judged = pipeline.justifier.justify(
                query=entry["query"], title=paper.title, abstract=paper.abstract
            )
            score = judged.get("relevance_score")
            if score is not None:
                pooled_relevance[doc_id] = score

        result["pooled_judgment"] = {
            "pool_size": pool_size,
            "pool_ids": pool_ids,
            "relevance": pooled_relevance,
            "hyde_ndcg_pooled@k": ndcg_at_k(hyde_pool, pooled_relevance, top_k),
            "raw_ndcg_pooled@k": ndcg_at_k(raw_pool, pooled_relevance, top_k),
        }

    return result


# ---------------------------------------------------------------------------
# Calibration: decoy discrimination
# ---------------------------------------------------------------------------

def evaluate_calibration(pipeline, gold, per_query_results, n_decoys, seed):
    all_records = [
        rec
        for result in per_query_results
        for rec in result.get("justifier_records", [])
    ]
    rel_scores = [r["relevance_score"] for r in all_records]
    spec_scores = [r["specificity_score"] for r in all_records]

    distribution = {
        "relevance_score": _describe(rel_scores),
        "specificity_score": _describe(spec_scores),
    }

    rng = random.Random(seed)
    all_ids = [m.arxiv_id for m in pipeline._all_meta]

    decoy_rows = []
    for entry, result in zip(gold, per_query_results):
        records = result.get("justifier_records", [])
        topk_scores = [r["relevance_score"] for r in records if r["relevance_score"] is not None]
        if not topk_scores:
            continue

        exclude = {r["arxiv_id"] for r in records} | set(entry["relevant_ids"])
        decoy_ids = []
        while len(decoy_ids) < n_decoys:
            candidate = rng.choice(all_ids)
            if candidate not in exclude and candidate not in decoy_ids:
                decoy_ids.append(candidate)

        decoy_scores = []
        for doc_id in decoy_ids:
            paper = pipeline._load_paper(doc_id)
            judged = pipeline.justifier.justify(
                query=entry["query"], title=paper.title, abstract=paper.abstract
            )
            score = judged.get("relevance_score")
            if score is not None:
                decoy_scores.append(score)

        topk_mean = statistics.mean(topk_scores)
        decoy_mean = statistics.mean(decoy_scores) if decoy_scores else None
        decoy_rows.append({
            "query": entry["query"],
            "decoy_ids": decoy_ids,
            "topk_mean_relevance": round(topk_mean, 3),
            "decoy_mean_relevance": round(decoy_mean, 3) if decoy_mean is not None else None,
            "gap": round(topk_mean - decoy_mean, 3) if decoy_mean is not None else None,
        })

    return distribution, decoy_rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_prefilter(per_query_results):
    print("\n=== Prefilter recall (does the Qwen keyword prefilter drop relevant papers?) ===")
    rows = []
    for r in per_query_results:
        p = r["prefilter"]
        rows.append([
            r["query"][:45],
            "-" if p["keyword_candidate_size"] is None else p["keyword_candidate_size"],
            _fmt(p["recall_keyword_prefilter"]),
            p["final_candidate_size"],
            _fmt(p["recall_final_candidates"]),
        ])
    _print_table(
        ["query", "kw_n", "kw_recall", "final_n", "final_recall"],
        rows,
    )
    mean_final = statistics.mean(r["prefilter"]["recall_final_candidates"] for r in per_query_results)
    print(f"\nmean recall_final_candidates={mean_final:.3f}")


def _wilcoxon_or_skip(a, b, label):
    if wilcoxon is not None and any(x != y for x, y in zip(a, b)):
        try:
            stat, p_value = wilcoxon(a, b)
            print(f"Wilcoxon signed-rank test ({label}): statistic={stat:.3f} p={p_value:.4f}")
        except ValueError as e:
            print(f"Wilcoxon test ({label}) skipped: {e}")
    else:
        print(f"Wilcoxon test ({label}) skipped (scipy unavailable or all differences are zero).")


def _diff_ci(hyde_values, raw_values, label):
    diffs = [h - r for h, r in zip(hyde_values, raw_values)]
    ci = bootstrap_ci(diffs)
    print(
        f"{label}: mean(hyde)={statistics.mean(hyde_values):.3f} "
        f"mean(raw)={statistics.mean(raw_values):.3f} "
        f"mean diff(hyde-raw)={ci['mean']:.3f} 95% CI=[{ci['low']:.3f}, {ci['high']:.3f}]"
    )


def report_hyde(per_query_results, top_k, rank_delta_top_n=0, pooled_judgments=False):
    print("\n=== HyDE ablation: dense-stage (pre-RRF) ===")
    rows = []
    hyde_ndcgs, raw_ndcgs = [], []
    wins = ties = losses = 0
    for r in per_query_results:
        h = r["hyde"]
        rows.append([
            r["query"][:45],
            _fmt(h["hyde_recall@k"]), _fmt(h["raw_recall@k"]),
            _fmt(h["hyde_ndcg@k"]), _fmt(h["raw_ndcg@k"]),
            _fmt(h["hyde_mrr"]), _fmt(h["raw_mrr"]),
        ])
        hyde_ndcgs.append(h["hyde_ndcg@k"])
        raw_ndcgs.append(h["raw_ndcg@k"])
        if h["hyde_ndcg@k"] > h["raw_ndcg@k"]:
            wins += 1
        elif h["hyde_ndcg@k"] < h["raw_ndcg@k"]:
            losses += 1
        else:
            ties += 1
    _print_table(
        ["query", "hyde_recall", "raw_recall", "hyde_ndcg", "raw_ndcg", "hyde_mrr", "raw_mrr"],
        rows,
    )
    print(f"\nHyDE vs raw-query NDCG@k: wins={wins} ties={ties} losses={losses}")
    _diff_ci(hyde_ndcgs, raw_ndcgs, "dense-stage NDCG@k")
    _wilcoxon_or_skip(hyde_ndcgs, raw_ndcgs, "dense-stage NDCG@k")

    # -- post-RRF end-to-end comparison (issue #2 item 1) --
    if all("hyde_e2e" in r for r in per_query_results):
        print(f"\n=== HyDE ablation: post-RRF end-to-end (top-{top_k}) ===")
        rows = []
        hyde_p, raw_p = [], []
        hyde_r, raw_r = [], []
        hyde_n, raw_n = [], []
        hyde_m, raw_m = [], []
        for r in per_query_results:
            e, he = r["e2e"], r["hyde_e2e"]
            rows.append([
                r["query"][:45],
                _fmt(e["precision@k"]), _fmt(he["raw_precision@k"]),
                _fmt(e["recall@k"]), _fmt(he["raw_recall@k"]),
                _fmt(e["ndcg@k"]), _fmt(he["raw_ndcg@k"]),
                _fmt(e["mrr"]), _fmt(he["raw_mrr"]),
            ])
            hyde_p.append(e["precision@k"]); raw_p.append(he["raw_precision@k"])
            hyde_r.append(e["recall@k"]); raw_r.append(he["raw_recall@k"])
            hyde_n.append(e["ndcg@k"]); raw_n.append(he["raw_ndcg@k"])
            hyde_m.append(e["mrr"]); raw_m.append(he["raw_mrr"])
        _print_table(
            ["query", "hyde_P", "raw_P", "hyde_R", "raw_R", "hyde_NDCG", "raw_NDCG", "hyde_MRR", "raw_MRR"],
            rows,
        )
        print()
        _diff_ci(hyde_p, raw_p, f"post-RRF P@{top_k}")
        _diff_ci(hyde_r, raw_r, f"post-RRF R@{top_k}")
        _diff_ci(hyde_n, raw_n, f"post-RRF NDCG@{top_k}")
        _diff_ci(hyde_m, raw_m, "post-RRF MRR")
        _wilcoxon_or_skip(hyde_n, raw_n, f"post-RRF NDCG@{top_k}")

    # -- rank-delta on relevant_ids at a large top_n (issue #2 item 2) --
    if rank_delta_top_n and all("rank_delta" in r for r in per_query_results):
        print(f"\n=== HyDE ablation: rank-delta @ top_n={rank_delta_top_n} ===")
        all_deltas = []
        finite_deltas = []
        for r in per_query_results:
            for item in r["rank_delta"]["per_id"]:
                all_deltas.append(item["delta"])
                if item["rank_hyde"] is not None or item["rank_raw"] is not None:
                    finite_deltas.append(item["delta"])
        ci = bootstrap_ci(all_deltas)
        print(
            f"n={len(all_deltas)} relevant_ids (rank_raw - rank_hyde, capped at "
            f"top_n+1 for ids absent from top {rank_delta_top_n}):"
        )
        print(f"  mean delta={ci['mean']:.1f}  95% CI=[{ci['low']:.1f}, {ci['high']:.1f}]")
        print("  (positive delta = HyDE ranks the relevant paper higher / better than raw query)")
        if finite_deltas:
            ci2 = bootstrap_ci(finite_deltas)
            print(
                f"  excluding ids absent from both lists (n={len(finite_deltas)}): "
                f"mean delta={ci2['mean']:.1f}  95% CI=[{ci2['low']:.1f}, {ci2['high']:.1f}]"
            )

    # -- stratified by terminology gap/alignment (issue #2 item 4) --
    print("\n=== HyDE ablation: stratified by query terminology (dense-stage NDCG@k) ===")
    for stratum in ("terminology_gap", "terminology_aligned"):
        subset = [r for r in per_query_results if r.get("stratum") == stratum]
        if not subset:
            continue
        hyde_vals = [r["hyde"]["hyde_ndcg@k"] for r in subset]
        raw_vals = [r["hyde"]["raw_ndcg@k"] for r in subset]
        _diff_ci(hyde_vals, raw_vals, f"{stratum} (n={len(subset)})")

    # -- TREC-style pooled relevance judgments (issue #2 item 3) --
    if pooled_judgments and all("pooled_judgment" in r for r in per_query_results):
        print(f"\n=== HyDE ablation: TREC-style pooled-judgment NDCG@{top_k} ===")
        rows = []
        hyde_vals, raw_vals = [], []
        for r in per_query_results:
            pj = r["pooled_judgment"]
            rows.append([
                r["query"][:45],
                pj["pool_size"],
                len(pj["pool_ids"]),
                _fmt(pj["hyde_ndcg_pooled@k"]),
                _fmt(pj["raw_ndcg_pooled@k"]),
            ])
            hyde_vals.append(pj["hyde_ndcg_pooled@k"])
            raw_vals.append(pj["raw_ndcg_pooled@k"])
        _print_table(["query", "pool_size", "pool_n", "hyde_ndcg", "raw_ndcg"], rows)
        print()
        _diff_ci(hyde_vals, raw_vals, "pooled-judgment NDCG@k")
        _wilcoxon_or_skip(hyde_vals, raw_vals, "pooled-judgment NDCG@k")

    print(
        "\nNote: Claude-as-judge relevance/specificity scores (used above for pooled "
        "judgments) are not calibrated across queries -- see docs/LIMITATIONS.md."
    )


def report_e2e(per_query_results, top_k):
    print(f"\n=== End-to-end relevance (top-{top_k} vs. gold relevant_ids) ===")
    rows = []
    for r in per_query_results:
        e = r["e2e"]
        rows.append([
            r["query"][:45],
            _fmt(e["precision@k"]), _fmt(e["recall@k"]), _fmt(e["ndcg@k"]), _fmt(e["mrr"]),
            ",".join(e["hits"]) or "-",
        ])
    _print_table(
        ["query", f"P@{top_k}", f"R@{top_k}", f"NDCG@{top_k}", "MRR", "hits"],
        rows,
    )
    for metric in ("precision@k", "recall@k", "ndcg@k", "mrr"):
        mean_val = statistics.mean(r["e2e"][metric] for r in per_query_results)
        print(f"mean {metric}={mean_val:.3f}", end="  ")
    print()


def report_calibration(distribution, decoy_rows):
    print("\n=== Justifier score calibration ===")
    print("Score distribution across all top-k results:")
    for name, stats in distribution.items():
        print(f"  {name}: n={stats['n']} mean={_fmt(stats['mean'])} stdev={_fmt(stats['stdev'])} "
              f"min={_fmt(stats['min'])} max={_fmt(stats['max'])}")

    print("\nDecoy discrimination (top-k mean relevance vs. random-paper mean relevance):")
    rows = [
        [d["query"][:45], _fmt(d["topk_mean_relevance"]), _fmt(d["decoy_mean_relevance"]), _fmt(d["gap"])]
        for d in decoy_rows
    ]
    _print_table(["query", "topk_mean", "decoy_mean", "gap"], rows)
    gaps = [d["gap"] for d in decoy_rows if d["gap"] is not None]
    if gaps:
        print(f"\nmean gap={statistics.mean(gaps):.3f} (larger gap = better discrimination; "
              f"near-zero or negative means the score doesn't separate relevant from random papers)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evals", nargs="+", choices=EVAL_CHOICES, default=EVAL_CHOICES)
    parser.add_argument("--gold", default="tests/eval/gold_queries.yaml")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--decoys", type=int, default=3, help="number of random decoy papers per query for calibration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/eval_report.json")
    parser.add_argument(
        "--rank-delta-top-n", type=int, default=0,
        help="if >0 and --evals includes 'hyde', also run an extra large-top_n dense "
             "search (HyDE-document and raw-query) and record rank_raw - rank_hyde for "
             "each relevant_id (issue #2 item 2). 0 disables (default).",
    )
    parser.add_argument(
        "--pooled-judgments", action="store_true",
        help="if set and --evals includes 'hyde', additionally pool the top "
             "--pool-size HyDE-fused and raw-fused results per query, judge any "
             "un-judged pool members with Claude, and compute NDCG@k against the "
             "pooled graded relevance (issue #2 item 3). Adds Claude API calls.",
    )
    parser.add_argument(
        "--pool-size", type=int, default=10,
        help="pool size per ranking for --pooled-judgments (default 10)",
    )
    args = parser.parse_args()

    load_dotenv()
    config = load_config()

    gold = load_gold_queries(args.gold)
    if not gold:
        raise SystemExit(f"No gold queries loaded from {args.gold}")

    evals = set(args.evals)
    needs_hyde_ablation = "hyde" in evals
    needs_justification = "e2e" in evals or "calibration" in evals
    if "calibration" in evals:
        evals.add("e2e")  # calibration consumes e2e's justifier records

    print(f"Loaded {len(gold)} gold queries from {args.gold}")
    print(f"evals={sorted(evals)} top_k={args.top_k}")

    pipeline = RagLiteraturePipeline(config)

    per_query_results = []
    for i, entry in enumerate(gold, start=1):
        print(f"\n[{i}/{len(gold)}] {entry['query']!r}")
        result = evaluate_query(
            pipeline,
            entry,
            top_k=args.top_k,
            hyde_ablation=needs_hyde_ablation,
            use_justification=needs_justification or args.pooled_judgments,
            rank_delta_top_n=args.rank_delta_top_n if needs_hyde_ablation else 0,
            pooled_judgments=needs_hyde_ablation and args.pooled_judgments,
            pool_size=args.pool_size,
        )
        per_query_results.append(result)

    report = {"config": vars(args), "per_query": per_query_results}

    if "prefilter" in evals:
        report_prefilter(per_query_results)
    if "hyde" in evals:
        report_hyde(
            per_query_results,
            args.top_k,
            rank_delta_top_n=args.rank_delta_top_n,
            pooled_judgments=args.pooled_judgments,
        )
    if "e2e" in evals:
        report_e2e(per_query_results, args.top_k)
    if "calibration" in evals:
        distribution, decoy_rows = evaluate_calibration(pipeline, gold, per_query_results, args.decoys, args.seed)
        report["calibration"] = {"distribution": distribution, "decoys": decoy_rows}
        report_calibration(distribution, decoy_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full report to {output_path}")


if __name__ == "__main__":
    main()
