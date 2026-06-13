"""
Dense-stage comparison of embedding models on the gold query set (issue #12).

Runs every query in tests/eval/gold_queries.yaml (raw query text, no HyDE/Claude)
against two benchmark ChromaDB indexes built by
scripts/build_embedding_benchmark.py -- one per embedding model -- over the same
fixed subset of the corpus, and reports recall@k / NDCG@k / MRR vs. the gold
`relevant_ids`, plus per-query encode+search latency.

This isolates embedding representation quality: both indexes cover the exact
same documents, so any difference in retrieval quality is attributable to the
embedding model, not corpus coverage.

Usage:
    python scripts/benchmark_embedding_models.py
    python scripts/benchmark_embedding_models.py --top-k 10 --output outputs/embedding_benchmark_report.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_lit.config import load_config
from src.rag_lit.dense_retriever import DenseRetriever
from src.rag_lit.eval_metrics import load_gold_queries, mrr, ndcg_at_k, recall_at_k

_GOLD_QUERIES_PATH = "tests/eval/gold_queries.yaml"
_MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"

# tests/eval/gold_queries.yaml is ordered: 20 CS/ML queries, then 2 biology,
# then 2 math, then 2 physics (26 total). There's no structured "domain"
# field, so domains are identified positionally.
_DOMAIN_SLICES = {
    "cs_ml": slice(0, 20),
    "biology": slice(20, 22),
    "math": slice(22, 24),
    "physics": slice(24, 26),
    "math_physics": slice(22, 26),
    "bio_math_physics": slice(20, 26),
    "all": slice(0, 26),
}

MODELS = {
    "minilm": {
        "persist_dir": "artifacts/benchmark_index_minilm",
        "model_key": "current",  # resolved from config at runtime
    },
    "mpnet": {
        "persist_dir": "artifacts/benchmark_index_mpnet",
        "model_key": _MPNET_MODEL,
    },
}


def _print(*args, **kwargs) -> None:
    print(*args, **kwargs, flush=True)


def evaluate_model(model_name: str, persist_dir: str, queries: List[dict], top_k: int) -> dict:
    _print(f"\n=== Evaluating {model_name} ({persist_dir}) ===")
    retriever = DenseRetriever(model_name=model_name, persist_dir=persist_dir)

    per_query = []
    for q in queries:
        relevant_ids = set(q.get("relevant_ids", []))

        start = time.time()
        results = retriever.search(query_text=q["query"], top_n=top_k)
        latency = time.time() - start

        retrieved_ids = [r["doc_id"] for r in results]
        relevance = {doc_id: 1.0 for doc_id in relevant_ids}

        per_query.append({
            "query": q["query"],
            "recall_at_k": round(recall_at_k(retrieved_ids, relevant_ids, top_k), 4),
            "ndcg_at_k": round(ndcg_at_k(retrieved_ids, relevance, top_k), 4),
            "mrr": round(mrr(retrieved_ids, relevant_ids), 4),
            "latency_seconds": round(latency, 4),
        })

    return {"model": model_name, "persist_dir": persist_dir, "per_query": per_query}


def _aggregate(per_query: List[dict], domain_slices: Dict[str, slice]) -> Dict[str, dict]:
    agg = {}
    for domain, sl in domain_slices.items():
        subset = per_query[sl]
        if not subset:
            continue
        n = len(subset)
        agg[domain] = {
            "n": n,
            "mean_recall_at_k": round(sum(r["recall_at_k"] for r in subset) / n, 4),
            "mean_ndcg_at_k": round(sum(r["ndcg_at_k"] for r in subset) / n, 4),
            "mean_mrr": round(sum(r["mrr"] for r in subset) / n, 4),
            "mean_latency_seconds": round(sum(r["latency_seconds"] for r in subset) / n, 4),
        }
    return agg


def _print_comparison(reports: Dict[str, dict], domain_slices: Dict[str, slice]) -> None:
    for domain in domain_slices:
        _print(f"\n--- {domain} ---")
        headers = ["model", "n", "recall@k", "ndcg@k", "mrr", "latency(s)"]
        rows = []
        for label, report in reports.items():
            agg = report["aggregate"].get(domain)
            if not agg:
                continue
            rows.append([
                label,
                agg["n"],
                f"{agg['mean_recall_at_k']:.3f}",
                f"{agg['mean_ndcg_at_k']:.3f}",
                f"{agg['mean_mrr']:.3f}",
                f"{agg['mean_latency_seconds']:.3f}",
            ])
        widths = [max(len(str(r[i])) for r in (rows + [headers])) for i in range(len(headers))]
        _print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        for row in rows:
            _print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="outputs/embedding_benchmark_report.json")
    args = parser.parse_args()

    config = load_config()
    queries = load_gold_queries(_GOLD_QUERIES_PATH)
    _print(f"Loaded {len(queries)} gold queries")

    reports = {}
    for label, spec in MODELS.items():
        model_name = config["models"]["embedding_model"] if spec["model_key"] == "current" else spec["model_key"]
        report = evaluate_model(model_name, spec["persist_dir"], queries, args.top_k)
        report["aggregate"] = _aggregate(report["per_query"], _DOMAIN_SLICES)
        reports[label] = report

    _print_comparison(reports, _DOMAIN_SLICES)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"top_k": args.top_k, "models": reports}, f, indent=2)
    _print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
