from typing import List, Dict, Any


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    scores = {}
    dense_ranks = {}
    bm25_ranks = {}

    for ranked_list in ranked_lists:
        for item in ranked_list:
            doc_id = item["doc_id"]
            rank = item["rank"]
            source = item.get("source")

            if doc_id not in scores:
                scores[doc_id] = 0.0

            scores[doc_id] += 1.0 / (k + rank)

            if source == "dense":
                dense_ranks[doc_id] = rank
            elif source == "bm25":
                bm25_ranks[doc_id] = rank

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "doc_id": doc_id,
            "rank": i + 1,
            "rrf_score": score,
            "dense_rank": dense_ranks.get(doc_id),
            "bm25_rank": bm25_ranks.get(doc_id),
        }
        for i, (doc_id, score) in enumerate(fused)
    ]