from unittest.mock import MagicMock

from scripts.evaluate_retrieval import evaluate_query
from src.rag_lit.schemas import Paper, PaperResult, RetrievalDebugInfo, RetrievalTrace, SearchResponse


GOLD_ENTRY = {
    "query": "transformer architectures for sequence modeling",
    "relevant_ids": ["A", "B", "C", "D"],
    "stratum": "terminology_gap",
}


def _make_response(hyde_ablation):
    results = [
        PaperResult(rank=1, arxiv_id="A", title="A", year=2020, rrf_score=0.05),
        PaperResult(rank=2, arxiv_id="X", title="X", year=2021, rrf_score=0.04,
                     relevance_score=8, specificity_score=7),
    ]
    debug = RetrievalDebugInfo(
        keyword_candidate_ids=["A", "B", "C", "D", "X", "Y"],
        final_candidate_ids=["A", "B", "C", "D", "X", "Y"],
        dense_results=[{"doc_id": "A", "rank": 1, "score": 0.9, "source": "dense"}],
        dense_results_raw_query=(
            [{"doc_id": "B", "rank": 1, "score": 0.8, "source": "dense"}] if hyde_ablation else None
        ),
        bm25_results=[{"doc_id": "A", "rank": 1, "score": 5.0, "source": "bm25"}],
        bm25_delta_results=[],
        canonical_results=[],
        fused_results_raw_query=(
            [
                {"doc_id": "B", "rank": 1, "rrf_score": 0.05, "dense_rank": 1, "bm25_rank": None},
                {"doc_id": "X", "rank": 2, "rrf_score": 0.04, "dense_rank": None, "bm25_rank": 1},
            ]
            if hyde_ablation else None
        ),
    )
    trace = RetrievalTrace(
        total_corpus_size=6,
        keyword_filtered_size=6,
        reduction_percent_after_keyword_filter=0.0,
        hyde_document="a hypothetical abstract",
    )
    return SearchResponse(query=GOLD_ENTRY["query"], results=results, trace=trace, debug=debug)


def _make_pipeline(hyde_ablation):
    pipeline = MagicMock()
    pipeline.run.return_value = _make_response(hyde_ablation)
    pipeline.dense.search.side_effect = [
        [{"doc_id": "A", "rank": 1, "score": 0.9, "source": "dense"},
         {"doc_id": "C", "rank": 2, "score": 0.5, "source": "dense"}],  # hyde large
        [{"doc_id": "B", "rank": 1, "score": 0.8, "source": "dense"}],  # raw large
    ]
    pipeline.justifier.justify.return_value = {"relevance_score": 6, "specificity_score": 5}
    pipeline._load_paper.return_value = Paper(
        arxiv_id="B", title="B", abstract="abstract B", year=2019
    )
    return pipeline


def test_evaluate_query_basic_e2e():
    pipeline = _make_pipeline(hyde_ablation=False)
    result = evaluate_query(pipeline, GOLD_ENTRY, top_k=2, hyde_ablation=False, use_justification=False)

    assert result["stratum"] == "terminology_gap"
    assert result["e2e"]["hits"] == ["A"]
    assert result["e2e"]["retrieved_ids"] == ["A", "X"]
    assert "hyde" not in result
    assert "hyde_e2e" not in result


def test_evaluate_query_hyde_ablation_post_rrf():
    pipeline = _make_pipeline(hyde_ablation=True)
    result = evaluate_query(pipeline, GOLD_ENTRY, top_k=2, hyde_ablation=True, use_justification=False)

    assert result["hyde"]["hyde_recall@k"] == 0.25  # "A" in dense_results
    assert result["hyde"]["raw_recall@k"] == 0.25  # "B" in dense_results_raw_query

    he = result["hyde_e2e"]
    assert he["raw_retrieved_ids"] == ["B", "X"]
    assert he["raw_hits"] == ["B"]
    assert he["raw_recall@k"] == 0.25


def test_evaluate_query_rank_delta():
    pipeline = _make_pipeline(hyde_ablation=True)
    result = evaluate_query(
        pipeline, GOLD_ENTRY, top_k=2, hyde_ablation=True, use_justification=False,
        rank_delta_top_n=5,
    )

    per_id = {item["arxiv_id"]: item for item in result["rank_delta"]["per_id"]}
    # "A" ranks 1 in hyde-large, absent (capped to 6) in raw-large -> delta = 6 - 1 = 5
    assert per_id["A"]["rank_hyde"] == 1
    assert per_id["A"]["rank_raw"] is None
    assert per_id["A"]["delta"] == 5
    # "B" ranks 1 in raw-large, absent (capped to 6) in hyde-large -> delta = 1 - 6 = -5
    assert per_id["B"]["rank_hyde"] is None
    assert per_id["B"]["rank_raw"] == 1
    assert per_id["B"]["delta"] == -5
    # "C" ranks 2 in hyde-large, absent (capped to 6) in raw-large -> delta = 6 - 2 = 4
    assert per_id["C"]["rank_hyde"] == 2
    assert per_id["C"]["rank_raw"] is None
    assert per_id["C"]["delta"] == 4
    # "D" absent from both -> delta = 6 - 6 = 0
    assert per_id["D"]["delta"] == 0


def test_evaluate_query_pooled_judgments():
    pipeline = _make_pipeline(hyde_ablation=True)
    result = evaluate_query(
        pipeline, GOLD_ENTRY, top_k=2, hyde_ablation=True, use_justification=True,
        pooled_judgments=True, pool_size=2,
    )

    pj = result["pooled_judgment"]
    # pool = hyde top-2 {A, X} | raw top-2 {B, X} = {A, B, X}
    assert pj["pool_ids"] == ["A", "B", "X"]
    # "X" was already scored (8) via justifier_records; "A" and "B" need new justify() calls (returns 6)
    assert pj["relevance"]["X"] == 8
    assert pj["relevance"]["A"] == 6
    assert pj["relevance"]["B"] == 6
    # justify() should only be called for the un-judged pool members (A and B)
    assert pipeline.justifier.justify.call_count == 2
