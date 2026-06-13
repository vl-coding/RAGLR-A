"""
Smoke tests for the pipeline using lightweight mocks.

Verifies pipeline wiring without requiring real models, API keys, or built indexes.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.rag_lit.pipeline import RagLiteraturePipeline, _PaperMeta
from src.rag_lit.schemas import Paper, SearchResponse


MINI_CONFIG = {
    "data": {
        "processed_path": "data/processed/arxiv_papers.jsonl",
        "delta_path": "data/processed/arxiv_delta.jsonl",
    },
    "paths": {
        "keyword_index": "artifacts/keyword_index.sqlite3",
        "metadata_db": "artifacts/metadata.sqlite3",
        "dense_index_dir": "artifacts/dense_index",
        "bm25_index": "artifacts/bm25_index",
        "bm25_delta": "artifacts/bm25_delta",
    },
    "models": {
        "qwen_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "claude_model": "claude-sonnet-4-6",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "retrieval": {
        "dense_candidates": 10,
        "bm25_candidates": 10,
        "rrf_k": 60,
        "min_prefilter_candidates": 5,
    },
}

SAMPLE_PAPERS = [
    Paper(
        arxiv_id=f"230{i}.0000{i}",
        title=f"Paper {i}",
        abstract=f"Abstract for paper {i}.",
        year=2023,
        categories=["cs.AI"],
        url=f"https://arxiv.org/abs/230{i}.0000{i}",
    )
    for i in range(1, 6)
]

_SAMPLE_META = [_PaperMeta(p.arxiv_id, tuple(p.categories)) for p in SAMPLE_PAPERS]
_SAMPLE_OFFSETS = {p.arxiv_id: i * 100 for i, p in enumerate(SAMPLE_PAPERS)}
_PAPER_LOOKUP = {p.arxiv_id: p for p in SAMPLE_PAPERS}


def _make_retriever_results(papers):
    dense = [
        {"doc_id": p.arxiv_id, "rank": i + 1, "score": 0.9 - i * 0.1, "source": "dense"}
        for i, p in enumerate(papers)
    ]
    bm25 = [
        {"doc_id": p.arxiv_id, "rank": i + 1, "score": 10.0 - i, "source": "bm25"}
        for i, p in enumerate(papers)
    ]
    return dense, bm25


def _build_mock_pipeline() -> RagLiteraturePipeline:
    """
    Constructs a RagLiteraturePipeline with all I/O and model calls mocked out.

    Patches that touch disk or the network (build_meta_index, open_keyword_index_db,
    BM25Retriever.load, and the three model constructors) are applied only during
    __init__ via context-manager patches.  Patches needed during run() (notably
    _maybe_reload and _load_paper) are set as instance attributes after construction,
    which survive the context-manager exit.
    """
    dense_results, bm25_results = _make_retriever_results(SAMPLE_PAPERS)

    mock_qwen = MagicMock()
    mock_qwen.generate_keywords.return_value = ["attention", "transformer"]

    mock_hyde = MagicMock()
    mock_hyde.generate.return_value = "A hypothetical abstract about AI."

    mock_justifier = MagicMock()
    mock_justifier.justify.return_value = {
        "contribution": "Proposes a new method.",
        "relevance_justification": "Directly relevant.",
        "relevance_score": 9,
        "specificity_score": 8,
    }

    mock_dense = MagicMock()
    mock_dense.search.return_value = dense_results

    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = bm25_results

    mock_kw_conn = MagicMock()
    mock_kw_conn.execute.return_value.fetchone.return_value = None

    with (
        patch.object(
            RagLiteraturePipeline,
            "_build_meta_index",
            return_value=(_SAMPLE_META[:], dict(_SAMPLE_OFFSETS)),
        ),
        patch.object(RagLiteraturePipeline, "_load_delta_meta_from"),
        patch("src.rag_lit.pipeline.open_keyword_index_db", return_value=mock_kw_conn),
        patch("src.rag_lit.pipeline.QwenKeywordExtractor", return_value=mock_qwen),
        patch("src.rag_lit.pipeline.ClaudeHyDE", return_value=mock_hyde),
        patch("src.rag_lit.pipeline.ClaudeJustifier", return_value=mock_justifier),
        patch("src.rag_lit.pipeline.DenseRetriever", return_value=mock_dense),
        patch("src.rag_lit.pipeline.BM25Retriever") as MockBM25Cls,
    ):
        MockBM25Cls.load.return_value = mock_bm25
        pipeline = RagLiteraturePipeline(MINI_CONFIG)

    # Patch instance methods/attributes used during run() (class-level patches expired above)
    pipeline._maybe_reload = MagicMock()
    pipeline._load_paper = MagicMock(side_effect=lambda aid: _PAPER_LOOKUP[aid])
    pipeline._qwen = mock_qwen

    return pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_returns_search_response():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="What are recent advances in transformer models?",
        top_k=3,
    )
    assert isinstance(response, SearchResponse)
    assert len(response.results) == 3


def test_pipeline_trace_fields():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="attention mechanisms in NLP",
        top_k=2,
    )
    trace = response.trace
    assert trace.total_corpus_size == len(SAMPLE_PAPERS)
    assert trace.total_latency_seconds >= 0


def test_pipeline_no_qwen():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="graph neural networks",
        top_k=2,
        use_qwen_prefilter=False,
    )
    assert isinstance(response, SearchResponse)
    assert response.trace.generated_keywords == []


def test_pipeline_no_justification():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="diffusion models",
        top_k=2,
        use_claude_justification=False,
    )
    for result in response.results:
        assert result.contribution is None
        assert result.relevance_justification is None


def test_pipeline_debug_output():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="contrastive learning",
        top_k=2,
        debug=True,
        hyde_ablation=True,
    )
    debug = response.debug
    assert debug is not None
    assert debug.final_candidate_ids
    assert set(debug.final_candidate_ids) == {p.arxiv_id for p in SAMPLE_PAPERS}
    assert debug.keyword_candidate_ids is not None
    assert debug.dense_results
    assert debug.dense_results_raw_query is not None
    assert debug.bm25_results
    assert debug.fused_results_raw_query is not None
    assert {item["doc_id"] for item in debug.fused_results_raw_query} == {
        p.arxiv_id for p in SAMPLE_PAPERS
    }


def test_pipeline_no_fused_raw_query_without_ablation():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="contrastive learning",
        top_k=2,
        debug=True,
        hyde_ablation=False,
    )
    assert response.debug.fused_results_raw_query is None


def test_pipeline_no_debug_by_default():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="contrastive learning",
        top_k=2,
    )
    assert response.debug is None


def test_pipeline_category_filter_matching_keeps_candidates():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="contrastive learning",
        top_k=2,
        debug=True,
        categories=["cs.AI"],
    )
    assert set(response.debug.final_candidate_ids) == {p.arxiv_id for p in SAMPLE_PAPERS}


def test_pipeline_category_filter_no_match_empties_candidates():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="contrastive learning",
        top_k=2,
        debug=True,
        categories=["cs.CV"],
    )
    assert response.debug.final_candidate_ids == []


def test_pipeline_result_ranks_are_sequential():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="contrastive learning",
        top_k=5,
    )
    ranks = [r.rank for r in response.results]
    assert ranks == list(range(1, len(ranks) + 1))
