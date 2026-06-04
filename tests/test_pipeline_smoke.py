"""
Smoke tests for the pipeline using lightweight mocks.

These tests verify the pipeline wiring without requiring real models,
API keys, or a built index.
"""
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest

from src.rag_lit.pipeline import RagLiteraturePipeline
from src.rag_lit.schemas import Paper, SearchResponse


MINI_CONFIG = {
    "data": {"processed_path": "data/processed/arxiv_papers.jsonl"},
    "paths": {
        "keyword_index": "artifacts/keyword_inverted_index.pkl",
        "dense_index_dir": "artifacts/dense_index",
        "bm25_index": "artifacts/bm25_index.pkl",
    },
    "models": {
        "qwen_model": "Qwen/Qwen2.5-3B-Instruct",
        "claude_model": "claude-sonnet-4-6",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "retrieval": {
        "dense_candidates": 10,
        "bm25_candidates": 10,
        "rrf_k": 60,
        "min_prefilter_candidates": 5,
    },
    "academic_fields": {
        "all": {"label": "All arXiv Fields", "categories": "*"},
        "computer_science": {
            "label": "Computer Science",
            "categories": ["cs.AI", "cs.LG"],
        },
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


def _build_mock_pipeline() -> RagLiteraturePipeline:
    with (
        patch("src.rag_lit.pipeline.load_papers_jsonl", return_value=SAMPLE_PAPERS),
        patch("src.rag_lit.pipeline.load_keyword_index", return_value={}),
        patch("src.rag_lit.pipeline.QwenKeywordExtractor") as MockQwen,
        patch("src.rag_lit.pipeline.ClaudeHyDE") as MockHyDE,
        patch("src.rag_lit.pipeline.ClaudeJustifier") as MockJustifier,
        patch("src.rag_lit.pipeline.DenseRetriever") as MockDense,
        patch("src.rag_lit.pipeline.BM25Retriever") as MockBM25,
    ):
        MockQwen.return_value.generate_keywords.return_value = ["attention", "transformer"]
        MockHyDE.return_value.generate.return_value = "A hypothetical abstract about AI."
        MockJustifier.return_value.justify.return_value = {
            "contribution": "Proposes a new method.",
            "relevance_justification": "Directly relevant.",
            "relevance_score": 9,
            "specificity_score": 8,
        }

        dense_results = [
            {"doc_id": p.arxiv_id, "rank": i + 1, "score": 0.9 - i * 0.1, "source": "dense"}
            for i, p in enumerate(SAMPLE_PAPERS)
        ]
        bm25_results = [
            {"doc_id": p.arxiv_id, "rank": i + 1, "score": 10.0 - i, "source": "bm25"}
            for i, p in enumerate(SAMPLE_PAPERS)
        ]

        MockDense.return_value.search.return_value = dense_results
        MockBM25.load.return_value.search.return_value = bm25_results

        return RagLiteraturePipeline(MINI_CONFIG)


def test_pipeline_returns_search_response():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="What are recent advances in transformer models?",
        selected_fields=["computer_science"],
        top_k=3,
    )
    assert isinstance(response, SearchResponse)
    assert len(response.results) == 3


def test_pipeline_trace_fields():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="attention mechanisms in NLP",
        selected_fields=["all"],
        top_k=2,
    )
    trace = response.trace
    assert trace.total_corpus_size == len(SAMPLE_PAPERS)
    assert trace.total_latency_seconds >= 0


def test_pipeline_no_qwen():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="graph neural networks",
        selected_fields=["computer_science"],
        top_k=2,
        use_qwen_prefilter=False,
    )
    assert isinstance(response, SearchResponse)
    assert response.trace.generated_keywords == []


def test_pipeline_no_justification():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="diffusion models",
        selected_fields=["computer_science"],
        top_k=2,
        use_claude_justification=False,
    )
    for result in response.results:
        assert result.contribution is None
        assert result.relevance_justification is None


def test_pipeline_result_ranks_are_sequential():
    pipeline = _build_mock_pipeline()
    response = pipeline.run(
        query="contrastive learning",
        selected_fields=["all"],
        top_k=5,
    )
    ranks = [r.rank for r in response.results]
    assert ranks == list(range(1, len(ranks) + 1))
