from src.rag_lit.schemas import Paper, PaperResult, RetrievalTrace, SearchResponse


def make_paper(**kwargs) -> Paper:
    defaults = dict(
        arxiv_id="2301.00001",
        title="Test Paper",
        abstract="This is a test abstract.",
        authors=["Alice", "Bob"],
        categories=["cs.AI", "cs.LG"],
        year=2023,
        url="https://arxiv.org/abs/2301.00001",
    )
    defaults.update(kwargs)
    return Paper(**defaults)


def make_trace(**kwargs) -> RetrievalTrace:
    defaults = dict(
        total_corpus_size=1000,
        field_filtered_size=500,
        keyword_filtered_size=100,
        reduction_percent_after_field_filter=50.0,
        reduction_percent_after_keyword_filter=90.0,
        generated_keywords=["transformer", "attention"],
        selected_fields=["computer_science"],
        hyde_document="A hypothetical abstract.",
        dense_latency_seconds=0.5,
        bm25_latency_seconds=0.1,
        total_latency_seconds=2.0,
    )
    defaults.update(kwargs)
    return RetrievalTrace(**defaults)


def test_paper_text_property():
    paper = make_paper()
    assert "Test Paper" in paper.text
    assert "test abstract" in paper.text


def test_paper_optional_fields_default():
    paper = make_paper()
    assert paper.primary_category is None
    assert paper.published_date is None
    assert paper.category_metadata == []


def test_paper_result_defaults():
    result = PaperResult(
        rank=1,
        arxiv_id="2301.00001",
        title="Test",
        year=2023,
        rrf_score=0.05,
    )
    assert result.categories == []
    assert result.dense_rank is None
    assert result.bm25_rank is None
    assert result.relevance_justification is None


def test_retrieval_trace():
    trace = make_trace()
    assert trace.total_corpus_size == 1000
    assert trace.reduction_percent_after_keyword_filter == 90.0


def test_search_response_serialization():
    trace = make_trace()
    result = PaperResult(
        rank=1,
        arxiv_id="2301.00001",
        title="Test",
        year=2023,
        rrf_score=0.05,
    )
    response = SearchResponse(
        query="test query",
        results=[result],
        trace=trace,
        metadata={"pipeline_version": "v1"},
    )
    dumped = response.model_dump()
    assert dumped["query"] == "test query"
    assert len(dumped["results"]) == 1
    assert dumped["metadata"]["pipeline_version"] == "v1"

    json_str = response.model_dump_json()
    assert "test query" in json_str
