from unittest.mock import MagicMock, patch

from src.rag_lit.dense_retriever import DenseRetriever, _MAX_OVERFETCH_MULTIPLIER


def _build_retriever(total: int, skip_filter_threshold: float = 0.4) -> DenseRetriever:
    with (
        patch("src.rag_lit.dense_retriever.SentenceTransformer") as MockModel,
        patch("src.rag_lit.dense_retriever.chromadb.PersistentClient") as MockClient,
    ):
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2]
        mock_model.encode.return_value.__getitem__.return_value.tolist.return_value = [0.1, 0.2]
        MockModel.return_value = mock_model

        mock_collection = MagicMock()
        mock_collection.count.return_value = total
        MockClient.return_value.get_or_create_collection.return_value = mock_collection

        retriever = DenseRetriever(
            model_name="dummy-model",
            persist_dir="dummy-dir",
            skip_filter_threshold=skip_filter_threshold,
        )

    return retriever


def _result(ids):
    return {
        "ids": [ids],
        "distances": [[0.1] * len(ids)],
    }


def test_search_single_query_when_first_pool_has_enough_hits():
    retriever = _build_retriever(total=10_000)
    candidate_ids = {f"id{i}" for i in range(100)}

    pool = [f"id{i}" for i in range(100)] + [f"other{i}" for i in range(9_900)]
    retriever.collection.query.return_value = _result(pool[:5000])

    output = retriever.search("query", candidate_ids=candidate_ids, top_n=10)

    assert retriever.collection.query.call_count == 1
    assert len(output) == 10
    assert all(item["doc_id"] in candidate_ids for item in output)


def test_search_doubles_pool_when_first_query_yields_too_few_hits():
    retriever = _build_retriever(total=10_000)
    candidate_ids = {"id1", "id2"}

    first_pool = ["other"] * 500 + ["id1"]
    second_pool = ["other"] * 1000 + ["id1", "id2"]

    retriever.collection.query.side_effect = [
        _result(first_pool),
        _result(second_pool),
    ]

    output = retriever.search("query", candidate_ids=candidate_ids, top_n=2)

    assert retriever.collection.query.call_count == 2
    second_call_n_results = retriever.collection.query.call_args_list[1].kwargs["n_results"]
    first_call_n_results = retriever.collection.query.call_args_list[0].kwargs["n_results"]
    assert second_call_n_results == first_call_n_results * 2
    assert {item["doc_id"] for item in output} == {"id1", "id2"}


def test_search_stops_doubling_at_total_corpus_size():
    total = 1000
    retriever = _build_retriever(total=total)
    candidate_ids = {"id1"}

    retriever.collection.query.return_value = _result(["other"] * 500)

    retriever.search("query", candidate_ids=candidate_ids, top_n=10)

    for call in retriever.collection.query.call_args_list:
        assert call.kwargs["n_results"] <= total


def test_search_stops_doubling_at_ceiling():
    top_n = 10
    retriever = _build_retriever(total=10_000_000)
    candidate_ids = {"id1"}

    retriever.collection.query.return_value = _result(["other"] * 100)

    retriever.search("query", candidate_ids=candidate_ids, top_n=top_n)

    ceiling = top_n * _MAX_OVERFETCH_MULTIPLIER
    for call in retriever.collection.query.call_args_list:
        assert call.kwargs["n_results"] <= ceiling
    assert retriever.collection.query.call_args_list[-1].kwargs["n_results"] == ceiling


def test_search_skips_filtering_when_candidate_set_above_threshold():
    total = 1000
    retriever = _build_retriever(total=total, skip_filter_threshold=0.4)
    candidate_ids = {f"cand{i}" for i in range(500)}  # 50% of corpus

    pool = [f"other{i}" for i in range(10)]
    retriever.collection.query.return_value = _result(pool)

    output = retriever.search("query", candidate_ids=candidate_ids, top_n=10)

    assert retriever.collection.query.call_count == 1
    n_results = retriever.collection.query.call_args_list[0].kwargs["n_results"]
    assert n_results == 10  # unfiltered top_n, not top_n * 50
    assert {item["doc_id"] for item in output} == set(pool)


def test_search_filters_normally_when_candidate_set_below_threshold():
    total = 1000
    retriever = _build_retriever(total=total, skip_filter_threshold=0.4)
    candidate_ids = {f"cand{i}" for i in range(100)}  # 10% of corpus

    pool = [f"cand{i}" for i in range(10)] + [f"other{i}" for i in range(90)]
    retriever.collection.query.return_value = _result(pool)

    output = retriever.search("query", candidate_ids=candidate_ids, top_n=10)

    n_results = retriever.collection.query.call_args_list[0].kwargs["n_results"]
    assert n_results == 10 * 50
    assert all(item["doc_id"] in candidate_ids for item in output)
