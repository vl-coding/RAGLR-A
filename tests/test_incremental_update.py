import json

import numpy as np

from scripts.incremental_update import (
    append_to_delta_jsonl,
    load_known_ids,
    merge_keyword_index,
    papers_to_jsonl_lines,
    rebuild_delta_bm25,
    save_known_ids,
)
from src.rag_lit.bm25_retriever import BM25Retriever
from src.rag_lit.keyword_index import (
    candidate_ids_from_keywords,
    open_keyword_index_db,
    save_keyword_index_db,
)
from src.rag_lit.schemas import Paper


RAW_PAPERS = [
    {
        "arxiv_id": "2401.00001",
        "title": "A New Paper",
        "abstract": "An abstract about transformer models.",
        "authors": ["Doe, Jane"],
        "categories": ["cs.AI"],
        "year": 2024,
        "url": "https://arxiv.org/abs/2401.00001",
        "published_date": "2024-01-01",
        "updated_date": "2024-01-01",
    },
    {
        "arxiv_id": "2401.00002",
        "title": "Another Paper",
        "abstract": "An abstract about diffusion models.",
        "authors": ["Roe, Sam"],
        "categories": ["cs.LG"],
        "year": 2024,
        "url": "https://arxiv.org/abs/2401.00002",
        "published_date": "2024-01-02",
        "updated_date": "2024-01-02",
    },
]


def test_papers_to_jsonl_lines_skips_malformed():
    raw = RAW_PAPERS + [{"title": "Missing arxiv_id"}]
    lines = papers_to_jsonl_lines(raw)

    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert {p["arxiv_id"] for p in parsed} == {"2401.00001", "2401.00002"}


def test_append_to_delta_jsonl_creates_file_and_appends(tmp_path):
    delta_path = tmp_path / "nested" / "arxiv_delta.jsonl"
    lines = papers_to_jsonl_lines(RAW_PAPERS)

    append_to_delta_jsonl(str(delta_path), lines[:1])
    append_to_delta_jsonl(str(delta_path), lines[1:])

    with open(delta_path, "r", encoding="utf-8") as f:
        written = [json.loads(line) for line in f if line.strip()]

    assert [p["arxiv_id"] for p in written] == ["2401.00001", "2401.00002"]


def test_load_known_ids_round_trip(tmp_path):
    known_ids_path = tmp_path / "known_ids.npy"
    save_known_ids(str(known_ids_path), {"1706.03762", "1810.04805"})

    loaded = load_known_ids(
        str(known_ids_path),
        bm25_ids_path=str(tmp_path / "missing_bm25"),
        delta_jsonl_path=str(tmp_path / "missing_delta.jsonl"),
    )

    assert loaded == {"1706.03762", "1810.04805"}


def test_load_known_ids_bootstraps_from_bm25_and_delta(tmp_path):
    bm25_index_path = tmp_path / "bm25_index"
    retriever = BM25Retriever()
    retriever.build_index(
        [Paper(arxiv_id="1706.03762", title="Attention", abstract="Transformer paper.", year=2017)]
    )
    retriever.save(str(bm25_index_path))

    delta_path = tmp_path / "arxiv_delta.jsonl"
    append_to_delta_jsonl(str(delta_path), papers_to_jsonl_lines(RAW_PAPERS))

    known_ids_path = tmp_path / "known_ids.npy"
    loaded = load_known_ids(
        str(known_ids_path),
        bm25_ids_path=str(bm25_index_path),
        delta_jsonl_path=str(delta_path),
    )

    assert loaded == {"1706.03762", "2401.00001", "2401.00002"}
    assert known_ids_path.exists()

    # The bootstrapped set is persisted, so a second load doesn't need to rescan.
    reloaded = load_known_ids(
        str(known_ids_path),
        bm25_ids_path=str(bm25_index_path),
        delta_jsonl_path=str(delta_path),
    )
    assert reloaded == loaded


def test_rebuild_delta_bm25_is_searchable(tmp_path):
    delta_path = tmp_path / "arxiv_delta.jsonl"
    append_to_delta_jsonl(str(delta_path), papers_to_jsonl_lines(RAW_PAPERS))

    bm25_delta_path = tmp_path / "bm25_delta"
    rebuild_delta_bm25(str(delta_path), str(bm25_delta_path))

    retriever = BM25Retriever.load(str(bm25_delta_path))
    assert set(retriever.arxiv_ids) == {"2401.00001", "2401.00002"}

    results = retriever.search("diffusion models", top_n=2)
    assert results[0]["doc_id"] == "2401.00002"


def test_merge_keyword_index_adds_new_postings(tmp_path):
    keyword_index_path = tmp_path / "keyword_index.sqlite3"
    save_keyword_index_db({"transformer": {"1706.03762"}}, str(keyword_index_path))

    merge_keyword_index(str(keyword_index_path), RAW_PAPERS)

    conn = open_keyword_index_db(str(keyword_index_path))
    try:
        # Pre-existing posting is untouched.
        assert candidate_ids_from_keywords(["transformer"], conn) == {
            "1706.03762",
            "2401.00001",
        }
        # New papers were merged in under their own tokens.
        assert candidate_ids_from_keywords(["diffusion"], conn) == {"2401.00002"}
    finally:
        conn.close()
