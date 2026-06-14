import json

from src.rag_lit.metadata_db import (
    build_metadata_db,
    candidate_ids_for_categories,
    load_metadata_db,
    open_metadata_db,
)

PAPERS = [
    {"arxiv_id": "p1", "categories": ["cs.AI"]},
    {"arxiv_id": "p2", "categories": ["cs.LG", "stat.ML"]},
    {"arxiv_id": "p3", "categories": ["stat.TH"]},
    {"arxiv_id": "p4", "categories": ["physics.optics"]},
    {"arxiv_id": "p5", "categories": ["cs.CL"]},
]


def _write_jsonl(path, papers):
    with open(path, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper) + "\n")


def test_build_and_load_metadata_db(tmp_path):
    jsonl_path = tmp_path / "papers.jsonl"
    db_path = tmp_path / "metadata.sqlite3"
    _write_jsonl(jsonl_path, PAPERS)

    count = build_metadata_db(str(jsonl_path), str(db_path))
    assert count == len(PAPERS)

    rows, offsets = load_metadata_db(str(db_path))
    assert {arxiv_id for arxiv_id, _ in rows} == {p["arxiv_id"] for p in PAPERS}
    assert set(offsets.keys()) == {p["arxiv_id"] for p in PAPERS}


def test_candidate_ids_for_categories_single(tmp_path):
    jsonl_path = tmp_path / "papers.jsonl"
    db_path = tmp_path / "metadata.sqlite3"
    _write_jsonl(jsonl_path, PAPERS)
    build_metadata_db(str(jsonl_path), str(db_path))

    conn = open_metadata_db(str(db_path))
    try:
        assert candidate_ids_for_categories(conn, ["stat.TH"]) == {"p3"}
    finally:
        conn.close()


def test_candidate_ids_for_categories_cross_listed(tmp_path):
    jsonl_path = tmp_path / "papers.jsonl"
    db_path = tmp_path / "metadata.sqlite3"
    _write_jsonl(jsonl_path, PAPERS)
    build_metadata_db(str(jsonl_path), str(db_path))

    conn = open_metadata_db(str(db_path))
    try:
        assert candidate_ids_for_categories(conn, ["stat.ML"]) == {"p2"}
    finally:
        conn.close()


def test_candidate_ids_for_categories_union_of_multiple(tmp_path):
    jsonl_path = tmp_path / "papers.jsonl"
    db_path = tmp_path / "metadata.sqlite3"
    _write_jsonl(jsonl_path, PAPERS)
    build_metadata_db(str(jsonl_path), str(db_path))

    conn = open_metadata_db(str(db_path))
    try:
        assert candidate_ids_for_categories(conn, ["cs.AI", "cs.CL"]) == {"p1", "p5"}
    finally:
        conn.close()


def test_candidate_ids_for_categories_no_match(tmp_path):
    jsonl_path = tmp_path / "papers.jsonl"
    db_path = tmp_path / "metadata.sqlite3"
    _write_jsonl(jsonl_path, PAPERS)
    build_metadata_db(str(jsonl_path), str(db_path))

    conn = open_metadata_db(str(db_path))
    try:
        assert candidate_ids_for_categories(conn, ["q-bio.NC"]) == set()
    finally:
        conn.close()


def test_candidate_ids_for_categories_empty_categories(tmp_path):
    jsonl_path = tmp_path / "papers.jsonl"
    db_path = tmp_path / "metadata.sqlite3"
    _write_jsonl(jsonl_path, PAPERS)
    build_metadata_db(str(jsonl_path), str(db_path))

    conn = open_metadata_db(str(db_path))
    try:
        assert candidate_ids_for_categories(conn, []) == set()
    finally:
        conn.close()
