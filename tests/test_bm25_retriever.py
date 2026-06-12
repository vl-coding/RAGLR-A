from src.rag_lit.bm25_retriever import BM25Retriever
from src.rag_lit.schemas import Paper


def _make_papers():
    return [
        Paper(
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            abstract="We propose the Transformer, a model architecture based on attention mechanisms.",
            year=2017,
        ),
        Paper(
            arxiv_id="1810.04805",
            title="BERT",
            abstract="We introduce a new language representation model called BERT.",
            year=2018,
        ),
        Paper(
            arxiv_id="2103.00020",
            title="CLIP",
            abstract="We learn visual concepts from natural language supervision.",
            year=2021,
        ),
    ]


def _build_and_save(tmp_path):
    retriever = BM25Retriever()
    retriever.build_index(_make_papers())
    out = tmp_path / "bm25_index"
    retriever.save(str(out))
    return out


def test_load_without_mmap(tmp_path):
    out = _build_and_save(tmp_path)
    retriever = BM25Retriever.load(str(out))
    results = retriever.search("transformer attention mechanism", top_n=3)
    assert results[0]["doc_id"] == "1706.03762"


def test_load_with_mmap_returns_same_results(tmp_path):
    out = _build_and_save(tmp_path)
    in_memory = BM25Retriever.load(str(out), mmap=False)
    mmapped = BM25Retriever.load(str(out), mmap=True)

    query = "language representation model"
    in_memory_results = in_memory.search(query, top_n=3)
    mmapped_results = mmapped.search(query, top_n=3)

    assert [r["doc_id"] for r in in_memory_results] == [r["doc_id"] for r in mmapped_results]
    assert [r["score"] for r in in_memory_results] == [r["score"] for r in mmapped_results]
