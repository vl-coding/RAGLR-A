from src.rag_lit.canonical_boost import load_canonical_papers, match_canonical_papers

_PAPERS = [
    {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "topics": ["transformer architecture", "self-attention", "sequence modeling"],
    },
    {
        "arxiv_id": "2106.09685",
        "title": "LoRA",
        "topics": ["lora", "low-rank adaptation", "parameter-efficient fine-tuning"],
    },
    {
        "arxiv_id": "2103.00020",
        "title": "CLIP",
        "topics": ["clip", "contrastive language-image pretraining", "zero-shot transfer"],
    },
]


def test_matches_query_against_topics():
    result = match_canonical_papers(
        query="transformer architectures for sequence modeling",
        keywords=[],
        canonical_papers=_PAPERS,
        corpus_ids={"1706.03762", "2106.09685", "2103.00020"},
    )
    assert result == [{"doc_id": "1706.03762", "rank": 1, "source": "canonical"}]


def test_matches_query_against_keywords():
    result = match_canonical_papers(
        query="how do I make my model cheaper to train",
        keywords=["lora", "parameter-efficient fine-tuning"],
        canonical_papers=_PAPERS,
        corpus_ids={"1706.03762", "2106.09685", "2103.00020"},
    )
    assert result == [{"doc_id": "2106.09685", "rank": 1, "source": "canonical"}]


def test_ranks_by_total_matched_topic_length():
    result = match_canonical_papers(
        query="contrastive language-image pretraining for zero-shot transfer with clip",
        keywords=[],
        canonical_papers=_PAPERS,
        corpus_ids={"1706.03762", "2106.09685", "2103.00020"},
    )
    assert result[0]["doc_id"] == "2103.00020"
    assert result[0]["rank"] == 1


def test_more_specific_topic_phrase_outranks_generic_one():
    papers = [
        {
            "arxiv_id": "2005.14165",
            "title": "GPT-3",
            "topics": ["gpt-3", "large language model"],
        },
        {
            "arxiv_id": "2106.09685",
            "title": "LoRA",
            "topics": ["lora", "parameter-efficient fine-tuning"],
        },
    ]
    result = match_canonical_papers(
        query="parameter-efficient fine-tuning of large language models",
        keywords=[],
        canonical_papers=papers,
        corpus_ids={"2005.14165", "2106.09685"},
    )
    assert result[0]["doc_id"] == "2106.09685"


def test_no_match_returns_empty_list():
    result = match_canonical_papers(
        query="single-cell transcriptomics analysis methods",
        keywords=["single-cell", "transcriptomics"],
        canonical_papers=_PAPERS,
        corpus_ids={"1706.03762", "2106.09685", "2103.00020"},
    )
    assert result == []


def test_excludes_papers_not_in_corpus():
    result = match_canonical_papers(
        query="transformer architectures for sequence modeling",
        keywords=[],
        canonical_papers=_PAPERS,
        corpus_ids={"2106.09685", "2103.00020"},
    )
    assert result == []


def test_empty_registry_returns_empty_list():
    assert match_canonical_papers("anything", [], [], {"x"}) == []


def test_load_canonical_papers_missing_path_returns_empty_list():
    assert load_canonical_papers("") == []
    assert load_canonical_papers("does/not/exist.yaml") == []


def test_load_canonical_papers_reads_real_registry():
    papers = load_canonical_papers("data/canonical_papers.yaml")
    assert len(papers) > 0
    assert all("arxiv_id" in p and "topics" in p for p in papers)
