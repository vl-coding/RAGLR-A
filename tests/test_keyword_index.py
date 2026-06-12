from src.rag_lit.keyword_index import tokenize


def test_tokenize_keeps_two_character_acronyms():
    assert tokenize("AI and ML models") == ["ai", "and", "ml", "models"]


def test_tokenize_keeps_alphanumeric_model_names():
    assert tokenize("T5 and CV tasks") == ["t5", "and", "cv", "tasks"]


def test_tokenize_keeps_hyphenated_model_names():
    assert tokenize("GPT-4 outperforms GPT4 on this benchmark") == [
        "gpt-4",
        "outperforms",
        "gpt4",
        "on",
        "this",
        "benchmark",
    ]


def test_tokenize_drops_single_character_tokens():
    assert tokenize("a b cd e") == ["cd"]


def test_tokenize_transliterates_greek_letters():
    assert tokenize("the α and β parameters") == [
        "the",
        "alpha",
        "and",
        "beta",
        "parameters",
    ]


def test_tokenize_transliterates_uppercase_greek_letters():
    assert tokenize("Δ-step updates with Σ aggregation") == [
        "delta",
        "step",
        "updates",
        "with",
        "sigma",
        "aggregation",
    ]
