import json
from unittest.mock import MagicMock, patch

import torch
from transformers import StoppingCriteriaList

from src.rag_lit.qwen_prefilter import QwenKeywordExtractor, _filter_keywords, _JSONArrayStoppingCriteria


def test_filter_keywords_drops_generic_single_words():
    assert _filter_keywords(["learning", "models", "graph neural networks"]) == [
        "graph neural networks"
    ]


def test_filter_keywords_keeps_multiword_phrase_with_generic_word():
    assert _filter_keywords(["contrastive learning", "vision transformers"]) == [
        "contrastive learning",
        "vision transformers",
    ]


def test_filter_keywords_dedupes_case_and_plural_near_duplicates():
    assert _filter_keywords(["Transformer", "transformers", "TRANSFORMER"]) == ["Transformer"]


def test_filter_keywords_drops_empty_and_whitespace_entries():
    assert _filter_keywords(["", "   ", "diffusion models"]) == ["diffusion models"]


def test_filter_keywords_preserves_first_seen_order():
    assert _filter_keywords(["b term", "a term", "b term"]) == ["b term", "a term"]


def test_filter_keywords_all_generic_returns_empty():
    assert _filter_keywords(["learning", "models", "approach"]) == []


def test_fallback_keywords_filters_generic_terms():
    extractor = QwenKeywordExtractor.__new__(QwenKeywordExtractor)
    assert extractor.fallback_keywords("deep learning methods for object detection") == [
        "object",
        "detection",
    ]


def test_fallback_keywords_all_generic_returns_empty():
    extractor = QwenKeywordExtractor.__new__(QwenKeywordExtractor)
    assert extractor.fallback_keywords("deep learning methods") == []


def _build_extractor_with_mock_model(decoded_output: str) -> QwenKeywordExtractor:
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = decoded_output

    mock_inputs = MagicMock()
    mock_inputs.__getitem__.return_value = torch.zeros((1, 5), dtype=torch.long)
    mock_inputs.to.return_value = mock_inputs
    mock_tokenizer.return_value = mock_inputs

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.zeros((1, 10), dtype=torch.long)

    with (
        patch("src.rag_lit.qwen_prefilter.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("src.rag_lit.qwen_prefilter.AutoModelForCausalLM.from_pretrained", return_value=mock_model),
    ):
        extractor = QwenKeywordExtractor("Qwen/Qwen2.5-0.5B-Instruct")

    return extractor


def test_generate_keywords_filters_model_output():
    decoded = json.dumps(
        ["learning", "vision transformers", "Transformer", "transformers", "contrastive learning"]
    )
    extractor = _build_extractor_with_mock_model(decoded)

    assert extractor.generate_keywords("some query") == [
        "vision transformers",
        "Transformer",
        "contrastive learning",
    ]


def test_generate_keywords_falls_back_on_unparseable_output():
    extractor = _build_extractor_with_mock_model("not valid json output")

    assert extractor.generate_keywords("deep learning methods for object detection") == [
        "object",
        "detection",
    ]


def test_generate_keywords_passes_stopping_criteria():
    decoded = json.dumps(["vision transformers"])
    extractor = _build_extractor_with_mock_model(decoded)

    extractor.generate_keywords("some query")

    _, kwargs = extractor.model.generate.call_args
    criteria_list = kwargs["stopping_criteria"]
    assert isinstance(criteria_list, StoppingCriteriaList)
    assert isinstance(criteria_list[0], _JSONArrayStoppingCriteria)


def _stopping_criteria(decoded_text: str) -> _JSONArrayStoppingCriteria:
    tokenizer = MagicMock()
    tokenizer.decode.return_value = decoded_text
    return _JSONArrayStoppingCriteria(tokenizer, prompt_length=0)


def test_json_array_stopping_criteria_false_before_any_bracket():
    criteria = _stopping_criteria("Here is some text without brackets")
    assert criteria(torch.zeros((1, 5), dtype=torch.long), None) is False


def test_json_array_stopping_criteria_false_for_unbalanced_array():
    criteria = _stopping_criteria('["transformer", "attention')
    assert criteria(torch.zeros((1, 5), dtype=torch.long), None) is False


def test_json_array_stopping_criteria_true_when_array_balances():
    criteria = _stopping_criteria('["transformer", "attention"]')
    assert criteria(torch.zeros((1, 5), dtype=torch.long), None) is True


def test_json_array_stopping_criteria_ignores_trailing_array_after_first_closes():
    criteria = _stopping_criteria('["transformer"]\nQuery: foo\nOutput: [')
    assert criteria(torch.zeros((1, 5), dtype=torch.long), None) is True
