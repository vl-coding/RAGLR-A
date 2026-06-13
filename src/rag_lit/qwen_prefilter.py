import json
from pathlib import Path
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .keyword_index import tokenize

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

# Single-word keywords that are too generic to usefully narrow the keyword
# prefilter's candidate set (their postings cover a large fraction of the
# corpus on their own). Multi-word phrases containing these words are kept --
# e.g. "graph neural networks" is fine even though "networks" is generic.
_GENERIC_TERMS = {
    "model", "models", "method", "methods", "learning", "network", "networks",
    "approach", "approaches", "system", "systems", "analysis", "algorithm",
    "algorithms", "data", "deep", "framework", "frameworks", "technique",
    "techniques", "based", "using", "via", "study", "review", "survey",
    "paper", "research", "performance",
}


def _dedup_key(keyword: str) -> str:
    """Normalize a keyword for near-duplicate detection (case + simple plural)."""
    normalized = keyword.strip().lower()
    if normalized.endswith("s") and len(normalized) > 1:
        normalized = normalized[:-1]
    return normalized


def _filter_keywords(keywords: List[str]) -> List[str]:
    """Drop empty/overly-generic single-word keywords and near-duplicates.

    Preserves the order of first occurrence. Multi-word phrases are kept even
    if one of their words is in `_GENERIC_TERMS`.
    """
    filtered: List[str] = []
    seen = set()

    for keyword in keywords:
        stripped = keyword.strip()
        if not stripped:
            continue

        key = _dedup_key(stripped)
        if key in seen:
            continue

        tokens = tokenize(stripped)
        if len(tokens) == 1 and _dedup_key(tokens[0]) in _GENERIC_TERMS:
            continue

        seen.add(key)
        filtered.append(stripped)

    return filtered


class QwenKeywordExtractor:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="cpu",
            )
        except Exception as e:
            print(f"Warning: Qwen model failed to load ({e}). Keyword extraction will use fallback.", flush=True)
            self.model = None
        self._prompt_template = (_PROMPTS_DIR / "qwen_keywords_v1.txt").read_text(encoding="utf-8").strip()

    def generate_keywords(self, query: str, max_keywords: int = 18) -> List[str]:
        if self.model is None:
            return self.fallback_keywords(query)

        prompt = self._prompt_template.format(max_keywords=max_keywords, query=query)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=160,
            temperature=0.1,
            do_sample=False
        )

        # Decode only the newly generated tokens. The prompt now includes a
        # literal JSON-list example, so decoding the full sequence (prompt +
        # completion) would make `text.index("[")` / `text.rindex("]")` below
        # span from the example's brackets to the real output's, breaking the
        # JSON parse.
        generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        try:
            # Parse only the first JSON array. The 0.5B model sometimes
            # continues generating additional hallucinated "Query:"/"Output:"
            # pairs (and a second, often-truncated JSON block) after the real
            # answer; the first array is the actual response to our query.
            start = text.index("[")
            end = text.index("]", start) + 1
            keywords = json.loads(text[start:end])
            parsed = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]
            return _filter_keywords(parsed)
        except Exception:
            return self.fallback_keywords(query)

    def fallback_keywords(self, query: str) -> List[str]:
        tokens = [
            token for token in query.lower().replace("/", " ").split()
            if len(token) > 3
        ]
        return _filter_keywords(tokens)