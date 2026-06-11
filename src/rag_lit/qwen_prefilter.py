import json
from pathlib import Path
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


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

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            keywords = json.loads(text[start:end])
            return [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]
        except Exception:
            return self.fallback_keywords(query)

    def fallback_keywords(self, query: str) -> List[str]:
        return [
            token for token in query.lower().replace("/", " ").split()
            if len(token) > 3
        ]