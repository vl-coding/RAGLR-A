import json
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QwenKeywordExtractor:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def generate_keywords(self, query: str, max_keywords: int = 18) -> List[str]:
        prompt = f"""
Extract {max_keywords} concise academic search keywords or phrases from the query below.

Return only a JSON list of strings.
Do not include explanations.

Query:
{query}
"""

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