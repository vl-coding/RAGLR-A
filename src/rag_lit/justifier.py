import os
import json
from anthropic import Anthropic


class ClaudeJustifier:
    def __init__(self, model_name: str):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name

    def justify(self, query: str, title: str, abstract: str) -> dict:
        prompt = f"""
You are evaluating whether an academic paper is useful for a literature review.

Return only valid JSON with these keys:
- contribution
- relevance_justification
- relevance_score
- specificity_score

Scores should be from 1 to 10.

User query:
{query}

Paper title:
{title}

Paper abstract:
{abstract}
"""

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=450,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        try:
            return json.loads(text)
        except Exception:
            return {
                "contribution": None,
                "relevance_justification": text,
                "relevance_score": None,
                "specificity_score": None,
            }