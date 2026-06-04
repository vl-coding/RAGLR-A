import os
from anthropic import Anthropic


class ClaudeHyDE:
    def __init__(self, model_name: str):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name

    def generate(self, query: str) -> str:
        prompt = f"""
You are helping improve academic paper retrieval.

Given the user's research query, write a concise hypothetical academic abstract
that would be highly relevant to the query.

Rules:
- Do not invent citations.
- Do not invent authors.
- Do not include bullet points.
- Write 3 to 5 sentences.
- Use academic language.

User query:
{query}
"""

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=250,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()