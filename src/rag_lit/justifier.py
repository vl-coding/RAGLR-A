import os
import json
from pathlib import Path
from anthropic import Anthropic

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class ClaudeJustifier:
    def __init__(self, model_name: str, max_retries: int = 5, timeout_seconds: float = 60.0):
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_retries=max_retries,
            timeout=timeout_seconds,
        )
        self.model_name = model_name

        template = (_PROMPTS_DIR / "claude_justifier_v1.txt").read_text(encoding="utf-8")
        system_part, user_part = template.split("---", 1)
        self._system = system_part.strip()
        self._user_template = user_part.strip()

    def justify(self, query: str, title: str, abstract: str) -> dict:
        user_text = self._user_template.format(query=query, title=title, abstract=abstract)
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=450,
            temperature=0.1,
            system=[{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_text}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text[len("```"):]
            if text.startswith("json"):
                text = text[len("json"):]
            if text.endswith("```"):
                text = text[: -len("```")]
            text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            return {
                "contribution": None,
                "relevance_justification": text,
                "relevance_score": None,
                "specificity_score": None,
            }