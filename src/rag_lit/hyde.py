import os
from pathlib import Path
from anthropic import Anthropic

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


class ClaudeHyDE:
    def __init__(self, model_name: str):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name

        template = (_PROMPTS_DIR / "claude_hyde_v1.txt").read_text(encoding="utf-8")
        system_part, user_part = template.split("---", 1)
        self._system = system_part.strip()
        self._user_template = user_part.strip()

    def generate(self, query: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=250,
            temperature=0.2,
            system=[{"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": self._user_template.format(query=query)}],
        )
        return response.content[0].text.strip()