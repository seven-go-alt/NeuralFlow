from __future__ import annotations

from litellm import acompletion

from app.config import get_settings


class LLMClient:
    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.litellm_model

    async def generate(self, prompt: str) -> str:
        response = await acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是 NeuralFlow 助手。"},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""
