from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any

import litellm

from app.core.llm import LLMClient
from app.ingestion.multimodal.extractor import ExtractedImage

logger = logging.getLogger(__name__)


@dataclass
class VisionDescriber:
    llm_client: LLMClient
    vision_model: str = "gpt-4o"
    prompt_template: str = "Describe this image in detail, focusing on content relevant to document understanding."
    _settings: Any = field(default=None, compare=False)

    async def describe_images(self, images: list[ExtractedImage]) -> list[str]:
        descriptions: list[str] = []
        for img in images:
            try:
                b64 = base64.b64encode(img.image_data).decode("utf-8")
                kwargs: dict[str, Any] = {
                    "model": self.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt_template},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/{img.format};base64,{b64}"},
                                },
                            ],
                        }
                    ],
                }
                if self._settings and hasattr(self._settings, "llm_api_base") and self._settings.llm_api_base:
                    kwargs["api_base"] = self._settings.llm_api_base
                if self._settings and hasattr(self._settings, "llm_api_key") and self._settings.llm_api_key:
                    kwargs["api_key"] = self._settings.llm_api_key

                response = await litellm.acompletion(**kwargs)
                desc = response.choices[0].message.content or "[No description generated]"
                descriptions.append(desc)
            except Exception as exc:
                logger.warning("failed to describe image %d: %s", img.image_index, exc)
                descriptions.append("[Image description unavailable]")
        return descriptions
