from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from string import Formatter
from typing import Any


class PromptRenderError(ValueError):
    """Raised when a prompt template cannot be rendered."""


@dataclass(slots=True)
class PromptTemplate:
    name: str
    version: str
    template: str
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def render(self, **kwargs: Any) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise PromptRenderError(
                f"Missing required field '{e.args[0]}' for template '{self.name}' (v{self.version})"
            ) from e

    @property
    def required_fields(self) -> list[str]:
        return [fname for _, fname, _, _ in Formatter().parse(self.template) if fname is not None]


class PromptRegistry:
    """Registry for versioned prompt templates."""

    def __init__(self) -> None:
        self._templates: dict[str, list[PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> None:
        if template.name not in self._templates:
            self._templates[template.name] = []
        self._templates[template.name].append(template)

    def get(self, name: str, version: str | None = None) -> PromptTemplate | None:
        versions = self._templates.get(name)
        if not versions:
            return None
        if version is None:
            return versions[-1]
        for t in versions:
            if t.version == version:
                return t
        return None

    def list_templates(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "version": t.version, "description": t.description, "tags": t.tags}
            for versions in self._templates.values()
            for t in versions
        ]

    def remove(self, name: str, version: str | None = None) -> bool:
        if name not in self._templates:
            return False
        if version is None:
            del self._templates[name]
            return True
        versions = self._templates[name]
        before = len(versions)
        self._templates[name] = [t for t in versions if t.version != version]
        if not self._templates[name]:
            del self._templates[name]
        return len(self._templates.get(name, [])) < before


# --- Built-in RAG prompts ---

RAG_SYSTEM_PROMPT = PromptTemplate(
    name="rag_system",
    version="1.0.0",
    description="Default RAG system prompt with context and citation instructions",
    tags=["rag", "system", "default"],
    template="""You are a helpful assistant. Answer the user's question based on the provided context.

Context:
{context}

Instructions:
- Answer concisely and accurately based only on the provided context.
- If the context does not contain enough information, say so.
- Cite sources using bracket notation like [1], [2] when referencing specific parts of the context.
- Do not make up information or citations that are not in the context.""",
)

RAG_SYSTEM_PROMPT_NO_CITATION = PromptTemplate(
    name="rag_system_no_citation",
    version="1.0.0",
    description="RAG system prompt without citation requirements",
    tags=["rag", "system", "no-citation"],
    template="""You are a helpful assistant. Answer the user's question based on the provided context.

Context:
{context}

Instructions:
- Answer concisely and accurately based only on the provided context.
- If the context does not contain enough information, say so.
- Do not make up information that is not in the context.""",
)
