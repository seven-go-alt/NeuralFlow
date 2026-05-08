from __future__ import annotations

import hashlib
import json


class EmbeddingCache:
    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    def build_key(self, model: str, text: str) -> str:
        return hashlib.sha256(json.dumps({"model": model, "text": text}, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        return self._store.get(self.build_key(model, text))

    def set(self, model: str, text: str, vector: list[float]) -> None:
        self._store[self.build_key(model, text)] = vector
