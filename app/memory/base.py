from abc import ABC, abstractmethod
from typing import Any


class MemoryStore(ABC):
    @abstractmethod
    def add(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError
