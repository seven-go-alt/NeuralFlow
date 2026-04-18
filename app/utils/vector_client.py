from chromadb import HttpClient
from chromadb.api import ClientAPI

from app.config import get_settings


class InMemoryCollection:
    def __init__(self) -> None:
        self.documents: list[dict] = []

    def add(self, documents, metadatas, ids):
        for document, metadata, item_id in zip(documents, metadatas, ids, strict=False):
            self.documents.append({"id": item_id, "document": document, "metadata": metadata})

    def query(self, query_texts, n_results=3, where=None):
        query = query_texts[0].lower()
        filtered = self._filter(where)
        ranked = [item for item in filtered if query in item["document"].lower()]
        top_docs = ranked[:n_results]
        return {
            "documents": [[item["document"] for item in top_docs]],
            "metadatas": [[item["metadata"] for item in top_docs]],
            "ids": [[item["id"] for item in top_docs]],
            "distances": [[0.0 for _ in top_docs]],
        }

    def get(self, where=None, include=None):
        matched = self._filter(where)
        return {
            "documents": [item["document"] for item in matched],
            "metadatas": [item["metadata"] for item in matched],
            "ids": [item["id"] for item in matched],
        }

    def _filter(self, where=None):
        if not where:
            return list(self.documents)
        if "$and" in where:
            return [item for item in self.documents if all(self._filter_clause(item["metadata"], clause) for clause in where["$and"])]
        return [item for item in self.documents if self._filter_clause(item["metadata"], where)]

    @staticmethod
    def _filter_clause(metadata, clause):
        return all(metadata.get(key) == value for key, value in clause.items())


class InMemoryVectorClient:
    def __init__(self) -> None:
        self.collections: dict[str, InMemoryCollection] = {}

    def get_or_create_collection(self, name: str):
        return self.collections.setdefault(name, InMemoryCollection())


def get_vector_client() -> ClientAPI | InMemoryVectorClient:
    settings = get_settings()
    return HttpClient(host=settings.chroma_host, port=settings.chroma_port)
