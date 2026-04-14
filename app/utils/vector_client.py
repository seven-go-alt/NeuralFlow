from chromadb import HttpClient
from chromadb.api import ClientAPI

from app.config import get_settings


class InMemoryCollection:
    def __init__(self) -> None:
        self.documents: list[dict] = []

    def add(self, documents, metadatas, ids):
        for document, metadata, item_id in zip(documents, metadatas, ids, strict=False):
            self.documents.append({"id": item_id, "document": document, "metadata": metadata})

    def query(self, query_texts, n_results=3):
        query = query_texts[0].lower()
        ranked = [item for item in self.documents if query in item["document"].lower()]
        top_docs = ranked[:n_results]
        return {
            "documents": [[item["document"] for item in top_docs]],
            "metadatas": [[item["metadata"] for item in top_docs]],
            "ids": [[item["id"] for item in top_docs]],
        }


class InMemoryVectorClient:
    def __init__(self) -> None:
        self.collections: dict[str, InMemoryCollection] = {}

    def get_or_create_collection(self, name: str):
        return self.collections.setdefault(name, InMemoryCollection())


def get_vector_client() -> ClientAPI | InMemoryVectorClient:
    settings = get_settings()
    return HttpClient(host=settings.chroma_host, port=settings.chroma_port)
