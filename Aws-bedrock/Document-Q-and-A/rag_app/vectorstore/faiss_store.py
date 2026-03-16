from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag_app.config import AppConfig


def _document_hash(document: Document) -> str:
    payload = {
        "content": document.page_content,
        "metadata": document.metadata,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class LocalFAISSStore:
    def __init__(self, config: AppConfig, embeddings):
        self.config = config
        self.embeddings = embeddings
        self.index_dir = config.index_dir
        self.cache_file = config.cache_dir / "embedding_manifest.json"

    def build(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("No documents were provided for indexing.")

        manifest = {"documents": [_document_hash(document) for document in documents]}
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_file.exists():
            current_manifest = json.loads(self.cache_file.read_text())
            if current_manifest == manifest and (self.index_dir / "index.faiss").exists():
                return

        vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(self.index_dir))
        self.cache_file.write_text(json.dumps(manifest, indent=2))

    def exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def load(self) -> FAISS:
        if not self.exists():
            raise FileNotFoundError("FAISS index not found. Run ingestion first.")
        return FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def load_documents(self) -> list[Document]:
        vectorstore = self.load()
        return list(vectorstore.docstore._dict.values())

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        vectorstore = self.load()
        results = vectorstore.similarity_search_with_score(query, k=k * 2)
        return [(document, float(score)) for document, score in results[:k]]
