from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, documents: Iterable[Document]):
        self.documents = list(documents)
        self.corpus = [_tokenize(document.page_content) for document in self.documents]
        self.index = BM25Okapi(self.corpus) if self.documents and BM25Okapi is not None else None

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if not self.index:
            return []

        scores = self.index.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[tuple[Document, float]] = []
        for doc_index, score in ranked:
            document = self.documents[doc_index]
            results.append((document, float(score)))
            if len(results) >= k:
                break
        return results

    @property
    def available(self) -> bool:
        return self.index is not None
