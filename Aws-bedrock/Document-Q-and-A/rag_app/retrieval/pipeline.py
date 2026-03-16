from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from langchain_core.documents import Document

from rag_app.config import AppConfig
from rag_app.retrieval.bm25 import BM25Index
from rag_app.retrieval.compression import SimpleContextualCompressor, is_exhaustive_query
from rag_app.retrieval.query_rewriter import QueryRewriter
from rag_app.retrieval.reranker import LLMReranker
from rag_app.vectorstore.faiss_store import LocalFAISSStore


@dataclass(slots=True)
class RetrievedContext:
    documents: list[Document]
    rewritten_queries: list[str]


class RetrievalPipeline:
    def __init__(
        self,
        config: AppConfig,
        vector_store: LocalFAISSStore,
        query_rewriter: QueryRewriter,
        reranker: LLMReranker,
        compressor: SimpleContextualCompressor,
    ):
        self.config = config
        self.vector_store = vector_store
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.compressor = compressor
        self._bm25_index: BM25Index | None = None

    def retrieve(self, question: str, chat_history: list[dict[str, str]]) -> RetrievedContext:
        documents = self.vector_store.load_documents()
        if self._bm25_index is None or len(self._bm25_index.documents) != len(documents):
            self._bm25_index = BM25Index(documents)

        exhaustive_query = is_exhaustive_query(question)
        queries = self.query_rewriter.rewrite(question=question, chat_history=chat_history)
        retrieval_k = self.config.retrieval_k * 2 if exhaustive_query else self.config.retrieval_k
        final_k = max(self.config.final_k, 12) if exhaustive_query else self.config.final_k
        fused_documents = self._hybrid_search(queries, retrieval_k)
        reranked_documents = self.reranker.rerank(question, fused_documents, final_k)
        compressed_documents = self.compressor.compress(question, reranked_documents)
        return RetrievedContext(documents=compressed_documents, rewritten_queries=queries)

    def _hybrid_search(self, queries: list[str], retrieval_k: int) -> list[Document]:
        fused_scores: dict[str, float] = defaultdict(float)
        doc_lookup: dict[str, Document] = {}
        bm25_weight, vector_weight = self.config.retriever_weights

        for query in queries:
            if self._bm25_index and self._bm25_index.available:
                for rank, (document, score) in enumerate(self._bm25_index.search(query, retrieval_k), start=1):
                    key = self._doc_key(document)
                    doc_lookup[key] = document
                    fused_scores[key] += bm25_weight * (1 / (rank + 20)) * max(score, 0.01)

            for rank, (document, score) in enumerate(
                self.vector_store.search(query, k=retrieval_k),
                start=1,
            ):
                key = self._doc_key(document)
                doc_lookup[key] = document
                similarity = max(1.0 - score, 0.01)
                fused_scores[key] += vector_weight * (1 / (rank + 20)) * similarity

        ranked_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [doc_lookup[key] for key in ranked_keys[:retrieval_k]]

    @staticmethod
    def _doc_key(document: Document) -> str:
        return f"{document.metadata.get('filename')}::{document.metadata.get('page')}::{document.metadata.get('chunk_id')}"
