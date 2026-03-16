from __future__ import annotations

import re

from langchain_core.documents import Document


def is_exhaustive_query(question: str) -> bool:
    normalized = question.lower()
    trigger_phrases = (
        "all projects",
        "list all",
        "show all",
        "display all",
        "entire document",
        "complete list",
        "full list",
        "every project",
        "all items",
        "all experience",
        "all skills",
    )
    return any(phrase in normalized for phrase in trigger_phrases)


class SimpleContextualCompressor:
    def compress(self, question: str, documents: list[Document], max_sentences: int = 4) -> list[Document]:
        if is_exhaustive_query(question):
            return documents

        question_terms = set(question.lower().split())
        compressed_docs: list[Document] = []
        for document in documents:
            sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", document.page_content) if sentence.strip()]
            prioritized = sorted(
                sentences,
                key=lambda sentence: len(question_terms.intersection(sentence.lower().split())),
                reverse=True,
            )
            shortened = ". ".join(prioritized[:max_sentences]).strip()
            compressed_docs.append(
                Document(
                    page_content=shortened if shortened else document.page_content[:800],
                    metadata=document.metadata,
                )
            )
        return compressed_docs
