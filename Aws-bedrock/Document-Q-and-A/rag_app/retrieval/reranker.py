from __future__ import annotations

import re

from langchain_core.documents import Document

from rag_app.generation.bedrock import invoke_with_backoff
from rag_app.prompts.templates import RERANK_PROMPT


class LLMReranker:
    def __init__(self, chat_model):
        self.chat_model = chat_model

    def rerank(self, question: str, documents: list[Document], top_k: int) -> list[Document]:
        if len(documents) <= top_k:
            return documents

        payload = "\n\n".join(
            f"DOC_{idx} ({doc.metadata.get('filename', 'unknown')} p.{doc.metadata.get('page', '?')}):\n{doc.page_content[:800]}"
            for idx, doc in enumerate(documents)
        )
        try:
            result = invoke_with_backoff(
                lambda: (RERANK_PROMPT | self.chat_model).invoke({"question": question, "documents": payload})
            )
        except Exception:
            return documents[:top_k]
        scores: dict[int, float] = {}
        for line in result.content.splitlines():
            match = re.match(r"DOC_(\d+)\s*:\s*([0-9]*\.?[0-9]+)", line.strip())
            if match:
                scores[int(match.group(1))] = float(match.group(2))

        ranked = sorted(
            enumerate(documents),
            key=lambda item: scores.get(item[0], 0.0),
            reverse=True,
        )
        return [document for _, document in ranked[:top_k]]
