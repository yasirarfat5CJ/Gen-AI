from __future__ import annotations

from langchain_core.documents import Document

from rag_app.generation.bedrock import invoke_with_backoff, is_bedrock_throttling_error
from rag_app.prompts.templates import GUARDRAIL_PROMPT


def has_sufficient_evidence(question: str, retrieved_docs: list[Document], min_docs: int = 2) -> bool:
    if len(retrieved_docs) < min_docs:
        return False
    normalized_question = set(question.lower().split())
    overlap_hits = 0
    for doc in retrieved_docs:
        doc_terms = set(doc.page_content.lower().split())
        if normalized_question & doc_terms:
            overlap_hits += 1
    return overlap_hits >= 1


def validate_answer(chat_model, answer: str, retrieved_docs: list[Document]) -> bool:
    sources = "\n\n".join(
        f"{doc.metadata.get('filename', 'unknown')} p.{doc.metadata.get('page', '?')}\n{doc.page_content[:600]}"
        for doc in retrieved_docs
    )
    try:
        verdict = invoke_with_backoff(
            lambda: (GUARDRAIL_PROMPT | chat_model).invoke({"answer": answer, "sources": sources})
        )
    except Exception as exc:
        if is_bedrock_throttling_error(exc):
            return True
        raise
    return verdict.content.strip().upper().startswith("SUPPORTED")


def safe_fallback() -> str:
    return "I don't know based on the retrieved documents."
