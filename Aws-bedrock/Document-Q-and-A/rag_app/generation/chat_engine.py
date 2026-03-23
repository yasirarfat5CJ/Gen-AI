from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from langchain_core.documents import Document

from rag_app.generation.bedrock import invoke_with_backoff, is_bedrock_throttling_error
from rag_app.generation.guardrails import has_sufficient_evidence, safe_fallback, validate_answer
from rag_app.prompts.templates import ANSWER_PROMPT
from rag_app.retrieval.pipeline import RetrievalPipeline, RetrievedContext


def _format_history(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return "No prior conversation."
    return "\n".join(f"{item['role']}: {item['content']}" for item in chat_history[-8:])


def _format_context(documents: list[Document], max_chars: int) -> str:
    blocks: list[str] = []
    total = 0
    for doc in documents:
        snippet = (
            f"[source:{doc.metadata.get('filename', 'unknown')} p.{doc.metadata.get('page', '?')}]\n"
            f"{doc.page_content.strip()}"
        )
        total += len(snippet)
        if total > max_chars:
            break
        blocks.append(snippet)
    return "\n\n".join(blocks)


@dataclass
class AnswerBundle:
    answer: str
    context: RetrievedContext


class ChatEngine:
    def __init__(self, retrieval_pipeline: RetrievalPipeline, chat_model, validator_model, max_context_chars: int):
        self.retrieval_pipeline = retrieval_pipeline
        self.chat_model = chat_model
        self.validator_model = validator_model
        self.max_context_chars = max_context_chars

    def answer(self, question: str, chat_history: list[dict[str, str]]) -> AnswerBundle:
        retrieved_context = self.retrieval_pipeline.retrieve(question=question, chat_history=chat_history)

        if not has_sufficient_evidence(question, retrieved_context.documents):
            return AnswerBundle(answer=safe_fallback(), context=retrieved_context)

        history_text = _format_history(chat_history)
        context_text = _format_context(retrieved_context.documents, self.max_context_chars)
        chain = ANSWER_PROMPT | self.chat_model
        response = invoke_with_backoff(
            lambda: chain.invoke(
                {
                    "history": history_text,
                    "filters": "None",
                    "context": context_text,
                    "question": question,
                }
            )
        )
        answer = response.content.strip()

        if not validate_answer(self.validator_model, answer, retrieved_context.documents):
            answer = safe_fallback()

        return AnswerBundle(answer=answer, context=retrieved_context)

    def stream_answer(self, question: str, chat_history: list[dict[str, str]]) -> tuple[Generator[str, None, None], RetrievedContext]:
        retrieved_context = self.retrieval_pipeline.retrieve(question=question, chat_history=chat_history)

        if not has_sufficient_evidence(question, retrieved_context.documents):
            def fallback() -> Generator[str, None, None]:
                yield safe_fallback()
            return fallback(), retrieved_context

        history_text = _format_history(chat_history)
        context_text = _format_context(retrieved_context.documents, self.max_context_chars)
        chain = ANSWER_PROMPT | self.chat_model

        def token_stream() -> Generator[str, None, None]:
            answer_parts: list[str] = []
            try:
                for chunk in chain.stream(
                    {
                        "history": history_text,
                        "filters": "None",
                        "context": context_text,
                        "question": question,
                    }
                ):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    answer_parts.append(text)
                    yield text
            except Exception as exc:
                if is_bedrock_throttling_error(exc):
                    yield "Bedrock is temporarily rate-limiting requests. Please wait a few seconds and try again."
                    return
                raise

            final_answer = "".join(answer_parts).strip()
            if not validate_answer(self.validator_model, final_answer, retrieved_context.documents):
                yield "\n\nI don't know based on the retrieved documents."

        return token_stream(), retrieved_context
