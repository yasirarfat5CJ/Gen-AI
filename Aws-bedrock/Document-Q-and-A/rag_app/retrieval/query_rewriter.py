from __future__ import annotations

from rag_app.generation.bedrock import invoke_with_backoff
from rag_app.prompts.templates import QUERY_REWRITE_PROMPT


class QueryRewriter:
    def __init__(self, chat_model):
        self.chat_model = chat_model

    def rewrite(self, question: str, chat_history: list[dict[str, str]]) -> list[str]:
        history = "\n".join(f"{item['role']}: {item['content']}" for item in chat_history[-6:]) or "No history."
        try:
            response = invoke_with_backoff(
                lambda: (QUERY_REWRITE_PROMPT | self.chat_model).invoke({"history": history, "question": question})
            )
            rewrites = [line.strip("- ").strip() for line in response.content.splitlines() if line.strip()]
        except Exception:
            rewrites = []
        unique_queries = []
        seen = {question.lower()}
        for candidate in [question, *rewrites]:
            lowered = candidate.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique_queries.append(candidate)
        return [question, *unique_queries][:4]
