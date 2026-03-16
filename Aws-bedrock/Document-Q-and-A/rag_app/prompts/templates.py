from langchain_core.prompts import ChatPromptTemplate


QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You rewrite user questions for retrieval. Generate 3 focused search variations. "
                "Keep the meaning intact. If the user asks to list, show all, summarize all, "
                "or enumerate items, create rewrites that maximize recall across the full document. "
                "Return each rewrite on a new line with no numbering."
            ),
        ),
        ("human", "Conversation summary:\n{history}\n\nOriginal question:\n{question}"),
    ]
)


RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Score how useful each document is for answering the user's question. "
                "If the user asks for all items, all projects, a full list, or a complete summary, "
                "prefer documents that add new non-overlapping facts over documents that repeat the same fact. "
                "Return one line per document as DOC_ID:score where score is 0 to 1."
            ),
        ),
        ("human", "Question:\n{question}\n\nDocuments:\n{documents}"),
    ]
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a retrieval-grounded assistant. Use only the supplied context for facts. "
                "Conversation history may help resolve references, but it is not evidence and must not add facts. "
                "If the context is insufficient, say you do not know. "
                "Cite supporting sources inline using [source:filename p.X]. "
                "Do not cite sources you did not use. Refuse to speculate. "
                "Never say you are assuming something was mentioned earlier in the conversation. "
                "If the user asks to list or display all items, extract every relevant item present in the context, "
                "merge duplicates, and present a complete list based only on the retrieved context. "
                "Do not stop after the first matching item. "
                "If the user asks for descriptions, details, or explanations, include the supporting description for each listed item. "
                "For resume project questions, return each project as a separate bullet with its title followed by its description bullets. "
                "If only part of the full list is visible in context, explicitly say the list may be incomplete."
            ),
        ),
        (
            "human",
            (
                "Conversation history:\n{history}\n\n"
                "Metadata filters:\n{filters}\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}"
            ),
        ),
    ]
)


GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Decide whether the answer is fully supported by the provided sources. "
                "If unsupported or missing citations, respond with UNSUPPORTED. "
                "Otherwise respond with SUPPORTED."
            ),
        ),
        ("human", "Answer:\n{answer}\n\nSources:\n{sources}"),
    ]
)
