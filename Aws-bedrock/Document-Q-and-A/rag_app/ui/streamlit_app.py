from __future__ import annotations

import streamlit as st

from rag_app.config import get_config
from rag_app.generation.bedrock import get_chat_model, get_embeddings
from rag_app.generation.chat_engine import ChatEngine
from rag_app.ingestion.pipeline import ingest_documents, save_uploaded_files
from rag_app.retrieval.bm25 import BM25Okapi
from rag_app.retrieval.compression import SimpleContextualCompressor
from rag_app.retrieval.pipeline import RetrievalPipeline
from rag_app.retrieval.query_rewriter import QueryRewriter
from rag_app.retrieval.reranker import LLMReranker
from rag_app.vectorstore.faiss_store import LocalFAISSStore


@st.cache_resource(show_spinner=False)
def _bootstrap_services():
    config = get_config()
    embeddings = get_embeddings(config)
    vector_store = LocalFAISSStore(config, embeddings)
    query_model = get_chat_model(config, streaming=False)
    answer_model = get_chat_model(config, streaming=True)
    validator_model = get_chat_model(config, streaming=False)

    retrieval_pipeline = RetrievalPipeline(
        config=config,
        vector_store=vector_store,
        query_rewriter=QueryRewriter(query_model),
        reranker=LLMReranker(query_model),
        compressor=SimpleContextualCompressor(),
    )

    chat_engine = ChatEngine(
        retrieval_pipeline=retrieval_pipeline,
        chat_model=answer_model,
        validator_model=validator_model,
        max_context_chars=config.max_context_chars,
    )
    return config, vector_store, chat_engine


def _ensure_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_queries", [])


def run():
    st.set_page_config(page_title="Production RAG on Bedrock", layout="wide")
    st.title("Ask Questions About Your PDF Documents")
    st.caption("A Bedrock-powered RAG assistant with hybrid retrieval, reranking, citations, and conversational memory.")
    _ensure_state()

    if BM25Okapi is None:
        st.warning("`rank-bm25` is not installed. The app is running with vector retrieval only.")

    try:
        config, vector_store, chat_engine = _bootstrap_services()
    except Exception as exc:
        st.error(f"Failed to initialize Bedrock services: {exc}")
        return

    with st.sidebar:
        st.subheader("Ingestion")
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if st.button("Build / Refresh Index", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one PDF before rebuilding the index.")
            else:
                with st.spinner("Ingesting and indexing documents..."):
                    saved_paths = save_uploaded_files(uploaded_files, config.data_dir)
                    ingestion_result = ingest_documents(config, saved_paths)
                    if ingestion_result.documents:
                        vector_store.build(ingestion_result.documents)
                        st.success(f"Indexed {len(ingestion_result.documents)} chunks from {len(saved_paths)} file(s).")
                    else:
                        st.error("No documents were indexed.")
                    for error in ingestion_result.errors:
                        st.warning(error)

        if st.button("Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_sources"] = []
            st.session_state["last_queries"] = []
            st.rerun()

        st.subheader("Retrieval Diagnostics")
        if st.session_state["last_queries"]:
            st.write("Rewritten queries:")
            for query in st.session_state["last_queries"]:
                st.code(query)

        if st.session_state["last_sources"]:
            st.write("Sources:")
            for source in st.session_state["last_sources"]:
                st.write(source)

    if not vector_store.exists():
        st.info("Upload PDFs and build the index to start chatting.")
        return

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your PDFs")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            stream, retrieved_context = chat_engine.stream_answer(
                question=question,
                chat_history=st.session_state["messages"][:-1],
            )
            response_text = placeholder.write_stream(stream)
        except Exception as exc:
            response_text = f"Request failed: {exc}"
            placeholder.markdown(response_text)
            retrieved_context = None

    st.session_state["messages"].append({"role": "assistant", "content": response_text})

    if retrieved_context:
        st.session_state["last_queries"] = retrieved_context.rewritten_queries
        st.session_state["last_sources"] = [
            f"{doc.metadata.get('filename', 'unknown')} p.{doc.metadata.get('page', '?')}"
            for doc in retrieved_context.documents
        ]
