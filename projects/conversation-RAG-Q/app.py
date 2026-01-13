import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# # Embeddings
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.title("Conversation RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")


# Input Groq API key
api_key = st.text_input("Enter your Groq API key", type="password")

if api_key:
    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0
    )

    session_id = st.text_input("Session ID", value="default_session")

    # Manage chat history
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        accept_multiple_files=False
    )

    if uploaded_file:
        documents = []

        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFDirectoryLoader("./")
        docs = loader.load()
        documents.extend(docs)

        # Splitting and embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever()

        # Contextualize question prompt
        contextualize_q_sys_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question that can be understood "
            "without the chat history. "
            "Do NOT answer the question. "
            "Only reformulate it if needed, otherwise return it as-is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_sys_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # Answer question prompt
        sys_prompt = (
            "You are a helpful AI assistant for question answering over provided documents.\n"
            "Use ONLY the given context to answer the user's question.\n"
            "If the answer is not present in the context, say:\n"
            "'I could not find the answer in the provided documents.'\n\n"
            "Be concise, factual, and clear.\n"
            "Do not make up information.\n"
            "If needed, quote or summarize relevant parts of the context.\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # Session-based chat
        def get_session_history(session_id: str):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        user_input = st.chat_input("Ask a question about the PDF")

        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.write("### Answer")
            st.write(st.session_state.store)
            st.write("Assistant",response["answer"])
            st.write("chat history",session_history.messages)
else:
    st.warning("plz enter the groq api key")