import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# LLM (Groq)
from langchain_groq import ChatGroq

# Embeddings (Ollama)
from langchain_ollama import OllamaEmbeddings

# Loaders & Splitters
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store
from langchain_community.vectorstores import FAISS

# Prompts & Chains
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

##load the groq api  key
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")
print("GROQ KEY LOADED:", os.getenv("GROQ_API_KEY"))


llm=ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)


##creating an  prompt

prompt=ChatPromptTemplate.from_template(
"""
Answer the question based on the context only.
Please provide the most accurate result based on question
<context>
{context}
<context>
Question:{input}
"""
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:

        # Embeddings
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Load PDFs
        st.session_state.loader = PyPDFDirectoryLoader("./research_papers")
        st.session_state.docs = st.session_state.loader.load()

        # Split documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )

        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_docs,
            st.session_state.embeddings
        )



user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.success("Vector database is ready")

import time

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please create document embeddings first")
    else:
        # Create document chain
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )

        # Create retriever
        retriever = st.session_state.vectors.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        # Measure response time
        start = time.process_time()
        res = retrieval_chain.invoke({"input": user_prompt})
        end = time.process_time()

        st.write("Answer")
        st.write(res["answer"])

        st.caption(f"⏱ Response time: {end - start:.2f} seconds")

        # Show retrieved documents
        with st.expander("📄 Document similarity search"):
            for i, doc in enumerate(res["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)



