import json
import os
import sys
import boto3
import streamlit as st
import shutil

## we will be using titan embedding models to generate embeddings
from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

## vector embeddings and vector store
from langchain_community.vectorstores import FAISS

## LLM models
from langchain_core.prompts import PromptTemplate

## Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock
)

# -------------------- DATA INGESTION --------------------
def data_ingestion():
    documents = []

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("data", file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    split_docs = text_splitter.split_documents(documents)
    return split_docs


# -------------------- VECTOR STORE --------------------
def get_vector_store(docs):
  
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

# -------------------- LLM --------------------
def get_llama2_llm():
    return Bedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"max_gen_len": 512}
    )

# -------------------- PROMPT --------------------
prompt_template = """
Human:
Use ONLY the context below to answer the question.
If the answer is not present in the context, say "I don't know".
Do NOT use prior knowledge.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -------------------- RAG CHAIN --------------------
def get_response_llm(llm, vectorstore_faiss, query):
    retriever = vectorstore_faiss.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15}   
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)

# -------------------- STREAMLIT APP --------------------
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    # ---------- PDF UPLOAD ----------
    st.subheader("Upload your PDF")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
      
        if os.path.exists("data"):
            shutil.rmtree("data")

        os.makedirs("data", exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success("PDF uploaded successfully. Click 'Vectors Update'.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.title("Update Or Create Vector Store")

        if st.button("Vectors Update"):
            with st.spinner("Processing documents..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated")

    # ---------- OUTPUT ----------
    if st.button("Output"):
        with st.spinner("Generating answer..."):
            faiss_index = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True
            )
            llm = get_llama2_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            st.write(response)

if __name__ == "__main__":
    main()
