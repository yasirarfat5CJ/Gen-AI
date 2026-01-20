import os
import boto3
import streamlit as st
import shutil

from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------- BEDROCK CLIENT --------------------
bedrock = boto3.client(service_name="bedrock-runtime")

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock
)

# -------------------- DATA INGESTION --------------------
def data_ingestion():
    documents = []

    if not os.path.exists("data"):
        return []

    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join("data", file))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Skipping {file}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return text_splitter.split_documents(documents)

# -------------------- VECTOR STORE --------------------
def get_vector_store(docs):
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

    if not docs:
        return

    vectorstore = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore.save_local("faiss_index")

# -------------------- LLM --------------------
def get_llm():
    return Bedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"max_gen_len": 512}
    )

# -------------------- PROMPT --------------------
PROMPT = PromptTemplate(
    template="""
Human:
Use ONLY the context below to answer the question.
If the answer is not present in the context, say "I don't know".
Do NOT use prior knowledge.

<context>
{context}
</context>

Question: {question}

Assistant:
""",
    input_variables=["context", "question"]
)

# -------------------- CONTEXT TRUNCATION --------------------
def truncate_context(docs, max_chars=12000):
    context = ""
    for doc in docs:
        if len(context) + len(doc.page_content) > max_chars:
            break
        context += doc.page_content + "\n"
    return context

# -------------------- RAG RESPONSE --------------------
def get_response(llm, vectorstore, query):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(query)
    context = truncate_context(docs)

    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})

# -------------------- STREAMLIT APP --------------------
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    st.subheader("Upload PDF files")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data", exist_ok=True)

        # Remove old PDFs
        for f in os.listdir("data"):
            try:
                os.remove(os.path.join("data", f))
            except Exception:
                pass

        # Save new PDFs
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success("PDFs uploaded successfully. Click Vectors Update.")

    question = st.text_input("Ask a question from the PDFs")

    with st.sidebar:
        st.title("Vector Store")

        if st.button("Vectors Update"):
            with st.spinner("Processing documents..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated")

    if st.button("Output"):
        with st.spinner("Generating answer..."):
            if not os.path.exists("faiss_index"):
                st.error("Vector store not found. Upload PDFs and click Vectors Update.")
                return

            vectorstore = FAISS.load_local(
                "faiss_index",
                bedrock_embeddings,
                allow_dangerous_deserialization=True
            )

            llm = get_llm()
            response = get_response(llm, vectorstore, question)
            st.write(response)

if __name__ == "__main__":
    main()
