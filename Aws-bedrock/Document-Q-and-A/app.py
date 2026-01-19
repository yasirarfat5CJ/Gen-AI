import json
import os 
import sys
import boto3
import streamlit as st

## we will be using titan embedding models  to generate embeddings 
from langchain_aws import BedrockEmbeddings
from langchain_community.llms import Bedrock

import  numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

##vector embeddings and vector store
from langchain_community.vectorstores import FAISS

##LLm models
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import retrieval_qa
from langchain_community.chat_models import BedrockChat

##Bedrock client
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    ## splitting
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    split_docs=text_splitter.split_documents(documents)

    return split_docs

## vecor embedding and vector store 
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

# def get_claude_llm():
#     llm = BedrockChat(
#         model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#         client=bedrock,
#         model_kwargs={
#             "max_tokens": 512,
#             "temperature": 0.5
#         }
#     )
#     return llm



def get_llama2_llm():
    llm=Bedrock(model_id="meta.llama3-70b-instruct-v1:0",client=bedrock,
                model_kwargs={'max_gen_len':512}
                )
    return llm

prompt_template="""

Human: Use the followingpieces of context to provide a 
xoncise answer to the question at the end explain in detail.
if you dont know the answer, just say that you don't know try 
to  make up an answer
<context>
{context}
<context>

Question:{question}

Assistant:
"""
PROMPT=PromptTemplate(
    template=prompt_template,input_variables={"context","question"}
)

def get_response_llm(llm, vectorstore_faiss, query):
    retriever = vectorstore_faiss.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query)


def main():
    st.set_page_config("Chat PDf")
    st.header("Chat with Pdf using AWS Bedrock")

    user_question=st.text_input("Ask a Question from the PDF  Files")

    with st.sidebar:
        st.title("Update Or create vector store")

        if st.button("Vectors Update"):
            with st.spinner("processing..."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("Done")
    if st.button("output"):
        with st.spinner("processing..."):
            faiss_index=FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_llama2_llm()

            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")
if __name__ =="__main__":
    main()