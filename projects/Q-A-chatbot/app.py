import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# =========================
# Load environment variables
# =========================
load_dotenv()

# =========================
# LangSmith tracing (ONLY LangChain)
# =========================
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# =========================
# Prompt Template
# =========================
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer user queries clearly."),
        ("human", "Question: {question}")
    ]
)

# =========================
# Response Generator
# =========================
def generate_response(question, model_name, temperature, max_tokens):
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        num_predict=max_tokens
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Ollama Q&A Bot", page_icon="🤖")
st.title(" Q&A Chatbot (Ollama + LangChain)")

# -------- Sidebar --------
st.sidebar.title("⚙️ Settings")
api_key=st.sidebar.text_input("Enter your  Ollama Api key:",type="password")

model_name = st.sidebar.selectbox(
    "Select Ollama Model",
    ["mistral", "llama2", "gemma:2b"]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=50,
    max_value=300,
    value=150
)

# -------- Main Input --------
question = st.text_input("Ask your question:")

if st.button("Generate Answer") and question:
    with st.spinner("Thinking..."):
        answer = generate_response(question, model_name, temperature, max_tokens)
    st.write("### Answer:")
    st.write(answer)
else:
    st.write("Please provide the query")