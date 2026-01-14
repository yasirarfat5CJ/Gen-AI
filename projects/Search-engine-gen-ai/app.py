import streamlit as st
from langchain_groq import ChatGroq

# Tools
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
)
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)

# Streamlit callback handler from community
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Core LangChain types
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


# =============================
# Sidebar
# =============================
st.set_page_config(page_title="Web Search Chatbot")

import os

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.getenv("GROQ_API_KEY", "")
)


# =============================
# Tools Setup
# =============================
search_tool = DuckDuckGoSearchRun(name="Search")

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=200)
)

arxiv_tool = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=200)
)

tools = {
    "search": search_tool,
    "wiki": wiki_tool,
    "arxiv": arxiv_tool,
}


# =============================
# Session State
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hi! I can search the web and reference tools.")
    ]

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)


# =============================
# Prompt Template
# =============================
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that uses tools when needed."),
        ("human", "{input}"),
    ]
)


# =============================
# Chat Input
# =============================
if user_input := st.chat_input("Ask anything..."):

    if not api_key:
        st.warning("⚠️ Groq API key is required.")
        st.stop()

    st.chat_message("user").write(user_input)
    st.session_state.messages.append(HumanMessage(content=user_input))

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True,
    )

    # Combine prompt + model
    chain = prompt | llm

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        result = chain.invoke(
            {"input": user_input},
            callbacks=[st_cb],
        )
        answer = result.content

        st.write(answer)
        st.session_state.messages.append(AIMessage(content=answer))
