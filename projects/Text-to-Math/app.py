import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool 
from langchain_community.utilities import WikipediaAPIWrapper


from langchain_classic.chains import LLMMathChain
from langchain_classic.agents import AgentExecutor, create_react_agent

from langchain_community.callbacks import StreamlitCallbackHandler

# --- Page Setup ---
st.set_page_config(page_title="Math & Data Assistant", page_icon="🧮")
st.title("Text to Math Problem Solver & Data Search")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")


template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

react_prompt = PromptTemplate.from_template(template)

# --- 2. Tool Definitions ---

# Wikipedia
wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="Useful for searching the internet for background information."
)

# Calculator
math_chain = LLMMathChain.from_llm(llm=llm)
calc_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for math questions. Input should be a math expression like '2+2'."
)

# Reasoning
reasoning_prompt = PromptTemplate(
    input_variables=["question"],
    template="Solve the math question logically with a detailed explanation.\nQuestion:{question}\nAnswer:"
)
reasoning_chain = reasoning_prompt | llm
reasoning_tool = Tool(
    name="Reasoning",
    func=reasoning_chain.invoke,
    description="Useful for explaining logic or step-by-step reasoning."
)

tools = [wiki_tool, calc_tool, reasoning_tool]

# --- 3. Agent Initialization ---

# Initialize the Agent using the manual react_prompt
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

# --- 4. Streamlit Chat UI ---

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I can help with math or research. What's on your mind?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

question = st.text_area("Enter your question:")

if st.button("Find my Answer"):
    if question:
        with st.spinner("Processing..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                
                # Execute the agent
                response = agent_executor.invoke(
                    {"input": question},
                    {"callbacks": [st_cb]}
                )
                
                final_answer = response["output"]
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.write(final_answer)
    else:
        st.warning("Please enter a question.")