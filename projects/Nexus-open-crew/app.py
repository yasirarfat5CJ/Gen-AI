import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nexus-Open-Crew", page_icon="🚀", layout="wide")

st.title("🌐 Nexus-Open-Crew: AI Research Engine")
st.markdown("Your personal research team powered by **Local Open Source Models**.")

# --- 1. MODEL CONFIGURATION ---
@st.cache_resource
def load_llms():
    scout_llm = LLM(model="ollama/deepseek-r1:7b", base_url="http://localhost:11434")
    writer_llm = LLM(model="ollama/llama3.2:3b", base_url="http://localhost:11434")
    return scout_llm, writer_llm

scout_llm, writer_llm = load_llms()

# --- 2. THE TOOL WRAPPER ---
class SearchTool(BaseTool):
    name: str = "Search_Tool"
    description: str = "Search the internet for the latest information on a given topic."

    def _run(self, query: str) -> str:
        search = DuckDuckGoSearchRun()
        return search.run(query)

search_tool = SearchTool()

# --- 3. UI INPUTS ---
with st.sidebar:
    st.header("Control Panel")
    topic = st.text_input("Enter Research Topic:", placeholder="e.g., Future of Robotics 2026")
    st.info("Status: Connected to Ollama (Local)")

if st.button("🚀 Run Research Crew"):
    if not topic:
        st.warning("Please enter a topic first.")
    else:
        # --- 4. AGENT & TASK DEFINITIONS (Inside the button trigger) ---
        scout = Agent(
            role="Deep Research Specialist",
            goal=f"Identify top 3 breakthrough trends in {topic} for 2026",
            backstory="Expert at pattern recognition using search tools.",
            llm=scout_llm,
            tools=[search_tool],
            verbose=True
        )

        writer = Agent(
            role="Lead Technical Writer",
            goal=f"Write a compelling executive summary about {topic}",
            backstory="Translates complex research into professional summaries.",
            llm=writer_llm,
            verbose=True
        )

        auditor = Agent(
            role="Senior Quality Auditor",
            goal="Ensure the report is accurate and professional.",
            backstory="Perfectionist editor checking for logical consistency.",
            llm=scout_llm,
            verbose=True,
            allow_delegation=True
        )

        research_task = Task(
            description=f"Deep dive into {topic} 2026 trends. Use search tool to verify.",
            expected_output="3 trends with 1-sentence explanation each.",
            agent=scout
        )

        writer_task = Task(
            description=f"Summarize scout research for {topic} under 200 words.",
            expected_output="Polished 3-paragraph executive summary.",
            agent=writer,
            context=[research_task]
        )

        audit_task = Task(
            description="Review the summary and ensure 3 trends are captured correctly.",
            expected_output="Final polished report approved for distribution.",
            agent=auditor,
            context=[writer_task]
        )

        # --- 5. EXECUTION WITH VISUAL FEEDBACK ---
        with st.status("🤖 Crew is working...", expanded=True) as status:
            st.write("🛰️ Scout is researching the live web...")
            nexus_crew = Crew(
                agents=[scout, writer, auditor],
                tasks=[research_task, writer_task, audit_task],
                process="sequential"
            )
            result = nexus_crew.kickoff()
            status.update(label="Mission Complete!", state="complete", expanded=False)

        # --- 6. DISPLAY RESULTS ---
        st.subheader("📋 Final Research Report")
        st.markdown(result)
        
        st.download_button(
            label="Download Report (.md)",
            data=str(result),
            file_name=f"research_{topic.replace(' ', '_')}.md",
            mime="text/markdown"
        )