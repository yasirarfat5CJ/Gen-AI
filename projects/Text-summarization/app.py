import os
import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader


from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.schema import Document

# YouTube transcript helper
from youtube_transcript_api import YouTubeTranscriptApi

## Streamlit APP Setup
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)
st.title("🦜 LangChain: Summarize Text")
st.subheader("Summarize URL")

## Sidebar for API Key
with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password",
        placeholder="gsk_..."
    )

generic_url = st.text_input("Enter URL (YouTube or Website)", placeholder="https://...")


if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
else:
    st.info("Please enter your Groq API Key in the sidebar to begin.")
    st.stop()

## LLaMA model using Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0
)

prompt_template = """
Provide a summary of the following content in about 300 words:
Content: {text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

def load_youtube_transcript(url: str):
    try:
        
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        elif "shorts/" in url:
            video_id = url.split("shorts/")[1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL")

        
        ytt_api = YouTubeTranscriptApi() 
        # Returns a FetchedTranscript object in v1.x
        transcript_obj = ytt_api.fetch(video_id)
        
        full_text = " ".join([snippet.text for snippet in transcript_obj])
        
        
        return [Document(page_content=full_text)]
    except Exception as e:
        raise Exception(f"Failed to get YouTube transcript: {str(e)}")


if st.button("Summarize the Content"):

    if not generic_url.strip():
        st.error("Please provide a URL.")
    elif not validators.url(generic_url):
        st.error("Invalid URL format.")
    else:
        try:
            with st.spinner("Processing content..."):
                # 1. Load Data
                if any(x in generic_url for x in ["youtube.com", "youtu.be"]):
                    docs = load_youtube_transcript(generic_url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                # 2. Content Check
                if not docs or not docs[0].page_content.strip():
                    st.error("No text could be extracted.")
                    st.stop()

                # 3. Summarization (Using langchain_classic)
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                # Modern invoke() syntax is standard in 2026
                output = chain.invoke({"input_documents": docs})
                
                summary_text = output["output_text"] if isinstance(output, dict) else output

                st.success("### Summary:")
                st.write(summary_text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
