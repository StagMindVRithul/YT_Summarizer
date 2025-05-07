import ssl
import certifi
import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from langchain.docstore.document import Document

# Streamlit App setup
st.set_page_config(page_title="Langchain: Youtube Summarizer", page_icon="🔎📖")
st.title("🔎📖 Youtube Summarizer using LangChain")
st.subheader('Summarize by entering a YouTube URL below ⬇️')

# Setting up the GROQ API KEY input and URL input
with st.sidebar.expander("Click to open Sidebar", expanded=True):
    groq_api_key = st.text_input('Give the GROQ API Key here!!!', type='password')
youtube_url = st.text_input("YouTube URL", label_visibility="collapsed")

def get_youtube_transcript(video_url):
    """Fetches transcript and metadata from a YouTube video."""
    video_id = video_url.split("v=")[-1]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US'])
        transcript_text = "\n".join([entry["text"] for entry in transcript])
    except Exception as e:
        transcript_text = f"Transcript Error: {e}"
    
    try:
        ydl_opts = {"quiet": True, "format": "bestaudio/best"}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            video_title = info.get("title", "Unknown")
    except Exception as e:
        video_title = f"Metadata Error: {e}"
    
    return video_title, transcript_text

if st.button("Summarize the YouTube Video"):
    if not groq_api_key.strip():
        st.error('Please provide the GROQ API KEY')
    elif not youtube_url.strip():
        st.error('Please provide the YouTube URL')
    elif not validators.url(youtube_url):
        st.error("Please enter a valid YouTube URL!!!!!")
    else:
        try:
            with st.spinner('Summarizing........'):
                # Loading YouTube data
                title, text = get_youtube_transcript(youtube_url)
                
                if not text:
                    st.error("Failed to load content from the YouTube video")
                    st.stop()

                # Create document in the required format
                document = Document(page_content=text, metadata={'title': title})

                # Initialize Chain for Summarization
                llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key, streaming=True)

                prompt_template = f"""
                Provide a summary of the following YouTube video in 300 words with a proper title.
                Title: {title}
                Content:{{text}}
                """
                prompt = PromptTemplate(input_variables=['text'], template=prompt_template)
                chain = load_summarize_chain(llm=llm, chain_type='stuff', verbose=True, prompt=prompt)

                # Get summary
                summary = chain.invoke({'input_documents': [document]})
                summary_text = summary.get("output_text", "Summarization failed")
                
                st.success(summary_text)

        except Exception as e:
            st.exception(f'Exception: {e}')
