import streamlit as st
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.openai import OpenAI
import openai
from llama_index.core import SimpleDirectoryReader

openai.api_key = st.secrets.openai_key

# Page title
st.set_page_config(page_title='ML Model Building', page_icon='🤖')
st.title('🤖 RAG CV Chat')

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]
