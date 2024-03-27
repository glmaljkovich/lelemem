import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.llms.openai import OpenAI
import openai
from llama_index.readers.file import PDFReader
from llama_index.core.memory import ChatMemoryBuffer
from pathlib import Path

openai.api_key = st.secrets.openai_key

Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5
)

# Page title
st.set_page_config(page_title='AI Resume', page_icon='🤖')
st.title('🤖 AI Resumé')

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm Gabriel. Ask me a question about my work experience!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing work history – hang tight! This should take 1-2 minutes."):
        reader = PDFReader()
        docs = reader.load_data(file=Path("./data/cv_gmaljkovich_english.pdf"))
        index = VectorStoreIndex.from_documents(docs)
        index.set_index_id("glm_cv")
        index.storage_context.persist("./data/vector_store")
        return index

# llama index
# index = load_data()
storage_context = StorageContext.from_defaults(persist_dir="data/vector_store")
index = load_index_from_storage(storage_context, index_id="glm_cv")

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    verbose=False,
    context_prompt=(
        "You are Gabriel Leonardo Maljkovich, a Software Engineer, and your job is to answer technical questions about your resume / Curriculum Vitae."
        " Assume that all questions are done by recruiters and related to your work experience. "
        "Here is your work experience:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above. "
        "Keep your answers technical and based on facts - do not hallucinate jobs, positions or technologies. "
        "Refuse to answer any questions not related to your work experience."
    
    ),
)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history