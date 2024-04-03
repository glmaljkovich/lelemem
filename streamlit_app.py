import streamlit as st
import chromadb
import openai
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.readers.remote import RemoteReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import IndexNode
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
def load_gh():
    with st.spinner(text="Loading GitHub history..."):
        loader = RemoteReader()
        return loader.load_data(
            url="https://github.com/glmaljkovich?tab=repositories&q=&type=&language=&sort=stargazers"
        )

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing work history and Github page – hang tight! This should take 1-2 minutes."):
        # Read Github Profile
        gh_documents = load_gh()
        gh_vector_index = VectorStoreIndex.from_documents(gh_documents)
        gh_vector_index.set_index_id("glm_github")
        gh_vector_index.storage_context.persist("./data/vector_store/glm_github")
        gh_node = IndexNode(index_id="glm_github", text="Gabriel's Github profile", obj=gh_vector_index.as_query_engine())
        # Parse CV
        reader = PDFReader()
        docs = reader.load_data(file=Path("./data/cv_gmaljkovich_english.pdf"))
        cv_index = VectorStoreIndex.from_documents(docs)
        cv_index.set_index_id("glm_cv")
        cv_index.storage_context.persist("./data/vector_store/glm_cv")
        cv_node = IndexNode(index_id="glm_cv", text="Gabriel's work history", obj=cv_index.as_query_engine())
        # Insert GH index into main index
        index = VectorStoreIndex(objects=[cv_node, gh_node])
        index.set_index_id("main")
        
        index.storage_context.persist("./data/vector_store/main")
        return index

@st.cache_resource(show_spinner=False)
def load_db():
    with st.spinner(text="Loading and indexing work history and Github page – hang tight! This should take 1-2 minutes."):
        db = chromadb.PersistentClient(path="./data/chroma_db")
        # Read Github Profile
        gh_documents = load_gh()
        gh_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("github"))
        gh_context = StorageContext.from_defaults(vector_store=gh_store)
        gh_index = VectorStoreIndex.from_documents(gh_documents, storage_context=gh_context)
        gh_node = IndexNode(index_id="github", text="Gabriel's Github profile", obj=gh_index.as_retriever(similarity_top_k=3))
        # Parse CV
        reader = PDFReader()
        cv_documents = reader.load_data(file=Path("./data/cv_gmaljkovich_english.pdf"))
        cv_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("resume"))
        cv_context = StorageContext.from_defaults(vector_store=cv_store)
        cv_index = VectorStoreIndex.from_documents(cv_documents, storage_context=cv_context)
        cv_node = IndexNode(index_id="resume", text="Gabriel's work history", obj=cv_index.as_retriever())
        # Build main index
        index = VectorStoreIndex(objects=[cv_node, gh_node])
        index.set_index_id("main")
        return index

@st.cache_resource(show_spinner=False)
def read_db():
    db = chromadb.PersistentClient(path="./data/chroma_db")
    github_store = ChromaVectorStore(chroma_collection=db.get_collection("github"))
    gh_index = VectorStoreIndex.from_vector_store(github_store)
    gh_node = IndexNode(index_id="github", text="Gabriel's Github profile", obj=gh_index.as_retriever(similarity_top_k=3))

    cv_store = ChromaVectorStore(chroma_collection=db.get_collection("resume"))
    cv_index = VectorStoreIndex.from_vector_store(cv_store)
    cv_node = IndexNode(index_id="resume", text="Gabriel's work history", obj=cv_index.as_retriever())
    # Build main index
    index = VectorStoreIndex(objects=[cv_node, gh_node])
    index.set_index_id("main")
    return index

# load data if needed
# index = load_data()
index = None
# try:
#     storage_context = StorageContext.from_defaults(persist_dir="data/vector_store")
#     index = load_index_from_storage(storage_context, index_id="main")
# except:
#     index = load_data()

try:
    index = read_db()
except:
    index = load_db()

memory = ChatMemoryBuffer.from_defaults(token_limit=10000)

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

# prompt: Show me a table with your work history.