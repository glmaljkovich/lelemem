import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage, SummaryIndex
from llama_index.llms.openai import OpenAI
import openai
from llama_index.readers.file import PDFReader
from pathlib import Path


from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.readers.remote import RemoteReader

openai.api_key = st.secrets.openai_key

Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    system_prompt=(
        "You are Gabriel Leonardo Maljkovich, a Software Engineer, and your job is to answer technical questions about your resume / Curriculum Vitae "
        "and Github contributions. "
        "Assume that all questions are done by recruiters and related to your work experience. "
        "\nInstruction: "
        "Keep your answers technical and based on facts - do not hallucinate jobs, positions or technologies. "
        "Refuse to answer any questions not related to your work experience and contributions."
        "You can use Markdown"
    )
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
        index2 = VectorStoreIndex.from_documents(docs)
        index2.set_index_id("glm_cv_summary")
        index2.storage_context.persist("./data/vector_store_summary")
        return index, index2

# llama index
# index = load_data()

# Resume
storage_context = None
storage_context_sum = None
try:
    storage_context = StorageContext.from_defaults(persist_dir="data/vector_store")
    storage_context_sum = StorageContext.from_defaults(persist_dir="data/vector_store_summary")
    index_loaded = True
except:
    load_data()
    storage_context = StorageContext.from_defaults(persist_dir="data/vector_store")
    storage_context_sum = StorageContext.from_defaults(persist_dir="data/vector_store_summary")

index = load_index_from_storage(storage_context, index_id="glm_cv")
summary_index = load_index_from_storage(storage_context_sum, index_id="glm_cv_summary")

query_engine = index.as_query_engine(
    llm=Settings.llm
)

summary_query_engine = summary_index.as_query_engine(
    llm=Settings.llm
)


summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    name="summary_tool",
    description=(
        "Useful for summarization questions related to the author's life"
    ),
)


resume_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="resume_tool",
    description=(
        "Useful for fetching facts about Gabriel's work history from his Resume"
    ),
)

# Github
loader = RemoteReader()

@st.cache_resource(show_spinner=False)
def load_gh():
    with st.spinner(text="Loading GitHub history..."):
        return loader.load_data(
            url="https://github.com/glmaljkovich?tab=repositories&q=&type=&language=&sort=stargazers"
        )
gh_documents = load_gh()

nodes = Settings.node_parser.get_nodes_from_documents(gh_documents)

gh_storage_context = StorageContext.from_defaults()
gh_storage_context.docstore.add_documents(nodes)

gh_vector_index = VectorStoreIndex(nodes, storage_context=gh_storage_context)
gh_query_engine = gh_vector_index.as_query_engine()

gh_tool = QueryEngineTool.from_defaults(
    query_engine=gh_query_engine,
    name="github_tool",
    description=(
        "Useful for retrieving stats about Gabriel's Open source contributions in Github"
    ),
)


agent = OpenAIAgent.from_tools(
    llm=Settings.llm,
    tools=[gh_tool, resume_tool],
    verbose=True,
    system_prompt=(
        "You are Gabriel Leonardo Maljkovich, a Software Engineer, and your job is to answer technical questions about your resume / Curriculum Vitae "
        "and Github contributions. "
        "Assume that all questions are done by recruiters and related to your work experience. "
        "\nInstruction: "
        "Keep your answers technical and based on facts - do not hallucinate jobs, positions or technologies. "
        "Refuse to answer any questions not related to your work experience and contributions."
        "You can use Markdown"
    )
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
            response = agent.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history