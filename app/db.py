from typing import Tuple
import streamlit as st
import chromadb
from chromadb.api import ClientAPI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, SummaryIndex
from llama_index.readers.file import PDFReader
from llama_index.readers.remote import RemoteReader
from llama_index.core.schema import IndexNode
from pathlib import Path

default_client = chromadb.PersistentClient(path="./data/chroma_db")

@st.cache_resource(show_spinner=False)
def load_gh(db: ClientAPI = default_client) -> VectorStoreIndex:
    with st.spinner(text="Loading GitHub history..."):
        loader = RemoteReader()
        gh_documents = loader.load_data(
            url="https://github.com/glmaljkovich?tab=repositories&q=&type=&language=&sort=stargazers"
        )
        gh_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("github"))
        gh_context = StorageContext.from_defaults(vector_store=gh_store)
        return VectorStoreIndex.from_documents(gh_documents, storage_context=gh_context)

def load_cv(db: ClientAPI = default_client) -> Tuple[VectorStoreIndex]:
    with st.spinner(text="Loading Resume..."):
        reader = PDFReader()
        cv_documents = reader.load_data(file=Path("./data/cv_gmaljkovich_english.pdf"))
        cv_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("resume"))
        cv_context = StorageContext.from_defaults(vector_store=cv_store)
        index = VectorStoreIndex.from_documents(cv_documents, storage_context=cv_context)

        cv_summary_store = ChromaVectorStore(chroma_collection=db.get_or_create_collection("resume_summary"))
        cv_summary_context = StorageContext.from_defaults(vector_store=cv_summary_store)
        summary_index = SummaryIndex.from_documents(cv_documents, storage_context=cv_summary_context)
        return index, summary_index

@st.cache_resource(show_spinner=False)
def load_db():
    """
    Initializes the Chroma DB
    """
    with st.spinner(text="Loading and indexing work history and Github page – hang tight! This should take 1-2 minutes."):
        # Read Github Profile
        gh_index = load_gh()
        gh_node = IndexNode(index_id="github", text="Gabriel's Github profile", obj=gh_index.as_query_engine())
        # Parse CV
        cv_index, summary_index = load_cv()
        cv_node = IndexNode(index_id="resume", text="Gabriel's work history", obj=cv_index.as_retriever())
        cv_summary_node = IndexNode(index_id="resume_summary", text="Gabriel's work history for summaries", obj=summary_index.as_retriever())
        # Build main index
        index = VectorStoreIndex(objects=[cv_node, gh_node, cv_summary_node])
        index.set_index_id("main")
        return index

def read_cv(db: ClientAPI) -> Tuple[VectorStoreIndex]:
    cv_store = ChromaVectorStore(chroma_collection=db.get_collection("resume"))
    cv_summary_store = ChromaVectorStore(chroma_collection=db.get_collection("resume_summary"))
    return VectorStoreIndex.from_vector_store(cv_store), SummaryIndex.from_vector_store(cv_summary_store)

def read_gh(db: ClientAPI) -> VectorStoreIndex:
    gh_store = ChromaVectorStore(chroma_collection=db.get_collection("github"))
    return VectorStoreIndex.from_vector_store(gh_store)

@st.cache_resource(show_spinner=False)
def read_db():
    """
    Reads an existing Chroma DB
    """
    gh_index = read_gh(default_client)
    gh_node = IndexNode(index_id="github", text="Gabriel's Github profile", obj=gh_index.as_query_engine())

    cv_index, summary_index = read_cv(default_client)
    cv_node = IndexNode(index_id="resume", text="Gabriel's work history", obj=cv_index.as_retriever())
    # Build main index
    index = VectorStoreIndex(objects=[cv_node, gh_node])
    index.set_index_id("main")
    return index
