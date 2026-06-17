import os
import time
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils import config, helpers
from rag.loader import load_documents
from rag.splitter import get_splitter
from rag.embeddings import get_embeddings
from rag.vectorstore import get_vectorstore, add_documents_to_store, save_vectorstore
from rag.chain import get_retrieval_chain

# Load environment variables
load_dotenv(config.PROJECT_ROOT / ".env")

st.set_page_config(page_title="🤖 DocChat", layout="wide")

# Custom CSS for modern look
custom_css = """
/* Rounded cards */
:root{
    --primary-color: #2596be;
    --secondary-background: #ddf0fb;
}
.chat-card {border-radius: 12px; padding: 12px; margin-bottom: 8px;}
.user-msg {background-color: var(--primary-color); color: white; align-self: flex-end;}
.assistant-msg {background-color: var(--secondary-background); color: black; align-self: flex-start;}
/* Scrollable chat area */
.chat-container {height: 70vh; overflow-y: auto; display: flex; flex-direction: column;}
/* Input area */
.input-area {position: fixed; bottom: 20px; width: 100%;}
"""
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_ids" not in st.session_state:
    st.session_state.doc_ids = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# Initial load of existing vectorstore
if st.session_state.vectorstore is None:
    try:
        embeddings = get_embeddings()
        st.session_state.vectorstore = get_vectorstore(embeddings)
    except Exception as e:
        print(f"Error initializing embeddings or vectorstore: {e}")

# Sidebar for file management
st.sidebar.title("Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs, DOCX, TXT, MD",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if st.sidebar.button("Clear All Documents"):
    helpers.clear_vectorstore(config.VECTORSTORE_PATH)
    
    # Also clear files in data dir
    if os.path.exists(config.DATA_DIR):
        for f in os.listdir(config.DATA_DIR):
            file_path = os.path.join(config.DATA_DIR, f)
            if os.path.isfile(file_path) and f != ".gitkeep":
                os.unlink(file_path)
                
    st.session_state.vectorstore = None
    st.session_state.doc_ids.clear()
    st.session_state.total_chunks = 0
    st.success("All documents cleared.")

# Process newly uploaded files
if uploaded_files:
    # Save files to data directory first to be processed by loaders
    saved_paths = []
    new_files = False
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.doc_ids:
            file_path = os.path.join(config.DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_paths.append(file_path)
            st.session_state.doc_ids.add(uploaded_file.name)
            new_files = True
            
    if new_files and saved_paths:
        with st.spinner("Processing uploaded files..."):
            new_docs = load_documents(saved_paths)
            splitter = get_splitter()
            chunks = splitter.split_documents(new_docs)
            st.session_state.total_chunks += len(chunks)
            
            embeddings = get_embeddings()
            st.session_state.vectorstore = add_documents_to_store(
                st.session_state.vectorstore, chunks, embeddings
            )
            save_vectorstore(st.session_state.vectorstore)
            st.success(f"Added {len(chunks)} chunks from new documents.")

# Display document summary
st.sidebar.markdown(f"**Total Documents:** {len(st.session_state.doc_ids)}")
st.sidebar.markdown(f"**Total Chunks:** {st.session_state.total_chunks}")

# Main chat area
st.title("🤖 DocChat")
chat_container = st.container()
with chat_container:
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f"<div class='chat-card user-msg'>{entry['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-card assistant-msg'>{entry['content']}</div>", unsafe_allow_html=True)
            if entry.get("sources"):
                with st.expander("Sources"):
                    for src in entry["sources"]:
                        st.write(src)

# Input box
user_input = st.chat_input("Ask a question")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Generating response..."):
        if st.session_state.vectorstore is None:
            response = "No documents have been indexed yet. Please upload files first."
            sources = []
        else:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
            qa_chain = get_retrieval_chain(retriever)
            result = qa_chain.invoke({"input": user_input})
            response = result.get("answer", "I could not find relevant information in the uploaded documents.")
            
            source_docs = result.get("context", [])
            sources = list(set([doc.metadata.get("source", "unknown") for doc in source_docs]))
            
        st.session_state.chat_history.append({"role": "assistant", "content": response, "sources": sources})
        st.rerun()
