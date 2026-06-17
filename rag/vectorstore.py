"""FAISS Vector Store wrapper."""
import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from utils.config import VECTORSTORE_PATH

def get_vectorstore(embeddings: Embeddings) -> FAISS:
    """Load existing FAISS index or create a new empty one."""
    if os.path.exists(VECTORSTORE_PATH) and os.path.exists(f"{VECTORSTORE_PATH}/index.faiss"):
        try:
            return FAISS.load_local(
                VECTORSTORE_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True  # Required for local FAISS loading
            )
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Creating a new one.")
            # Fallback to creating a new one if load fails, though ideally this shouldn't happen.
            # We return a dummy empty FAISS store using a trick, but it's better to let add_documents_to_store handle creation.
            return None 
    return None

def add_documents_to_store(vectorstore: FAISS, documents: List[Document], embeddings: Embeddings) -> FAISS:
    """Add documents to an existing vectorstore or create a new one."""
    if not documents:
        return vectorstore
        
    if vectorstore is None:
        return FAISS.from_documents(documents, embeddings)
    
    vectorstore.add_documents(documents)
    return vectorstore

def save_vectorstore(vectorstore: FAISS) -> None:
    """Save the vectorstore to disk."""
    if vectorstore is not None:
        vectorstore.save_local(VECTORSTORE_PATH)
