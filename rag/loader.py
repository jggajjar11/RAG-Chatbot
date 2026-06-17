"""Load PDF, DOCX, TXT, and Markdown files into LangChain Documents."""
import os
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize newlines."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def load_file(file_path: str) -> List[Document]:
    """Load a single file and return a list of Documents."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext in {".txt", ".text"}:
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in {".md", ".markdown"}:
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    # Clean text in every page/chunk
    for doc in docs:
        doc.page_content = _clean_text(doc.page_content)
        doc.metadata["source"] = os.path.basename(file_path)
    return docs


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load multiple files and return a flat list of Documents."""
    all_docs: List[Document] = []
    for fp in file_paths:
        try:
            all_docs.extend(load_file(fp))
        except Exception as e:
            print(f"[loader] Skipping {fp}: {e}")
    return all_docs
