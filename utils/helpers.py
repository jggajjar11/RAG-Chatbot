"""Helper utilities for the RAG chatbot."""
import os
import shutil
from pathlib import Path


def is_supported_file(filename: str) -> bool:
    """Return True if the file extension is supported."""
    return Path(filename).suffix.lower() in {".pdf", ".docx", ".txt", ".md"}


def human_readable_size(num_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def clear_vectorstore(path: str) -> None:
    """Delete and recreate the FAISS index directory."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
