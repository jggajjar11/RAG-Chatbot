"""Text splitter for chunking documents."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP


def get_splitter() -> RecursiveCharacterTextSplitter:
    """Return a RecursiveCharacterTextSplitter with configured chunk size and overlap."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
