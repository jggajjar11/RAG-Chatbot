"""Embeddings configuration using Google Gemini."""
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.config import GOOGLE_API_KEY, GEMINI_EMBED_MODEL

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return configured Gemini embeddings model."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY must be set in .env")
        
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
