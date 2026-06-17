from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.config import GOOGLE_API_KEY, GEMINI_EMBED_MODEL

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

result = embeddings.embed_query("hello world")
print(len(result))