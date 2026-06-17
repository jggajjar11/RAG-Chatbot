"""RAG Chain construction and execution."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils.config import GOOGLE_API_KEY, GEMINI_LLM_MODEL

def get_retrieval_chain(retriever):
    """Build and return the RetrievalQA chain."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY must be set in .env")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        streaming=True
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know or that the information is not in the uploaded documents. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain
