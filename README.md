# 🤖 DocChat — RAG Chatbot with LangChain, Gemini & Streamlit

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload documents and chat with them using Google Gemini. Built with LangChain, FAISS, and Streamlit.

---

## ✨ Features

- 📁 **Multi-format document upload** — PDF, DOCX, TXT, Markdown
- 🔍 **Semantic search** — FAISS vector store with Gemini Embeddings
- 💬 **Conversational UI** — ChatGPT-style chat with streaming responses
- 📌 **Source citations** — Every answer references the source document(s)
- 🧠 **Grounded answers** — Strictly answers from uploaded context, no hallucinations
- ⚡ **Cached embeddings** — Incremental indexing for performance
- 🗂️ **Session persistence** — Chat history maintained during session

---

## 🗂️ Project Structure

```
project/
│
├── app.py                  # Streamlit entry point
├── rag/
│   ├── loader.py           # Document loaders (PDF, DOCX, TXT, MD)
│   ├── splitter.py         # RecursiveCharacterTextSplitter config
│   ├── embeddings.py       # Gemini Embeddings setup
│   ├── vectorstore.py      # FAISS save/load logic
│   ├── retriever.py        # Top-K similarity retriever
│   └── chain.py            # LangChain RAG chain
│
├── utils/
│   ├── config.py           # Env vars and constants
│   └── helpers.py          # Utility functions
│
├── data/                   # Temporary uploaded files
├── vectorstore/            # Persisted FAISS index
├── requirements.txt
├── .env
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RAG-Chatbot.git
cd RAG-Chatbot.git
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google Gemini Embeddings |
| RAG Framework | LangChain |
| Vector Store | FAISS (local) |
| Text Splitting | RecursiveCharacterTextSplitter |
| File Parsing | PyPDF2, python-docx, unstructured |

---

## ⚙️ RAG Pipeline

```
User uploads file
       ↓
Text Extraction (PDF / DOCX / TXT / MD)
       ↓
Text Cleaning
       ↓
Chunking (size: 1000, overlap: 200)
       ↓
Gemini Embeddings
       ↓
FAISS Vector Store (persisted locally)
       ↓
User sends a message
       ↓
Top-5 Similarity Search
       ↓
LangChain RAG Chain (Gemini LLM)
       ↓
Grounded response + source citations
```

---

## 📦 Requirements

```
streamlit
langchain
langchain-google-genai
langchain-community
google-generativeai
faiss-cpu
pypdf2
python-docx
unstructured
python-dotenv
tiktoken
```

> Full list in `requirements.txt`.

---

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Your Google Gemini API key |

---

## 💡 Usage

1. **Upload documents** using the left sidebar (PDF, DOCX, TXT, or MD).
2. Wait for indexing to complete — a progress indicator shows status.
3. **Ask questions** in the chat input at the bottom.
4. Answers are grounded strictly in your uploaded documents.
5. Expand **Sources** under each response to see which files were referenced.
6. Use **Clear Documents** in the sidebar to reset the vector store.

---

## 🛡️ Error Handling

| Scenario | Behavior |
|---|---|
| Missing API key | Friendly error with setup instructions |
| Unsupported file type | Warning shown, file skipped |
| Empty document | Skipped with notification |
| Vector store load failure | Auto-reinitialization |
| No relevant context found | `"I could not find relevant information in the uploaded documents."` |

---

## 🧪 Local Development Tips

- The FAISS index is saved to `vectorstore/` — delete this folder to reset embeddings.
- Uploaded files are temporarily stored in `data/` during processing.
- Use `st.cache_resource` for embedding model caching (already configured).

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🙌 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Google Gemini](https://deepmind.google/technologies/gemini/)
- [Streamlit](https://streamlit.io/)
- [FAISS by Meta AI](https://faiss.ai/)