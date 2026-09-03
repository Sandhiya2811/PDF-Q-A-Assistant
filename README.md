# 📄 PDF Q&A Assistant

A Streamlit web app that lets you upload a PDF and ask questions about its content. It uses a **RAG (Retrieval-Augmented Generation)** pipeline — the PDF is chunked, embedded, and stored in a local vector database, then relevant chunks are retrieved and passed to an LLM (via Groq) to generate grounded answers.

If the answer isn't found in the document, the app explicitly says **"Information not found"** instead of guessing.

---

## ✨ Features

- 📤 Upload any PDF and automatically build a searchable knowledge base from it
- 💬 Ask natural language questions about the document
- 📚 View the source chunks (with similarity distance scores) used to generate each answer
- 🌗 Light / Dark / Device theme toggle
- 🧠 Chat history within the session
- ⚡ Fast inference using Groq's LLM API

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| PDF Loading | `langchain_community` (`PyMuPDFLoader`) |
| Text Splitting | `langchain_text_splitters` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `langchain_huggingface` |
| Vector Store | [Chroma](https://www.trychroma.com/) |
| LLM | [Groq](https://groq.com/) (`openai/gpt-oss-20b`) via `langchain_groq` |

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt` yet, create one with:
   ```txt
   streamlit
   langchain
   langchain-community
   langchain-text-splitters
   langchain-huggingface
   langchain-groq
   chromadb
   pymupdf
   sentence-transformers
   ```

---

## 🔑 Configuration

This app requires a **Groq API key** to generate answers.

You have two options:

**Option A — Environment variable**
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

**Option B — Streamlit secrets**

Create a file at `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

**Option C — Enter it in the app**

If no key is found in the environment or secrets, the app will show a password field in the sidebar to enter your key manually.

> ⚠️ **Never commit your API key to source control.** Make sure `.streamlit/secrets.toml` and any `.env` files are listed in `.gitignore`.

---

## ▶️ Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`).

1. Upload a PDF from the sidebar.
2. Wait for the app to build the knowledge base (chunking + embedding).
3. Type your question in the search bar and hit **➤**.
4. Expand **📄 View sources** under any answer to see which parts of the PDF were used.

---

## ⚙️ Internal Settings

These are fixed in code (not exposed in the UI) but can be changed in `app.py`:

| Setting | Value | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per text chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `TOP_K` | 3 | Number of chunks retrieved per query |
| `NOT_FOUND_TEXT` | `"Information not found"` | Fallback response when the answer isn't in the document |

---

## 📁 Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── chroma_db/           # Local vector store (auto-created on first run)
└── README.md
```

---

## ⚠️ Known Limitations / Notes

- The vector store persists to a local `./chroma_db` folder and reuses the same collection name (`MDT51_Vector_Collection`) across uploads — consider clearing this folder or using per-file collections if you plan to use the app with many different PDFs over time.
- Currently supports one PDF at a time per session.
- Answer quality depends on chunking and retrieval quality (`TOP_K = 3`), so very large or complex PDFs may need tuning.

---

## 📄 License

Add your preferred license here (e.g. MIT).
