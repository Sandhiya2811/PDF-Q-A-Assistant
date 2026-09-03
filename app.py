import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# =========================================================
# Fixed internal settings (not exposed in the UI)
# =========================================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
NOT_FOUND_TEXT = "Information not found"

ACCENT = "#0ea5e9"        
ACCENT_HOVER = "#0284c7"  

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="centered")


# =========================================================
# Theme handling (Light / Dark / Device)
# =========================================================
if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Device"

THEMES = {
    "Light": {
        "bg": "#ffffff",
        "text": "#1a1a1a",
        "muted": "#6b7280",
        "card": "#f8f9fb",
        "sidebar_bg": "#f3f4f6",  # Light mode sidebar background
        "border": "#e5e7eb",
        "input_bg": "#ffffff",
        "toggle_bg": "#ffffff",
        "toggle_border": "#e5e7eb",
    },
    "Dark": {
        "bg": "#0e1117",
        "text": "#f3f4f6",
        "muted": "#9ca3af",
        "card": "#1a1d24",
        "sidebar_bg": "#161a21",  # Dark mode sidebar background
        "border": "#2d313a",
        "input_bg": "#161a21",
        "toggle_bg": "#1a1d24",
        "toggle_border": "#2d313a",
    },
}


def theme_css(choice: str) -> str:
    """Build the CSS block for the selected theme (Light / Dark / Device)."""

    def vars_block(t):
        return f"""
            --bg-color:{t['bg']};
            --text-color:{t['text']};
            --muted-color:{t['muted']};
            --card-bg:{t['card']};
            --sidebar-bg:{t['sidebar_bg']};
            --border-color:{t['border']};
            --input-bg:{t['input_bg']};
            --toggle-bg:{t['toggle_bg']};
            --toggle-border:{t['toggle_border']};
        """

    if choice == "Light":
        root_vars = f":root {{{vars_block(THEMES['Light'])}}}"
    elif choice == "Dark":
        root_vars = f":root {{{vars_block(THEMES['Dark'])}}}"
    else:  # Device -> follow OS preference
        root_vars = f"""
            :root {{{vars_block(THEMES['Light'])}}}
            @media (prefers-color-scheme: dark) {{
                :root {{{vars_block(THEMES['Dark'])}}}
            }}
        """

    return f"""
    <style>
        {root_vars}

        .stApp {{
            background-color: var(--bg-color);
            color: var(--text-color);
        }}

        /* Sidebar background and text styling */
        section[data-testid="stSidebar"] {{
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--border-color);
        }}

        section[data-testid="stSidebar"] * {{
            color: var(--text-color) !important;
        }}

        /* Padding-top so Streamlit main header bar won't overlap the title */
        .block-container {{
            padding-top: 5rem;
            padding-bottom: 3rem;
            max-width: 760px;
        }}

        h1, h1 span, .stMarkdown h1 {{
            font-weight: 700;
            color: var(--text-color);
            letter-spacing: -0.01em;
            font-size: 2.1rem !important;
            white-space: nowrap;
        }}

        .section-label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted-color);
            margin-bottom: 0.3rem;
            margin-top: 0.8rem;
        }}

        /* Theme toggle button styling */
        div[data-testid="stButton"] button {{
            background-color: var(--toggle-bg) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--toggle-border) !important;
            border-radius: 8px;
            width: 40px;
            height: 40px;
            padding: 0px !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        div[data-testid="stButton"] button:hover {{
            border-color: {ACCENT} !important;
            color: {ACCENT} !important;
        }}

        /* File uploader compact styling */
        section[data-testid="stFileUploaderDropzone"] {{
            background-color: var(--card-bg);
            border: 1px dashed var(--border-color);
            border-radius: 10px;
            padding: 0.2rem !important;
        }}
        section[data-testid="stFileUploaderDropzone"] * {{
            color: var(--text-color) !important;
        }}
        section[data-testid="stFileUploaderDropzone"] button {{
            background-color: var(--card-bg) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--border-color) !important;
            box-shadow: none !important;
            border-radius: 6px !important;
        }}
        section[data-testid="stFileUploaderDropzone"] button:hover {{
            border-color: {ACCENT} !important;
            color: {ACCENT} !important;
        }}

        /* Text inputs */
        div[data-testid="stTextInput"] input {{
            background-color: var(--input-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 0.65rem 0.9rem;
        }}
        
        div[data-testid="stTextInput"] input::placeholder {{
            color: #9ca3af !important;
            opacity: 1;
        }}

        hr, div[data-testid="stDivider"] {{
            border-color: var(--border-color) !important;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }}

        /* Arrow submit button */
        div[data-testid="stFormSubmitButton"] button {{
            background-color: {ACCENT};
            color: #ffffff;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.55rem 1.05rem;
            transition: background-color 0.15s ease;
            height: 46px;
        }}
        div[data-testid="stFormSubmitButton"] button:hover {{
            background-color: {ACCENT_HOVER};
            color: #ffffff;
        }}

        /* Expander/Source box styling */
        div[data-testid="stExpander"] {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }}
        div[data-testid="stExpander"] * {{
            color: var(--text-color) !important;
        }}

        .question-label {{
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 0.3rem;
            margin-top: 0.4rem;
        }}
        .answer-text {{
            color: var(--text-color);
            margin-top: 0.1rem;
            margin-bottom: 0.5rem;
            line-height: 1.6;
        }}
    </style>
    """


st.markdown(theme_css(st.session_state.theme_choice), unsafe_allow_html=True)


# =========================================================
# Header (Centered Title with small square theme toggle & gray sub-caption)
# =========================================================
top_col1, top_col2 = st.columns([6, 1])

with top_col1:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.5rem;">
            <div style="background: linear-gradient(135deg, #0ea5e9, #6366f1); width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 1.2rem; box-shadow: 0 8px 16px -4px rgba(14,165,233,0.4);">AI</div>
            <div>
                <div style="font-size: 0.8rem; color: var(--muted-color); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Intelligent Search</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: var(--text-color); line-height: 1.1;">PDF Q&A Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with top_col2:
    toggle_icon = "🌙" if st.session_state.theme_choice == "Light" else "☀️"
    if st.button(toggle_icon, use_container_width=True):
        st.session_state.theme_choice = "Dark" if st.session_state.theme_choice == "Light" else "Light"
        st.rerun()

# =========================================================
# Sidebar for Document Upload & API Key Management
# =========================================================
with st.sidebar:
    st.markdown("### 📁 Document Control")
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password", help="Required to generate answers.")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )
    
    if uploaded_file is not None:
        st.success(f"Loaded: {uploaded_file.name}")


# =========================================================
# Session state init
# =========================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None
if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None
if "processed_chunks" not in st.session_state:
    st.session_state.processed_chunks = None


# =========================================================
# Build vectorstore (cached embeddings model)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(pdf_path: str, chunk_size: int, chunk_overlap: int):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs_chunks = text_splitter.split_documents(docs)

    embedding = load_embeddings()

    vectorstore = Chroma.from_documents(
        documents=docs_chunks,
        embedding=embedding,
        collection_name="MDT51_Vector_Collection",
        persist_directory="./chroma_db",
    )
    return vectorstore, len(docs_chunks)


if uploaded_file is not None:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state.processed_file_id != file_id:
        with st.spinner("Reading PDF and building knowledge base..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            vectorstore, n_chunks = build_vectorstore(tmp_path, CHUNK_SIZE, CHUNK_OVERLAP)
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            st.session_state.processed_file_id = file_id
            st.session_state.processed_file_name = uploaded_file.name
            st.session_state.processed_chunks = n_chunks
            st.session_state.chat_history = []

            os.remove(tmp_path)


# =========================================================
# Prompt + RAG chain
# =========================================================
prompt_template = """
You are an assistant for question-answering tasks.

Use the following pieces of retrieved context to answer the question.

If the answer is not present in the context, strictly respond with
"{not_found_text}".

Do not use outside knowledge or assume anything.

Context:

{{context}}

Question: {{question}}

Answer:
""".format(not_found_text=NOT_FOUND_TEXT)

prompt = ChatPromptTemplate.from_template(prompt_template)


def get_rag_chain(retriever, api_key):
    llm = ChatGroq(
        groq_api_key="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",
        model_name="openai/gpt-oss-20b",
        temperature=0,
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# =========================================================
# Main Q&A area (Centered Search Bar)
# =========================================================
st.divider()

if st.session_state.retriever is None:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    st.markdown('<div class="section-label">Ask a question</div>', unsafe_allow_html=True)

    with st.form(key="qa_form", clear_on_submit=False):
        col1, col2 = st.columns([6, 1])
        with col1:
            question = st.text_input(
                "Ask a question about your document",
                placeholder="e.g. What is the main conclusion?",
                label_visibility="collapsed",
            )
        with col2:
            ask_clicked = st.form_submit_button("➤")

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not groq_api_key:
            st.error("Please provide a Groq API key.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                rag_chain = get_rag_chain(st.session_state.retriever, groq_api_key)
                response = rag_chain.invoke(question)

                results_with_scores = st.session_state.vectorstore.similarity_search_with_score(
                    question, k=1
                )

            st.session_state.chat_history.append(
                {"question": question, "answer": response, "sources": results_with_scores}
            )

    # Display chat history, most recent first
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f'<div class="question-label">❓ {entry["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-text">{entry["answer"]}</div>', unsafe_allow_html=True)

        if entry["answer"].strip().lower() != NOT_FOUND_TEXT.lower():
            with st.expander("📄 View sources"):
                for doc, score in entry["sources"]:
                    page = doc.metadata.metadata.get("page", "Unknown") if hasattr(doc.metadata, "metadata") else doc.metadata.get("page", "Unknown")
                    st.write(f"**Page {page}** | Distance: {score:.4f}")
                    st.caption(doc.page_content[:300] + "...")

        st.divider()