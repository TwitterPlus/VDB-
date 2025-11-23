import os
from pathlib import Path
from typing import List

import chromadb
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import uuid


# -------------------------
# Setup & configuration
# -------------------------

# Base directory (where this app.py lives)
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env in the project directory
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment. Please set it in your .env file.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
LLM_MODEL_NAME = "gemini-2.5-flash"
llm = genai.GenerativeModel(LLM_MODEL_NAME)

# Create a persistent Chroma client (stored locally in 'chroma_db' under the project dir)
chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))

COLLECTION_NAME = "policies"
POLICIES_FILE = BASE_DIR / "policies.txt"


@st.cache_resource(show_spinner=True)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def ensure_policies_collection():
    """Ensure the 'policies' collection exists and is populated with paragraph chunks.

    - If the collection does not exist, create it.
    - If it exists but is empty, load policies.txt, chunk, embed, and populate.
    """
    # Get or create the collection
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        collection = chroma_client.create_collection(name=COLLECTION_NAME)

    # Check if collection has any documents
    existing = collection.get()
    if existing and existing.get("ids"):
        return collection

    # Otherwise, populate from policies.txt
    if not POLICIES_FILE.exists():
        raise FileNotFoundError(f"{POLICIES_FILE} not found in the project directory.")

    with POLICIES_FILE.open("r", encoding="utf-8") as f:
        policy_text = f.read()

    # Split into paragraphs separated by blank lines
    raw_chunks: List[str] = [p.strip() for p in policy_text.split("\n\n") if p.strip()]

    model = get_embedding_model()

    ids = [str(uuid.uuid4()) for _ in raw_chunks]
    embeddings = model.encode(raw_chunks).tolist()

    collection.add(
        ids=ids,
        documents=raw_chunks,
        embeddings=embeddings,
        metadatas=[{"source": str(POLICIES_FILE), "type": "paragraph"} for _ in raw_chunks],
    )

    return collection


def retrieve_policies(question: str, k: int = 3):
    """Retrieve top-k relevant policy paragraphs from Chroma for a question."""
    model = get_embedding_model()
    query_embedding = model.encode([question]).tolist()

    collection = ensure_policies_collection()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    return results


def generate_answer(question: str, results) -> str:
    """Use Gemini to generate an answer based on retrieved policy paragraphs."""
    docs = results.get("documents", [[]])[0]

    if not docs:
        return "I couldn't find any relevant policy information to answer that question."

    context = "\n\n".join(docs)

    prompt = f"""You are a helpful assistant for a retail store.
Use ONLY the information in the CONTEXT section below to answer the QUESTION.
If the answer is not clearly present in the context, say that you don't know based on the given policies.

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and concisely.
"""

    response = llm.generate_content(prompt)
    return getattr(response, "text", str(response))


# -------------------------
# Streamlit Chatbot UI
# -------------------------

st.set_page_config(page_title="Retail Policy RAG Chatbot", page_icon="📄", layout="wide")

st.title("Retail Policy RAG Chatbot")
st.write(
    "Chat with a bot that knows the store policies. Your question is embedded, "
    "relevant policy paragraphs are retrieved via Chroma, and Gemini answers "
    "using only that context."
)

# Sidebar: simple settings only (no direct reference to underlying docs)
with st.sidebar:
    st.header("Settings")
    k_default = 3
    k = st.slider(
        "Amount of internal context to use (top-k paragraphs):",
        min_value=1,
        max_value=5,
        value=k_default,
    )

# Ensure collection is ready (done once per app session)
with st.spinner("Initializing vector store (Chroma) and embeddings..."):
    collection = ensure_policies_collection()
    count = len(collection.get().get("ids", []))

st.success(f"Chroma collection '{COLLECTION_NAME}' is ready with {count} paragraphs.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question about the store policies...")

if user_input:
    # Store and display user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve context and generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = retrieve_policies(user_input, k=k)
            answer = generate_answer(user_input, results)

            st.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})
