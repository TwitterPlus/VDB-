import os
import uuid
from pathlib import Path
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
NOTE_COLLECTION_NAME = "notes"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=NOTE_COLLECTION_NAME)


@st.cache_resource(show_spinner=False)
def get_llm():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set in .env. Please create a .env file with GEMINI_API_KEY=your_key."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


def load_notes_text() -> str:
    notes_path = BASE_DIR / "Demystifying The Myth.txt"
    if not notes_path.exists():
        raise FileNotFoundError(f"Demystifying The Myth.txt not found at {notes_path}")
    return notes_path.read_text(encoding="utf-8")


def ensure_notes_indexed(collection, model: SentenceTransformer) -> None:
    # If the collection already has vectors, assume indexing has been done
    try:
        if collection.count() > 0:
            return
    except Exception:
        # If count is not available for some reason, fall back to always indexing
        pass

    notes_text = load_notes_text()
    raw_chunks: List[str] = [p.strip() for p in notes_text.split("\n\n") if p.strip()]
    if not raw_chunks:
        return

    ids = [str(uuid.uuid4()) for _ in raw_chunks]
    embeddings = model.encode(raw_chunks).tolist()

    metadatas = [{"source": "Demystifying The Myth.txt", "type": "paragraph"} for _ in raw_chunks]

    collection.add(
        ids=ids,
        documents=raw_chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def answer_question_with_rag(
    question: str,
    k: int,
    collection,
    model: SentenceTransformer,
    llm,
) -> Tuple[str, List[str]]:
    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    documents = results.get("documents", [[]])[0]

    context = "\n\n".join(documents)

    prompt = f"""You are a helpful teaching assistant for a class.
Use ONLY the information in the CONTEXT section below to answer the QUESTION.
If the answer is not clearly present in the context, say that you don't know based on the given notes.

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and concisely.
"""

    response = llm.generate_content(prompt)
    answer_text = getattr(response, "text", str(response))

    return answer_text, documents


def main() -> None:
    st.set_page_config(page_title="AI Q&A (RAG)")
    st.title("AI Q&A (RAG over Chroma + Gemini)")
    st.write(
        "Ask questions about AI. Answers are generated using "
        "SentenceTransformer embeddings + Chroma + Gemini, grounded in Demystifying The Myth.txt."
    )

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Number of notes chunks to retrieve", min_value=1, max_value=10, value=3)

    with st.spinner("Loading models and preparing vector store..."):
        model = get_embedding_model()
        collection = get_chroma_collection()
        llm = get_llm()
        ensure_notes_indexed(collection, model)

    question = st.text_input("Your question about the notes:")

    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            answer, docs = answer_question_with_rag(question.strip(), top_k, collection, model, llm)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Show retrieved notes paragraphs"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Result {i}:**")
                st.write(doc)
                st.markdown("---")


if __name__ == "__main__":
    main()
