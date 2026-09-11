"""Prep AI - Streamlit entry point.

This file intentionally contains only orchestration/UI logic. Document ingestion,
embeddings, vector search, retrieval, generation, validation, and quiz logic stay
in their own modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


# -----------------------------------------------------------------------------
# Streamlit Cloud / local import bootstrap
# -----------------------------------------------------------------------------
# Streamlit Cloud executes app.py from the checked-out repository. Explicitly
# adding the directory containing this file makes local packages such as
# `ingestion`, `embeddings`, and `vectorstore` importable even when the process
# working directory differs from the repository root.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


st.set_page_config(
    page_title="Prep AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Dependency imports with a deployment-friendly error message
# -----------------------------------------------------------------------------
try:
    from config import AppConfig
    from ingestion.pdf_loader import extract_pdf_pages
    from ingestion.chunker import build_chunks
    from embeddings.embedding_model import get_embedding_model
    from vectorstore.faiss_store import build_vector_store
    from retrieval.retriever import retrieve_context
    from generation.mcq_generator import generate_mcqs
    from validation.mcq_validator import validate_mcqs
    from quiz.quiz_manager import render_quiz
except ModuleNotFoundError as exc:
    st.error("Prep AI could not load one of its project modules.")
    st.code(str(exc), language="text")
    st.markdown(
        """
Make sure the **complete project structure** is pushed to the same GitHub
repository as `app.py`. In particular, these folders must exist in the repo:

`ingestion/`, `embeddings/`, `vectorstore/`, `retrieval/`, `generation/`,
`validation/`, and `quiz/`.

Each of those folders should also contain its `__init__.py` file.
"""
    )
    st.stop()
except ImportError as exc:
    st.error("A Prep AI module exists but one of its imports could not be loaded.")
    st.code(str(exc), language="text")
    st.stop()


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
try:
    config = AppConfig.from_environment()
except Exception as exc:
    st.error("Application configuration could not be loaded.")
    st.exception(exc)
    st.stop()


# -----------------------------------------------------------------------------
# Cached resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    """Load the open-source SentenceTransformer model once per app process."""
    return get_embedding_model(model_name)


# -----------------------------------------------------------------------------
# Session-state helpers
# -----------------------------------------------------------------------------
def clear_document_state() -> None:
    """Clear state tied to the previously processed document."""
    keys = (
        "vector_store",
        "document_name",
        "document_signature",
        "document_stats",
        "mcqs",
        "quiz_submitted",
        "retrieved_chunks",
        "active_topic",
    )
    for key in keys:
        st.session_state.pop(key, None)


def file_signature(uploaded_file) -> str:
    """Return a lightweight signature so same-named replacement files reset state."""
    size = getattr(uploaded_file, "size", None)
    return f"{uploaded_file.name}:{size}"


# -----------------------------------------------------------------------------
# Header / sidebar
# -----------------------------------------------------------------------------
st.title("📚 Prep AI")
st.caption("Document-grounded MDCAT entry-test preparation using RAG")

with st.sidebar:
    st.header("App settings")
    st.caption("Embedding model")
    st.code(config.embedding_model_name, language="text")
    st.caption("Groq model")
    st.code(config.groq_model, language="text")
    st.info(
        "Questions are generated from retrieved content in your uploaded PDF. "
        "Version 1 supports text-based PDFs."
    )


# -----------------------------------------------------------------------------
# Step 1: Upload and index PDF
# -----------------------------------------------------------------------------
st.subheader("1. Upload study material")

uploaded_file = st.file_uploader(
    "Upload a text-based PDF textbook, notes, or exam preparation document.",
    type=["pdf"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    signature = file_signature(uploaded_file)
    old_signature = st.session_state.get("document_signature")

    if old_signature is not None and old_signature != signature:
        clear_document_state()

    process_clicked = st.button(
        "📥 Process document",
        type="primary",
        use_container_width=False,
    )

    if process_clicked:
        try:
            with st.spinner("Extracting text from PDF..."):
                # Rewind in case Streamlit or another operation has read the buffer.
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                pages = extract_pdf_pages(uploaded_file)

            if not pages:
                st.error("No readable pages were extracted from the PDF.")
                st.stop()

            total_chars = sum(len(str(page.get("text", ""))) for page in pages)

            if total_chars < config.min_document_chars:
                st.error(
                    "Very little readable text was extracted. The PDF may be scanned "
                    "or image-only. Prep AI Version 1 requires a text-based PDF."
                )
                st.stop()

            with st.spinner("Splitting the document into chunks..."):
                chunks = build_chunks(
                    pages=pages,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )

            if not chunks:
                st.error("No usable text chunks were created from the document.")
                st.stop()

            with st.spinner("Loading the open-source embedding model..."):
                embedding_model = load_embedding_model(config.embedding_model_name)

            with st.spinner("Creating embeddings and FAISS index..."):
                vector_store = build_vector_store(chunks, embedding_model)

            st.session_state["vector_store"] = vector_store
            st.session_state["document_name"] = uploaded_file.name
            st.session_state["document_signature"] = signature
            st.session_state["document_stats"] = {
                "pages": len(pages),
                "characters": total_chars,
                "chunks": len(chunks),
            }
            st.session_state.pop("mcqs", None)
            st.session_state.pop("quiz_submitted", None)
            st.session_state.pop("retrieved_chunks", None)
            st.session_state.pop("active_topic", None)

            st.success("Document processed successfully and indexed in FAISS.")

        except Exception as exc:
            st.error("Document processing failed.")
            st.exception(exc)


# -----------------------------------------------------------------------------
# Step 2: Topic selection and RAG retrieval
# -----------------------------------------------------------------------------
if "vector_store" in st.session_state:
    stats = st.session_state.get("document_stats", {})
    document_name = st.session_state.get("document_name", "Uploaded document")

    st.success(
        f"Ready: {document_name} | "
        f"{stats.get('pages', 0)} pages | "
        f"{stats.get('chunks', 0)} chunks"
    )

    st.subheader("2. Choose a chapter or topic")

    topic = st.text_input(
        "Enter a chapter name or topic that appears in the uploaded document.",
        placeholder="e.g. Cell Biology, Genetics, Thermodynamics",
        key="topic_input",
    )

    col1, col2 = st.columns(2)

    with col1:
        default_count = min(20, config.max_questions)
        question_count = st.slider(
            "Number of MCQs",
            min_value=5,
            max_value=max(5, config.max_questions),
            value=max(5, default_count),
            step=5,
        )

    with col2:
        difficulty = st.selectbox(
            "Difficulty",
            options=[
                "MDCAT",
                "MDCAT - Moderate",
                "MDCAT - Challenging",
            ],
            index=0,
        )

    generate_clicked = st.button(
        "🧠 Generate MCQs",
        type="primary",
    )

    if generate_clicked:
        clean_topic = topic.strip()

        if not clean_topic:
            st.warning("Please enter a chapter name or topic first.")
            st.stop()

        try:
            # Fail early with a friendly message before doing retrieval work.
            config.validate_api_key()

            embedding_model = load_embedding_model(config.embedding_model_name)

            with st.spinner("Retrieving relevant material from FAISS..."):
                retrieved = retrieve_context(
                    query=clean_topic,
                    vector_store=st.session_state["vector_store"],
                    embedding_model=embedding_model,
                    top_k=config.retrieval_top_k,
                )

            if not retrieved:
                st.warning(
                    "No relevant material was found for that topic. Try the exact "
                    "chapter heading or a broader topic from the uploaded document."
                )
                st.stop()

            context_chars = sum(len(str(item.get("text", ""))) for item in retrieved)
            if context_chars < config.min_context_chars:
                st.warning(
                    "Too little relevant document content was retrieved to generate "
                    "reliable MCQs. Try a broader chapter or topic."
                )
                st.stop()

            st.session_state["retrieved_chunks"] = retrieved
            st.session_state["active_topic"] = clean_topic

            with st.spinner("Generating document-grounded MDCAT MCQs with Groq..."):
                generated = generate_mcqs(
                    topic=clean_topic,
                    context_chunks=retrieved,
                    number_of_questions=question_count,
                    difficulty=difficulty,
                    config=config,
                )

            allowed_chunk_ids = {
                str(item.get("chunk_id"))
                for item in retrieved
                if item.get("chunk_id") is not None
            }

            validated = validate_mcqs(
                generated,
                allowed_chunk_ids=allowed_chunk_ids,
            )

            minimum_required = min(5, question_count)
            if len(validated) < minimum_required:
                st.error(
                    "The model did not return enough valid, grounded questions. "
                    "Try a broader topic or request fewer MCQs."
                )
                st.stop()

            st.session_state["mcqs"] = validated
            st.session_state["quiz_submitted"] = False

            if len(validated) < question_count:
                st.warning(
                    f"Generated {len(validated)} valid MCQs out of the requested "
                    f"{question_count}. Invalid or duplicate questions were removed."
                )
            else:
                st.success(f"Generated {len(validated)} valid MCQs.")

        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error("MCQ generation failed.")
            st.exception(exc)


# -----------------------------------------------------------------------------
# Step 3: Quiz
# -----------------------------------------------------------------------------
if st.session_state.get("mcqs"):
    st.divider()
    st.subheader("3. Practice test")

    active_topic = st.session_state.get("active_topic")
    if active_topic:
        st.caption(f"Topic: {active_topic}")

    render_quiz(
        st.session_state["mcqs"],
        show_sources=config.show_sources,
    )
