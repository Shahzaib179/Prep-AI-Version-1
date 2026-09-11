import streamlit as st

from config import AppConfig
from ingestion.pdf_loader import extract_pdf_pages
from ingestion.chunker import build_chunks
from embeddings.embedding_model import get_embedding_model
from vectorstore.faiss_store import build_vector_store
from retrieval.retriever import retrieve_context
from generation.mcq_generator import generate_mcqs
from validation.mcq_validator import validate_mcqs
from quiz.quiz_manager import render_quiz

st.set_page_config(
    page_title="Prep AI",
    page_icon="📚",
    layout="wide",
)

config = AppConfig.from_environment()

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    return get_embedding_model(model_name)

def reset_state():
    for key in [
        "vector_store", "document_name", "document_stats",
        "topics", "mcqs", "quiz_submitted"
    ]:
        st.session_state.pop(key, None)

st.title("📚 Prep AI")
st.caption("Document-grounded MDCAT practice question generator")

with st.sidebar:
    st.header("Settings")
    st.write("**Embedding model:**")
    st.code(config.embedding_model_name, language="text")
    st.write("**Groq model:**")
    st.code(config.groq_model, language="text")
    st.info(
        "Prep AI uses your uploaded document as the knowledge source. "
        "The LLM is instructed not to use outside information."
    )

st.subheader("1. Upload your study material")
uploaded_file = st.file_uploader(
    "Upload a text-based PDF textbook, notes, or exam material.",
    type=["pdf"],
)

if uploaded_file is not None:
    current_name = st.session_state.get("document_name")
    if current_name != uploaded_file.name:
        reset_state()

    if st.button("📥 Process Document", type="primary"):
        try:
            with st.spinner("Extracting PDF text..."):
                pages = extract_pdf_pages(uploaded_file)

            total_chars = sum(len(p["text"]) for p in pages)
            if total_chars < config.min_document_chars:
                st.error(
                    "Very little text was extracted. The PDF may be scanned/image-only "
                    "or contain too little usable text. Version 1 requires a text-based PDF."
                )
                st.stop()

            with st.spinner("Creating chunks..."):
                chunks = build_chunks(
                    pages,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )

            if not chunks:
                st.error("No usable text chunks were created.")
                st.stop()

            with st.spinner("Loading open-source embedding model..."):
                model = load_embedding_model(config.embedding_model_name)

            with st.spinner("Creating embeddings and FAISS index..."):
                store = build_vector_store(chunks, model)

            st.session_state.vector_store = store
            st.session_state.document_name = uploaded_file.name
            st.session_state.document_stats = {
                "pages": len(pages),
                "characters": total_chars,
                "chunks": len(chunks),
            }
            st.session_state.pop("mcqs", None)

            st.success("Document processed successfully.")

        except Exception as exc:
            st.error(f"Document processing failed: {exc}")

if "vector_store" in st.session_state:
    stats = st.session_state["document_stats"]
    st.success(
        f"Ready: {st.session_state['document_name']} | "
        f"{stats['pages']} pages | {stats['chunks']} chunks"
    )

    st.subheader("2. Choose a topic")
    topic = st.text_input(
        "Enter a chapter name or topic from the uploaded document.",
        placeholder="e.g. Cell Biology, Genetics, Thermodynamics",
    )

    col1, col2 = st.columns(2)
    with col1:
        question_count = st.slider(
            "Number of MCQs",
            min_value=5,
            max_value=config.max_questions,
            value=20,
            step=5,
        )
    with col2:
        difficulty = st.selectbox(
            "Difficulty",
            ["MDCAT", "MDCAT - Moderate", "MDCAT - Challenging"],
        )

    if st.button("🧠 Generate MCQs", type="primary"):
        if not topic.strip():
            st.warning("Please enter a chapter or topic.")
            st.stop()

        try:
            model = load_embedding_model(config.embedding_model_name)

            with st.spinner("Retrieving relevant study material..."):
                retrieved = retrieve_context(
                    query=topic.strip(),
                    vector_store=st.session_state["vector_store"],
                    embedding_model=model,
                    top_k=config.retrieval_top_k,
                )

            if not retrieved:
                st.warning(
                    "No relevant material was found for this topic. "
                    "Try a chapter name or a more specific topic."
                )
                st.stop()

            context_chars = sum(len(item["text"]) for item in retrieved)
            if context_chars < config.min_context_chars:
                st.warning(
                    "The retrieved material is too small to generate reliable questions. "
                    "Try a broader topic."
                )
                st.stop()

            with st.spinner("Generating MDCAT questions with Groq..."):
                generated = generate_mcqs(
                    topic=topic.strip(),
                    context_chunks=retrieved,
                    number_of_questions=question_count,
                    difficulty=difficulty,
                    config=config,
                )

            validated = validate_mcqs(
                generated,
                allowed_chunk_ids={item["chunk_id"] for item in retrieved},
            )

            if len(validated) < min(5, question_count):
                st.error(
                    "The model did not produce enough valid questions. "
                    "Try a broader topic or a smaller question count."
                )
                st.stop()

            st.session_state["mcqs"] = validated
            st.session_state["quiz_submitted"] = False

        except Exception as exc:
            st.error(f"MCQ generation failed: {exc}")

if "mcqs" in st.session_state:
    st.divider()
    st.subheader("3. Practice Test")
    render_quiz(
        st.session_state["mcqs"],
        show_sources=config.show_sources,
    )
