from __future__ import annotations

import io
import json
import os
import re
from typing import Any

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# Prep AI - Self-contained Streamlit application
# This version intentionally has NO local-project imports.
# It can run on Streamlit Cloud even if package folders are
# accidentally omitted from the GitHub deployment.
# ============================================================

st.set_page_config(
    page_title="Prep AI - MDCAT",
    page_icon="📚",
    layout="wide",
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
MAX_QUESTIONS = 100


def get_secret(name: str, default: str = "") -> str:
    """Read Streamlit Cloud secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
        if value is not None:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(uploaded_file) -> list[dict[str, Any]]:
    uploaded_file.seek(0)
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

    return pages


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[dict[str, Any]]:
    """Create word-based overlapping chunks while retaining page metadata."""
    chunks = []
    chunk_id = 0

    for page in pages:
        words = page["text"].split()
        if not words:
            continue

        start = 0
        local_index = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            text = " ".join(words[start:end]).strip()

            if text:
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id}",
                        "page_number": page["page_number"],
                        "local_index": local_index,
                        "text": text,
                    }
                )
                chunk_id += 1
                local_index += 1

            if end >= len(words):
                break

            start = max(end - overlap, start + 1)

    return chunks


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype="float32")


def build_index(
    chunks: list[dict[str, Any]],
    model: SentenceTransformer,
):
    embeddings = embed_texts(model, [c["text"] for c in chunks])
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(
    query: str,
    index,
    chunks: list[dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    query_embedding = embed_texts(model, [query])
    k = min(top_k, len(chunks))
    scores, ids = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(chunks[int(idx)])
        item["score"] = float(score)
        results.append(item)

    return results


def make_context(results: list[dict[str, Any]]) -> str:
    blocks = []
    for item in results:
        blocks.append(
            f"[{item['chunk_id']} | page {item['page_number']}]\n"
            f"{item['text']}"
        )
    return "\n\n".join(blocks)


def extract_json(text: str) -> Any:
    text = text.strip()

    # Remove markdown fences if the model adds them.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try the first JSON object/array in the response.
    candidates = []
    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj >= 0 and last_obj > first_obj:
        candidates.append(text[first_obj:last_obj + 1])

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr >= 0 and last_arr > first_arr:
        candidates.append(text[first_arr:last_arr + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError("The AI returned invalid JSON.")


def build_prompt(
    context: str,
    topic: str,
    count: int,
    difficulty: str,
    previous_questions: list[str],
) -> str:
    previous = "\n".join(f"- {q}" for q in previous_questions[-30:])
    if not previous:
        previous = "None"

    return f"""
You are an expert MDCAT question writer.

Create exactly {count} high-quality MDCAT-level multiple-choice questions
about this topic:

TOPIC:
{topic}

DIFFICULTY:
{difficulty}

STRICT SOURCE RULE:
Use ONLY the supplied textbook context. Do not introduce facts that are
not supported by the context.

REQUIREMENTS:
1. Every question must be self-contained.
2. Create exactly four options: A, B, C, D.
3. Exactly one option must be correct.
4. Options must be plausible and from the same subject/domain.
5. Avoid duplicate or near-duplicate questions.
6. Questions should test understanding, application, comparison, reasoning,
   or important factual knowledge appropriate for MDCAT.
7. Give a concise explanation for the correct answer.
8. Give a source_chunk_ids array containing only chunk IDs supplied below.
9. Return JSON only. No markdown and no extra commentary.

OUTPUT FORMAT:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "correct_answer": "A",
      "explanation": "Why the answer is correct.",
      "source_chunk_ids": ["chunk_0"]
    }}
  ]
}}

PREVIOUS QUESTION STEMS TO AVOID:
{previous}

TEXTBOOK CONTEXT:
{context}
""".strip()


def call_groq(
    api_key: str,
    model_name: str,
    prompt: str,
) -> list[dict[str, Any]]:
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.35,
        max_tokens=8192,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate source-grounded MDCAT MCQs. "
                    "Follow the requested JSON schema exactly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content or ""
    data = extract_json(raw)

    if isinstance(data, dict):
        questions = data.get("questions", [])
    elif isinstance(data, list):
        questions = data
    else:
        questions = []

    if not isinstance(questions, list):
        return []

    return questions


def normalize_question(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    question = str(item.get("question", "")).strip()
    options = item.get("options", {})
    answer = str(item.get("correct_answer", "")).strip().upper()
    explanation = str(item.get("explanation", "")).strip()
    source_ids = item.get("source_chunk_ids", [])

    if not question or not isinstance(options, dict):
        return None

    normalized_options = {}
    for letter in ["A", "B", "C", "D"]:
        value = str(options.get(letter, "")).strip()
        if not value:
            return None
        normalized_options[letter] = value

    if answer not in {"A", "B", "C", "D"}:
        return None

    if not explanation:
        explanation = "The selected option is supported by the supplied textbook context."

    if not isinstance(source_ids, list):
        source_ids = []

    source_ids = [str(x) for x in source_ids if str(x).strip()]

    # Prevent two identical options.
    if len(set(v.lower() for v in normalized_options.values())) != 4:
        return None

    return {
        "question": question,
        "options": normalized_options,
        "correct_answer": answer,
        "explanation": explanation,
        "source_chunk_ids": source_ids,
    }


def is_duplicate_question(question: str, existing: list[str]) -> bool:
    def normalize(s: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", s.lower())
        return set(words)

    current = normalize(question)
    if not current:
        return True

    for old in existing:
        old_set = normalize(old)
        if not old_set:
            continue
        similarity = len(current & old_set) / max(1, len(current | old_set))
        if similarity >= 0.82:
            return True

    return False


def validate_questions(
    questions: list[dict[str, Any]],
    allowed_chunk_ids: set[str],
) -> list[dict[str, Any]]:
    valid = []
    stems = []

    for raw in questions:
        item = normalize_question(raw)
        if not item:
            continue

        if item["source_chunk_ids"]:
            if not set(item["source_chunk_ids"]).issubset(allowed_chunk_ids):
                continue

        if is_duplicate_question(item["question"], stems):
            continue

        stems.append(item["question"])
        valid.append(item)

    return valid


def reset_app():
    for key in [
        "pages",
        "chunks",
        "index",
        "embedding_model",
        "document_name",
        "questions",
        "quiz_submitted",
        "answers",
        "score",
    ]:
        st.session_state.pop(key, None)


def render_quiz(questions: list[dict[str, Any]]):
    st.subheader("MDCAT Quiz")

    if not questions:
        st.warning("No valid questions were generated.")
        return

    with st.form("mdcat_quiz"):
        answers = {}

        for i, q in enumerate(questions, start=1):
            st.markdown(f"### Q{i}. {q['question']}")

            labels = {
                letter: f"{letter}. {q['options'][letter]}"
                for letter in ["A", "B", "C", "D"]
            }

            answers[i] = st.radio(
                "Select one answer:",
                options=["A", "B", "C", "D"],
                format_func=lambda x, labels=labels: labels[x],
                key=f"answer_{i}",
                index=None,
            )

            st.divider()

        submitted = st.form_submit_button(
            "Submit Quiz",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        score = 0
        st.session_state["answers"] = answers
        st.session_state["quiz_submitted"] = True

        for i, q in enumerate(questions, start=1):
            if answers.get(i) == q["correct_answer"]:
                score += 1

        st.session_state["score"] = score

    if st.session_state.get("quiz_submitted"):
        score = st.session_state.get("score", 0)
        st.success(f"Score: {score}/{len(questions)}")

        for i, q in enumerate(questions, start=1):
            selected = st.session_state["answers"].get(i)
            correct = q["correct_answer"]

            if selected == correct:
                st.success(f"Q{i}: Correct — {correct}")
            elif selected is None:
                st.warning(f"Q{i}: Not answered — correct answer: {correct}")
            else:
                st.error(
                    f"Q{i}: Your answer: {selected} | Correct answer: {correct}"
                )

            st.info(q["explanation"])


# ============================================================
# UI
# ============================================================

st.title("📚 Prep AI")
st.caption("RAG-powered MDCAT preparation from your own textbook or notes")

with st.sidebar:
    st.header("Settings")

    embedding_model_name = get_secret(
        "EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL,
    )

    groq_model = get_secret(
        "GROQ_MODEL",
        DEFAULT_GROQ_MODEL,
    )

    top_k = st.slider("Retrieved context chunks", 3, 12, 8)
    chunk_size = st.slider("Chunk size (words)", 400, 1400, 900, step=100)
    overlap = st.slider("Chunk overlap (words)", 50, 250, 120, step=10)

    if st.button("Reset application", use_container_width=True):
        reset_app()
        st.rerun()

st.header("1. Upload your study PDF")

uploaded_file = st.file_uploader(
    "Upload a text-based PDF textbook, notes, or chapter",
    type=["pdf"],
)

if uploaded_file is not None:
    already_loaded = (
        st.session_state.get("document_name") == uploaded_file.name
        and "index" in st.session_state
    )

    if not already_loaded:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Extracting text from PDF..."):
                pages = extract_pdf(uploaded_file)

            if not pages:
                st.error(
                    "No selectable text was found in this PDF. "
                    "If it is a scanned/image-only PDF, OCR is required."
                )
                st.stop()

            total_chars = sum(len(p["text"]) for p in pages)

            if total_chars < 500:
                st.error(
                    "Very little text was extracted. Please upload a "
                    "text-based PDF with readable content."
                )
                st.stop()

            with st.spinner("Creating chunks..."):
                chunks = chunk_pages(
                    pages,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

            if not chunks:
                st.error("The PDF could not be divided into usable chunks.")
                st.stop()

            with st.spinner("Loading open-source embedding model..."):
                model = load_embedding_model(embedding_model_name)

            with st.spinner("Building FAISS vector index..."):
                index = build_index(chunks, model)

            st.session_state["pages"] = pages
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["embedding_model"] = model
            st.session_state["document_name"] = uploaded_file.name
            st.session_state.pop("questions", None)
            st.session_state.pop("quiz_submitted", None)

            st.success(
                f"PDF processed successfully: {len(pages)} pages, "
                f"{len(chunks)} chunks."
            )

if "index" not in st.session_state:
    st.info("Upload and process a PDF to continue.")
    st.stop()

st.success(f"Loaded document: **{st.session_state['document_name']}**")

st.header("2. Generate MDCAT MCQs")

topic = st.text_input(
    "Chapter / topic",
    placeholder="Example: Cell membrane, Thermodynamics, Genetics",
)

col1, col2 = st.columns(2)

with col1:
    question_count = st.number_input(
        "Number of questions",
        min_value=5,
        max_value=MAX_QUESTIONS,
        value=20,
        step=5,
    )

with col2:
    difficulty = st.selectbox(
        "Difficulty",
        ["MDCAT Standard", "Moderate", "Challenging"],
    )

if st.button("Generate MCQs", type="primary", use_container_width=True):
    if not topic.strip():
        st.warning("Please enter a chapter or topic.")
        st.stop()

    api_key = get_secret("GROQ_API_KEY")

    if not api_key:
        st.error(
            "GROQ_API_KEY is not configured. Add it to Streamlit Cloud "
            "Secrets before generating questions."
        )
        st.stop()

    model = st.session_state["embedding_model"]
    chunks = st.session_state["chunks"]
    index = st.session_state["index"]

    with st.spinner("Retrieving relevant textbook content..."):
        results = retrieve(
            topic.strip(),
            index,
            chunks,
            model,
            top_k=top_k,
        )

    if not results:
        st.error("No relevant textbook content was retrieved.")
        st.stop()

    context = make_context(results)
    allowed_ids = {r["chunk_id"] for r in results}

    st.caption(
        f"Retrieved {len(results)} relevant chunks from pages "
        + ", ".join(str(r["page_number"]) for r in results)
    )

    target = int(question_count)
    all_questions = []
    previous_stems = []

    # Generate in small batches to reduce malformed output and improve
    # reliability for larger question counts.
    batch_size = 10
    progress = st.progress(0)

    with st.spinner("Generating MDCAT questions with Groq..."):
        while len(all_questions) < target:
            remaining = target - len(all_questions)
            current_batch = min(batch_size, remaining)

            prompt = build_prompt(
                context=context,
                topic=topic.strip(),
                count=current_batch,
                difficulty=difficulty,
                previous_questions=previous_stems,
            )

            try:
                generated = call_groq(
                    api_key=api_key,
                    model_name=groq_model,
                    prompt=prompt,
                )
            except Exception as exc:
                st.error(f"Groq generation failed: {exc}")
                break

            valid = validate_questions(generated, allowed_ids)

            for q in valid:
                if len(all_questions) >= target:
                    break
                all_questions.append(q)
                previous_stems.append(q["question"])

            progress.progress(min(len(all_questions) / target, 1.0))

            # Avoid an endless loop if the model repeatedly returns
            # unusable/duplicate questions.
            if not valid:
                break

    progress.empty()

    if not all_questions:
        st.error(
            "The model did not return valid source-grounded MCQs. "
            "Try a more specific topic or a smaller question count."
        )
    else:
        st.session_state["questions"] = all_questions

        if len(all_questions) < target:
            st.warning(
                f"Generated {len(all_questions)} valid questions instead of "
                f"{target}. This usually happens when the source material "
                f"does not contain enough distinct information."
            )
        else:
            st.success(f"Generated {len(all_questions)} valid MCQs.")

if "questions" in st.session_state:
    st.header("3. Attempt the quiz")
    render_quiz(st.session_state["questions"])

st.markdown("---")
st.caption(
    "Prep AI uses PDF text extraction, Sentence Transformers embeddings, "
    "FAISS retrieval, and Groq for source-grounded MCQ generation."
)
