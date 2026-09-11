# Prep AI

Prep AI is a modular RAG-based MDCAT entry-test preparation application.

A student uploads a text-based PDF, enters a chapter/topic, and receives
MDCAT-level multiple-choice questions generated from the uploaded material.

## Architecture

PDF
-> text extraction
-> cleaning/chunking
-> open-source embeddings
-> FAISS vector search
-> topic retrieval
-> Groq-hosted open-weight LLM
-> MCQ validation
-> Streamlit quiz

## Technology

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Groq Python SDK
- Open-weight LLM through Groq

The default embedding model is:
`sentence-transformers/all-MiniLM-L6-v2`

The default Groq model is:
`openai/gpt-oss-120b`

The project targets Python 3.11 (`runtime.txt`) for deployment stability.

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/prep-ai.git
cd prep-ai
```

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure Groq

Copy `.env.example` to `.env`.

Windows:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Open `.env` and add:

```text
GROQ_API_KEY=your_real_key
```

Never commit `.env`.

## 5. Run locally

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## 6. Use the app

1. Upload a text-based PDF.
2. Click `Process Document`.
3. Enter a chapter or topic.
4. Select the number of questions.
5. Select difficulty.
6. Click `Generate MCQs`.
7. Attempt the test.
8. Submit to see the score and answer review.

## Streamlit Cloud deployment

Push the repository to GitHub and create a Streamlit Cloud app using `app.py`
as the main file.

In Streamlit Cloud, open the app's Secrets settings and add:

```toml
GROQ_API_KEY = "your_real_key"
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = "900"
CHUNK_OVERLAP = "120"
RETRIEVAL_TOP_K = "8"
MAX_QUESTIONS = "100"
GENERATION_BATCH_SIZE = "10"
MIN_DOCUMENT_CHARS = "500"
MIN_CONTEXT_CHARS = "500"
SHOW_SOURCES = "true"
```

The application reads local environment variables / `.env` and Streamlit Cloud
`st.secrets` values.

## Important deployment note

FAISS is intentionally used in-memory for this version. An uploaded document
is processed during the user's Streamlit session. It is not a permanent
multi-user document database.

This keeps Version 1 simple and suitable for a hackathon/prototype.

## Limitations

- Version 1 expects text-based PDFs.
- Scanned/image-only PDFs need OCR, which can be added later.
- The quality of questions depends on the uploaded material and model.
- "Maximum questions" is implemented as batch generation up to the configured
  maximum, with duplicate filtering.
- The Groq model name can change. Keep it configurable in environment/secrets.

## GitHub files

Commit these project files:

```text
app.py
config.py
requirements.txt
README.md
.gitignore
.env.example

ingestion/__init__.py
ingestion/pdf_loader.py
ingestion/chunker.py

embeddings/__init__.py
embeddings/embedding_model.py

vectorstore/__init__.py
vectorstore/faiss_store.py

retrieval/__init__.py
retrieval/retriever.py

generation/__init__.py
generation/prompts.py
generation/groq_client.py
generation/mcq_generator.py

validation/__init__.py
validation/mcq_validator.py

quiz/__init__.py
quiz/quiz_manager.py
```

Do NOT commit:

```text
.env
.streamlit/secrets.toml
__pycache__/
.venv/
data/*
*.faiss
*.pkl
```
