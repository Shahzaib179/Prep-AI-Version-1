# Prep AI — MDCAT RAG Application

A simple RAG-based MDCAT preparation application built with Streamlit.

## Important fix in this version

This version uses a **single self-contained `app.py`** and intentionally does not import local modules such as:

```python
from ingestion.pdf_loader import ...
```

Therefore the previous Streamlit Cloud error:

```text
ModuleNotFoundError: No module named 'ingestion'
```

cannot occur because `app.py` no longer depends on the `ingestion`, `embeddings`, `retrieval`, `generation`, or other local package folders.

This is the recommended deployment version for the hackathon/demo.

## Features

1. Upload a text-based PDF.
2. Extract PDF text with PyPDF.
3. Split the document into overlapping chunks.
4. Generate embeddings with the open-source Sentence Transformers model:
   `sentence-transformers/all-MiniLM-L6-v2`
5. Store embeddings in FAISS.
6. Enter a chapter/topic.
7. Retrieve relevant textbook chunks.
8. Generate MDCAT-level MCQs using Groq.
9. Validate questions and remove duplicates.
10. Attempt the generated quiz and receive a score.

## Project structure

```text
Prep-AI/
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

## GitHub / Streamlit Cloud deployment

Upload these files directly to the root of the GitHub repository.

Your repository should look like:

```text
app.py
requirements.txt
runtime.txt
README.md
.gitignore
```

On Streamlit Cloud, set the main file to:

```text
app.py
```

Then add the Groq API key under the app's Secrets.

Example:

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

Do not put your API key inside `app.py` or commit it to GitHub.

## Local setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Set your API key.

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_key_here"
```

Run:

```bash
streamlit run app.py
```

## Notes

### Scanned PDFs

Version 1 requires selectable text. Image-only/scanned PDFs will not work without OCR.

### FAISS storage

The FAISS index is built in memory for the current Streamlit session. It is intentionally simple for a hackathon application.

### Question count

The app supports up to 100 questions and generates them in batches of 10. The model may return fewer valid questions if the uploaded source does not contain enough distinct information.

## Security

Never commit:

```text
.env
.streamlit/secrets.toml
```

and never paste your Groq API key into source code.
