import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str
    groq_model: str
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    max_questions: int
    generation_batch_size: int
    min_document_chars: int
    min_context_chars: int
    show_sources: bool

    @classmethod
    def from_environment(cls):
        # Streamlit Cloud stores secrets in st.secrets; local development
        # normally uses environment variables / .env.
        secrets = {}
        try:
            import streamlit as st
            secrets = dict(st.secrets)
        except Exception:
            secrets = {}

        def setting(name: str, default: str) -> str:
            return os.getenv(name, secrets.get(name, default))

        return cls(
            groq_api_key=setting("GROQ_API_KEY", "").strip(),
            groq_model=setting(
                "GROQ_MODEL", "openai/gpt-oss-120b"
            ).strip(),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ).strip(),
            chunk_size=int(setting("CHUNK_SIZE", "900")),
            chunk_overlap=int(setting("CHUNK_OVERLAP", "120")),
            retrieval_top_k=int(setting("RETRIEVAL_TOP_K", "8")),
            max_questions=int(setting("MAX_QUESTIONS", "100")),
            generation_batch_size=int(setting("GENERATION_BATCH_SIZE", "10")),
            min_document_chars=int(setting("MIN_DOCUMENT_CHARS", "500")),
            min_context_chars=int(setting("MIN_CONTEXT_CHARS", "500")),
            show_sources=setting("SHOW_SOURCES", "true").lower() == "true",
        )

    def validate_api_key(self):
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Add it to .env locally "
                "or Streamlit Cloud Secrets in deployment."
            )
