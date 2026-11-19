from app.config import settings
from app.providers.stt import groq, openai


def transcribe(data: bytes, filename: str) -> dict:
    for name, key, provider in [("groq", settings.groq_api_key, groq), ("openai", settings.openai_api_key, openai)]:
        if key and settings.stt_provider in ("auto", name):
            return provider.transcribe(data, filename)
    raise LookupError("stt_unavailable")
