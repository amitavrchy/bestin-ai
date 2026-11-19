from app.config import settings
from app.providers.stt.base import transcribe_with


def transcribe(data: bytes, filename: str) -> dict:
    return transcribe_with(settings.groq_base_url, settings.groq_api_key, "whisper-large-v3-turbo", "groq", data, filename)
