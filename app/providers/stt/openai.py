from app.config import settings
from app.providers.stt.base import transcribe_with


def transcribe(data: bytes, filename: str) -> dict:
    return transcribe_with(settings.openai_base_url, settings.openai_api_key, "whisper-1", "openai", data, filename)
