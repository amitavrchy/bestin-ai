import importlib.util
import logging

from app.config import settings
from app.providers.stt import groq, openai, local


def candidates():
    providers = []
    for name, key, provider in [("groq", settings.groq_api_key, groq), ("openai", settings.openai_api_key, openai)]:
        if key and settings.stt_provider in ("auto", name):
            providers.append((name, provider))
    if settings.local_stt_enabled and settings.stt_provider in ("auto", "local", "groq", "openai") and importlib.util.find_spec("faster_whisper"):
        providers.append(("local", local))
    return providers


def transcribe(data: bytes, filename: str) -> dict:
    choices = candidates()
    if not choices:
        raise LookupError("stt_unavailable")
    for index, (name, provider) in enumerate(choices):
        try:
            return provider.transcribe(data, filename)
        except Exception:
            if index == len(choices) - 1:
                raise
            logging.exception("STT provider %s failed; trying fallback", name)
