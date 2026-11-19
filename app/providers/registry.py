import logging
import re
import threading
import time
from pathlib import Path

from openai import OpenAI

from app.config import settings

log = logging.getLogger(__name__)
_pollinations_lock = threading.Lock()
_pollinations_last = 0.0


def client(base_url: str, key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=key or "not-needed",
        timeout=settings.request_timeout,
        max_retries=0,
    )


def llm_candidates():
    candidates = [
        ("groq", settings.groq_api_key, settings.groq_base_url, settings.groq_model),
        (
            "gemini",
            settings.gemini_api_key,
            settings.gemini_base_url,
            settings.gemini_model,
        ),
        (
            "openai",
            settings.openai_api_key,
            settings.openai_base_url,
            settings.openai_model,
        ),
        (
            "pollinations",
            "keyless",
            settings.pollinations_base_url,
            settings.pollinations_model,
        ),
    ]
    available = [c for c in candidates if c[1]]
    if settings.llm_provider != "auto":
        available.sort(key=lambda c: c[0] != settings.llm_provider)
    return available


def chat(messages: list[dict]) -> tuple[str, str, str] | None:
    global _pollinations_last
    for name, key, url, model in llm_candidates():
        if name == "pollinations":
            with _pollinations_lock:
                now = time.monotonic()
                if now - _pollinations_last < 30:
                    continue
                _pollinations_last = now
        try:
            response = client(url, key).chat.completions.create(
                model=model, messages=messages, max_tokens=400
            )
            content = response.choices[0].message.content
            if content:
                content = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", content, flags=re.I | re.S).strip()
                if content:
                    return content, name, model
        except Exception as exc:
            log.warning("LLM provider %s unavailable: %s", name, type(exc).__name__)
    return None


class Embeddings:
    def __init__(self):
        self.provider = settings.embedding_provider
        self.model = (
            settings.embedding_model
            if self.provider == "fastembed"
            else (
                "text-embedding-3-small"
                if self.provider == "openai"
                else "gemini-embedding-001"
            )
        )
        self._local = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "fastembed":
            if self._local is None:
                from fastembed import TextEmbedding

                cache = Path(settings.data_dir) / "models"
                cache.mkdir(parents=True, exist_ok=True)
                self._local = TextEmbedding(model_name=self.model, cache_dir=str(cache))
            return [v.tolist() for v in self._local.embed(texts, batch_size=32)]
        if self.provider not in ("openai", "gemini"):
            raise ValueError("Unknown embedding provider")
        key = (
            settings.openai_api_key
            if self.provider == "openai"
            else settings.gemini_api_key
        )
        url = (
            settings.openai_base_url
            if self.provider == "openai"
            else settings.gemini_base_url
        )
        if not key:
            raise ValueError("Embedding provider key is missing")
        result = client(url, key).embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in result.data]


def transcribe(data: bytes, filename: str) -> dict:
    choices = [
        (
            "groq",
            settings.groq_api_key,
            settings.groq_base_url,
            "whisper-large-v3-turbo",
        ),
        ("openai", settings.openai_api_key, settings.openai_base_url, "whisper-1"),
    ]
    for name, key, url, model in choices:
        if key and settings.stt_provider in ("auto", name):
            result = client(url, key).audio.transcriptions.create(
                model=model, file=(filename, data), response_format="verbose_json"
            )
            return {
                "text": result.text,
                "language": getattr(result, "language", None),
                "provider": name,
            }
    raise LookupError("stt_unavailable")
