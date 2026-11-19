import logging
import re
import threading
import time

from app.config import settings
from app.providers.llm import groq, gemini, openai, pollinations
from app.providers.llm.base import client

log = logging.getLogger(__name__)
_pollinations_lock = threading.Lock()
_pollinations_last = 0.0


def llm_candidates():
    candidates = [groq.configuration(), gemini.configuration(), openai.configuration(), pollinations.configuration()]
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

