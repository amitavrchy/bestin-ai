from openai import OpenAI

from app.config import settings


def client(base_url: str, key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=key or "not-needed", timeout=settings.request_timeout, max_retries=0)
