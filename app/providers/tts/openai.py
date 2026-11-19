from app.config import settings
from app.providers.llm.base import client


def synthesize(text: str, voice: str) -> bytes:
    response = client(settings.openai_base_url, settings.openai_api_key).audio.speech.create(model="tts-1", voice=voice, input=text)
    return response.content
