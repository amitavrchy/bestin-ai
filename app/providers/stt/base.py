from app.providers.llm.base import client


def transcribe_with(base_url: str, key: str, model: str, name: str, data: bytes, filename: str) -> dict:
    result = client(base_url, key).audio.transcriptions.create(model=model, file=(filename, data), response_format="verbose_json")
    return {"text": result.text, "language": getattr(result, "language", None), "provider": name}
