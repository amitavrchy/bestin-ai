import io
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.config import settings
from app.providers.base import ChatProvider, EmbeddingProvider, STTProvider, TTSProvider


def _client() -> OpenAI:
    if not settings.openai_is_configured():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy backend/.env.example to backend/.env and add your key."
        )
    return OpenAI(api_key=settings.openai_api_key.strip())


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.openai_embedding_model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = _client()
        vectors: List[List[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(model=self.model, input=batch)
            # The API returns data sorted by index, but sort defensively.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([item.embedding for item in ordered])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class OpenAIChatProvider(ChatProvider):
    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.openai_chat_model

    def complete(self, system: str, user: str) -> str:
        client = _client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class OpenAISTTProvider(STTProvider):
    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.openai_stt_model

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        mime: Optional[str] = None,
    ) -> str:
        client = _client()
        buffer = io.BytesIO(audio_bytes)
        # The OpenAI SDK reads a name attribute to infer format.
        buffer.name = filename or "audio.webm"
        response = client.audio.transcriptions.create(
            model=self.model,
            file=buffer,
        )
        return (response.text or "").strip()

    def transcribe_words(self, audio_path: str) -> Optional[List[Dict[str, Any]]]:
        client = _client()
        with open(audio_path, "rb") as handle:
            response = client.audio.transcriptions.create(
                model=self.model,
                file=handle,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        words = getattr(response, "words", None) or []
        timed = []
        for item in words:
            if isinstance(item, dict):
                word = item.get("word") or item.get("text") or ""
                start = float(item.get("start", 0.0))
                end = float(item.get("end", start))
            else:
                word = getattr(item, "word", None) or getattr(item, "text", "") or ""
                start = float(getattr(item, "start", 0.0))
                end = float(getattr(item, "end", start))
            if word:
                timed.append({"word": word, "start": start, "end": max(end, start)})
        return timed or None


class OpenAITTSProvider(TTSProvider):
    def __init__(self, model: Optional[str] = None, voice: Optional[str] = None):
        self.model = model or settings.openai_tts_model
        self.voice = voice or settings.openai_tts_voice

    def synthesize(self, text: str, out_path: str) -> Dict[str, Any]:
        client = _client()
        # Standard OpenAI TTS does not return visemes or word timings.
        # Those are added later by the lip-sync pipeline.
        with client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="mp3",
        ) as response:
            response.stream_to_file(out_path)
        return {
            "path": out_path,
            "mime": "audio/mpeg",
            "visemes": None,
            "word_timings": None,
        }
