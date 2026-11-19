"""Keyless transcription with a lazily loaded, shared Whisper model."""
from io import BytesIO
from threading import Lock

from app.config import settings

_model = None
_lock = Lock()


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(settings.local_stt_model, device="cpu", compute_type="int8")
    return _model


def transcribe(data: bytes, filename: str) -> dict:
    segments, info = get_model().transcribe(BytesIO(data), vad_filter=True)
    return {"text": " ".join(segment.text.strip() for segment in segments).strip(),
            "language": info.language, "provider": "local"}
