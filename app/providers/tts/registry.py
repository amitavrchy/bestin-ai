import base64
import hashlib
import io
import json
import re
from pathlib import Path

from mutagen.mp3 import MP3

from app.config import settings
from app.providers.tts import edge, openai
from app.services.lipsync import build

def normalize_text(text: str) -> str:
    """Remove citations and common markup before sending text to speech."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"!??\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#~]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


async def synthesize(text: str, voice: str | None = None) -> dict:
    """Synthesize normalized speech and cache audio and timing data."""
    text = normalize_text(text)
    if not text:
        raise ValueError("No speakable text")
    key = hashlib.sha256(json.dumps([text, voice, settings.tts_provider, settings.lipsync_backend]).encode()).hexdigest()
    cache = Path(settings.data_dir) / "cache" / (key + ".json")
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except (ValueError, OSError):
            pass
    if settings.tts_provider == "openai":
        voice = voice or "nova"
        audio = openai.synthesize(text, voice)
        words = []
    else:
        voice = voice or "en-US-AriaNeural"
        audio, words = await edge.synthesize(text, voice)
    duration = MP3(io.BytesIO(audio)).info.length
    cues, backend = build(audio, text, duration, words)
    result = {
        "audio_base64": base64.b64encode(audio).decode(),
        "mime": "audio/mpeg",
        "duration": duration,
        "visemes": cues,
        "words": words,
        "voice": voice,
        "lipsync_backend": backend,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + "." + __import__("uuid").uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(result))
        tmp.replace(cache)
    finally:
        tmp.unlink(missing_ok=True)
    return result
