import base64
import io
import hashlib
import json
import re
from pathlib import Path

import edge_tts
from mutagen.mp3 import MP3

from app.config import settings
from app.providers.registry import client
from app.services.lipsync import build

VOICES = [
    {"id": i, "name": n, "gender": g, "locale": loc, "provider": "edge"}
    for i, n, g, loc in [
        ("en-US-AriaNeural", "Aria", "Female", "en-US"),
        ("en-US-JennyNeural", "Jenny", "Female", "en-US"),
        ("en-US-GuyNeural", "Guy", "Male", "en-US"),
        ("en-US-ChristopherNeural", "Christopher", "Male", "en-US"),
        ("en-GB-SoniaNeural", "Sonia", "Female", "en-GB"),
        ("en-GB-RyanNeural", "Ryan", "Male", "en-GB"),
        ("en-AU-NatashaNeural", "Natasha", "Female", "en-AU"),
        ("en-AU-WilliamNeural", "William", "Male", "en-AU"),
        ("en-IN-NeerjaNeural", "Neerja", "Female", "en-IN"),
        ("en-IN-PrabhatNeural", "Prabhat", "Male", "en-IN"),
        ("bn-BD-NabanitaNeural", "Nabanita", "Female", "bn-BD"),
        ("hi-IN-SwaraNeural", "Swara", "Female", "hi-IN"),
        ("fr-FR-DeniseNeural", "Denise", "Female", "fr-FR"),
        ("es-ES-ElviraNeural", "Elvira", "Female", "es-ES"),
    ]
]


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
    words = []
    if settings.tts_provider == "openai":
        voice = voice or "nova"
        response = client(
            settings.openai_base_url, settings.openai_api_key
        ).audio.speech.create(model="tts-1", voice=voice, input=text)
        audio = response.content
    else:
        voice = voice or "en-US-AriaNeural"
        audio_parts = []
        async for item in edge_tts.Communicate(
            text, voice, boundary="WordBoundary"
        ).stream():
            if item["type"] == "audio":
                audio_parts.append(item["data"])
            elif item["type"] == "WordBoundary":
                words.append(
                    {
                        "word": item["text"],
                        "start": item["offset"] / 10_000_000,
                        "end": (item["offset"] + item["duration"]) / 10_000_000,
                    }
                )
        audio = b"".join(audio_parts)
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
