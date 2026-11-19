import os

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.providers.embeddings.fastembed import FastEmbed
from app.providers.llm import registry
from app.providers.stt import registry as stt
from app.providers.tts import edge
from app.services.rag import extractive_answer


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "test-groq")
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(registry, "_pollinations_last", 0.0)


def completion(content, **extra):
    return {"choices": [{"message": {"role": "assistant", "content": content, **extra}, "index": 0, "finish_reason": "stop"}]}


def test_http_failure_chain_returns_extractive(configured):
    with respx.mock(assert_all_called=True) as router:
        groq = router.post(f"{settings.groq_base_url}/chat/completions").mock(return_value=httpx.Response(429, headers={"Retry-After": "3600"}, json={"error": "rate limit"}))
        gemini = router.post(f"{settings.gemini_base_url.rstrip('/')}/chat/completions").mock(return_value=httpx.Response(500, json={"error": "unavailable"}))
        pollinations = router.post(f"{settings.pollinations_base_url}/chat/completions").mock(return_value=httpx.Response(402, headers={"Retry-After": "3600"}, json={"error": "quota"}))
        result = registry.chat([{"role": "user", "content": "Who?"}])
        assert result is None
        assert groq.call_count == gemini.call_count == pollinations.call_count == 1
        answer = extractive_answer("Who?", [{"text": "Alice followed the rabbit.", "chunk_id": "1"}])
        assert "Alice" in answer


def test_http_groq_success_and_reasoning(configured):
    with respx.mock(assert_all_called=True) as router:
        groq = router.post(f"{settings.groq_base_url}/chat/completions").mock(return_value=httpx.Response(200, json=completion("<think>private</think>Grounded answer [1]", reasoning="hidden")))
        result = registry.chat([{"role": "user", "content": "Who?"}])
        assert result == ("Grounded answer [1]", "groq", settings.groq_model)
        assert groq.call_count == 1


@pytest.mark.parametrize("content,expected", [
    ("<think>secret</think> Answer", "Answer"),
    ("<THINK>secret\nthought</THINK> Answer", "Answer"),
    ("Before <think>secret</think> after", "Before  after"),
    ("<think type='x'>secret</think> Answer", "Answer"),
    ("Ordinary answer", "Ordinary answer"),
])
def test_reasoning_removed(configured, content, expected):
    with respx.mock:
        respx.post(f"{settings.groq_base_url}/chat/completions").mock(return_value=httpx.Response(200, json=completion(content, reasoning="private")))
        assert registry.chat([{"role": "user", "content": "Hi"}])[0] == expected


def test_stt_groq_http_routing(configured, monkeypatch):
    monkeypatch.setattr(settings, "stt_provider", "auto")
    with respx.mock:
        route = respx.post(f"{settings.groq_base_url}/audio/transcriptions").mock(return_value=httpx.Response(200, json={"text": "Hello", "language": "en", "duration": 1.0}))
        with TestClient(app) as client:
            response = client.post("/v1/stt", headers={"X-Internal-Token": settings.ai_internal_token}, files={"file": ("sample.wav", b"audio", "audio/wav")})
        assert response.json() == {"text": "Hello", "language": "en", "provider": "groq"}
        assert route.call_count == 1
        assert b"whisper-large-v3-turbo" in route.calls[0].request.content


@pytest.mark.asyncio
async def test_edge_word_boundaries(monkeypatch):
    class Communicate:
        def __init__(self, text, voice, boundary):
            assert (text, voice, boundary) == ("Hello", "voice", "WordBoundary")

        async def stream(self):
            yield {"type": "audio", "data": b"mp3"}
            yield {"type": "WordBoundary", "text": "Hello", "offset": 2_000_000, "duration": 3_000_000}

    monkeypatch.setattr(edge.edge_tts, "Communicate", Communicate)
    audio, words = await edge.synthesize("Hello", "voice")
    assert audio == b"mp3"
    assert words == [{"word": "Hello", "start": 0.2, "end": 0.5}]


@pytest.mark.parametrize("provider", ["none", "openai"])
def test_stt_missing_key(configured, monkeypatch, provider):
    monkeypatch.setattr(settings, "stt_provider", provider)
    with pytest.raises(LookupError, match="stt_unavailable"):
        stt.transcribe(b"audio", "a.wav")


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to load the real model")
def test_real_fastembed():
    vectors = FastEmbed(settings.embedding_model).embed(["Alice followed the rabbit.", "A rabbit ran away."])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) == 384
    assert any(vectors[0])
