from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, settings
from app.main import app
from app.providers import registry
from app.services import library
from app.services.ingestion import chunk_sections, parse_file
from app.services.lipsync import timeline


def test_settings():
    assert Settings(_env_file=None).ai_internal_token == "dev-internal-token"


def test_auth_and_health():
    with TestClient(app) as c:
        assert c.get("/health").json()["service"] == "bestin-ai"
        assert c.get("/v1/voices").status_code == 401
        assert (
            c.get(
                "/v1/voices", headers={"X-Internal-Token": settings.ai_internal_token}
            ).status_code
            == 200
        )


def test_parser_and_chunking(tmp_path):
    path = tmp_path / "book.md"
    path.write_text("Chapter 1\n" + "A sentence. " * 200)
    sections, pages = parse_file(path)
    assert pages is None and sections[-1]["chapter"] == "Chapter 1"
    chunks = chunk_sections(sections)
    assert len(chunks) > 1 and all(len(c["text"]) <= 900 for c in chunks)
    assert (
        chunks[0]["text"][-100:] in chunks[1]["text"]
        or chunks[1]["text"][:50] in chunks[0]["text"]
    )


def test_pdf_parser(tmp_path):
    from pypdf import PdfWriter

    path = tmp_path / "book.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(path)
    sections, pages = parse_file(path)
    assert pages == 1 and sections[0]["page"] == 1


def test_epub_parser(tmp_path):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("test")
    book.set_title("Test")
    book.set_language("en")
    chapter = epub.EpubHtml(
        title="Start",
        file_name="start.xhtml",
        content="<h1>Start</h1><p>Hello book</p>",
    )
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.spine = [chapter]
    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)
    sections, _ = parse_file(path)
    assert any("Hello book" in s["text"] and s["chapter"] == "Start" for s in sections)


@pytest.mark.parametrize("words", [[], [{"word": "hello", "start": 0.2, "end": 0.8}]])
def test_visemes(words):
    cues = timeline("hello there", 1.2, words)
    assert cues
    assert all(
        0 <= c["start"] < c["end"] <= 1.2 and c["value"] in "ABCDEFGHX" for c in cues
    )
    assert all(a["end"] <= b["start"] + 1e-9 for a, b in zip(cues, cues[1:]))


def test_provider_fallback(monkeypatch):
    monkeypatch.setattr(registry.settings, "llm_provider", "auto")
    monkeypatch.setattr(registry.settings, "groq_api_key", "")
    monkeypatch.setattr(registry.settings, "gemini_api_key", "")
    monkeypatch.setattr(registry.settings, "openai_api_key", "")
    fake = Mock()
    fake.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="grounded [1]"))
    ]
    monkeypatch.setattr(registry, "client", lambda *args: fake)
    assert registry.chat([{"role": "user", "content": "Hi"}]) == (
        "grounded [1]",
        "pollinations",
        settings.pollinations_model,
    )


def test_routes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(
        library.embeddings, "embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )
    monkeypatch.setattr("app.main.chat", lambda messages: None)
    headers = {"X-Internal-Token": settings.ai_internal_token}
    with TestClient(app) as c:
        assert (
            c.post(
                "/v1/ask",
                headers=headers,
                json={"book_id": 1, "question": "Who?", "speak": False},
            ).status_code
            == 404
        )
        with Path("tests/fixtures/sample.txt").open("rb") as f:
            assert (
                c.post(
                    "/v1/books/1/ingest",
                    headers=headers,
                    files={"file": ("sample.txt", f, "text/plain")},
                ).status_code
                == 202
            )
        assert c.get("/v1/books/1/status", headers=headers).json()["status"] == "ready"
        response = c.post(
            "/v1/ask",
            headers=headers,
            json={"book_id": 1, "question": "Who?", "speak": False},
        )
        assert (
            response.status_code == 200 and response.json()["provider"] == "extractive"
        )
        assert response.json()["citations"][0]["chapter"] == "Chapter 1"
        assert (
            c.post(
                "/v1/stt", headers=headers, files={"file": ("a.wav", b"a")}
            ).status_code
            == 503
        )
        assert c.delete("/v1/books/1", headers=headers).status_code == 204
        assert (
            c.get("/v1/books/1/status", headers=headers).json()["status"] == "not_found"
        )
