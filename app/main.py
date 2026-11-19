import base64
import os
import uuid
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.providers.openai_provider import (
    OpenAIChatProvider,
    OpenAIEmbeddingProvider,
    OpenAISTTProvider,
    OpenAITTSProvider,
)
from app.schemas import AskResponse, BookListResponse, BookSummary, HealthResponse
from app.services.ingestion import ALLOWED_EXTENSIONS, detect_kind, ingest_file
from app.services.library import list_books
from app.services.lipsync import available_backends, build_timeline
from app.services.rag import VectorStore, answer_question

app = FastAPI(
    title="Bestin",
    description="Voice RAG over an uploaded book, with a lip-synced 2D avatar.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.audio_dir, exist_ok=True)
os.makedirs(settings.chroma_dir, exist_ok=True)

app.mount("/audio", StaticFiles(directory=settings.audio_dir), name="audio")


def _embeddings() -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider()


_store_instance = None  # type: ignore


def _store() -> VectorStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore(_embeddings())
    return _store_instance


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    info = available_backends()
    info["mode"] = settings.lipsync_backend
    return HealthResponse(
        status="ok",
        service="bestin",
        openai_configured=settings.openai_is_configured(),
        lipsync=info,
    )


@app.get("/books", response_model=BookListResponse)
def get_books() -> BookListResponse:
    return BookListResponse(books=list_books())


@app.post("/books", response_model=BookSummary)
async def upload_book(file: UploadFile = File(...)) -> BookSummary:
    filename = file.filename or "upload.bin"
    ext = "." + detect_kind(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a PDF, EPUB, or TXT file.",
        )
    book_id_name = uuid.uuid4().hex[:12]
    dest_name = "{}{}".format(book_id_name, ext)
    dest_path = os.path.join(settings.upload_dir, dest_name)
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file was empty.")
    # ~50 MB guard so a huge scan cannot blow up local ingestion unexpectedly.
    if len(payload) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File is larger than the 50 MB upload limit.")
    with open(dest_path, "wb") as handle:
        handle.write(payload)
    try:
        record, chunks = ingest_file(dest_path, filename)
        store = _store()
        store.add_chunks(record["id"], record["title"], chunks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Ingestion failed: {}".format(exc))
    return BookSummary(
        id=record["id"],
        title=record["title"],
        filename=record["filename"],
        kind=record["kind"],
        pages=record["pages"],
        chunks=record["chunks"],
        created_at=record["created_at"],
    )


async def _parse_ask(request: Request) -> Any:
    content_type = request.headers.get("content-type", "")
    question = None
    book_id = None
    audio_bytes = None
    audio_name = "question.webm"
    audio_mime = None
    if "application/json" in content_type:
        body = await request.json()
        question = (body.get("question") or "").strip() or None
        book_id = (body.get("book_id") or "").strip() or None
    else:
        form = await request.form()
        raw_q = form.get("question")
        question = (str(raw_q).strip() if raw_q is not None else "") or None
        raw_id = form.get("book_id")
        book_id = (str(raw_id).strip() if raw_id is not None else "") or None
        audio = form.get("audio")
        if audio is not None and hasattr(audio, "read"):
            audio_bytes = await audio.read()
            audio_name = getattr(audio, "filename", None) or "question.webm"
            audio_mime = getattr(audio, "content_type", None)
    return question, book_id, audio_bytes, audio_name, audio_mime


@app.post("/ask", response_model=AskResponse)
async def ask(request: Request) -> AskResponse:
    question, book_id, audio_bytes, audio_name, audio_mime = await _parse_ask(request)

    if audio_bytes and not question:
        try:
            question = OpenAISTTProvider().transcribe(audio_bytes, audio_name, audio_mime)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Speech recognition failed: {}".format(exc))

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Provide a text question or an audio recording.",
        )

    try:
        answer, sources = answer_question(
            question,
            _store(),
            OpenAIChatProvider(),
            book_id=book_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="RAG failed: {}".format(exc))

    audio_id = uuid.uuid4().hex
    audio_path = os.path.join(settings.audio_dir, "{}.mp3".format(audio_id))
    tts_info = {}
    audio_b64 = None
    audio_mime_out = None
    audio_url = None
    visemes = []
    backend_name = None

    spoken = answer.strip()
    if spoken:
        try:
            tts = OpenAITTSProvider()
            tts_info = tts.synthesize(spoken, audio_path)
            visemes, backend_name = build_timeline(
                audio_path,
                spoken,
                mode=(settings.lipsync_backend or "auto").strip().lower(),
                tts_info=tts_info,
                stt=OpenAISTTProvider(),
            )
            with open(audio_path, "rb") as handle:
                audio_b64 = base64.b64encode(handle.read()).decode("ascii")
            audio_mime_out = tts_info.get("mime") or "audio/mpeg"
            audio_url = "/audio/{}.mp3".format(audio_id)
        except Exception as exc:
            # Still return the text answer if TTS or lip-sync fails.
            backend_name = "none (tts failed: {})".format(exc)

    return AskResponse(
        question=question,
        answer=answer,
        sources=sources,
        audio_base64=audio_b64,
        audio_mime=audio_mime_out,
        audio_url=audio_url,
        visemes=visemes,
        lipsync_backend=backend_name,
    )
