import logging
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field

from app.config import settings
from app.providers.registry import chat, llm_candidates, transcribe
from app.providers.speech import VOICES, synthesize
from app.services import library

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Jobs cannot resume after a process restart; make the interruption visible.
    directory = library.root() / "status"
    if directory.exists():
        for path in directory.glob("*.json"):
            record = library.status(path.stem)
            if record["status"] in ("queued", "processing"):
                record.update(
                    status="failed", error="Ingestion interrupted by service restart"
                )
                library.save(record)
    yield


app = FastAPI(title="Bestin AI", version="1.0.0", lifespan=lifespan)


def auth(x_internal_token: str = Header(default="")):
    if not secrets.compare_digest(x_internal_token, settings.ai_internal_token):
        raise HTTPException(401, "invalid_internal_token")


def valid_id(book_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", str(book_id)):
        raise HTTPException(400, "invalid_book_id")
    return str(book_id)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Ask(BaseModel):
    book_id: str | int
    question: str = Field(min_length=1, max_length=10000)
    history: list[Message] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    speak: bool = True
    voice: str | None = None
    style: Literal["concise", "detailed"] = "concise"


class TTS(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    voice: str | None = None


@app.get("/health")
def health():
    candidates = llm_candidates()
    stt = (
        "groq"
        if settings.groq_api_key
        else "openai"
        if settings.openai_api_key
        else "unavailable"
    )
    return {
        "status": "ok",
        "service": "bestin-ai",
        "version": "1.0.0",
        "providers": {
            "llm": candidates[0][0] if candidates else "extractive",
            "embeddings": settings.embedding_provider,
            "stt": stt,
            "tts": settings.tts_provider,
            "lipsync": settings.lipsync_backend,
        },
    }


@app.get("/v1/voices", dependencies=[Depends(auth)])
def voices():
    return VOICES


@app.post("/v1/books/{book_id}/ingest", status_code=202, dependencies=[Depends(auth)])
async def ingest(
    book_id: str,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(default=""),
):
    book_id = valid_id(book_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".epub", ".txt", ".md"):
        raise HTTPException(400, "unsupported_file_type")
    data = await file.read(50 * 1024 * 1024 + 1)
    if not data or len(data) > 50 * 1024 * 1024:
        raise HTTPException(400, "empty_or_oversized_file")
    with library.lock:
        if library.status(book_id)["status"] in ("queued", "processing"):
            raise HTTPException(409, "ingestion_in_progress")
        upload = library.root() / "uploads"
        upload.mkdir(exist_ok=True)
        path = upload / f"{book_id}{suffix}"
        path.write_bytes(data)
        record = {
            "book_id": book_id,
            "status": "queued",
            "progress": 0,
            "chunks": 0,
            "pages": None,
            "error": None,
            "embedding_model": library.embeddings.model,
        }
        library.save(record)
    background.add_task(
        library.ingest, book_id, path, title or Path(file.filename or "Book").stem
    )
    return {"book_id": book_id, "status": "queued"}


@app.get("/v1/books/{book_id}/status", dependencies=[Depends(auth)])
def book_status(book_id: str):
    return library.status(valid_id(book_id))


@app.delete("/v1/books/{book_id}", status_code=204, dependencies=[Depends(auth)])
def delete_book(book_id: str):
    library.delete(valid_id(book_id))


@app.post("/v1/ask", dependencies=[Depends(auth)])
async def ask(body: Ask):
    book_id = valid_id(str(body.book_id))
    record = library.status(book_id)
    if record["status"] == "not_found":
        raise HTTPException(404, "book_not_found")
    if record["status"] != "ready":
        raise HTTPException(409, "book_not_ready")
    try:
        coll = library.db().get_collection(library.collection_name(book_id))
        metadata = coll.metadata or {}
        if (
            metadata.get("embedding_model") != library.embeddings.model
            or metadata.get("embedding_provider") != library.embeddings.provider
        ):
            raise HTTPException(409, "embedding_model_changed_reingest_required")
        query = library.embeddings.embed([body.question])
        result = coll.query(
            query_embeddings=query, n_results=min(body.top_k, coll.count())
        )
    except HTTPException:
        raise
    except Exception:
        logging.exception("Retrieval failed")
        raise HTTPException(500, "retrieval_failed")
    citations = []
    for chunk_id, text, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        citations.append(
            {
                "chunk_id": chunk_id,
                "text": text[:600],
                "page": meta.get("page") or None,
                "chapter": meta.get("chapter") or None,
                "score": float(1 - distance),
            }
        )
    sources = "\n\n".join(
        f"[{i}] {text}" for i, text in enumerate(result["documents"][0], 1)
    )
    messages = [
        {
            "role": "system",
            "content": "Answer only from the numbered book sources. Cite factual claims with [1], [2], etc. If the answer is not in the sources, say you do not know. "
            + (
                "Keep the answer around 120 words or less."
                if body.style == "concise"
                else "Give a detailed answer."
            )
            + "\n\nSources:\n"
            + sources,
        }
    ]
    messages.extend(m.model_dump() for m in body.history[-6:])
    messages.append({"role": "user", "content": body.question})
    response = chat(messages)
    if response:
        answer, provider, model = response
    else:
        answer = (
            "I could not generate an answer. Here is a relevant passage from the book: "
            + citations[0]["text"]
            + " [1]"
            if citations
            else "I do not know based on this book."
        )
        provider, model = "extractive", "extractive"
    speech = None
    if body.speak:
        try:
            speech = await synthesize(answer, body.voice)
        except Exception:
            logging.exception("Speech synthesis failed")
    return {
        "answer": answer,
        "citations": citations,
        "provider": provider,
        "model": model,
        "speech": speech,
    }


@app.post("/v1/tts", dependencies=[Depends(auth)])
async def tts(body: TTS):
    try:
        return await synthesize(body.text, body.voice)
    except Exception:
        logging.exception("Speech synthesis failed")
        raise HTTPException(503, "tts_unavailable")


@app.post("/v1/stt", dependencies=[Depends(auth)])
async def stt(file: UploadFile = File(...)):
    try:
        return transcribe(await file.read(), file.filename or "audio.webm")
    except LookupError:
        raise HTTPException(503, "stt_unavailable")
    except Exception:
        logging.exception("Transcription failed")
        raise HTTPException(502, "stt_failed")
