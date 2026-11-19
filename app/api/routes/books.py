import uuid
from pathlib import Path
from fastapi import BackgroundTasks, File, Form, UploadFile
from app.services import library
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import auth, valid_id

router = APIRouter()

@router.post("/v1/books/{book_id}/ingest", status_code=202, dependencies=[Depends(auth)])
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
        job_id = uuid.uuid4().hex
        (library.root() / "status" / f"{book_id}.job").write_text(job_id)
    background.add_task(
        library.ingest,
        book_id,
        path,
        title or Path(file.filename or "Book").stem,
        job_id,
    )
    return {"book_id": book_id, "status": "queued"}


@router.get("/v1/books/{book_id}/status", dependencies=[Depends(auth)])
def book_status(book_id: str):
    return library.status(valid_id(book_id))


@router.delete("/v1/books/{book_id}", status_code=204, dependencies=[Depends(auth)])
def delete_book(book_id: str):
    library.delete(valid_id(book_id))


