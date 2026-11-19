"""Application factory and lifecycle."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.routes import ask, books, health, speech
from app.core import logging as structured_logging  # noqa: F401
from app.services import library

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


def create_app() -> FastAPI:
    """Create the internal AI service."""
    application = FastAPI(title="Bestin AI", version="1.0.0", lifespan=lifespan)
    for module in (health, books, ask, speech):
        application.include_router(module.router)
    return application


app = create_app()


