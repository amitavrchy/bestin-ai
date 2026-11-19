from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VisemeCue(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    viseme: str = Field(..., description="Rhubarb-style viseme: A-H or X")


class Source(BaseModel):
    book_id: str
    title: str
    page: Optional[int] = None
    locator: str
    excerpt: str


class BookSummary(BaseModel):
    id: str
    title: str
    filename: str
    kind: str
    pages: int
    chunks: int
    created_at: str


class HealthResponse(BaseModel):
    status: str
    service: str
    openai_configured: bool
    lipsync: Dict[str, Any]


class BookListResponse(BaseModel):
    books: List[BookSummary]


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    audio_base64: Optional[str] = None
    audio_mime: Optional[str] = None
    audio_url: Optional[str] = None
    visemes: List[VisemeCue] = Field(default_factory=list)
    lipsync_backend: Optional[str] = None
