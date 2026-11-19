from typing import Literal
from pydantic import BaseModel, Field

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


