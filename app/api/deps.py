import re
import secrets

from fastapi import Header, HTTPException
from app.config import settings

def auth(x_internal_token: str = Header(default="")):
    if not secrets.compare_digest(x_internal_token, settings.ai_internal_token):
        raise HTTPException(401, "invalid_internal_token")


def valid_id(book_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", str(book_id)):
        raise HTTPException(400, "invalid_book_id")
    return str(book_id)


