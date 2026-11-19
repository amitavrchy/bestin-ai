import json
import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.config import settings
from app.schemas import BookSummary

_lock = threading.Lock()


def _index_path() -> str:
    path = settings.books_index
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def _read() -> List[Dict]:
    path = _index_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        return data
    return []


def _write(books: List[Dict]) -> None:
    path = _index_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(books, handle, indent=2)
    os.replace(tmp, path)


def list_books() -> List[BookSummary]:
    with _lock:
        books = _read()
    return [BookSummary(**item) for item in books]


def get_book(book_id: str) -> Optional[BookSummary]:
    for book in list_books():
        if book.id == book_id:
            return book
    return None


def add_book(record: Dict) -> BookSummary:
    with _lock:
        books = _read()
        books = [item for item in books if item.get("id") != record["id"]]
        books.insert(0, record)
        _write(books)
    return BookSummary(**record)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
