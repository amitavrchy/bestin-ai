import os
import re
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.config import settings
from app.services.library import add_book, now_iso


@dataclass
class Page:
    page: Optional[int]
    locator: str
    text: str


@dataclass
class Chunk:
    text: str
    page: Optional[int]
    locator: str


ALLOWED_EXTENSIONS = {".pdf", ".epub", ".txt"}


def detect_kind(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return ext.lstrip(".")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: str) -> Tuple[str, List[Page]]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    title = ""
    if reader.metadata and reader.metadata.title:
        title = str(reader.metadata.title).strip()
    pages: List[Page] = []
    for index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = normalize_whitespace(raw)
        if not text:
            continue
        pages.append(Page(page=index, locator="p.{}".format(index), text=text))
    return title, pages


def parse_epub(path: str) -> Tuple[str, List[Page]]:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(path)
    title = ""
    try:
        titles = book.get_metadata("DC", "title")
        if titles:
            title = str(titles[0][0]).strip()
    except Exception:
        title = ""

    pages: List[Page] = []
    chapter_no = 0
    for item in book.get_items():
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        chapter_no += 1
        soup = BeautifulSoup(item.get_content(), "lxml")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        text = normalize_whitespace(soup.get_text("\n"))
        if not text:
            continue
        heading = soup.find(["h1", "h2", "h3"])
        if heading and heading.get_text(strip=True):
            locator = "ch.{} ({})".format(chapter_no, heading.get_text(strip=True)[:80])
        else:
            locator = "ch.{}".format(chapter_no)
        pages.append(Page(page=chapter_no, locator=locator, text=text))
    return title, pages


def parse_txt(path: str) -> Tuple[str, List[Page]]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = normalize_whitespace(handle.read())
    if not text:
        return "", []
    return "", [Page(page=1, locator="txt", text=text)]


def parse_book(path: str, filename: str) -> Tuple[str, List[Page]]:
    kind = detect_kind(filename)
    if kind == "pdf":
        return parse_pdf(path)
    if kind == "epub":
        return parse_epub(path)
    if kind == "txt":
        return parse_txt(path)
    raise ValueError(
        "Unsupported file type '{}'. Please upload a PDF, EPUB, or TXT file.".format(
            kind or "unknown"
        )
    )


def _split_text(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            window = text[start:end]
            cut = max(
                window.rfind(". "),
                window.rfind("? "),
                window.rfind("! "),
                window.rfind("\n"),
                window.rfind(" "),
            )
            if cut >= int(size * 0.5):
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(
    pages: List[Page],
    size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[Chunk]:
    size = size if size is not None else settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap
    chunks: List[Chunk] = []
    for page in pages:
        for part in _split_text(page.text, size, overlap):
            chunks.append(Chunk(text=part, page=page.page, locator=page.locator))
    return chunks


def title_from_filename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    cleaned = re.sub(r"[._-]+", " ", stem).strip()
    return cleaned or "Untitled book"


def ingest_file(path: str, original_name: str) -> Tuple[dict, List[Chunk]]:
    title, pages = parse_book(path, original_name)
    if not pages:
        raise ValueError("No extractable text was found in that file.")
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("The file was parsed but produced no text chunks.")
    book_id = uuid.uuid4().hex[:12]
    record = {
        "id": book_id,
        "title": title or title_from_filename(original_name),
        "filename": os.path.basename(original_name),
        "kind": detect_kind(original_name),
        "pages": len(pages),
        "chunks": len(chunks),
        "created_at": now_iso(),
        "path": path,
    }
    add_book(record)
    return record, chunks
