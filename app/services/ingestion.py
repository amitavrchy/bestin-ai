import re
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from pypdf import PdfReader


def parse_file(path: Path) -> tuple[list[dict], int | None]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(path))
        return [
            {"text": p.extract_text() or "", "page": i + 1, "chapter": None}
            for i, p in enumerate(reader.pages)
        ], len(reader.pages)
    if ext == ".epub":
        book = epub.read_epub(str(path))
        sections = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            heading = soup.find(["h1", "h2", "h3", "title"])
            sections.append(
                {
                    "text": soup.get_text(" ", strip=True),
                    "page": None,
                    "chapter": heading.get_text(" ", strip=True) if heading else None,
                }
            )
        return sections, None
    if ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8-sig")
        parts = re.split(r"(?im)^(?:#{1,3}\s*)?(chapter\s+[^\n]+)\s*$", text)
        sections = [{"text": parts[0], "page": None, "chapter": None}]
        sections.extend(
            {"text": parts[i + 1], "page": None, "chapter": parts[i].strip()}
            for i in range(1, len(parts) - 1, 2)
        )
        return sections, None
    raise ValueError("Unsupported file type")


def chunk_sections(
    sections: list[dict], size: int = 900, overlap: int = 150
) -> list[dict]:
    chunks = []
    for section in sections:
        text = re.sub(r"\s+", " ", section["text"]).strip()
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = max(
                    text.rfind(". ", start + size // 2, end),
                    text.rfind("? ", start + size // 2, end),
                    text.rfind("! ", start + size // 2, end),
                )
                if boundary > start:
                    end = boundary + 1
            chunks.append({**section, "text": text[start:end].strip()})
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks
