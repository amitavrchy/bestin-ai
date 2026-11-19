"""Extract readable, attributed sections from supported book formats."""
import re
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Repair line-wrapped hyphens and normalize whitespace."""
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def parse_file(path: Path) -> tuple[list[dict], int | None]:
    """Return nonempty sections and the original PDF page count, if applicable."""
    ext = path.suffix.lower()
    if ext == '.pdf':
        reader = PdfReader(str(path))
        sections = []
        for i, page in enumerate(reader.pages, 1):
            text = clean_text(page.extract_text() or '')
            if text:
                sections.append({'text': text, 'page': i, 'chapter': None})
        return sections, len(reader.pages)
    if ext == '.epub':
        book = epub.read_epub(str(path))
        sections = []
        for item_id, *_ in book.spine:
            item = book.get_item_with_id(item_id)
            if item is None or item.get_type() != ITEM_DOCUMENT:
                continue
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            heading = soup.find(['h1', 'h2', 'h3', 'title'])
            text = clean_text(soup.get_text(' ', strip=True))
            if text:
                sections.append({'text': text, 'page': None, 'chapter': heading.get_text(' ', strip=True) if heading else None})
        return sections, None
    if ext in ('.txt', '.md'):
        raw = path.read_bytes()
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw.decode('latin-1')
        parts = re.split(r'(?im)^(?:#{1,6}\s*)?(chapter\s+[^\n]+?)\s*#*\s*$', text)
        sections = [{'text': clean_text(parts[0]), 'page': None, 'chapter': None}]
        sections.extend({'text': clean_text(parts[i+1]), 'page': None, 'chapter': parts[i].strip()} for i in range(1, len(parts)-1, 2))
        return [s for s in sections if s['text']], None
    raise ValueError('Unsupported file type')
