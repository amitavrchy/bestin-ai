"""Sentence-aware section chunking."""
import re

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
