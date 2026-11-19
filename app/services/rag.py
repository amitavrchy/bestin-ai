import os
from typing import List, Optional, Tuple

import chromadb

from app.config import settings
from app.providers.base import ChatProvider, EmbeddingProvider
from app.schemas import Source
from app.services.ingestion import Chunk
from app.services.library import get_book


SYSTEM_PROMPT = (
    "You are Bestin, a friendly voice guide for a book the listener has uploaded. "
    "Answer ONLY using the provided excerpts. If the excerpts do not contain the answer, "
    "say you cannot find that in the book and offer a nearby idea only if the excerpts support it. "
    "Cite sources in plain language using the locator in brackets, like [p.12] or [ch.3]. "
    "Write for speech: short, clear sentences, no markdown, no bullet lists, no URLs. "
    "Stay in the book's world; do not invent page numbers or quotations."
)


class VectorStore:
    def __init__(self, embeddings: EmbeddingProvider, path: Optional[str] = None):
        self.embeddings = embeddings
        persist = path or settings.chroma_dir
        os.makedirs(persist, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist)
        self.collection = self.client.get_or_create_collection(
            name="bestin_books",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, book_id: str, title: str, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        ids = ["{}:{}".format(book_id, index) for index in range(len(chunks))]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "book_id": book_id,
                "title": title,
                "page": chunk.page if chunk.page is not None else -1,
                "locator": chunk.locator,
            }
            for chunk in chunks
        ]
        batch_size = 32
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            batch_docs = documents[start:end]
            vectors = self.embeddings.embed_texts(batch_docs)
            self.collection.add(
                ids=ids[start:end],
                embeddings=vectors,
                documents=batch_docs,
                metadatas=metadatas[start:end],
            )

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        book_id: Optional[str] = None,
    ) -> List[Source]:
        k = k if k is not None else settings.retrieval_k
        # Chroma requires n_results <= collection size; clamp after a peek.
        count = self.collection.count()
        if count == 0:
            return []
        k = min(k, count)
        query_vec = self.embeddings.embed_query(question)
        kwargs = {
            "query_embeddings": [query_vec],
            "n_results": k,
            "include": ["documents", "metadatas"],
        }
        if book_id:
            kwargs["where"] = {"book_id": book_id}
        result = self.collection.query(**kwargs)
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        sources: List[Source] = []
        for doc, meta in zip(documents, metadatas):
            meta = meta or {}
            page_val = meta.get("page", -1)
            page = page_val if isinstance(page_val, int) and page_val >= 0 else None
            excerpt = (doc or "").strip()
            if len(excerpt) > 420:
                excerpt = excerpt[:417].rstrip() + "..."
            sources.append(
                Source(
                    book_id=str(meta.get("book_id") or book_id or ""),
                    title=str(meta.get("title") or ""),
                    page=page,
                    locator=str(meta.get("locator") or ("p.{}".format(page) if page else "")),
                    excerpt=excerpt,
                )
            )
        return sources


def build_user_prompt(question: str, sources: List[Source]) -> str:
    blocks = []
    for index, source in enumerate(sources, start=1):
        locator = source.locator or "excerpt {}".format(index)
        title = source.title or "the book"
        blocks.append(
            "Excerpt {n} - {title} [{locator}]:\n{text}".format(
                n=index,
                title=title,
                locator=locator,
                text=source.excerpt,
            )
        )
    joined = "\n\n".join(blocks) if blocks else "(No excerpts were retrieved.)"
    return "Excerpts:\n{excerpts}\n\nQuestion: {question}".format(
        excerpts=joined,
        question=question.strip(),
    )

def answer_question(
    question: str,
    store: VectorStore,
    chat: ChatProvider,
    book_id: Optional[str] = None,
) -> Tuple[str, List[Source]]:
    if book_id and get_book(book_id) is None:
        raise ValueError("Unknown book_id: {}".format(book_id))
    sources = store.query(question, book_id=book_id)
    if not sources:
        return (
            "I could not find any passages from the book to answer with. Please upload a book first.",
            [],
        )
    answer = chat.complete(SYSTEM_PROMPT, build_user_prompt(question, sources))
    return answer, sources
