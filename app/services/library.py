import json
import logging
import threading
from pathlib import Path

import chromadb

from app.config import settings
from app.providers.registry import Embeddings
from app.services.ingestion import chunk_sections, parse_file

log = logging.getLogger(__name__)
lock = threading.RLock()
embeddings = Embeddings()


def root() -> Path:
    path = Path(settings.data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def status(book_id: str) -> dict:
    path = root() / 'status' / f'{book_id}.json'
    with lock:
        if path.exists():
            return json.loads(path.read_text())
    return {'book_id': book_id, 'status': 'not_found', 'progress': 0, 'chunks': 0, 'pages': None, 'error': None, 'embedding_model': embeddings.model}


def save(record: dict):
    with lock:
        path = root() / 'status' / f"{record['book_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(record))
        tmp.replace(path)


def collection_name(book_id: str) -> str:
    return f'book_{book_id}'


def db():
    return chromadb.PersistentClient(path=str(root() / 'chroma'))


def delete(book_id: str):
    with lock:
        try:
            db().delete_collection(collection_name(book_id))
        except ValueError:
            pass
        (root() / 'status' / f'{book_id}.json').unlink(missing_ok=True)
        for path in (root() / 'uploads').glob(f'{book_id}.*'):
            path.unlink(missing_ok=True)


def ingest(book_id: str, path: Path, title: str):
    record = status(book_id)
    try:
        record.update(status='processing', progress=0.1)
        save(record)
        sections, pages = parse_file(path)
        chunks = chunk_sections(sections)
        if not chunks:
            raise ValueError('No readable text found in book')
        vectors = []
        for i in range(0, len(chunks), 32):
            vectors.extend(embeddings.embed([c['text'] for c in chunks[i:i + 32]]))
            record['progress'] = 0.1 + 0.7 * min(i + 32, len(chunks)) / len(chunks)
            save(record)
        with lock:
            if not (root() / 'status' / f'{book_id}.json').exists():
                return
            database = db()
            try:
                database.delete_collection(collection_name(book_id))
            except ValueError:
                pass
            coll = database.create_collection(collection_name(book_id), metadata={'hnsw:space': 'cosine', 'embedding_model': embeddings.model, 'embedding_provider': embeddings.provider})
            for i in range(0, len(chunks), 100):
                batch = chunks[i:i + 100]
                coll.add(ids=[str(j) for j in range(i, i + len(batch))], documents=[c['text'] for c in batch], embeddings=vectors[i:i + 100], metadatas=[{'page': c['page'] or 0, 'chapter': c['chapter'] or ''} for c in batch])
            record.update(status='ready', progress=1, chunks=len(chunks), pages=pages, error=None,
                          suggested_questions=[f'What is {title} about?', 'What are the main ideas?', 'What are the key lessons?', 'How does the book begin?'])
            save(record)
    except Exception as exc:
        log.exception('Ingestion failed for %s', book_id)
        if (root() / 'status' / f'{book_id}.json').exists():
            record.update(status='failed', error=str(exc))
            save(record)
