from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.providers.speech import normalize_text
from app.services import library
from app.services.chunking import chunk_sections
from app.services.lipsync import timeline
from app.services.parsing import clean_text, parse_file
from app.services.prompts import build_messages
from app.services.rag import extractive_answer, rewrite_query


@pytest.mark.parametrize(('raw', 'expected'), [
    ('Hello [1] world', 'Hello world'), ('**Bold** text', 'Bold text'),
    ('[link](https://example.com)', 'link'), ('Visit https://example.com now', 'Visit now'),
    ('# Heading\nText', 'Heading Text'), ('`code`', 'code'),
])
def test_normalize(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize('word', ['thin', 'ship', 'church', 'moon', 'feet', 'out', 'now', 'phone', 'when', 'queen', 'mama', 'very', 'light'])
def test_timeline_covers_audio(word):
    cues = timeline(word, 1.5, [{'word': word, 'start': .2, 'end': 1.1}])
    assert cues[0]['start'] == 0 and cues[-1]['end'] == 1.5
    assert all(a['end'] == pytest.approx(b['start']) for a, b in zip(cues, cues[1:]))
    assert all(c['value'] in 'ABCDEFGHX' and c['start'] < c['end'] for c in cues)


@pytest.mark.parametrize('suffix', ['.txt', '.md'])
def test_latin1_chapters(tmp_path, suffix):
    path = tmp_path / ('book' + suffix)
    path.write_bytes('Chapter 1\nCafé.\nChapter 2\nFin.'.encode('latin-1'))
    sections, pages = parse_file(path)
    assert pages is None and sections[0]['chapter'] == 'Chapter 1'
    assert 'Café' in sections[0]['text'] and sections[1]['chapter'] == 'Chapter 2'


def test_hyphenation():
    assert clean_text('exam-\nple  text') == 'example text'


@pytest.mark.parametrize('size,overlap', [(100, 10), (200, 25), (900, 150)])
def test_chunk_bounds(size, overlap):
    chunks = chunk_sections([{'text': 'One sentence here. ' * 200, 'page': 2, 'chapter': 'A'}], size, overlap)
    assert len(chunks) > 1 and all(0 < len(c['text']) <= size and c['page'] == 2 for c in chunks)


def test_prompt_citations():
    messages = build_messages('Why?', [], ['first', 'second'], 'concise')
    assert '[1] first' in messages[0]['content'] and '[2] second' in messages[0]['content']
    assert 'Bestin' in messages[0]['content']


def test_rewrite_history():
    history = [SimpleNamespace(role='user', content='Who is Alice?')]
    assert rewrite_query('Where is she?', history) == 'Who is Alice? Where is she?'


def test_extractive_cites():
    answer = extractive_answer('Where was Alice?', [{'text': 'Alice was sitting by the river. She was bored.'}])
    assert 'Alice' in answer and '[1]' in answer


def test_restart_marks_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'data_dir', str(tmp_path))
    library.save({'book_id': '1', 'status': 'processing', 'progress': .5})
    with TestClient(app):
        assert library.status('1')['status'] == 'failed'
        assert 're-ingest' in library.status('1')['error']


@pytest.mark.parametrize('route', ['/v1/voices', '/v1/books/1/status'])
def test_private_routes_require_token(route):
    with TestClient(app) as client:
        assert client.get(route).status_code == 401


def test_empty_book_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'data_dir', str(tmp_path))
    headers = {'X-Internal-Token': settings.ai_internal_token}
    with TestClient(app) as client:
        response = client.post('/v1/books/empty/ingest', headers=headers, files={'file': ('empty.txt', b'   ')})
        assert response.status_code == 202
    assert library.status('empty')['status'] == 'failed'


def test_provider_fallback_strips_reasoning(monkeypatch):
    from unittest.mock import Mock
    from app.providers import registry

    monkeypatch.setattr(settings, 'llm_provider', 'auto')
    monkeypatch.setattr(settings, 'groq_api_key', 'test')
    monkeypatch.setattr(settings, 'gemini_api_key', 'test')
    monkeypatch.setattr(settings, 'openai_api_key', '')
    monkeypatch.setattr(registry, '_pollinations_last', 0.0)
    def fake_client(url, key):
        fake = Mock()
        if url == settings.gemini_base_url:
            fake.chat.completions.create.return_value.choices = [Mock(message=Mock(content='<think>secret</think> Answer [1]'))]
        else:
            fake.chat.completions.create.side_effect = RuntimeError('unavailable')
        return fake
    monkeypatch.setattr(registry, 'client', fake_client)
    assert registry.chat([{'role': 'user', 'content': 'hello'}])[:2] == ('Answer [1]', 'gemini')


def test_provider_exhaustion(monkeypatch):
    from app.providers import registry
    monkeypatch.setattr(settings, 'groq_api_key', 'test')
    monkeypatch.setattr(settings, 'gemini_api_key', 'test')
    monkeypatch.setattr(settings, 'openai_api_key', '')
    monkeypatch.setattr(registry, '_pollinations_last', 0.0)
    def unavailable(*args):
        raise RuntimeError('unavailable')
    monkeypatch.setattr(registry, 'client', unavailable)
    assert registry.chat([{'role': 'user', 'content': 'hello'}]) is None


def test_vector_collection_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'data_dir', str(tmp_path))
    collection = library.db().create_collection('book_roundtrip', metadata={'hnsw:space': 'cosine'})
    collection.add(ids=['1'], documents=['A story'], embeddings=[[1.0, 0.0]])
    assert collection.query(query_embeddings=[[1.0, 0.0]], n_results=1)['documents'][0] == ['A story']
    library.delete('roundtrip')
    assert not any(c.name == 'book_roundtrip' for c in library.db().list_collections())
