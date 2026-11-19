"""Small deterministic retrieval and answer helpers."""
import re


def rewrite_query(question: str, history: list) -> str:
    """Add recent user context to short or referential follow-up questions."""
    if len(question.split()) <= 5 or re.search(r'\b(it|they|them|he|she|that|those|this|his|her|their)\b', question, re.I):
        previous = next((m.content for m in reversed(history) if m.role == 'user'), '')
        if previous:
            return previous + ' ' + question
    return question


def extractive_answer(question: str, citations: list[dict]) -> str:
    """Select a few relevant sentences with source numbers as a safe fallback."""
    terms = set(re.findall(r'[a-z]{3,}', question.lower()))
    candidates = []
    for i, citation in enumerate(citations, 1):
        for sentence in re.split(r'(?<=[.!?])\s+', citation['text']):
            sentence = sentence.strip()
            if len(sentence) >= 20:
                score = len(terms & set(re.findall(r'[a-z]{3,}', sentence.lower())))
                candidates.append((score, i, sentence))
    candidates.sort(key=lambda x: -x[0])
    selected = candidates[:3]
    return ' '.join(f'{sentence} [{i}]' for _, i, sentence in selected) if selected else 'I do not know based on this book.'
