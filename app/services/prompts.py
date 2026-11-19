"""Grounded, speech-friendly prompt construction."""

def build_messages(question: str, history: list, sources: list[str], style: str) -> list[dict]:
    """Build a numbered source prompt and include recent conversation context."""
    instruction = ('You are Bestin, a helpful guide to this book. Answer only from the numbered sources. '
                   'Cite each factual claim using [n], and say you do not know if the sources do not answer. '
                   'Use spoken-friendly plain language: no markdown, tables or URLs. ')
    instruction += 'Keep the answer under about 120 words. ' if style == 'concise' else 'Give a useful detailed answer. '
    instruction += '\n\nSources:\n' + '\n\n'.join(f'[{i}] {text}' for i, text in enumerate(sources, 1))
    return [{'role': 'system', 'content': instruction}, *[m.model_dump() for m in history[-6:]], {'role': 'user', 'content': question}]
