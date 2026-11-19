from app.config import settings
from app.providers.embeddings import fastembed, gemini, openai


class Embeddings:
    def __init__(self):
        self.provider = settings.embedding_provider
        if self.provider == "fastembed":
            self._provider = fastembed.FastEmbed(settings.embedding_model)
        elif self.provider == "openai":
            self._provider = openai.Provider()
        elif self.provider == "gemini":
            self._provider = gemini.Provider()
        else:
            raise ValueError("Unknown embedding provider")
        self.model = self._provider.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)
