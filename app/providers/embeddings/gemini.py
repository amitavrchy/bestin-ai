from app.config import settings
from app.providers.llm.base import client


class Provider:
    model = "gemini-embedding-001"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not settings.gemini_api_key:
            raise ValueError("Embedding provider key is missing")
        result = client(settings.gemini_base_url, settings.gemini_api_key).embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in result.data]
