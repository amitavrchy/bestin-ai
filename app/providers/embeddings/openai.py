from app.config import settings
from app.providers.llm.base import client


class Provider:
    model = "text-embedding-3-small"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not settings.openai_api_key:
            raise ValueError("Embedding provider key is missing")
        result = client(settings.openai_base_url, settings.openai_api_key).embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in result.data]
