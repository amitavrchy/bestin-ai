from pathlib import Path

from app.config import settings


class FastEmbed:
    def __init__(self, model: str):
        self.model = model
        self._local = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._local is None:
            from fastembed import TextEmbedding

            cache = Path(settings.data_dir) / "models"
            cache.mkdir(parents=True, exist_ok=True)
            self._local = TextEmbedding(model_name=self.model, cache_dir=str(cache))
        return [v.tolist() for v in self._local.embed(texts, batch_size=32)]
