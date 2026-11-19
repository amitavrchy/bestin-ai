from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_tts_model: str = "tts-1"
    openai_tts_voice: str = "nova"
    openai_stt_model: str = "whisper-1"
    lipsync_backend: str = "auto"
    chroma_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"
    audio_dir: str = "./data/audio"
    books_index: str = "./data/books.json"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 6

    def cors_origin_list(self) -> List[str]:
        return [part.strip() for part in self.cors_origins.split(",") if part.strip()]

    def openai_is_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())


settings = Settings()
