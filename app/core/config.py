from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    ai_internal_token: str = "dev-internal-token"
    llm_provider: str = "auto"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_model: str = "gemini-3.1-flash-lite"
    pollinations_base_url: str = "https://text.pollinations.ai/openai"
    pollinations_model: str = "openai-fast"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    stt_provider: str = "auto"
    local_stt_model: str = "base.en"
    local_stt_enabled: bool = True
    tts_provider: str = "edge"
    lipsync_backend: str = "auto"
    data_dir: str = "data"
    request_timeout: float = 20.0


settings = Settings()
