from app.config import settings


def configuration() -> tuple[str, str, str, str]:
    return ("openai", settings.openai_api_key, settings.openai_base_url, settings.openai_model)
