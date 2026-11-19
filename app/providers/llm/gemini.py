from app.config import settings


def configuration() -> tuple[str, str, str, str]:
    return ("gemini", settings.gemini_api_key, settings.gemini_base_url, settings.gemini_model)
