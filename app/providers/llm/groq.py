from app.config import settings


def configuration() -> tuple[str, str, str, str]:
    return ("groq", settings.groq_api_key, settings.groq_base_url, settings.groq_model)
