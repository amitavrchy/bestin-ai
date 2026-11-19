from app.config import settings


def configuration() -> tuple[str, str, str, str]:
    return ("pollinations", "keyless", settings.pollinations_base_url, settings.pollinations_model)
