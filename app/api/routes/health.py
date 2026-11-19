from fastapi import APIRouter
from app.config import settings
from app.providers.llm.registry import llm_candidates

router = APIRouter()

@router.get("/health")
def health():
    candidates = llm_candidates()
    stt = (
        "groq"
        if settings.groq_api_key
        else "openai"
        if settings.openai_api_key
        else "unavailable"
    )
    return {
        "status": "ok",
        "service": "bestin-ai",
        "version": "1.0.0",
        "providers": {
            "llm": candidates[0][0] if candidates else "extractive",
            "embeddings": settings.embedding_provider,
            "stt": stt,
            "tts": settings.tts_provider,
            "lipsync": settings.lipsync_backend,
        },
    }


