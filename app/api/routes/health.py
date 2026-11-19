from fastapi import APIRouter
from app.config import settings
from app.providers.llm.registry import llm_candidates
from app.providers.stt.registry import candidates as stt_candidates

router = APIRouter()

@router.get("/health")
def health():
    candidates = llm_candidates()
    stt = stt_candidates()
    return {
        "status": "ok",
        "service": "bestin-ai",
        "version": "1.0.0",
        "providers": {
            "llm": candidates[0][0] if candidates else "extractive",
            "embeddings": settings.embedding_provider,
            "stt": stt[0][0] if stt else "unavailable",
            "tts": settings.tts_provider,
            "lipsync": settings.lipsync_backend,
        },
    }


