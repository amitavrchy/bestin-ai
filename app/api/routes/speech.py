import logging
from fastapi import File, UploadFile
from app.schemas.requests import TTS
from app.providers.registry import transcribe
from app.providers.speech import VOICES, synthesize
from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import auth

router = APIRouter()

@router.get("/v1/voices", dependencies=[Depends(auth)])
def voices():
    return VOICES


@router.post("/v1/tts", dependencies=[Depends(auth)])
async def tts(body: TTS):
    try:
        return await synthesize(body.text, body.voice)
    except Exception:
        logging.exception("Speech synthesis failed")
        raise HTTPException(503, "tts_unavailable")


@router.post("/v1/stt", dependencies=[Depends(auth)])
async def stt(file: UploadFile = File(...)):
    try:
        return transcribe(await file.read(), file.filename or "audio.webm")
    except LookupError:
        raise HTTPException(503, "stt_unavailable")
    except Exception:
        logging.exception("Transcription failed")
        raise HTTPException(502, "stt_failed")
