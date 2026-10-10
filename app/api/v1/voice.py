"""Voice to Text API routes using Google Cloud Speech-to-Text."""

import os
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.speech.service import speech_service
from app.core.logging import logger
from app.core.config import settings

router = APIRouter(prefix="/voice", tags=["Voice"])

@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe an audio file using Google Cloud STT."""
    if not settings.GCP_STT_ENABLED:
        raise HTTPException(status_code=501, detail="Speech-to-Text is not enabled.")

    # Validate file size or type if needed
    try:
        content = await file.read()
        transcript = await speech_service.transcribe(content)
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"STT Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process audio.")
