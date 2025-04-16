from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import tempfile
import os
import logging

from app.services.whisper_service import transcribe_audio
from app.services.storage_service import storage_service
from app.db.database import get_db
from app.models.audio_model import AudioTranscription
from app.api.auth.auth import get_current_user
from app.models.user_model import User
from app.services.whisperx_service import transcribe_with_whisperx

router = APIRouter()
logger = logging.getLogger(__name__)


class TranscriptionResponse(BaseModel):
    id: int
    text: str
    audio_url: str
    language: Optional[str] = None
    duration: Optional[float] = None
    filename: str
    segments: Optional[List[Dict[str, Any]]] = None
    word_segments: Optional[List[Dict[str, Any]]] = None


class TranscriptionList(BaseModel):
    items: List[TranscriptionResponse]


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_file(
        file: UploadFile = File(...),
        language: str = "kk",
        task: str = "transcribe",
        use_whisperx: bool = True,
        diarize: bool = False,
        align_words: bool = True,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    try:
        file_info = await storage_service.upload_file(file)

        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            await file.seek(0)
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Transcribe audio with WhisperX or standard Whisper
        if use_whisperx:
            result = transcribe_with_whisperx(
                temp_path,
                language=language,
                task=task,
                diarize=diarize,
                align_words=align_words
            )
        else:
            result = transcribe_audio(temp_path, language=language, task=task)

        # Clean up temp file
        os.unlink(temp_path)

        # Store in database
        transcription = AudioTranscription(
            original_filename=file_info["original_filename"],
            s3_filename=file_info["s3_filename"],
            s3_url=file_info["s3_url"],
            file_size=file_info["size"],
            duration=result.get("duration"),
            language=result.get("language"),
            transcription=result["text"],
            segments=result.get("segments"),
            word_segments=result.get("word_segments")
        )

        db.add(transcription)
        db.commit()
        db.refresh(transcription)

        return TranscriptionResponse(
            id=transcription.id,
            text=transcription.transcription,
            audio_url=transcription.s3_url,
            language=transcription.language,
            duration=transcription.duration,
            filename=transcription.original_filename,
            segments=transcription.segments,
            word_segments=transcription.word_segments
        )

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@router.get("/transcriptions", response_model=TranscriptionList)
def get_transcriptions(skip: int = 0, limit: int = 10, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    transcriptions = db.query(AudioTranscription).offset(skip).limit(limit).all()
    return TranscriptionList(items=[
        TranscriptionResponse(
            id=t.id,
            text=t.transcription,
            audio_url=t.s3_url,
            language=t.language,
            duration=t.duration,
            filename=t.original_filename
        ) for t in transcriptions
    ])


@router.get("/transcription/{transcription_id}", response_model=TranscriptionResponse)
def get_transcription(transcription_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    transcription = db.query(AudioTranscription).filter(AudioTranscription.id == transcription_id).first()
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")

    return TranscriptionResponse(
        id=transcription.id,
        text=transcription.transcription,
        audio_url=transcription.s3_url,
        language=transcription.language,
        duration=transcription.duration,
        filename=transcription.original_filename
    )