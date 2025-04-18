from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
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
    formatted_text: Optional[str] = None  # Текст с указанием спикеров
    speakers: Optional[List[str]] = None


class TranscriptionList(BaseModel):
    items: List[TranscriptionResponse]


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio_file(
        file: UploadFile = File(...),
        language: str = Form("kk"),
        task: str = Form("transcribe"),
        enable_diarization: bool = Form(True),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    try:
        file_info = await storage_service.upload_file(file)

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            await file.seek(0)
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Передаем параметр enable_diarization
        result = transcribe_audio(temp_path, language=language, task=task, enable_diarization=enable_diarization)

        os.unlink(temp_path)

        # Сохраняем данные диаризации, если они есть
        segments_data = result.get("segments")
        formatted_text = result.get("formatted_text")
        speakers_list = result.get("speakers")

        transcription = AudioTranscription(
            original_filename=file_info["original_filename"],
            s3_filename=file_info["s3_filename"],
            s3_url=file_info["s3_url"],
            file_size=file_info["size"],
            duration=result.get("duration"),
            language=result.get("language"),
            transcription=result["text"],
            formatted_transcription=formatted_text,
            speakers=speakers_list,
            diarization_data=segments_data
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
            segments=transcription.diarization_data,
            formatted_text=transcription.formatted_transcription,
            speakers=transcription.speakers
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
            filename=t.original_filename,
            segments=t.diarization_data
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
        filename=transcription.original_filename,
        segments=transcription.diarization_data,
        formatted_text=transcription.formatted_transcription,
        speakers=transcription.speakers
    )