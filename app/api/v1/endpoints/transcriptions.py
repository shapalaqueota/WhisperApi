# app/api/v1/endpoints/transcriptions.py

import os
import tempfile
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.services.storage_service import storage_service
from app.services.diarization_service import diarize_file
from app.services.whisper_service import transcribe_full, transcribe_segment
from app.services.emotion_service import detect_emotion
from app.services.polishing_service import polish_text, polish_segments
from app.db.database import get_db
from app.models.audio_model import AudioTranscription
from app.api.auth.auth import get_current_user

router = APIRouter(
    prefix="/api/v1",
    tags=["transcriptions"]
)

logger = logging.getLogger(__name__)


class SegmentOut(BaseModel):
    start: float
    end: float
    speaker: str
    text: str
    emotion: str
    polished_text: str


class TranscriptionResponse(BaseModel):
    id: int
    text: str
    audio_url: str
    language: str
    duration: float
    filename: str
    segments: list[SegmentOut]
    formatted_text: str
    speakers: list[str]
    overall_emotion: str
    polished_text: str


async def transcribe_audio_file(
    file: UploadFile,
    language: str = "kk",
    task: str = "transcribe",
    enable_diarization: bool = True,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
) -> TranscriptionResponse:
    # 1. Проверка
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        raise HTTPException(status_code=400, detail="Only audio files allowed")

    # 2. Загрузка
    file_info = await storage_service.upload_file(file)

    # 3. Временный файл
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        await file.seek(0)
        tmp.write(await file.read())
        temp_path = tmp.name

    segments_data = []
    speakers = []
    overall_emotion = ""
    formatted = ""
    full_text = ""
    full_polished = ""

    # 4. Диаризация или целиком
    if enable_diarization:
        raw = diarize_file(temp_path)
        for seg in raw:
            if seg['speaker'] not in speakers:
                speakers.append(seg['speaker'])
            txt = transcribe_segment(temp_path, seg['start'], seg['end'], task=task)
            emo = detect_emotion(temp_path)
            segments_data.append({
                "start": seg['start'],
                "end": seg['end'],
                "speaker": seg['speaker'],
                "text": txt,
                "emotion": emo
            })

        formatted = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments_data)
        full_text = " ".join(s['text'] for s in segments_data)

        # 5. Полировка по сегментам
        polished_segments = polish_segments(segments_data, language)
        full_polished = " ".join(s['polished_text'] for s in polished_segments)
        segments_data = polished_segments

    else:
        full_text = transcribe_full(temp_path, task=task)
        formatted = full_text
        overall_emotion = detect_emotion(temp_path)
        full_polished = polish_text(full_text, language)
        speakers = []
        segments_data = []

    os.unlink(temp_path)

    # 6. Сохранение
    transcription = AudioTranscription(
        original_filename       = file_info["original_filename"],
        s3_filename             = file_info["s3_filename"],
        s3_url                  = file_info["s3_url"],
        file_size               = file_info["size"],
        duration                = None,
        language                = language,
        transcription           = full_text,
        formatted_transcription = formatted,
        speakers                = speakers,
        diarization_data        = segments_data,
        overall_emotion         = overall_emotion,
        polished_text           = full_polished
    )
    db.add(transcription)
    db.commit()
    db.refresh(transcription)

    return TranscriptionResponse(
        id               = transcription.id,
        text             = transcription.transcription,
        audio_url        = transcription.s3_url,
        language         = transcription.language,
        duration         = transcription.duration or 0.0,
        filename         = transcription.original_filename,
        segments         = transcription.diarization_data,
        formatted_text   = transcription.formatted_transcription,
        speakers         = transcription.speakers,
        overall_emotion  = transcription.overall_emotion,
        polished_text    = transcription.polished_text
    )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str = Form("kk"),
    task: str = Form("transcribe"),
    enable_diarization: bool = Form(True),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    return await transcribe_audio_file(
        file, language, task, enable_diarization, db, current_user
    )


async def transcribe_audio_demo(
    file: UploadFile,
    language: str = "kk",
    task: str = "transcribe",
    enable_diarization: bool = True
) -> TranscriptionResponse:
    # Копия функции transcribe_audio_file без сохранения в БД
    
    # 1. Проверка формата файла
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        raise HTTPException(status_code=400, detail="Only audio files allowed")

    # 2. Загрузка
    file_info = await storage_service.upload_file(file)
    
    # 3-5. Обработка аудио (как в оригинальной функции)
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        await file.seek(0)
        tmp.write(await file.read())
        temp_path = tmp.name

    segments_data = []
    speakers = []
    overall_emotion = ""
    formatted = ""
    full_text = ""
    full_polished = ""

    # 4. Диаризация или целиком
    if enable_diarization:
        raw = diarize_file(temp_path)
        for seg in raw:
            if seg['speaker'] not in speakers:
                speakers.append(seg['speaker'])
            txt = transcribe_segment(temp_path, seg['start'], seg['end'], task=task)
            emo = detect_emotion(temp_path)
            segments_data.append({
                "start": seg['start'],
                "end": seg['end'],
                "speaker": seg['speaker'],
                "text": txt,
                "emotion": emo
            })

        formatted = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments_data)
        full_text = " ".join(s['text'] for s in segments_data)

        # 5. Полировка по сегментам
        polished_segments = polish_segments(segments_data, language)
        full_polished = " ".join(s['polished_text'] for s in polished_segments)
        segments_data = polished_segments

    else:
        full_text = transcribe_full(temp_path, task=task)
        formatted = full_text
        overall_emotion = detect_emotion(temp_path)
        full_polished = polish_text(full_text, language)
        speakers = []
        segments_data = []

    os.unlink(temp_path)

    # Возвращаем ответ без сохранения в БД
    return TranscriptionResponse(
        id=0,  # Демо ID
        text=full_text,
        audio_url=file_info["s3_url"],
        language=language,
        duration=0.0,  # Демо значение
        filename=file_info["original_filename"],
        segments=segments_data,
        formatted_text=formatted,
        speakers=speakers,
        overall_emotion=overall_emotion,
        polished_text=full_polished
    )


@router.post("/transcribe-demo", response_model=TranscriptionResponse)
async def transcribe_demo_endpoint(
    file: UploadFile = File(...),
    language: str = Form("kk"),
    task: str = Form("transcribe"),
    enable_diarization: bool = Form(True)
):
    return await transcribe_audio_demo(
        file, language, task, enable_diarization
    )

