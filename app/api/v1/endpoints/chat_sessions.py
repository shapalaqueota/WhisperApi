from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models.user_model import User
from app.services.chat_service import (
    create_chat_session, get_chat_sessions, get_chat_session, add_transcription_to_chat, get_chat_history
)
from app.api.auth.auth import get_current_user
from app.api.v1.endpoints.transcriptions import transcribe_audio_file
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class ChatSessionResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    id: int
    message: str
    is_system: int
    created_at: datetime
    audio_url: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    chat_id: int
    title: str
    messages: List[ChatMessageResponse]


@router.post("/sessions", response_model=ChatSessionResponse)
def create_session(
        title: str = "Новый чат",
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    session = create_chat_session(db, current_user.id, title)
    return session


@router.get("/sessions", response_model=List[ChatSessionResponse])
def get_sessions(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    return get_chat_sessions(db, current_user.id, skip, limit)


@router.get("/sessions/{session_id}", response_model=ChatHistoryResponse)
def get_session_history(
        session_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    session = get_chat_session(db, session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Чат-сессия не найдена")

    history = get_chat_history(db, session_id, current_user.id)
    messages = []

    for msg in history:
        audio_url = None
        if msg.audio_transcription:
            audio_url = msg.audio_transcription.s3_url

        messages.append(ChatMessageResponse(
            id=msg.id,
            message=msg.message,
            is_system=msg.is_system,
            created_at=msg.created_at,
            audio_url=audio_url
        ))

    return ChatHistoryResponse(
        chat_id=session.id,
        title=session.title,
        messages=messages
    )


@router.post("/sessions/{session_id}/transcribe")
async def transcribe_in_session(
        session_id: int,
        file: UploadFile = File(...),
        language: str = Form("kk"),
        task: str = Form("transcribe"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    # Проверяем, существует ли сессия
    session = get_chat_session(db, session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Чат-сессия не найдена")

    # Транскрибируем аудио
    transcription_result = await transcribe_audio_file(file, language, task, db, current_user)

    # Добавляем результат в чат
    add_transcription_to_chat(
        db,
        session_id,
        transcription_result.id,
        transcription_result.text
    )

    # Обновляем время последней активности сессии
    session.updated_at = datetime.now()
    db.commit()

    return transcription_result