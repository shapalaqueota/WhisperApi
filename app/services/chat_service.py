# app/services/chat_service.py
from sqlalchemy.orm import Session
from app.models.chat_session_model import ChatSession, ChatTranscription
from app.models.audio_model import AudioTranscription
from datetime import datetime
from typing import List, Optional


def create_chat_session(db: Session, user_id: int, title: str = "Новый чат") -> ChatSession:
    chat_session = ChatSession(user_id=user_id, title=title)
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)
    return chat_session


def get_chat_sessions(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[ChatSession]:
    return db.query(ChatSession).filter(ChatSession.user_id == user_id).offset(skip).limit(limit).all()


def get_chat_session(db: Session, session_id: int, user_id: int) -> Optional[ChatSession]:
    return db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()


def add_transcription_to_chat(
        db: Session,
        chat_session_id: int,
        audio_transcription_id: int,
        message: str
) -> ChatTranscription:
    chat_message = ChatTranscription(
        chat_session_id=chat_session_id,
        audio_transcription_id=audio_transcription_id,
        message=message
    )
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)
    return chat_message


def get_chat_history(db: Session, chat_session_id: int, user_id: int) -> List[ChatTranscription]:
    # Проверяем, принадлежит ли чат пользователю
    chat = get_chat_session(db, chat_session_id, user_id)
    if not chat:
        return []

    return db.query(ChatTranscription).filter(
        ChatTranscription.chat_session_id == chat_session_id
    ).order_by(ChatTranscription.created_at).all()