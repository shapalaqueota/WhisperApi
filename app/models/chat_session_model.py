from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.models.audio_model import Base

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(255), default="Новый чат")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Отношения
    user = relationship("User", back_populates="chat_sessions")
    transcriptions = relationship("ChatTranscription", back_populates="chat_session", cascade="all, delete-orphan")

class ChatTranscription(Base):
    __tablename__ = "chat_transcriptions"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    audio_transcription_id = Column(Integer, ForeignKey("audio_transcriptions.id"), nullable=True)
    message = Column(Text)
    is_system = Column(Integer, default=0)  # 0 - пользовательское сообщение, 1 - системное
    created_at = Column(DateTime, default=datetime.now)

    # Отношения
    chat_session = relationship("ChatSession", back_populates="transcriptions")
    audio_transcription = relationship("AudioTranscription")