import os
import logging
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints.transcriptions import router as transcriptions_router
from app.api.v1.endpoints.chat_sessions import router as chat_sessions_router
from app.api.auth.auth import router as auth_router
from app.models.audio_model import Base
from app.db.database import engine
import secrets
from starlette.middleware.sessions import SessionMiddleware

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Audio Transcription API",
    description="REST API for Kazakh audio transcription using faster-whisper",
    version="1.0.0"
)

# Добавляем поддержку сессий
app.add_middleware(
    SessionMiddleware, 
    secret_key=secrets.token_urlsafe(32),
    session_cookie="audio_transcription_session"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    transcriptions_router,
    prefix="/api/v1",
    tags=["transcription"]
)

app.include_router(
    chat_sessions_router,
    prefix="/api/v1/chat",
    tags=["chat"]
)

# Добавляем роутер авторизации
app.include_router(
    auth_router,
    prefix="/api/v1/auth",
    tags=["authentication"]
)

# Create .env file or set these environment variables
if "S3_ACCESS_KEY" not in os.environ:
    logging.warning("S3 credentials not found in environment variables. Set them before running in production.")

@app.get("/")
async def root():
    return {"message": "Audio Transcription API is running"}