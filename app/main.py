from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import transcriptions

app = FastAPI(
    title="Whisper API",
    version="12.0.0",
    description="Whisper API HUI",
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить запросы с любых источников
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST, и т.д.)
    allow_headers=["*"],  # Разрешить все заголовки
)

# Подключение маршрутов
app.include_router(transcriptions.router, prefix="/api/v1/transcriptions", tags=["Transcriptions"])
