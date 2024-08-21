from fastapi import FastAPI
from app.api.v1.endpoints import transcriptions

app = FastAPI(
    title="Whisper API",
    version="1.0.0",
)


app.include_router(transcriptions.router, prefix="/api/v1/transcriptions", tags=["Transcriptions"])