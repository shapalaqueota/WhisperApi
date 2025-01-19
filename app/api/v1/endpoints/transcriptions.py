from fastapi import APIRouter, UploadFile, File, HTTPException, status, WebSocket
from app.services.whisper_service import transcribe_audio_with_progress
import os
import librosa
import uuid
from tempfile import NamedTemporaryFile


router = APIRouter()

transcriptions_store = {}

@router.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()  # Ожидание данных от клиента
            await websocket.send_text("Processing your audio...")
    except Exception as e:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason=str(e))

@router.post("/")
async def transcribe(file: UploadFile = File(...)):
    try:
        transcription_id = str(uuid.uuid4())

        temp_file = NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            temp_file.write(await file.read())
            temp_file.close()

            y, sr = librosa.load(temp_file.name, sr=16000)

            transcription = await transcribe_audio_with_progress(y, sr)

            transcriptions_store[transcription_id] = transcription

            return {"transcription_id": transcription_id}
        finally:
            os.unlink(temp_file.name)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while transcribing audio: {e}"
        )
