from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.whisper_service import transcribe_audio
import io
import librosa

router = APIRouter()

@router.post("/")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio, sr=None)

        transcription = await transcribe_audio(y, sr)
        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while transcribing audio: {e}"
        )
