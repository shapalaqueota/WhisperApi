from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.whisper_service import transcribe_audio
import os
import io
import librosa
import uuid
from tempfile import NamedTemporaryFile

router = APIRouter()

# Хранилище для временных результатов транскрипции
transcriptions_store = {}


@router.post("/")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Генерация уникального ID для транскрипции
        transcription_id = str(uuid.uuid4())

        # Сохранение файла во временную папку
        temp_file = NamedTemporaryFile(delete=False, suffix=".wav")
        try:
            temp_file.write(await file.read())
            temp_file.close()

            # Загрузка и обработка аудиофайла
            y, sr = librosa.load(temp_file.name, sr=16000)

            # Транскрипция
            transcription = await transcribe_audio(y, sr)

            # Сохранение результата транскрипции во временное хранилище
            transcriptions_store[transcription_id] = transcription

            return {"transcription_id": transcription_id}
        finally:
            # Удаление временного файла
            os.unlink(temp_file.name)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while transcribing audio: {e}"
        )


@router.get("/{transcription_id}")
async def get_transcription(transcription_id: str):
    try:
        # Получение результата транскрипции по ID
        transcription = transcriptions_store.get(transcription_id)

        if not transcription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transcription not found"
            )

        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while retrieving transcription: {e}"
        )
