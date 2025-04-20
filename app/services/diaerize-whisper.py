# main.py
import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline
from pyannote.audio import Pipeline

app = FastAPI(title="ASR + Diarization Service")

# 1) Инициализируем пайплайн ASR с вашей fine-tuned моделью
asr = pipeline(
    "automatic-speech-recognition",
    model="your-username/your-fine-tuned-whisper",  # ← замените на свой репозиторий
    chunk_length_s=30,           # при необходимости разбиваем длинные файлы
    stride_length_s=(5, 5)       # перекрытие между чанками
)

# 2) Инициализируем пайплайн спикер-диаризации
diarization = Pipeline.from_pretrained("pyannote/speaker-diarization")


@app.post("/transcribe-alan")
async def transcribe(audio_file: UploadFile = File(...)):
    # Проверка типа файла
    if audio_file.content_type not in (
        "audio/wav", "audio/mpeg", "audio/mp3", "audio/flac"
    ):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Сохраняем во временный файл
    suffix = os.path.splitext(audio_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio_file.read())
        tmp_path = tmp.name

    try:
        # 3) Спикер-диаризация
        diariza = diarization(tmp_path)

        # 4) Транскрипция
        result = asr(tmp_path)
        text = result["text"] if isinstance(result, dict) else str(result)

        # 5) Собираем список сегментов
        segments = []
        for turn, _, speaker in diariza.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })

        return {
            "transcription": text,
            "segments": segments
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        # Чистим временный файл
        os.remove(tmp_path)
