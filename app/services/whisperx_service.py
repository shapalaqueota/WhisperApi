import whisperx
import torch
import gc
import os
from typing import Dict, Any, Optional


def transcribe_with_whisperx(
        audio_path: str,
        language: str = "kk",
        task: str = "transcribe",
        compute_type: str = "int8",
        batch_size: int = 16,
        diarize: bool = False,
        align_words: bool = True
) -> Dict[str, Any]:
    # Проверяем доступность CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")

    # Загружаем основную модель
    model = whisperx.load_model(
        "large-v2", device, compute_type=compute_type, language=language
    )

    # Транскрибируем аудио
    result = model.transcribe(
        audio_path,
        batch_size=batch_size,
        language=language,
        task=task
    )

    # Выравнивание на уровне слов, если включено
    if align_words and result["segments"]:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio_path,
            device
        )
        del align_model

    # Диаризация (определение говорящих), если включено
    if diarize and result["segments"]:
        try:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=os.environ.get("HF_TOKEN"),
                device=device
            )
            diarize_segments = diarize_model(audio_path)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            print(f"Ошибка диаризации: {e}")

    # Очистка памяти
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Форматируем результаты для совместимости с исходным API
    text = " ".join([segment["text"] for segment in result["segments"]])

    return {
        "text": text,
        "language": result.get("language"),
        "segments": result.get("segments", []),
        "word_segments": result.get("word_segments", []),
        "duration": result.get("duration", None)
    }